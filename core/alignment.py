import os
import gc
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from types import SimpleNamespace
import numpy as np
from core.models import StepResult

# whisperx ve torch yalnızca hizalama alt sürecinde import edilir (bkz. align_lyrics).

ALIGN_SR = 16000

# Hizalama modeli sesi bu uzunlukta pencerelerle işler; her pencerenin iki yanına
# CONTEXT_SEC kadar bağlam eklenir ve sınır bölgeleri atılır (kelimeler kesilmesin diye).
CHUNK_SEC = 30.0
CONTEXT_SEC = 3.0


def _frame_geometry(model) -> tuple[int, int]:
    """
    Modelin bir çıktı karesinin kaç örnek ilerlediğini (hop) ve kaç örneğe
    baktığını (field) döner. wav2vec2 için 320 / 400 (16 kHz'de 20 ms / 25 ms).
    torchaudio modellerinde config yok; onlar da aynı varsayılan yapıyı kullanır.
    """
    config = getattr(model, "config", None)
    kernels = getattr(config, "conv_kernel", None) or (10, 3, 3, 3, 3, 2, 2)
    strides = getattr(config, "conv_stride", None) or (5, 2, 2, 2, 2, 2, 2)
    hop, field = 1, 1
    for kernel, stride in zip(kernels, strides):
        field += (kernel - 1) * hop
        hop *= stride
    return hop, field


class _ChunkedAlignModel:
    """
    Hizalama modelini (wav2vec2) sesin tamamı yerine örtüşen pencerelerle çalıştırır
    ve karelik harf olasılıklarını tek bir matriste birleştirir.

    Modelin bellek ihtiyacı girdi uzunluğuyla hızla büyüdüğü için tüm şarkıyı tek
    seferde vermek 14 GB'ı aşabiliyor. Pencereleme sadece bu akustik adıma uygulanır:
    sözlerin zamana yerleştirilmesi whisperx.align içinde yine tüm şarkı ve tüm sözler
    üzerinden tek seferde yapılır, sözleri pencerelere dağıtmak gerekmez.

    whisperx.align modeli yalnızca `model(ses)` şeklinde çağırdığı için bu sınıf
    gerçek modelin yerine geçer.
    """

    def __init__(self, model, model_type: str):
        self.model = model
        self.model_type = model_type
        self.hop, self.field = _frame_geometry(model)
        self.chunk_frames = int(CHUNK_SEC * ALIGN_SR) // self.hop
        self.context_frames = int(CONTEXT_SEC * ALIGN_SR) // self.hop

    def _run(self, waveform, lengths=None):
        if self.model_type == "torchaudio":
            emissions, _ = self.model(waveform, lengths=lengths)
            return emissions
        return self.model(waveform).logits

    def __call__(self, waveform, lengths=None):
        logits = self._chunked_logits(waveform, lengths)
        if self.model_type == "torchaudio":
            return logits, None
        return SimpleNamespace(logits=logits)

    def _chunked_logits(self, waveform, lengths=None):
        import torch

        n_samples = waveform.shape[-1]
        window_frames = self.chunk_frames + 2 * self.context_frames
        if n_samples <= (window_frames - 1) * self.hop + self.field:
            return self._run(waveform, lengths)

        # Pencereler kare ızgarasına hizalı başlar; böylece pencerenin k. karesi
        # tam olarak tüm şarkının (başlangıç + k). karesine denk gelir.
        total_frames = (n_samples - self.field) // self.hop + 1
        parts = []
        for f0 in range(0, total_frames, self.chunk_frames):
            f1 = min(f0 + self.chunk_frames, total_frames)
            w0 = max(0, f0 - self.context_frames)
            w1 = min(total_frames, f1 + self.context_frames)
            window = waveform[..., w0 * self.hop:(w1 - 1) * self.hop + self.field]
            parts.append(self._run(window)[:, f0 - w0:f1 - w0])
        return torch.cat(parts, dim=1)


def align_lyrics(
    audio_path: str,
    lyrics_lines: list[str],
    language_code: str = "tr",
    device: str = "cpu",
    compute_type: str = "int8",
) -> StepResult:
    """
    Verilen lyrics satırlarını audio_path ses dosyasıyla hizalar.

    Hizalama ayrı bir alt süreçte çalışır: PyTorch hesaplama sırasında ayırdığı
    belleği (~3 GB) model silinse bile süreç içinde tutuyor ve aynı süreçte sonra
    gelen LLM adımına yer kalmıyordu. Alt süreç bitince bellek tamamen işletim
    sistemine döner; yerel bir çökme de çağıran programı (main.py / API) düşürmez.
    Alt süreç "spawn" ile başlatıldığı için çağıran script'in ana kodu
    `if __name__ == "__main__":` koruması altında olmalıdır.
    """
    try:
        with ProcessPoolExecutor(max_workers=1, mp_context=multiprocessing.get_context("spawn")) as pool:
            return pool.submit(
                _align_lyrics_in_process, audio_path, lyrics_lines, language_code, device
            ).result()
    except BrokenProcessPool as e:
        return StepResult(
            success=False,
            message="Hizalama süreci beklenmedik şekilde kapandı (bellek yetersiz olabilir).",
            error=str(e)
        )


def _align_lyrics_in_process(
    audio_path: str,
    lyrics_lines: list[str],
    language_code: str,
    device: str,
) -> StepResult:
    """
    align_lyrics'in alt süreçte çalışan asıl işi. Akustik model sesi pencerelerle
    işler (bkz. _ChunkedAlignModel); bellek kullanımı şarkı uzunluğundan bağımsızdır.
    """
    try:
        import whisperx

        # whisperx.load_audio zaten 16 kHz mono float32 döner — orijinal dosyaya dokunmaz
        audio = whisperx.load_audio(audio_path)
        duration = len(audio) / ALIGN_SR

        full_text = " ".join(lyrics_lines)
        segments = [{"text": full_text, "start": 0.0, "end": duration}]

        model_a, metadata = whisperx.load_align_model(
            language_code=language_code,
            device=device,
        )
        
        result = whisperx.align(
            segments,
            _ChunkedAlignModel(model_a, metadata["type"]),
            metadata,
            audio,
            device,
            return_char_alignments=False,
        )
        word_segments = result.get("word_segments", [])

        return StepResult(
            success=True,
            data={"word_segments": word_segments, "duration": duration},
            message=f"{len(word_segments)} kelime hizalandı. Ses süresi: {duration:.1f}s"
        )
    except Exception as e:
        return StepResult(
            success=False,
            message="WhisperX hizalaması sırasında hata oluştu.",
            error=str(e)
        )


def build_line_timestamps(lyrics_lines: list[str], word_segments: list[dict]) -> StepResult:
    """
    word_segments'ten satır bazlı timestamp listesi oluşturur.
    """
    try:
        result = []
        word_idx = 0
        for line in lyrics_lines:
            words_in_line = line.split()
            line_start = None
            line_end   = None
            line_words = []

            for word in words_in_line:
                if word_idx < len(word_segments):
                    w = word_segments[word_idx]
                    w_start = w.get("start", None)
                    w_end   = w.get("end",   None)
                    if w_start is not None and line_start is None:
                        line_start = float(w_start)
                    if w_end is not None:
                        line_end = float(w_end)
                    line_words.append({
                        "word":  w.get("word", word),
                        "start": float(w_start) if w_start is not None else 0.0,
                        "end":   float(w_end)   if w_end   is not None else 0.0,
                        "score": float(w.get("score", 0.0)),
                    })
                word_idx += 1

            result.append({
                "line":  line,
                "start": line_start if line_start is not None else 0.0,
                "end":   line_end   if line_end   is not None else 0.0,
                "words": line_words,
            })
        return StepResult(
            success=True,
            data={"line_timestamps": result},
            message=f"Satır bazlı timestamp listesi oluşturuldu: {len(result)} satır."
        )
    except Exception as e:
        return StepResult(
            success=False,
            message="Satır bazlı timestamp listesi oluşturulurken hata oluştu.",
            error=str(e)
        )
    finally:
        gc.collect()
