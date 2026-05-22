import os
import gc
import whisperx
import numpy as np
from core.models import StepResult

try:
    import torch
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False

ALIGN_SR = 16000

def align_lyrics(
    audio_path: str,
    lyrics_lines: list[str],
    language_code: str = "tr",
    device: str = "cpu",
    compute_type: str = "int8",
) -> StepResult:
    """
    Verilen lyrics satırlarını audio_path ses dosyasıyla hizalar.
    """
    model_a = None
    metadata = None
    result = None
    audio = None
    try:
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
            model_a,
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
    finally:
        if model_a is not None:
            del model_a
        if metadata is not None:
            del metadata
        if result is not None:
            del result
        if audio is not None:
            del audio
        gc.collect()
        if _TORCH_AVAILABLE:
            try:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass

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
