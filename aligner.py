"""
aligner.py
----------
WhisperX ile şarkı sözlerini ses dosyasına timestamp'ler.
Sistem kaynağını korumak için hizalama öncesi sesi düşük örnekleme hızında yükler,
ancak orijinal dosyayı dokunmadan bırakır.
"""

import os
import tempfile
import whisperx
import soundfile as sf
import numpy as np


# Alignment için düşük örnekleme hızı (sistem kaynağı tasarrufu)
ALIGN_SR = 16000  # WhisperX zaten 16 kHz bekler — native, no downsampling waste


def align_lyrics(
    audio_path: str,
    lyrics_lines: list[str],
    language_code: str = "en",
    device: str = "cpu",
    compute_type: str = "int8",
) -> list[dict]:
    """
    Verilen lyrics satırlarını audio_path ses dosyasıyla hizalar.

    Returns:
        word_segments: [{"word": str, "start": float, "end": float, "score": float}, ...]
    """
    print(f"  [align] Ses yükleniyor (16 kHz, CPU): {os.path.basename(audio_path)}")
    # whisperx.load_audio zaten 16 kHz mono float32 döner — orijinal dosyaya dokunmaz
    audio = whisperx.load_audio(audio_path)
    duration = len(audio) / ALIGN_SR
    print(f"  [align] Ses süresi: {duration:.1f}s")

    full_text = " ".join(lyrics_lines)
    segments = [{"text": full_text, "start": 0.0, "end": duration}]

    print(f"  [align] Alignment modeli yükleniyor (dil={language_code})…")
    model_a, metadata = whisperx.load_align_model(
        language_code=language_code,
        device=device,
    )
    print("  [align] Hizalama yapılıyor…")
    result = whisperx.align(
        segments,
        model_a,
        metadata,
        audio,
        device,
        return_char_alignments=False,
    )
    word_segments = result.get("word_segments", [])
    print(f"  [align] {len(word_segments)} kelime hizalandı.")
    return word_segments


def build_line_timestamps(lyrics_lines: list[str], word_segments: list[dict]) -> list[dict]:
    """
    word_segments'ten satır bazlı timestamp listesi oluşturur.

    Returns:
        [{"line": str, "start": float, "end": float, "words": [...]}, ...]
    """
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
    return result
