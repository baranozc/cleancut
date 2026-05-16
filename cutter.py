"""
cutter.py
---------
crossfader.py mantığını modül olarak saran, BPM'e göre crossfade
süresini otomatik ayarlayan kesme motoru.
"""

import os
import librosa
import numpy as np
from pydub import AudioSegment


# ── BPM → crossfade ms dönüşüm tablosu ──────────────────────────────────────
# Bir beat'in milisaniye cinsinden süresi = 60000 / bpm
# Crossfade ≈ yarım beat kadar tutulur; minimum 80 ms, maksimum 600 ms.
def bpm_to_crossfade_ms(bpm: float) -> int:
    beat_ms = 60_000.0 / max(bpm, 1.0)
    cf = int(beat_ms * 0.5)
    return max(80, min(cf, 600))


def _find_nearest_beat(timestamp_sec: float, beat_times: np.ndarray) -> float:
    idx = np.argmin(np.abs(beat_times - timestamp_sec))
    return float(beat_times[idx])


def apply_cuts(
    input_file: str,
    output_file: str,
    cuts: list[list[float]],  # [[start1, end1], [start2, end2], ...]
) -> None:
    """
    Orijinal ses dosyasını kullanarak verilen [start, end] aralıklarını
    BPM-aware crossfade ile keser ve output_file'a kaydeder.

    Args:
        input_file:  Orijinal (yüksek kaliteli) ses dosyası
        output_file: Çıktı dosyası (.mp3)
        cuts:        Kesilecek aralıkların listesi [[s1,e1], [s2,e2], ...]
    """
    print(f"  [cut] Orijinal ses yükleniyor: {os.path.basename(input_file)}")
    y, sr = librosa.load(input_file, sr=None, mono=True)
    tempo_arr, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
    bpm = float(np.squeeze(tempo_arr)) if hasattr(tempo_arr, '__len__') else float(tempo_arr)
    beat_times = librosa.frames_to_time(beat_frames, sr=sr)
    crossfade_ms = bpm_to_crossfade_ms(bpm)

    print(f"  [cut] BPM: {bpm:.1f} → crossfade: {crossfade_ms} ms")
    print(f"  [cut] {len(cuts)} kesim uygulanacak…")

    # PyDub ile yükle (ms cinsinden)
    audio = AudioSegment.from_file(input_file)

    # Kesimler küçükten büyüğe sıralı olmalı, iç içe geçmemeli
    sorted_cuts = sorted(cuts, key=lambda c: c[0])

    # Örtüşen aralıkları birleştir
    merged: list[list[float]] = []
    for start, end in sorted_cuts:
        # Beat hizalaması
        cut_start = _find_nearest_beat(start, beat_times) if len(beat_times) else start
        cut_end   = _find_nearest_beat(end,   beat_times) if len(beat_times) else end
        if merged and cut_start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], cut_end)
        else:
            merged.append([cut_start, cut_end])

    # Parçaları kes
    result = audio[:int(merged[0][0] * 1000)]
    for i, (cs, ce) in enumerate(merged):
        after_start = int(ce * 1000)
        if i + 1 < len(merged):
            after_end = int(merged[i + 1][0] * 1000)
            chunk = audio[after_start:after_end]
        else:
            chunk = audio[after_start:]
        result = result.append(chunk, crossfade=crossfade_ms)

    os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)
    result.export(output_file, format="mp3")
    print(f"  [cut] Kaydedildi: {output_file}")


def find_flagged_timestamps(
    flagged: list[str],
    word_segments: list[dict],
) -> list[list[float]]:
    """
    LLM'in işaretlediği kelime/kelime-gruplarını word_segments içinde arayıp
    [[start, end], ...] formatında döner.

    Kelime grubu için art arda gelen kelimelerin başı ve sonu alınır.
    """
    results: list[list[float]] = []

    # Normalize: küçük harf, strip
    ws_words = [w.get("word", "").lower().strip() for w in word_segments]

    for phrase in flagged:
        phrase_words = phrase.lower().split()
        n = len(phrase_words)
        found = False

        for i in range(len(ws_words) - n + 1):
            window = ws_words[i:i + n]
            if window == phrase_words:
                seg_start = word_segments[i]
                seg_end   = word_segments[i + n - 1]
                t_start = seg_start.get("start", None)
                t_end   = seg_end.get("end",     None)
                if t_start is not None and t_end is not None:
                    results.append([float(t_start), float(t_end)])
                    found = True
                    # Aynı ifade birden fazla geçiyorsa hepsini al
        if not found:
            print(f"  [cut] UYARI – timestamp bulunamadı: '{phrase}'")

    return results
