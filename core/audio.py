import os
import gc
import librosa
import numpy as np
from pydub import AudioSegment
from core.models import StepResult

def bpm_to_crossfade_ms(bpm: float) -> int:
    beat_ms = 60_000.0 / max(bpm, 1.0)
    cf = int(beat_ms * 0.5)
    return max(80, min(cf, 600))

def _beat_before(timestamp_sec: float, beat_times: np.ndarray) -> float:
    """Verilen zamandan ÖNCE (strict) gelen en yakın beat'i döner."""
    before = beat_times[beat_times < timestamp_sec]
    if len(before) == 0:
        return timestamp_sec
    return float(before[-1])

def _beat_after(timestamp_sec: float, beat_times: np.ndarray) -> float:
    """Verilen zamandan SONRA (strict) gelen en yakın beat'i döner."""
    after = beat_times[beat_times > timestamp_sec]
    if len(after) == 0:
        return timestamp_sec
    return float(after[0])

def apply_cuts(
    input_file: str,
    output_file: str,
    cuts: list[list[float]],
) -> StepResult:
    """
    Orijinal ses dosyasını kullanarak verilen [start, end] aralıklarını
    BPM-aware crossfade ile keser ve output_file'a kaydeder.
    """
    y = None
    audio = None
    result = None
    try:
        y, sr = librosa.load(input_file, sr=None, mono=True)
        tempo_arr, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
        bpm = float(np.squeeze(tempo_arr)) if hasattr(tempo_arr, '__len__') else float(tempo_arr)
        beat_times = librosa.frames_to_time(beat_frames, sr=sr)
        crossfade_ms = bpm_to_crossfade_ms(bpm)

        audio = AudioSegment.from_file(input_file)

        sorted_cuts = sorted(cuts, key=lambda c: c[0])

        merged: list[list[float]] = []
        for start, end in sorted_cuts:
            cut_start = _beat_before(start, beat_times) if len(beat_times) else start
            cut_end   = _beat_after(end,    beat_times) if len(beat_times) else end
            if merged and cut_start <= merged[-1][1]:
                merged[-1][1] = max(merged[-1][1], cut_end)
            else:
                merged.append([cut_start, cut_end])

        if not merged:
            os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)
            audio.export(output_file, format="mp3")
            return StepResult(
                success=True,
                data={"output_file": output_file, "cuts_applied": 0},
                message=f"Hiçbir kesim uygulanmadı. Orijinal dosya kopyalandı: {output_file}"
            )

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

        return StepResult(
            success=True,
            data={"output_file": output_file, "cuts_applied": len(merged), "bpm": bpm, "crossfade_ms": crossfade_ms},
            message=f"Şarkı kesildi ve kaydedildi ({len(merged)} aralık): {output_file}"
        )
    except Exception as e:
        return StepResult(
            success=False,
            message="Ses dosyası kesilirken hata oluştu.",
            error=str(e)
        )
    finally:
        if y is not None:
            del y
        if audio is not None:
            del audio
        if result is not None:
            del result
        gc.collect()

def find_flagged_timestamps(
    flagged: list[str],
    word_segments: list[dict],
) -> StepResult:
    """
    LLM'in işaretlediği kelime/kelime-gruplarını word_segments içinde arayıp
    [[start, end], ...] formatında döner.
    """
    try:
        results: list[list[float]] = []
        warnings: list[str] = []

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
            if not found:
                warnings.append(phrase)

        return StepResult(
            success=True,
            data={"cuts": results, "unmatched_phrases": warnings},
            message=f"{len(results)} uygunsuz içerik aralığı bulundu." + (f" Eşleşmeyenler: {warnings}" if warnings else "")
        )
    except Exception as e:
        return StepResult(
            success=False,
            message="Timestamp aralıkları aranırken hata oluştu.",
            error=str(e)
        )
    finally:
        gc.collect()
