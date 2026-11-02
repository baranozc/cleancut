import os
import gc
from pathlib import Path
import librosa
import numpy as np
from core.audio_io import decode, encode_like, probe
from core.models import DEFAULT_CUT_MODE, CutMode, StepResult
from core.text import find_sequence, normalize_word, normalize_words

# Kelime zamanlarındaki hizalama hatasına karşı küfrün iki yanında bırakılan pay.
SAFETY_SEC = 0.10
# Kesim noktasındaki geçiş süresi. Geçiş vuruştan hemen önce biter; gelen tarafın
# kullandığı kısa bölüm silinen aralığın sonundaki güvenlik payından alınır, bu yüzden
# SAFETY_SEC'ten kısa olmalı (küfrün kendisi hiçbir zaman duyulmaz).
XFADE_SEC = 0.05
assert XFADE_SEC < SAFETY_SEC

# Kesim şarkının başına/sonuna dayandığında kullanılan sanal vuruş indeksleri
_SONG_START = -1


def _plan_cuts(cuts: list[list[float]], beats: np.ndarray, duration: float, beats_per_cut: int) -> list[tuple[float, float]]:
    """
    Kelime aralıklarını vuruş ızgarasına oturtur. Her kesim bir vuruşta başlar ve bir
    vuruşta biter; silinen vuruş sayısı beats_per_cut'ın katıdır. Böylece kesimden sonraki
    vuruş, ölçüdeki aynı konuma düşer. Şarkının başına/sonuna dayanan kesimler o uçta
    vuruşa bağlı değildir.
    """
    song_end = len(beats)

    def to_time(i: int) -> float:
        if i == _SONG_START:
            return 0.0
        if i == song_end:
            return duration
        return float(beats[i])

    def round_up(i0: int, i1: int) -> list[int]:
        if i0 == _SONG_START or i1 == song_end:
            return [i0, i1]
        extra = -(i1 - i0) % beats_per_cut
        if i1 + extra < song_end:
            return [i0, i1 + extra]       # tercihen ileri doğru uzat
        if i0 - extra >= 0:
            return [i0 - extra, i1]
        return [i0, song_end]

    intervals = []
    for start, end in cuts:
        i0 = int(np.searchsorted(beats, start - SAFETY_SEC, side="right")) - 1  # son vuruş ≤ başlangıç - pay
        i1 = int(np.searchsorted(beats, end + SAFETY_SEC, side="left"))         # ilk vuruş ≥ bitiş + pay
        intervals.append([max(i0, _SONG_START), i1])

    # Uzatma yeni çakışmalar doğurabildiği için birleştirme + uzatma sonuç değişmeyene kadar tekrarlanır.
    while True:
        merged: list[list[int]] = []
        for i0, i1 in sorted(intervals):
            if merged and i0 <= merged[-1][1]:
                merged[-1][1] = max(merged[-1][1], i1)
            else:
                merged.append([i0, i1])
        snapped = [round_up(i0, i1) for i0, i1 in merged]
        if snapped == intervals:
            return [(to_time(i0), to_time(i1)) for i0, i1 in snapped]
        intervals = snapped


def _plan_cuts_without_beats(cuts: list[list[float]], duration: float) -> list[tuple[float, float]]:
    """Vuruş bulunamayan kayıtlarda (ör. a cappella) kelime aralıklarını güvenlik payıyla keser."""
    merged: list[list[float]] = []
    for start, end in sorted(cuts):
        cs, ce = max(0.0, start - SAFETY_SEC), min(duration, end + SAFETY_SEC)
        if merged and cs <= merged[-1][1] + XFADE_SEC:   # geçişler üst üste binmesin
            merged[-1][1] = max(merged[-1][1], ce)
        else:
            merged.append([cs, ce])
    return [(cs, ce) for cs, ce in merged]


def _render(y: np.ndarray, sr: int, plan: list[tuple[float, float]]) -> np.ndarray:
    """
    Kesimleri uygular. Kesimden sonraki vuruş tam olarak kesimin başladığı vuruşun yerine
    düşer (ritim ızgarası kaymaz). Geçiş, vuruştan hemen önceki XFADE_SEC içinde doğrusal
    crossfade'dir: gelen tarafın vuruşu (kick/snare) tam sesle çalar ve iki tarafın toplamı
    kaynak seviyesini aşmaz (yeni clipping oluşmaz).
    """
    n = len(y)
    xf = max(1, int(round(XFADE_SEC * sr)))
    ramp = np.linspace(0.0, 1.0, xf, endpoint=False, dtype=np.float32)[:, None]
    parts = []
    pos = 0                 # sıradaki korunan bölümün kaynaktaki başlangıcı
    fade_in = fade_out = False
    for cs, ce in plan:
        a, b = int(round(cs * sr)), min(n, int(round(ce * sr)))
        if a <= xf:         # şarkının başından kesiliyor: sonraki bölüm kısa fade-in ile başlar
            pos, fade_in = b, True
            continue
        if b >= n:          # şarkının sonuna kadar kesiliyor: kısa fade-out ile biter
            parts.append(y[pos:a])
            pos, fade_out = n, True
            break
        parts.append(y[pos:a - xf])
        parts.append(y[a - xf:a] * (1.0 - ramp) + y[b - xf:b] * ramp)
        pos = b
    parts.append(y[pos:])
    out = np.concatenate(parts)
    if fade_in and len(out) >= xf:
        out[:xf] *= ramp
    if fade_out and len(out) >= xf:
        out[-xf:] *= ramp[::-1]
    return out


def apply_cuts(
    input_file: str,
    output_file: str,
    cuts: list[list[float]],
    cut_mode: CutMode | str = DEFAULT_CUT_MODE,
) -> StepResult:
    """
    Verilen [start, end] (saniye) aralıklarını vuruş ızgarasına oturtarak keser ve
    output_file'a kaydeder. cut_mode silinen vuruş sayısının 2'nin (even_beats) ya da
    4'ün (full_bars) katı olacağını belirler.

    Çıktı kaynakla aynı formatta ve en az kaynağın kalitesinde yazılır, metadata korunur
    (bkz. core.audio_io). output_file'ın uzantısı kaynağınkiyle aynı değilse kaynağınki
    kullanılır; gerçek yol data["output_file"] içinde döner.
    """
    y = None
    out = None
    try:
        cut_mode = CutMode(cut_mode)
        output_file = str(Path(output_file).with_suffix(Path(input_file).suffix))

        info = probe(input_file)
        y = decode(input_file, info)
        duration = len(y) / info.sample_rate

        tempo_arr, beat_frames = librosa.beat.beat_track(y=y.mean(axis=1), sr=info.sample_rate)
        bpm = float(np.squeeze(tempo_arr))
        beat_times = librosa.frames_to_time(beat_frames, sr=info.sample_rate)

        if len(beat_times) >= 2:
            plan = _plan_cuts(cuts, beat_times, duration, cut_mode.beats)
        else:
            plan = _plan_cuts_without_beats(cuts, duration)
        out = _render(y, info.sample_rate, plan) if plan else y

        os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)
        encode_like(out, info, input_file, output_file)

        beat_sec = 60.0 / bpm if bpm > 0 else 0.0
        removed = sum(ce - cs for cs, ce in plan)
        return StepResult(
            success=True,
            data={
                "output_file": output_file,
                "cuts_applied": len(plan),
                "cut_mode": cut_mode.value,
                "cut_ranges": [[round(cs, 3), round(ce, 3)] for cs, ce in plan],
                "removed_beats": [round((ce - cs) / beat_sec) if beat_sec else None for cs, ce in plan],
                "removed_sec": round(removed, 3),
                "bpm": bpm,
                "crossfade_ms": XFADE_SEC * 1000,
            },
            message=(f"Şarkı kesildi ve kaydedildi ({len(plan)} aralık, {removed:.1f} sn, "
                     f"{cut_mode.label.lower()}): {output_file}") if plan else
                    f"Hiçbir kesim uygulanmadı, orijinal ses aynı kalitede yazıldı: {output_file}"
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
        if out is not None:
            del out
        gc.collect()

def find_flagged_timestamps(
    flagged: list[str],
    word_segments: list[dict],
) -> StepResult:
    """
    LLM'in işaretlediği kelime/kelime-gruplarını word_segments içinde arayıp
    [[start, end], ...] formatında döner. Karşılaştırmada noktalama ve büyük/küçük
    harf farkları yok sayılır ("piçin," ↔ "piçin"); bkz. core.text.normalize_word.
    """
    try:
        results: list[list[float]] = []
        warnings: list[str] = []

        # Yalnızca noktalamadan oluşan kelimeler atlanır; seg_idx orijinal sıraya bağlar.
        normalized = [(normalize_word(w.get("word", "")), i) for i, w in enumerate(word_segments)]
        normalized = [(word, i) for word, i in normalized if word]
        ws_words = [word for word, _ in normalized]
        seg_idx  = [i for _, i in normalized]

        for phrase in flagged:
            phrase_words = normalize_words(phrase)
            n = len(phrase_words)
            found = False

            for k in find_sequence(ws_words, phrase_words):
                seg_start = word_segments[seg_idx[k]]
                seg_end   = word_segments[seg_idx[k + n - 1]]
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
