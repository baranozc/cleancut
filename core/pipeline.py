from typing import Callable, Optional
from core.models import StepResult
from core.genius import fetch_and_save_lyrics
from core.alignment import align_lyrics, build_line_timestamps
from core.llm import check_profanity
from core.audio import find_flagged_timestamps, apply_cuts

def run_pipeline(
    audio_path: str,
    song_name: str,
    artist_name: str,
    genius_token: str,
    system_prompt: str,
    sources_dir: str,
    output_path: str,
    on_progress: Callable[[str, str], None],
    language_code: str = "tr",
    device: str = "cpu",
    compute_type: str = "int8",
    ollama_url: str = "http://localhost:11434/v1/chat/completions",
    model_name: str = "gemma4:e4b",
) -> StepResult:
    """
    Run the entire CleanCut pipeline sequentially.
    """
    # ── Step 1: Genius Lyrics ───────────────────────────────────────────────
    on_progress("genius", f"Genius'tan şarkı sözleri indiriliyor: «{song_name}» - {artist_name}…")
    genius_res = fetch_and_save_lyrics(
        song_name=song_name,
        artist_name=artist_name,
        sources_dir=sources_dir,
        genius_token=genius_token
    )
    if not genius_res.success:
        return genius_res
    
    lyrics_lines = genius_res.data.get("lyrics_lines", [])

    # ── Step 2: WhisperX Alignment ──────────────────────────────────────────
    on_progress("alignment", "WhisperX ile timestamp hizalaması yapılıyor (Bu işlem zaman alabilir)…")
    align_res = align_lyrics(
        audio_path=audio_path,
        lyrics_lines=lyrics_lines,
        language_code=language_code,
        device=device,
        compute_type=compute_type
    )
    if not align_res.success:
        return align_res
    
    word_segments = align_res.data.get("word_segments", [])

    # ── Step 3: Line Timestamps ──────────────────────────────────────────────
    on_progress("timestamps", "Kelime hizalamalarından satır bazlı zaman damgaları oluşturuluyor…")
    ts_res = build_line_timestamps(
        lyrics_lines=lyrics_lines,
        word_segments=word_segments
    )
    if not ts_res.success:
        return ts_res
    
    line_timestamps = ts_res.data.get("line_timestamps", [])

    # ── Step 4: Content Checker (LLM) ────────────────────────────────────────
    on_progress("llm", f"Yerel LLM ({model_name}) ile küfür & içerik denetimi yapılıyor…")
    llm_res = check_profanity(
        lyrics_lines=lyrics_lines,
        system_prompt=system_prompt,
        ollama_url=ollama_url,
        model_name=model_name
    )
    if not llm_res.success:
        return llm_res
    
    flagged = llm_res.data.get("flagged", [])
    if not flagged:
        return StepResult(
            success=True,
            data={
                "lyrics_lines": lyrics_lines,
                "line_timestamps": line_timestamps,
                "flagged": [],
                "unmatched_phrases": [],
                "cuts": [],
                "cuts_applied": 0,
                "output_file": None
            },
            message="Hiçbir uygunsuz içerik tespit edilmedi. Şarkı temiz! 🎉"
        )

    # ── Step 5: Timestamp Matcher ────────────────────────────────────────────
    on_progress("timestamps_match", f"İşaretlenen {len(flagged)} kelime için zaman damgaları çıkartılıyor…")
    match_res = find_flagged_timestamps(
        flagged=flagged,
        word_segments=word_segments
    )
    if not match_res.success:
        return match_res
    
    cuts = match_res.data.get("cuts", [])
    unmatched_phrases = match_res.data.get("unmatched_phrases", [])

    if not cuts:
        return StepResult(
            success=True,
            data={
                "lyrics_lines": lyrics_lines,
                "line_timestamps": line_timestamps,
                "flagged": flagged,
                "unmatched_phrases": unmatched_phrases,
                "cuts": [],
                "cuts_applied": 0,
                "output_file": None
            },
            message="Zaman damgası eşleştirilemedi — kesim yapılamadı."
        )

    # ── Step 6: Apply Cuts ───────────────────────────────────────────────────
    on_progress("audio_cut", f"Beat-aware crossfade ile {len(cuts)} kesim aralığı uygulanıyor…")
    cut_res = apply_cuts(
        input_file=audio_path,
        output_file=output_path,
        cuts=cuts
    )
    if not cut_res.success:
        return cut_res

    return StepResult(
        success=True,
        data={
            "lyrics_lines": lyrics_lines,
            "line_timestamps": line_timestamps,
            "flagged": flagged,
            "unmatched_phrases": unmatched_phrases,
            "cuts": cuts,
            "cuts_applied": cut_res.data.get("cuts_applied", 0),
            "output_file": output_path
        },
        message=f"Final sürüm başarıyla kaydedildi: {output_path}"
    )
