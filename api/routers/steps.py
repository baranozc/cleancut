"""
Pipeline adımlarının her biri ayrı bir endpoint olarak. Bir adımın çıktısı
bir sonrakinin girdisi olarak kullanılabilir:

  /lyrics → /alignment → /alignment/lines
  /lyrics → /llm/check → /audio/match (+ /alignment çıktısı) → /audio/cut
"""
from pathlib import Path
from fastapi import APIRouter, HTTPException
from api import jobs
from api.config import (
    MUSIC_DIR, OUTPUT_DIR, SOURCES_DIR,
    genius_token, load_system_prompt, output_name_for, parse_song_artist,
)
from api.schemas import (
    AlignRequest, CutRequest, LineTimestampsRequest, LLMCheckRequest,
    LyricsRequest, MatchRequest, StepResponse,
)
from api.utils import resolve_in, unwrap

router = APIRouter()


@router.post("/lyrics", response_model=StepResponse, tags=["1. Şarkı sözleri"])
def fetch_lyrics(req: LyricsRequest):
    """Sözleri Genius'tan çeker (sources/ klasöründe varsa oradan okur)."""
    from core.genius import fetch_and_save_lyrics

    token = genius_token()
    if not token:
        raise HTTPException(500, ".env dosyasında GENIUS_API_KEY veya GENIUS_TOKEN tanımlı değil.")
    result = fetch_and_save_lyrics(
        song_name=req.song_name,
        artist_name=req.artist_name,
        sources_dir=str(SOURCES_DIR),
        genius_token=token,
    )
    return unwrap(result, failure_status=502)


@router.post("/alignment", response_model=StepResponse, tags=["2. Hizalama"])
def align(req: AlignRequest):
    """WhisperX ile sözleri sese hizalar, kelime bazlı zaman damgaları döner. Uzun sürebilir."""
    from core.alignment import align_lyrics

    audio_path = resolve_in(MUSIC_DIR, req.song)
    result = jobs.run_exclusive(
        align_lyrics,
        audio_path=str(audio_path),
        lyrics_lines=req.lyrics_lines,
        language_code=req.language_code,
        device=req.device,
        compute_type=req.compute_type,
    )
    return unwrap(result)


@router.post("/alignment/lines", response_model=StepResponse, tags=["2. Hizalama"])
def line_timestamps(req: LineTimestampsRequest):
    """Kelime zaman damgalarından satır bazlı zaman damgaları üretir."""
    from core.alignment import build_line_timestamps

    return unwrap(build_line_timestamps(req.lyrics_lines, req.word_segments))


@router.post("/llm/check", response_model=StepResponse, tags=["3. İçerik denetimi"])
def check_lyrics(req: LLMCheckRequest):
    """Sözleri yerel LLM'e (Ollama) gönderir, uygunsuz kelime/kelime gruplarını döner."""
    from core.llm import check_profanity

    system_prompt = req.system_prompt or load_system_prompt()
    if not system_prompt:
        raise HTTPException(500, "system_prompt.txt bulunamadı veya boş; istekte system_prompt gönderin.")
    result = check_profanity(
        lyrics_lines=req.lyrics_lines,
        system_prompt=system_prompt,
        ollama_url=req.ollama_url,
        model_name=req.model_name,
    )
    return unwrap(result, failure_status=502)


@router.post("/audio/match", response_model=StepResponse, tags=["4. Ses"])
def match_flagged(req: MatchRequest):
    """İşaretlenen kelimeleri kelime zaman damgalarında arar, kesim aralıklarını döner."""
    from core.audio import find_flagged_timestamps

    return unwrap(find_flagged_timestamps(req.flagged, req.word_segments))


@router.post("/audio/cut", response_model=StepResponse, tags=["4. Ses"])
def cut_audio(req: CutRequest):
    """
    Verilen aralıkları vuruş ızgarasına oturtarak keser, sonucu cleancutted/ klasörüne
    kaynakla aynı formatta ve kalitede, metadata'sı korunarak yazar.
    """
    from core.audio import apply_cuts

    audio_path = resolve_in(MUSIC_DIR, req.song)
    if req.output_name:
        output_name = Path(req.output_name).stem + audio_path.suffix
    else:
        output_name = output_name_for(*parse_song_artist(audio_path.name), ext=audio_path.suffix)

    result = jobs.run_exclusive(
        apply_cuts,
        input_file=str(audio_path),
        output_file=str(OUTPUT_DIR / output_name),
        cuts=[list(c) for c in req.cuts],
        cut_mode=req.cut_mode,
    )
    response = unwrap(result)
    response["data"]["output_name"] = Path(response["data"]["output_file"]).name
    return response
