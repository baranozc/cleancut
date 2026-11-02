import asyncio
import json
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse
from api import jobs
from api.config import (
    MUSIC_DIR, OUTPUT_DIR, SOURCES_DIR,
    genius_token, load_system_prompt, output_name_for, parse_song_artist,
)
from api.schemas import JobOut, JobRequest
from api.utils import resolve_in

router = APIRouter(prefix="/jobs", tags=["Tam pipeline"])


def _get_job_or_404(job_id: str) -> jobs.Job:
    job = jobs.get_job(job_id)
    if job is None:
        raise HTTPException(404, f"İş bulunamadı: {job_id}")
    return job


@router.post("", response_model=JobOut, status_code=202)
def create_job(req: JobRequest):
    """Tüm pipeline'ı arka planda başlatır. İlerleme için /jobs/{id} veya /jobs/{id}/events."""
    audio_path = resolve_in(MUSIC_DIR, req.song)

    token = genius_token()
    if not token:
        raise HTTPException(500, ".env dosyasında GENIUS_API_KEY veya GENIUS_TOKEN tanımlı değil.")
    system_prompt = load_system_prompt()
    if not system_prompt:
        raise HTTPException(500, "system_prompt.txt bulunamadı veya boş.")

    parsed_song, parsed_artist = parse_song_artist(audio_path.name)
    song_name = req.song_name or parsed_song
    artist_name = req.artist_name or parsed_artist

    request = req.model_dump() | {"song_name": song_name, "artist_name": artist_name}
    run_kwargs = {
        "audio_path": str(audio_path),
        "song_name": song_name,
        "artist_name": artist_name,
        "genius_token": token,
        "system_prompt": system_prompt,
        "sources_dir": str(SOURCES_DIR),
        "output_path": str(OUTPUT_DIR / output_name_for(song_name, artist_name, ext=audio_path.suffix)),
        "language_code": req.language_code,
        "device": req.device,
        "compute_type": req.compute_type,
        "ollama_url": req.ollama_url,
        "model_name": req.model_name,
        "cut_mode": req.cut_mode,
    }
    return jobs.submit_pipeline(request, run_kwargs).to_dict()


@router.get("", response_model=list[JobOut])
def list_jobs():
    return [job.to_dict() for job in jobs.list_jobs()]


@router.get("/{job_id}", response_model=JobOut)
def get_job(job_id: str):
    return _get_job_or_404(job_id).to_dict()


@router.get("/{job_id}/events")
async def job_events(job_id: str, request: Request):
    """İlerleme olaylarını Server-Sent Events olarak akıtır; iş bitince akış kapanır."""
    job = _get_job_or_404(job_id)

    async def stream():
        sent = 0
        while True:
            while sent < len(job.events):
                yield f"data: {json.dumps(job.events[sent], ensure_ascii=False)}\n\n"
                sent += 1
            if job.finished and sent >= len(job.events):
                break
            if await request.is_disconnected():
                break
            await asyncio.sleep(0.5)

    return StreamingResponse(stream(), media_type="text/event-stream", headers={"Cache-Control": "no-cache"})
