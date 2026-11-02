import shutil
from pathlib import Path
from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from api.config import MUSIC_DIR, OUTPUT_DIR, SUPPORTED_EXT, parse_song_artist
from api.schemas import OutputInfo, SongInfo
from api.utils import resolve_in

songs = APIRouter(prefix="/songs", tags=["Şarkılar (music/)"])
outputs = APIRouter(prefix="/outputs", tags=["Çıktılar (cleancutted/)"])


def _audio_files(directory: Path) -> list[Path]:
    return sorted(
        (f for f in directory.iterdir() if f.is_file() and f.suffix.lower() in SUPPORTED_EXT),
        key=lambda f: f.name.lower(),
    )


def _song_info(path: Path) -> SongInfo:
    song, artist = parse_song_artist(path.name)
    return SongInfo(filename=path.name, song_name=song, artist_name=artist, size=path.stat().st_size)


# ── Şarkılar ─────────────────────────────────────────────────────────────────

@songs.get("", response_model=list[SongInfo])
def list_songs():
    return [_song_info(f) for f in _audio_files(MUSIC_DIR)]


@songs.post("", response_model=SongInfo, status_code=201)
def upload_song(file: UploadFile = File(..., description="'şarkı - sanatçı.mp3' formatı önerilir")):
    name = Path(file.filename or "").name
    if Path(name).suffix.lower() not in SUPPORTED_EXT:
        raise HTTPException(400, f"Desteklenmeyen format. Desteklenenler: {', '.join(sorted(SUPPORTED_EXT))}")
    dest = MUSIC_DIR / name
    if dest.exists():
        raise HTTPException(409, f"Bu isimde bir şarkı zaten var: {name}")
    with dest.open("wb") as f:
        shutil.copyfileobj(file.file, f)
    return _song_info(dest)


@songs.get("/{filename}/audio")
def get_song_audio(filename: str):
    return FileResponse(resolve_in(MUSIC_DIR, filename))


@songs.delete("/{filename}", status_code=204)
def delete_song(filename: str):
    resolve_in(MUSIC_DIR, filename).unlink()


# ── Çıktılar ─────────────────────────────────────────────────────────────────

@outputs.get("", response_model=list[OutputInfo])
def list_outputs():
    return [OutputInfo(filename=f.name, size=f.stat().st_size) for f in _audio_files(OUTPUT_DIR)]


@outputs.get("/{filename}")
def download_output(filename: str):
    path = resolve_in(OUTPUT_DIR, filename)
    return FileResponse(path, filename=path.name)


@outputs.delete("/{filename}", status_code=204)
def delete_output(filename: str):
    resolve_in(OUTPUT_DIR, filename).unlink()
