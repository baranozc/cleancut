import os
import re
from pathlib import Path
from dotenv import load_dotenv
from core.models import CutMode, DEFAULT_CUT_MODE as CORE_DEFAULT_CUT_MODE

ROOT_DIR = Path(__file__).resolve().parent.parent
MUSIC_DIR = ROOT_DIR / "music"
SOURCES_DIR = ROOT_DIR / "sources"
OUTPUT_DIR = ROOT_DIR / "cleancutted"
SYSTEM_PROMPT_PATH = ROOT_DIR / "system_prompt.txt"

SUPPORTED_EXT = {".mp3", ".wav", ".flac", ".m4a", ".ogg", ".aac"}

_env_path = ROOT_DIR / ".env"
if not _env_path.exists():
    _env_path = ROOT_DIR.parent / ".env"
load_dotenv(dotenv_path=_env_path)

DEFAULT_LANGUAGE     = os.getenv("CLEANCUT_LANGUAGE", "tr")
DEFAULT_DEVICE       = os.getenv("CLEANCUT_DEVICE", "cpu")
DEFAULT_COMPUTE_TYPE = os.getenv("CLEANCUT_COMPUTE_TYPE", "int8")
DEFAULT_MODEL        = os.getenv("OLLAMA_MODEL", "gemma4:e4b")
DEFAULT_OLLAMA_URL   = os.getenv("OLLAMA_URL", "http://localhost:11434/v1/chat/completions")
DEFAULT_CUT_MODE     = CutMode(os.getenv("CLEANCUT_CUT_MODE", CORE_DEFAULT_CUT_MODE.value))


def ensure_dirs() -> None:
    for directory in (MUSIC_DIR, SOURCES_DIR, OUTPUT_DIR):
        directory.mkdir(exist_ok=True)


def genius_token() -> str | None:
    return os.getenv("GENIUS_API_KEY") or os.getenv("GENIUS_TOKEN")


def load_system_prompt() -> str | None:
    """system_prompt.txt her istekte okunur; düzenlemek için sunucuyu yeniden başlatmak gerekmez."""
    if not SYSTEM_PROMPT_PATH.exists():
        return None
    return SYSTEM_PROMPT_PATH.read_text(encoding="utf-8").strip() or None


def parse_song_artist(filename: str) -> tuple[str, str]:
    """'şarkı - sanatçı.mp3' formatındaki dosya adından şarkı ve sanatçı adını çıkarır."""
    stem = Path(filename).stem
    if " - " in stem:
        song, artist = stem.split(" - ", 1)
        return song.strip(), artist.strip()
    return stem.strip(), ""


def output_name_for(song_name: str, artist_name: str, ext: str) -> str:
    """Çıktı adı; ext kaynak dosyanın uzantısıdır (çıktı kaynakla aynı formatta yazılır)."""
    safe = f"{song_name}-{artist_name}".replace(" ", "_")
    safe = re.sub(r'[\\/*?:"<>|]', "", safe)
    return f"{safe}_cleancutted{ext}"
