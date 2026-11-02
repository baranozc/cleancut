from typing import Any
from pydantic import BaseModel, Field
from core.models import CutMode
from api.config import (
    DEFAULT_LANGUAGE, DEFAULT_DEVICE, DEFAULT_COMPUTE_TYPE, DEFAULT_MODEL, DEFAULT_OLLAMA_URL,
    DEFAULT_CUT_MODE,
)

_CUT_MODE_DOC = "Kesim modu: " + "; ".join(f"{m.value} = {m.label} ({m.description})" for m in CutMode)


class StepResponse(BaseModel):
    message: str
    data: dict[str, Any]


class SongInfo(BaseModel):
    filename: str
    song_name: str
    artist_name: str
    size: int


class OutputInfo(BaseModel):
    filename: str
    size: int


class LyricsRequest(BaseModel):
    song_name: str = Field(min_length=1)
    artist_name: str = ""


class AlignRequest(BaseModel):
    song: str = Field(description="music/ klasöründeki dosya adı")
    lyrics_lines: list[str] = Field(min_length=1)
    language_code: str = DEFAULT_LANGUAGE
    device: str = DEFAULT_DEVICE
    compute_type: str = DEFAULT_COMPUTE_TYPE


class LineTimestampsRequest(BaseModel):
    lyrics_lines: list[str]
    word_segments: list[dict[str, Any]]


class LLMCheckRequest(BaseModel):
    lyrics_lines: list[str] = Field(min_length=1)
    system_prompt: str | None = Field(default=None, description="Boş bırakılırsa system_prompt.txt kullanılır")
    model_name: str = DEFAULT_MODEL
    ollama_url: str = DEFAULT_OLLAMA_URL


class MatchRequest(BaseModel):
    flagged: list[str]
    word_segments: list[dict[str, Any]]


class CutRequest(BaseModel):
    song: str = Field(description="music/ klasöründeki dosya adı")
    cuts: list[tuple[float, float]] = Field(description="[[başlangıç, bitiş], ...] saniye cinsinden")
    cut_mode: CutMode = Field(default=DEFAULT_CUT_MODE, description=_CUT_MODE_DOC)
    output_name: str | None = Field(default=None, description="Boş bırakılırsa şarkı adından üretilir; "
                                                              "uzantı her zaman kaynakla aynı olur")


class JobRequest(BaseModel):
    song: str = Field(description="music/ klasöründeki dosya adı")
    song_name: str | None = Field(default=None, description="Boş bırakılırsa dosya adından çıkarılır")
    artist_name: str | None = Field(default=None, description="Boş bırakılırsa dosya adından çıkarılır")
    language_code: str = DEFAULT_LANGUAGE
    device: str = DEFAULT_DEVICE
    compute_type: str = DEFAULT_COMPUTE_TYPE
    model_name: str = DEFAULT_MODEL
    ollama_url: str = DEFAULT_OLLAMA_URL
    cut_mode: CutMode = Field(default=DEFAULT_CUT_MODE, description=_CUT_MODE_DOC)


class CutModeOption(BaseModel):
    value: CutMode
    label: str
    description: str
    beats: int = Field(description="Silinen vuruş sayısının katı olacağı sayı")


class OptionsOut(BaseModel):
    cut_modes: list[CutModeOption]
    defaults: dict[str, Any]


class JobEvent(BaseModel):
    step: str
    message: str
    time: float


class JobOut(BaseModel):
    id: str
    status: str = Field(description="queued | running | done | failed")
    step: str | None
    request: dict[str, Any]
    events: list[JobEvent]
    result: StepResponse | None
    error: dict[str, Any] | None
    created_at: float
    finished_at: float | None
