import logging
import time
import uuid
import threading
from dataclasses import dataclass, field, asdict
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable
from api.utils import to_jsonable

# WhisperX ve librosa makineyi tek başına meşgul ettiği için ağır işler
# (pipeline, hizalama, kesim) tek bir worker üzerinde sırayla çalışır.
_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="cleancut-worker")

logger = logging.getLogger("cleancut")

_jobs: dict[str, "Job"] = {}
_jobs_lock = threading.Lock()


@dataclass
class Job:
    id: str
    request: dict[str, Any]
    status: str = "queued"
    step: str | None = None
    events: list[dict[str, Any]] = field(default_factory=list)
    result: dict[str, Any] | None = None
    error: dict[str, Any] | None = None
    created_at: float = field(default_factory=time.time)
    finished_at: float | None = None

    def add_event(self, step: str, message: str) -> None:
        self.step = step
        self.events.append({"step": step, "message": message, "time": time.time()})
        logger.info("  [iş %s] %s: %s", self.id, step, message)

    @property
    def finished(self) -> bool:
        return self.status in ("done", "failed")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def run_exclusive(fn: Callable, *args, **kwargs):
    """Fonksiyonu worker üzerinde çalıştırıp sonucunu bekler (diğer ağır işlerle çakışmaz)."""
    return _executor.submit(fn, *args, **kwargs).result()


def submit_pipeline(request: dict[str, Any], run_kwargs: dict[str, Any]) -> Job:
    """
    request: kullanıcıya gösterilen parametreler.
    run_kwargs: run_pipeline'a giden tüm parametreler (token gibi gizli değerler dahil).
    """
    job = Job(id=uuid.uuid4().hex[:12], request=request)
    with _jobs_lock:
        _jobs[job.id] = job
    _executor.submit(_run_pipeline, job, run_kwargs)
    return job


def get_job(job_id: str) -> Job | None:
    return _jobs.get(job_id)


def list_jobs() -> list[Job]:
    return sorted(_jobs.values(), key=lambda j: j.created_at, reverse=True)


def _run_pipeline(job: Job, run_kwargs: dict[str, Any]) -> None:
    job.status = "running"
    try:
        # whisperx/librosa import'u yavaş; sunucunun hızlı açılması için ilk kullanımda yüklenir.
        from core.pipeline import run_pipeline

        result = run_pipeline(on_progress=job.add_event, **run_kwargs)
        if result.success:
            data = to_jsonable(result.data)
            if data.get("output_file"):
                data["output_name"] = Path(data["output_file"]).name
            job.result = {"message": result.message, "data": data}
            _finish(job, "done", result.message)
        else:
            job.error = {"message": result.message, "error": result.error}
            _finish(job, "failed", result.message)
    except Exception as e:
        job.error = {"message": "Pipeline beklenmeyen bir hatayla durdu.", "error": str(e)}
        _finish(job, "failed", job.error["message"])


def _finish(job: Job, status: str, message: str) -> None:
    # Son olay status'tan önce eklenir; böylece SSE akışı kapanmadan son mesajı da gönderir.
    job.add_event(status, message)
    job.finished_at = time.time()
    job.status = status
