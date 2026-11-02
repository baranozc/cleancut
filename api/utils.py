import logging
import math
from pathlib import Path
from typing import Any
from fastapi import HTTPException
from core.models import StepResult

logger = logging.getLogger("cleancut")


def to_jsonable(value: Any) -> Any:
    """numpy tiplerini ve NaN/inf değerlerini JSON'a uygun hale getirir (WhisperX çıktıları için)."""
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(v) for v in value]
    if hasattr(value, "tolist"):
        return to_jsonable(value.tolist())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def unwrap(result: StepResult, failure_status: int = 500) -> dict:
    """Core'dan dönen StepResult'u HTTP yanıtına çevirir; başarısızsa hata fırlatır."""
    if not result.success:
        logger.warning("  HATA: %s | %s", result.message, result.error)
        raise HTTPException(
            status_code=failure_status,
            detail={"message": result.message, "error": result.error},
        )
    logger.info("  • %s", result.message)
    return {"message": result.message, "data": to_jsonable(result.data)}


def resolve_in(directory: Path, name: str) -> Path:
    """Dosya adını verilen klasör içinde çözer; klasör dışına çıkan yolları (../) reddeder."""
    path = (directory / name).resolve()
    if path.parent != directory.resolve() or not path.is_file():
        raise HTTPException(status_code=404, detail=f"Dosya bulunamadı: {name}")
    return path
