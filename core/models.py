from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Dict, Any

@dataclass
class StepResult:
    success: bool
    data: Dict[str, Any] = field(default_factory=dict)
    message: str = ""
    error: Optional[str] = None


class CutMode(str, Enum):
    """Kesimin kaç vuruşun katı olacağı; ritmin kesimden sonra nasıl devam edeceğini belirler."""
    EVEN_BEATS = "even_beats"
    FULL_BARS = "full_bars"

    @property
    def beats(self) -> int:
        return {CutMode.EVEN_BEATS: 2, CutMode.FULL_BARS: 4}[self]

    @property
    def label(self) -> str:
        return {CutMode.EVEN_BEATS: "Çift vuruş", CutMode.FULL_BARS: "Tam ölçü"}[self]

    @property
    def description(self) -> str:
        return {
            CutMode.EVEN_BEATS: "Silinen vuruş sayısı 2'nin katı olur. Kick/snare sırası korunur; "
                                "küfrün çevresinden daha az söz gider.",
            CutMode.FULL_BARS: "Silinen vuruş sayısı 4'ün katı olur. Ölçü yapısı tamamen korunur, "
                               "ritim en doğal haliyle devam eder; küfrün çevresinden daha çok söz gider.",
        }[self]


DEFAULT_CUT_MODE = CutMode.FULL_BARS
