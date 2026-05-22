from dataclasses import dataclass, field
from typing import Optional, Dict, Any

@dataclass
class StepResult:
    success: bool
    data: Dict[str, Any] = field(default_factory=dict)
    message: str = ""
    error: Optional[str] = None
