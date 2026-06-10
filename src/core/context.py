from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class Context:
    session_id: str
    metadata: tuple[tuple[str, Any], ...] = field(default_factory=tuple)
