from dataclasses import dataclass, field
from typing import Any

@dataclass
class SourceRef:
    file_name: str
    file_id: str
    location: str
    chunk_id: str
    text: str
    score: float
    extra: dict[str, Any] = field(default_factory=dict)

@dataclass
class ParsedUnit:
    text: str
    metadata: dict[str, Any]
