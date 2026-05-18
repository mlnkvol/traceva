from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel


class TaskStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    DONE = "done"
    ERROR = "error"


class VectorizeMode(str, Enum):
    AUTO = "auto"
    LOGO = "logo"
    SEMANTIC = "semantic"


class VectorizeRequest(BaseModel):
    tolerance: float = 1.0
    max_layers: int = 5
    simplify: bool = True
    mode: VectorizeMode = VectorizeMode.AUTO


class LayerInfo(BaseModel):
    name: str
    node_count: int
    color: str


class VectorizeResult(BaseModel):
    task_id: str
    status: TaskStatus
    svg_url: Optional[str] = None
    layers: Optional[List[LayerInfo]] = None
    metrics: Optional[Dict[str, Any]] = None

    # requested_mode — що користувач обрав на фронті: auto / logo / semantic
    requested_mode: Optional[str] = None

    # mode — що реально використав backend: logo / semantic
    mode: Optional[str] = None

    error: Optional[str] = None


class TaskStatusResponse(BaseModel):
    task_id: str
    status: TaskStatus
    progress: int = 0

    # requested_mode — що користувач обрав на фронті
    requested_mode: Optional[str] = None

    # mode — фактичний режим після auto-визначення
    mode: Optional[str] = None

    error: Optional[str] = None