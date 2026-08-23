"""Provider-independent inference request and response types."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Generic, Literal, Mapping, TypeAlias, TypeVar


T = TypeVar("T")
Role: TypeAlias = Literal["system", "user", "assistant"]
ResponseFormat: TypeAlias = type[T] | Mapping[str, Any]


class Provider(str, Enum):
    """Inference backends supported by :func:`call_model`."""

    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GEMINI = "gemini"
    MODAL = "modal"


@dataclass(frozen=True, slots=True)
class Message:
    """A text-only chat message in the common request format."""

    role: Role
    content: str

    def __post_init__(self) -> None:
        if self.role not in ("system", "user", "assistant"):
            raise ValueError(f"unsupported message role: {self.role!r}")
        if not isinstance(self.content, str) or not self.content:
            raise ValueError("message content must be a non-empty string")

    def to_dict(self) -> dict[str, str]:
        return {"role": self.role, "content": self.content}


@dataclass(frozen=True, slots=True)
class TokenUsage:
    """Normalized token counts when reported by the provider."""

    input_tokens: int | None = None
    output_tokens: int | None = None

    @property
    def total_tokens(self) -> int | None:
        if self.input_tokens is None and self.output_tokens is None:
            return None
        return (self.input_tokens or 0) + (self.output_tokens or 0)


@dataclass(frozen=True, slots=True)
class ModelResponse(Generic[T]):
    """A normalized model response.

    ``parsed`` is populated only when ``response_format`` was supplied. It is a
    validated Pydantic instance for Pydantic response types and a dictionary or
    list for raw JSON Schema inputs.
    """

    text: str
    provider: Provider
    model: str
    parsed: T | None = None
    request_id: str | None = None
    finish_reason: str | None = None
    usage: TokenUsage = field(default_factory=TokenUsage)
    raw: Any = field(default=None, repr=False, compare=False)


@dataclass(frozen=True, slots=True)
class ModalConfig:
    """Location of a deployed Modal inference function.

    The remote function must accept the keyword arguments documented in
    ``providers/modal.py`` and return either a string or a response dictionary.
    """

    app_name: str
    function_name: str = "generate"
    environment_name: str | None = None
    transport: Literal["function", "openai_web"] = "function"

    def __post_init__(self) -> None:
        if not self.app_name:
            raise ValueError("Modal app_name cannot be empty")
        if not self.function_name:
            raise ValueError("Modal function_name cannot be empty")
        if self.transport not in ("function", "openai_web"):
            raise ValueError(f"unsupported Modal transport: {self.transport!r}")
