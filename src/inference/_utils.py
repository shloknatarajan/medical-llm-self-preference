"""Internal normalization and structured-response helpers."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Any

from .exceptions import ResponseParseError
from .types import Message, ResponseFormat


def normalize_messages(
    input: str | Sequence[Message | Mapping[str, str]],
    system: str | None,
) -> tuple[str | None, list[Message]]:
    """Normalize a prompt/chat input and separate system instructions."""

    if isinstance(input, str):
        messages = [Message(role="user", content=input)]
    else:
        messages = [
            item
            if isinstance(item, Message)
            else Message(role=item["role"], content=item["content"])  # type: ignore[arg-type]
            for item in input
        ]

    if not messages:
        raise ValueError("input must contain at least one message")

    embedded_system = [message.content for message in messages if message.role == "system"]
    non_system = [message for message in messages if message.role != "system"]
    system_parts = ([system] if system else []) + embedded_system
    system_text = "\n\n".join(system_parts) or None
    if not non_system:
        raise ValueError("input must contain at least one non-system message")
    return system_text, non_system


def json_schema(response_format: ResponseFormat[Any]) -> dict[str, Any]:
    """Convert a Pydantic type or schema mapping to JSON Schema."""

    if isinstance(response_format, Mapping):
        return dict(response_format)
    schema_builder = getattr(response_format, "model_json_schema", None)
    if schema_builder is None:
        raise TypeError("response_format must be a Pydantic model type or JSON Schema mapping")
    return dict(schema_builder())


def parse_structured(text: str, response_format: ResponseFormat[Any] | None) -> Any:
    """Parse JSON and, for Pydantic types, validate the generated value."""

    if response_format is None:
        return None
    try:
        if isinstance(response_format, Mapping):
            return json.loads(text)
        validator = getattr(response_format, "model_validate_json", None)
        if validator is None:
            raise TypeError("response_format type does not support model_validate_json")
        return validator(text)
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ResponseParseError(f"model returned an invalid structured response: {exc}") from exc


def schema_name(response_format: ResponseFormat[Any]) -> str:
    if isinstance(response_format, Mapping):
        name = response_format.get("title", "response")
    else:
        name = getattr(response_format, "__name__", "response")
    safe = "".join(character if character.isalnum() else "_" for character in str(name))
    return (safe or "response")[:64]


def attr(value: Any, name: str, default: Any = None) -> Any:
    """Read a field from either an SDK object or a dictionary."""

    if isinstance(value, Mapping):
        return value.get(name, default)
    return getattr(value, name, default)


def enum_value(value: Any) -> str | None:
    if value is None:
        return None
    raw = getattr(value, "value", value)
    return str(raw)
