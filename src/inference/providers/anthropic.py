"""Anthropic Messages API adapter."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .._utils import attr, enum_value, json_schema, parse_structured
from ..exceptions import ProviderDependencyError
from ..types import Message, ModelResponse, Provider, ResponseFormat, TokenUsage


def call_anthropic(
    *,
    model: str,
    messages: list[Message],
    system: str | None,
    response_format: ResponseFormat[Any] | None,
    max_output_tokens: int | None,
    client: Any = None,
    options: dict[str, Any],
) -> ModelResponse[Any]:
    if client is None:
        try:
            from anthropic import Anthropic
        except ImportError as exc:  # pragma: no cover - depends on installation
            raise ProviderDependencyError("install the 'anthropic' package to call Anthropic") from exc
        client = Anthropic()

    request: dict[str, Any] = {
        "model": model,
        "max_tokens": max_output_tokens or 1024,
        "messages": [message.to_dict() for message in messages],
        **options,
    }
    if system is not None:
        request["system"] = system

    if response_format is not None and not isinstance(response_format, Mapping):
        response = client.messages.parse(**request, output_format=response_format)
    else:
        if response_format is not None:
            request["output_config"] = {
                "format": {"type": "json_schema", "schema": json_schema(response_format)}
            }
        response = client.messages.create(**request)

    text = "".join(
        str(attr(block, "text", ""))
        for block in attr(response, "content", [])
        if attr(block, "type") == "text"
    )
    usage = attr(response, "usage")
    return ModelResponse(
        text=text,
        parsed=parse_structured(text, response_format),
        provider=Provider.ANTHROPIC,
        model=str(attr(response, "model", model)),
        request_id=attr(response, "id"),
        finish_reason=enum_value(attr(response, "stop_reason")),
        usage=TokenUsage(
            input_tokens=attr(usage, "input_tokens"),
            output_tokens=attr(usage, "output_tokens"),
        ),
        raw=response,
    )
