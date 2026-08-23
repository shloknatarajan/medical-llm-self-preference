"""OpenAI Responses API adapter."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .._utils import attr, parse_structured, schema_name
from ..exceptions import ProviderDependencyError
from ..types import Message, ModelResponse, Provider, ResponseFormat, TokenUsage


def call_openai(
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
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover - depends on installation
            raise ProviderDependencyError("install the 'openai' package to call OpenAI") from exc
        client = OpenAI()

    request: dict[str, Any] = {
        "model": model,
        "input": [message.to_dict() for message in messages],
        **options,
    }
    if system is not None:
        request["instructions"] = system
    if max_output_tokens is not None:
        request["max_output_tokens"] = max_output_tokens

    if response_format is not None and not isinstance(response_format, Mapping):
        response = client.responses.parse(**request, text_format=response_format)
    else:
        if response_format is not None:
            request["text"] = {
                "format": {
                    "type": "json_schema",
                    "name": schema_name(response_format),
                    "schema": dict(response_format),
                    "strict": True,
                }
            }
        response = client.responses.create(**request)

    text = str(attr(response, "output_text", ""))
    parsed = parse_structured(text, response_format)
    usage = attr(response, "usage")
    return ModelResponse(
        text=text,
        parsed=parsed,
        provider=Provider.OPENAI,
        model=str(attr(response, "model", model)),
        request_id=attr(response, "id"),
        finish_reason=attr(response, "status"),
        usage=TokenUsage(
            input_tokens=attr(usage, "input_tokens"),
            output_tokens=attr(usage, "output_tokens"),
        ),
        raw=response,
    )
