"""Google Gemini generate-content adapter."""

from __future__ import annotations

from typing import Any

from .._utils import attr, enum_value, json_schema, parse_structured
from ..exceptions import ProviderDependencyError
from ..types import Message, ModelResponse, Provider, ResponseFormat, TokenUsage


def call_gemini(
    *,
    model: str,
    messages: list[Message],
    system: str | None,
    response_format: ResponseFormat[Any] | None,
    max_output_tokens: int | None,
    client: Any = None,
    options: dict[str, Any],
) -> ModelResponse[Any]:
    try:
        from google.genai import types
        if client is None:
            from google import genai
            client = genai.Client()
    except ImportError as exc:  # pragma: no cover - depends on installation
        raise ProviderDependencyError("install the 'google-genai' package to call Gemini") from exc

    contents = [
        types.Content(
            role="model" if message.role == "assistant" else "user",
            parts=[types.Part.from_text(text=message.content)],
        )
        for message in messages
    ]
    config: dict[str, Any] = dict(options)
    if system is not None:
        config["system_instruction"] = system
    if max_output_tokens is not None:
        config["max_output_tokens"] = max_output_tokens
    if response_format is not None:
        config["response_mime_type"] = "application/json"
        config["response_json_schema"] = json_schema(response_format)

    response = client.models.generate_content(model=model, contents=contents, config=config)
    text = str(attr(response, "text", ""))
    usage = attr(response, "usage_metadata")
    candidates = attr(response, "candidates", [])
    finish_reason = enum_value(attr(candidates[0], "finish_reason")) if candidates else None
    return ModelResponse(
        text=text,
        parsed=parse_structured(text, response_format),
        provider=Provider.GEMINI,
        model=str(attr(response, "model_version", model)),
        request_id=attr(response, "response_id"),
        finish_reason=finish_reason,
        usage=TokenUsage(
            input_tokens=attr(usage, "prompt_token_count"),
            output_tokens=attr(usage, "candidates_token_count"),
        ),
        raw=response,
    )
