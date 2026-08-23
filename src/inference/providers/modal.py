"""Adapters for deployed Modal functions and OpenAI-compatible web endpoints.

The deployed function is called with these keyword arguments::

    generate(
        model: str,
        messages: list[dict[str, str]],
        system: str | None,
        max_output_tokens: int | None,
        response_schema: dict | None,
        **provider_options,
    )

It may return a plain string, or a dictionary containing ``text`` and optional
``model``, ``request_id``, ``finish_reason``, and ``usage`` fields.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .._utils import attr, json_schema, parse_structured, schema_name
from ..exceptions import ProviderDependencyError
from ..types import Message, ModalConfig, ModelResponse, Provider, ResponseFormat, TokenUsage


def call_modal(
    *,
    model: str,
    messages: list[Message],
    system: str | None,
    response_format: ResponseFormat[Any] | None,
    max_output_tokens: int | None,
    config: ModalConfig | None,
    remote_function: Any = None,
    client: Any = None,
    options: dict[str, Any],
) -> ModelResponse[Any]:
    if config is not None and config.transport == "openai_web":
        return _call_openai_web(
            model=model,
            messages=messages,
            system=system,
            response_format=response_format,
            max_output_tokens=max_output_tokens,
            config=config,
            client=client,
            options=options,
        )

    if remote_function is None:
        if config is None:
            raise ValueError("modal_config is required for Modal inference")
        try:
            import modal
        except ImportError as exc:  # pragma: no cover - depends on installation
            raise ProviderDependencyError("install the 'modal' package to call Modal") from exc
        lookup_options = (
            {"environment_name": config.environment_name}
            if config.environment_name is not None
            else {}
        )
        remote_function = modal.Function.from_name(
            config.app_name,
            config.function_name,
            **lookup_options,
        )

    result = remote_function.remote(
        model=model,
        messages=[message.to_dict() for message in messages],
        system=system,
        max_output_tokens=max_output_tokens,
        response_schema=json_schema(response_format) if response_format is not None else None,
        **options,
    )
    if isinstance(result, str):
        result_dict: Mapping[str, Any] = {"text": result}
        raw = result
    elif isinstance(result, Mapping):
        result_dict = result
        raw = result
    else:
        raise TypeError("Modal inference function must return a string or mapping")

    text = str(result_dict.get("text", ""))
    usage = result_dict.get("usage", {})
    return ModelResponse(
        text=text,
        parsed=parse_structured(text, response_format),
        provider=Provider.MODAL,
        model=str(result_dict.get("model", model)),
        request_id=attr(result_dict, "request_id"),
        finish_reason=attr(result_dict, "finish_reason"),
        usage=TokenUsage(
            input_tokens=attr(usage, "input_tokens"),
            output_tokens=attr(usage, "output_tokens"),
        ),
        raw=raw,
    )


def _call_openai_web(
    *,
    model: str,
    messages: list[Message],
    system: str | None,
    response_format: ResponseFormat[Any] | None,
    max_output_tokens: int | None,
    config: ModalConfig,
    client: Any,
    options: dict[str, Any],
) -> ModelResponse[Any]:
    """Call a Modal-hosted vLLM server through its OpenAI-compatible API."""

    if client is None:
        try:
            import modal
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover - depends on installation
            raise ProviderDependencyError(
                "install the 'modal' and 'openai' packages for Modal web inference"
            ) from exc
        lookup_options = (
            {"environment_name": config.environment_name}
            if config.environment_name is not None
            else {}
        )
        function = modal.Function.from_name(
            config.app_name,
            config.function_name,
            **lookup_options,
        ).hydrate()
        web_url = function.get_web_url()
        if not web_url:
            raise ValueError("the configured Modal function has no web endpoint URL")
        client = OpenAI(base_url=web_url.rstrip("/") + "/v1", api_key="modal")

    request_messages = [message.to_dict() for message in messages]
    if system is not None:
        request_messages.insert(0, {"role": "system", "content": system})
    request: dict[str, Any] = {
        "model": model,
        "messages": request_messages,
        **options,
    }
    if max_output_tokens is not None:
        request["max_tokens"] = max_output_tokens
    if response_format is not None:
        request["response_format"] = {
            "type": "json_schema",
            "json_schema": {
                "name": schema_name(response_format),
                "schema": json_schema(response_format),
            },
        }

    response = client.chat.completions.create(**request)
    choices = attr(response, "choices", [])
    if not choices:
        raise RuntimeError("Modal vLLM endpoint returned no choices")
    choice = choices[0]
    text = str(attr(attr(choice, "message"), "content", "") or "")
    usage = attr(response, "usage")
    return ModelResponse(
        text=text,
        parsed=parse_structured(text, response_format),
        provider=Provider.MODAL,
        model=str(attr(response, "model", model)),
        request_id=attr(response, "id"),
        finish_reason=attr(choice, "finish_reason"),
        usage=TokenUsage(
            input_tokens=attr(usage, "prompt_tokens"),
            output_tokens=attr(usage, "completion_tokens"),
        ),
        raw=response,
    )
