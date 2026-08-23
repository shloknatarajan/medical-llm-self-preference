"""Public provider-independent model calling API."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, TypeVar, overload

from ._utils import normalize_messages
from .providers.anthropic import call_anthropic
from .providers.gemini import call_gemini
from .providers.modal import call_modal
from .providers.openai import call_openai
from .routing import resolve_provider
from .types import Message, ModalConfig, ModelResponse, Provider, ResponseFormat


T = TypeVar("T")
Input = str | Sequence[Message | Mapping[str, str]]


@overload
def call_model(
    model: str,
    input: Input,
    *,
    response_format: type[T],
    provider: Provider | str | None = None,
    system: str | None = None,
    max_output_tokens: int | None = 1024,
    modal_config: ModalConfig | None = None,
    client: Any = None,
    remote_function: Any = None,
    **provider_options: Any,
) -> ModelResponse[T]: ...


@overload
def call_model(
    model: str,
    input: Input,
    *,
    response_format: Mapping[str, Any] | None = None,
    provider: Provider | str | None = None,
    system: str | None = None,
    max_output_tokens: int | None = 1024,
    modal_config: ModalConfig | None = None,
    client: Any = None,
    remote_function: Any = None,
    **provider_options: Any,
) -> ModelResponse[Any]: ...


def call_model(
    model: str,
    input: Input,
    *,
    response_format: ResponseFormat[Any] | None = None,
    provider: Provider | str | None = None,
    system: str | None = None,
    max_output_tokens: int | None = 1024,
    modal_config: ModalConfig | None = None,
    client: Any = None,
    remote_function: Any = None,
    **provider_options: Any,
) -> ModelResponse[Any]:
    """Call a model and return a normalized response.

    The provider is inferred from common model names or an explicit
    ``provider/model`` prefix. Pass ``provider`` for aliases that do not encode
    their backend. Provider SDK clients may be injected with ``client`` for
    custom configuration and testing.
    """

    selected_provider, native_model = resolve_provider(model, provider)
    system_text, messages = normalize_messages(input, system)
    common = {
        "model": native_model,
        "messages": messages,
        "system": system_text,
        "response_format": response_format,
        "max_output_tokens": max_output_tokens,
        "options": provider_options,
    }

    if selected_provider is Provider.OPENAI:
        return call_openai(**common, client=client)
    if selected_provider is Provider.ANTHROPIC:
        return call_anthropic(**common, client=client)
    if selected_provider is Provider.GEMINI:
        return call_gemini(**common, client=client)
    return call_modal(
        **common,
        config=modal_config,
        remote_function=remote_function,
        client=client,
    )
