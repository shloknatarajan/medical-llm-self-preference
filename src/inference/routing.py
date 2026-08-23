"""Model-name based provider routing."""

from __future__ import annotations

from .exceptions import ProviderNotFoundError
from .types import Provider


_PREFIXES = {
    "openai": Provider.OPENAI,
    "anthropic": Provider.ANTHROPIC,
    "claude": Provider.ANTHROPIC,
    "gemini": Provider.GEMINI,
    "google": Provider.GEMINI,
    "modal": Provider.MODAL,
}


def resolve_provider(
    model: str,
    provider: Provider | str | None = None,
) -> tuple[Provider, str]:
    """Return the selected provider and provider-native model name.

    Explicit names can use ``provider/model`` (for example,
    ``modal/Qwen3.5-35B``). Well-known API model families may omit the prefix.
    """

    if not isinstance(model, str) or not model.strip():
        raise ValueError("model must be a non-empty string")
    model = model.strip()

    prefix, separator, unprefixed = model.partition("/")
    prefixed_provider = _PREFIXES.get(prefix.lower()) if separator else None

    if provider is not None:
        try:
            selected = provider if isinstance(provider, Provider) else Provider(provider.lower())
        except ValueError as exc:
            raise ProviderNotFoundError(f"unsupported provider: {provider!r}") from exc
        if prefixed_provider is not None and prefixed_provider is not selected:
            raise ValueError(
                f"model prefix selects {prefixed_provider.value}, but provider selects {selected.value}"
            )
        return selected, unprefixed if prefixed_provider else model

    if prefixed_provider is not None:
        if not unprefixed:
            raise ValueError("provider-prefixed model name cannot be empty")
        return prefixed_provider, unprefixed

    lowered = model.lower()
    if lowered.startswith(("gpt-", "o1", "o3", "o4")):
        return Provider.OPENAI, model
    if lowered.startswith("claude-"):
        return Provider.ANTHROPIC, model
    if lowered.startswith("gemini-"):
        return Provider.GEMINI, model
    if lowered.startswith(("qwen", "llama", "mistral", "deepseek")):
        return Provider.MODAL, model

    raise ProviderNotFoundError(
        f"cannot infer a provider for {model!r}; use provider='...' or a provider/model prefix"
    )
