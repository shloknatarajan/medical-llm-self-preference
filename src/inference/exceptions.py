"""Exceptions raised by the unified inference interface."""


class InferenceError(RuntimeError):
    """Base exception for inference configuration and response errors."""


class ProviderNotFoundError(InferenceError):
    """Raised when a provider cannot be inferred from a model name."""


class ProviderDependencyError(InferenceError):
    """Raised when the selected provider's optional SDK is unavailable."""


class ResponseParseError(InferenceError):
    """Raised when a structured response cannot be parsed or validated."""
