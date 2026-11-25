"""Default sampling parameters for the OpenAI-compatible translator."""

from __future__ import annotations

DEFAULT_LLM_PROVIDER = "OpenAI-Compatible"

_SAMPLING_DEFAULTS: dict[str, float | int] = {"temperature": 0.1, "top_p": 0.95, "top_k": 40}


def get_provider_sampling_defaults(provider: str | None = None) -> dict[str, float | int]:
    """Return a copy of the sampling defaults."""
    return dict(_SAMPLING_DEFAULTS)


__all__ = ["DEFAULT_LLM_PROVIDER", "get_provider_sampling_defaults"]
