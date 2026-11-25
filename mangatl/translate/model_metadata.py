import re


def get_max_tokens_cap(provider: str, model_name: str | None) -> int | None:
    """
    Get the maximum allowed max_tokens value for a specific provider/model combination.

    Returns:
        - 16384 for OpenAI models with "chat" in the name
        - 32768 for Moonshot Kimi K2.x models
        - None for all other models (no cap, use existing 63488 max)
    """
    if not model_name:
        return None

    model_lower = model_name.lower()

    is_openai_model = provider == "OpenAI" or (
        provider == "OpenRouter" and ("openai/" in model_lower or model_lower.startswith("gpt-"))
    )
    if is_openai_model and "chat" in model_lower:
        return 16384
    if provider == "Moonshot AI" and "kimi-k2." in model_lower:
        return 32768
    return None


def is_gpt5_series(model_name: str | None) -> bool:
    """Check if a model is any GPT-5 variant (5, 5.2, 5.4, etc.)."""
    if not model_name:
        return False
    lm = model_name.lower()
    return lm.startswith("gpt-5") or "/gpt-5" in lm


def is_gpt5_chat_variant(model_name: str | None) -> bool:
    """Check if a GPT-5 model is a chat variant (non-reasoning)."""
    return is_gpt5_series(model_name) and "chat" in (model_name or "").lower()


def is_openai_model_family(model_name: str | None) -> bool:
    """Check if a model name is OpenAI-family, including OpenRouter-prefixed IDs."""
    if not model_name:
        return False
    lm = model_name.lower()
    return "openai/" in lm or lm.startswith("gpt-") or "/gpt-" in lm


def is_anthropic_model_family(model_name: str | None) -> bool:
    """Check if a model name is Anthropic/Claude-family, including OpenRouter-prefixed IDs."""
    if not model_name:
        return False
    lm = model_name.lower()
    return "anthropic/" in lm or lm.startswith("claude-")


_GPT5_GEN_RE = re.compile(r"gpt-(5(?:\.\d+)?)", re.IGNORECASE)


def get_gpt5_generation(model_name: str | None) -> str | None:
    """Extract the GPT-5 generation string.

    Returns '5.2', '5.3', '5.4', '5.5', '5.6', etc. or None if not a GPT-5 model.
    """
    if not model_name:
        return None
    m = _GPT5_GEN_RE.search(model_name)
    return m.group(1) if m else None


def _parse_gpt5_gen_parts(gen: str | None) -> tuple[int, int] | None:
    if not gen:
        return None
    try:
        if "." in gen:
            major, minor = (int(part) for part in gen.split(".", 1))
            return major, minor
        return int(gen), 0
    except ValueError:
        return None


def is_gpt6_series(model_name: str | None) -> bool:
    """Check if a model is any GPT-6 variant (Astra, Sol, Terra, Luna)."""
    if not model_name:
        return False
    lm = model_name.lower()
    return "gpt-6" in lm or "/gpt-6" in lm


def is_gpt6_astra(model_name: str | None) -> bool:
    """Check if a model is GPT-6 Astra."""
    if not model_name:
        return False
    lm = model_name.lower()
    return "gpt-6-astra" in lm


def is_gpt61_sol(model_name: str | None) -> bool:
    """Check if a model is GPT-6.1 Sol."""
    if not model_name:
        return False
    lm = model_name.lower()
    return "gpt-6.1-sol" in lm


def supports_openai_original_image_detail(model_name: str | None) -> bool:
    """Whether an OpenAI model supports image detail='original'."""
    if is_gpt6_series(model_name):
        return True

    if is_gpt5_series(model_name):
        lm = (model_name or "").lower()
        if any(token in lm for token in ("mini", "chat")):
            return False

        parts = _parse_gpt5_gen_parts(get_gpt5_generation(model_name))
        if parts is None:
            return False

        return parts >= (5, 4)

    return False


def supports_openai_verbosity(model_name: str | None) -> bool:
    """Whether an OpenAI model supports the text.verbosity parameter (GPT-5 series, GPT-6 series)."""
    if not model_name or is_gpt5_chat_variant(model_name):
        return False
    return is_gpt5_series(model_name) or is_gpt6_series(model_name)


def is_openai_reasoning_model(model_name: str | None) -> bool:
    """Check if an OpenAI model is reasoning-capable (GPT-5 series, GPT-6 series)."""
    if not model_name:
        return False
    lm = model_name.lower()
    return lm.startswith(("gpt-5", "gpt-6")) or "/gpt-5" in lm or "/gpt-6" in lm or is_gpt6_series(model_name)


def clamp_gpt6_reasoning_effort(model_name: str | None, effort: str) -> str:
    """GPT-6 has no "minimal" effort, and Astra / 6.1 Sol have no "none"; both become "low"."""
    if effort == "minimal" or (effort == "none" and (is_gpt6_astra(model_name) or is_gpt61_sol(model_name))):
        return "low"
    return effort


def is_azure_url(url: str | None) -> bool:
    """Check if a URL is an Azure OpenAI or Azure AI Services endpoint."""
    if not url:
        return False
    url_lower = url.lower()
    azure_domains = (
        ".azure.com",
        ".azure-api.net",
        ".azure.us",
        ".azure.cn",
        ".microsoftazure.de",
    )
    return any(domain in url_lower for domain in azure_domains)


def is_openai_compatible_reasoning_model(model_name: str | None) -> bool:
    """Check if an OpenAI-Compatible model is reasoning-capable."""
    if not model_name:
        return False
    lm = model_name.lower()
    return (
        "thinking" in lm
        or "reasoning" in lm
        or is_openai_reasoning_model(model_name)
        or is_anthropic_reasoning_model(model_name)
    )


def is_anthropic_reasoning_model(model_name: str | None) -> bool:
    """Check if an Anthropic model is reasoning-capable.

    Uses `in` (not `startswith`) so OpenRouter-prefixed names also match.
    """
    if not model_name:
        return False
    lm = model_name.lower()
    return (
        "claude-opus-4" in lm
        or is_sonnet_46_model(model_name)
        or is_sonnet_5_model(model_name)
        or is_sonnet_55_model(model_name)
        or is_opus_5_model(model_name)
        or is_opus_55_model(model_name)
        or is_fable_5_model(model_name)
        or is_haiku_55_model(model_name)
    )


def is_opus_45_model(model_name: str | None) -> bool:
    """Check if a model is Claude Opus 4.5 (supports effort parameter)."""
    if not model_name:
        return False
    lm = model_name.lower()
    if "claude" not in lm or "opus" not in lm:
        return False
    return ("4.5" in lm) or ("4-5" in lm)


def is_opus_46_model(model_name: str | None) -> bool:
    """Check if a model is Claude Opus 4.6 (adaptive thinking, max effort)."""
    if not model_name:
        return False
    lm = model_name.lower()
    if "claude" not in lm or "opus" not in lm:
        return False
    return ("4.6" in lm) or ("4-6" in lm)


def is_opus_47_model(model_name: str | None) -> bool:
    """Check if a model is Claude Opus 4.7 (adaptive thinking, xhigh effort, no sampling params)."""
    if not model_name:
        return False
    lm = model_name.lower()
    if "claude" not in lm or "opus" not in lm:
        return False
    return ("4.7" in lm) or ("4-7" in lm)


def is_opus_48_model(model_name: str | None) -> bool:
    """Check if a model is Claude Opus 4.8 (same API constraints as Opus 4.7)."""
    if not model_name:
        return False
    lm = model_name.lower()
    if "claude" not in lm or "opus" not in lm:
        return False
    return ("4.8" in lm) or ("4-8" in lm)


def is_opus_5_model(model_name: str | None) -> bool:
    """Check if a model is Claude Opus 5 (adaptive thinking on by default, max effort support, no sampling params)."""
    if not model_name:
        return False
    lm = model_name.lower()
    if "claude" not in lm or "opus" not in lm:
        return False
    return "claude-opus-5" in lm


def is_opus_55_model(model_name: str | None) -> bool:
    """Check if a model is Claude Opus 5.5 (always-on adaptive thinking, max effort support, default effort medium, no sampling params)."""
    if not model_name:
        return False
    lm = model_name.lower()
    if "claude" not in lm or "opus" not in lm:
        return False
    return ("5.5" in lm) or ("5-5" in lm)


def is_fable_5_model(model_name: str | None) -> bool:
    """Check if a model is Claude Fable 5+ (always-on adaptive thinking, no sampling params)."""
    if not model_name:
        return False
    lm = model_name.lower()
    return "claude-fable-5" in lm


def is_sonnet_5_model(model_name: str | None) -> bool:
    """Check if a model is Claude Sonnet 5 (adaptive thinking on by default, no sampling params)."""
    if not model_name:
        return False
    lm = model_name.lower()
    if "claude" not in lm or "sonnet" not in lm:
        return False
    if is_sonnet_55_model(model_name):
        return False
    return "claude-sonnet-5" in lm


def is_sonnet_55_model(model_name: str | None) -> bool:
    """Check if a model is Claude Sonnet 5.5 (adaptive thinking on by default, effort parameter with max effort, no sampling params)."""
    if not model_name:
        return False
    lm = model_name.lower()
    if "claude" not in lm or "sonnet" not in lm:
        return False
    return ("5.5" in lm) or ("5-5" in lm)


def is_haiku_55_model(model_name: str | None) -> bool:
    """Check if a model is Claude Haiku 5.5 (adaptive thinking on by default, no sampling params)."""
    if not model_name:
        return False
    lm = model_name.lower()
    return "claude-haiku-5-5" in lm or "claude-haiku-5.5" in lm


def anthropic_model_flags(model_name: str | None) -> dict[str, bool]:
    """Cumulative Claude API capability flags for generation_config / OpenRouter metadata.

    Tiers (each includes the capabilities of tiers below):
    - Opus 4.5: is_claude_effort
    - Sonnet/Opus 4.6: + is_claude_effort_max
    - Sonnet/Opus 4.7/4.8/5: + is_claude_effort_xhigh (also strips sampling params)
    - Sonnet/Opus 5, Haiku 5.5: + is_claude_adaptive_default
    - Fable 5/5.1, Opus 5.5: same as 4.7+ plus is_claude_omit_thinking
    """
    if not model_name:
        return {}
    if is_fable_5_model(model_name) or is_opus_55_model(model_name):
        return {
            "is_claude_effort": True,
            "is_claude_effort_max": True,
            "is_claude_effort_xhigh": True,
            "is_claude_omit_thinking": True,
        }
    if is_sonnet_55_model(model_name):
        return {
            "is_claude_effort": True,
            "is_claude_effort_max": True,
            "is_claude_effort_xhigh": True,
            "is_claude_adaptive_default": True,
            "is_claude_between_tools": True,
        }
    if is_sonnet_5_model(model_name) or is_opus_5_model(model_name) or is_haiku_55_model(model_name):
        return {
            "is_claude_effort": True,
            "is_claude_effort_max": True,
            "is_claude_effort_xhigh": True,
            "is_claude_adaptive_default": True,
        }
    if is_opus_47_model(model_name) or is_opus_48_model(model_name):
        return {
            "is_claude_effort": True,
            "is_claude_effort_max": True,
            "is_claude_effort_xhigh": True,
        }
    if is_46_model(model_name):
        return {
            "is_claude_effort": True,
            "is_claude_effort_max": True,
        }
    if is_opus_45_model(model_name):
        return {"is_claude_effort": True}
    return {}


def is_sonnet_46_model(model_name: str | None) -> bool:
    """Check if a model is Claude Sonnet 4.6 (adaptive thinking, max effort)."""
    if not model_name:
        return False
    lm = model_name.lower()
    if "claude" not in lm or "sonnet" not in lm:
        return False
    return ("4.6" in lm) or ("4-6" in lm)


def is_46_model(model_name: str | None) -> bool:
    """Check if a model is any Claude 4.6 variant (Opus or Sonnet)."""
    return is_opus_46_model(model_name) or is_sonnet_46_model(model_name)


def is_gemini_4_model(model_name: str | None) -> bool:
    """Check if a model is any Gemini 4 variant (e.g., gemini-4-argon)."""
    if not model_name:
        return False
    return "gemini-4" in model_name.lower()


def is_gemini_38_flash_model(model_name: str | None) -> bool:
    """Check if a model is Gemini 3.8 Flash."""
    if not model_name:
        return False
    lm = model_name.lower()
    return "gemini-3.8" in lm and "flash" in lm


def is_gemini_36_flash_model(model_name: str | None) -> bool:
    """Check if a model is Gemini 3.6 Flash."""
    if not model_name:
        return False
    lm = model_name.lower()
    return "gemini-3.6" in lm and "flash" in lm


def is_gemini_35_flash_lite_model(model_name: str | None) -> bool:
    """Check if a model is Gemini 3.5 Flash Lite."""
    if not model_name:
        return False
    lm = model_name.lower()
    return "gemini-3.5" in lm and "flash" in lm and "lite" in lm


def is_gemini_no_sampling_model(model_name: str | None) -> bool:
    """Check if a Gemini model does not support custom sampling parameters."""
    return (
        is_gemini_38_flash_model(model_name)
        or is_gemini_36_flash_model(model_name)
        or is_gemini_35_flash_lite_model(model_name)
        or is_gemini_4_model(model_name)
    )
