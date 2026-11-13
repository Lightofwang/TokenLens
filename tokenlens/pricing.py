from __future__ import annotations

from typing import Any

# Price in USD per 1,000,000 tokens: (input, cache_read, cache_write, output)
MODEL_PRICING: dict[str, tuple[float, float, float, float]] = {
    "claude-opus": (15.0, 1.5, 18.75, 75.0),
    "claude-sonnet": (3.0, 0.3, 3.75, 15.0),
    "claude-haiku": (0.8, 0.08, 1.0, 4.0),
    "gpt-5": (2.5, 0.25, 2.5, 10.0),
    "gpt-6": (2.5, 0.25, 2.5, 10.0),
    "gpt-4o": (2.5, 1.25, 2.5, 10.0),
    "gpt-4.1": (2.0, 0.5, 2.0, 8.0),
    "o3": (2.0, 0.5, 2.0, 8.0),
    "o4-mini": (1.1, 0.275, 1.1, 4.4),
    "gemini-2.5-pro": (1.25, 0.31, 1.25, 10.0),
    "gemini-3": (1.25, 0.31, 1.25, 10.0),
    "gemini-flash": (0.15, 0.0375, 0.15, 0.6),
    "deepseek": (0.27, 0.07, 0.27, 1.10),
    "qwen": (0.4, 0.1, 0.4, 1.2),
    "glm": (0.5, 0.1, 0.5, 2.0),
    "kimi": (0.6, 0.15, 0.6, 2.5),
    "minimax": (0.3, 0.08, 0.3, 1.2),
    "sensenova": (0.3, 0.08, 0.3, 1.0),
}

DEFAULT_PRICE = (1.50, 0.30, 1.50, 6.00)


def lookup_price(model: str, custom_pricing: dict[str, Any] | None = None) -> tuple[float, float, float, float]:
    m = (model or "").lower().strip()
    if custom_pricing:
        for k, v in custom_pricing.items():
            if k.lower() in m and isinstance(v, (list, tuple)) and len(v) == 4:
                return tuple(float(x) for x in v)  # type: ignore[return-value]

    if "opus" in m:
        return MODEL_PRICING["claude-opus"]
    if "sonnet" in m:
        return MODEL_PRICING["claude-sonnet"]
    if "haiku" in m:
        return MODEL_PRICING["claude-haiku"]
    if "flash" in m:
        if "deepseek" in m:
            return MODEL_PRICING["deepseek"]
        return MODEL_PRICING["gemini-flash"]
    if "gemini" in m:
        return MODEL_PRICING["gemini-3"]
    if "deepseek" in m:
        return MODEL_PRICING["deepseek"]
    if "qwen" in m:
        return MODEL_PRICING["qwen"]
    if "glm" in m:
        return MODEL_PRICING["glm"]
    if "kimi" in m:
        return MODEL_PRICING["kimi"]
    if "minimax" in m:
        return MODEL_PRICING["minimax"]
    if "gpt-6" in m:
        return MODEL_PRICING["gpt-6"]
    if "gpt-5" in m:
        return MODEL_PRICING["gpt-5"]
    if "gpt-4o" in m:
        return MODEL_PRICING["gpt-4o"]
    if "gpt-4" in m:
        return MODEL_PRICING["gpt-4.1"]
    if "o3" in m or "o4" in m:
        return MODEL_PRICING["o3"]
    return DEFAULT_PRICE


def estimate_cost_usd(
    model: str,
    uncached_input: int,
    cache_read: int,
    cache_write: int,
    output: int,
    custom_pricing: dict[str, Any] | None = None,
) -> tuple[float, float]:
    """Return (actual_cost_usd, cache_savings_usd)."""
    p_in, p_cr, p_cw, p_out = lookup_price(model, custom_pricing)
    actual = (
        uncached_input * p_in
        + cache_read * p_cr
        + cache_write * p_cw
        + output * p_out
    ) / 1_000_000.0
    without_cache = (cache_read * max(0.0, p_in - p_cr)) / 1_000_000.0
    return actual, without_cache
