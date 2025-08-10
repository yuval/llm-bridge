"""
Parameter normalization for llm-bridge.

Public API
- Users pass a dict to `params` on `client.chat` or `client.stream`.

Contract
- Standard keys work across providers:
  temperature: float
  max_tokens: int
  top_p: float
  stream: bool
  tools: list
  tool_choice: str | dict
  stop: str | list[str]
  response_format: dict
  user: str

- Provider specific keys go under `extra` and pass through unchanged.
  Examples:
    extra.reasoning_effort: "low" | "medium" | "high"
    extra.verbosity: "low" | "medium" | "high"
    extra.logit_bias: dict

Unknown top-level keys are moved into extra.
Unknown extra keys are forwarded as-is.
"""

from __future__ import annotations

from typing import Any

# Type alias for chat messages
ChatMessage = dict[str, Any]

STANDARD_KEYS = {
    "temperature",
    "max_tokens",
    "top_p",
    "stream",
    "tools",
    "tool_choice",
    "stop",
    "response_format",
    "user",
    "frequency_penalty",
    "presence_penalty",
    "parallel_tool_calls",
    "seed",
}


def normalize_params(params: dict | None) -> dict:
    """
    Normalize a user-supplied params dict to a single internal shape.

    Returns a dict with only standard keys plus an `extra` dict.
    Defaults:
      stream defaults to False
      extra defaults to {}
    Rules:
      - Keys not in STANDARD_KEYS are moved into extra
      - If the caller already passed an `extra` dict it is merged last
      - None values are kept so adapters can decide to drop them

    Example
    -------
    >>> normalize_params({
    ...   "temperature": 0.2,
    ...   "max_tokens": 4000,
    ...   "reasoning_effort": "high",
    ...   "extra": {"verbosity": "high"}
    ... })
    {'temperature': 0.2, 'max_tokens': 4000, 'stream': False,
     'extra': {'reasoning_effort': 'high', 'verbosity': 'high'}}
    """
    if params is None:
        return {"stream": False, "extra": {}}
    if not isinstance(params, dict):
        raise TypeError(f"params must be a dict, got {type(params).__name__}")

    std: dict = {}
    extra: dict = {}

    # Pull user-provided extra first so later moves can override
    user_extra = params.get("extra") or {}
    if user_extra and not isinstance(user_extra, dict):
        raise TypeError("params['extra'] must be a dict")

    for key, value in params.items():
        if key == "extra":
            continue
        if key in STANDARD_KEYS:
            std[key] = value
        else:
            extra[key] = value

    # Defaults
    std.setdefault("stream", False)

    # Final extra merge: moved unknowns first, then user-provided extra wins
    merged_extra = {**extra, **user_extra}
    std["extra"] = merged_extra

    return std


def merge_params(defaults: dict | None, overrides: dict | None) -> dict:
    """
    Shallow-merge client defaults with per-call overrides, then normalize.

    Rules:
      - Top-level keys are overwritten by overrides
      - `extra` is merged with overrides winning per key
    """
    base: dict = dict(defaults or {})
    if overrides:
        base_extra = dict(base.get("extra") or {})
        over_extra = dict(overrides.get("extra") or {})

        # Apply top-level overrides except extra
        for k, v in overrides.items():
            if k != "extra":
                base[k] = v

        # Merge extra
        base["extra"] = {**base_extra, **over_extra}

    return normalize_params(base)
