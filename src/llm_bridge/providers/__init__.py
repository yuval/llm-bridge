from __future__ import annotations

import os
from enum import StrEnum
from typing import Final

from dotenv import load_dotenv

load_dotenv()

class Provider(StrEnum):
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GEMINI = "gemini"

_ENV_VARS: Final[dict[Provider, str]] = {
    Provider.OPENAI: "OPENAI_API_KEY",
    Provider.ANTHROPIC: "ANTHROPIC_API_KEY",
    Provider.GEMINI: "GEMINI_API_KEY",
}

def get_api_key(provider: Provider) -> str:
    """Return the API key for *provider* or raise RuntimeError."""
    try:
        env_var = _ENV_VARS[provider]
    except KeyError:
        raise RuntimeError(f"No config for {provider!s}") from None

    try:
        return os.environ[env_var]
    except KeyError as exc:
        raise RuntimeError(f"{env_var} missing") from exc

__all__ = ["Provider", "get_api_key"]