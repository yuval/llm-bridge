from enum import Enum
from dotenv import load_dotenv
import os

class Provider(str, Enum):
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GEMINI = "gemini"

_ENV_VARS: dict[Provider, str] = {
    Provider.OPENAI: "OPENAI_API_KEY",
    Provider.ANTHROPIC: "ANTHROPIC_API_KEY",
    Provider.GEMINI: "GEMINI_API_KEY",
}

def get_api_key(provider: Provider) -> str:
    load_dotenv()
    env = _ENV_VARS.get(provider)
    if not env:
        raise RuntimeError(f"No config for {provider}")
    key = os.getenv(env)
    if not key:
        raise RuntimeError(f"{env} missing")
    return key
