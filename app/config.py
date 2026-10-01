"""Runtime configuration, loaded only from environment variables.

No secret ever lives in this repo. Keys come from the process environment,
optionally seeded from a local `.env` file (which is git-ignored). Real
environment variables always win over `.env`, so production secrets injected
by the platform cannot be shadowed by a stray file.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

from dotenv import load_dotenv

# Which env var each LangChain provider prefix needs. Used to fail fast at
# startup with a clear message instead of failing on the first request.
PROVIDER_KEY_VARS: dict[str, str] = {
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "google_genai": "GOOGLE_API_KEY",
    "groq": "GROQ_API_KEY",
    "mistralai": "MISTRAL_API_KEY",
}


# Same idea as init_chat_model's own inference for bare model names.
_NAME_PREFIXES: tuple[tuple[tuple[str, ...], str], ...] = (
    (("gpt-", "o1", "o3", "o4", "chatgpt"), "openai"),
    (("claude",), "anthropic"),
    (("gemini",), "google_genai"),
    (("mistral",), "mistralai"),
)


def provider_of(spec: str) -> str | None:
    """Provider of a "provider:model" string, or a best guess for a bare name."""
    if ":" in spec:
        return spec.split(":", 1)[0]
    name = spec.lower()
    for prefixes, provider in _NAME_PREFIXES:
        if name.startswith(prefixes):
            return provider
    return None


class ConfigError(RuntimeError):
    """Raised when the service cannot start because configuration is missing."""


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    return float(raw) if raw else default


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    return int(raw) if raw else default


@dataclass(frozen=True)
class Settings:
    # Models use LangChain's "provider:model" format so the provider can be
    # swapped from the environment without touching code.
    router_model: str = "openai:gpt-5.6-luna"
    answer_model: str = "openai:gpt-5.6-luna"
    # Optional. Only passed to the model when set, since not every model
    # accepts it.
    reasoning_effort: str | None = None

    # Start the general answer while the router is still deciding.
    speculative_routing: bool = True

    llm_timeout_s: float = 30.0
    llm_max_retries: int = 2
    # Max silence before the first chunk and between chunks. Raise it for
    # reasoning models on high effort, which can think for a while first.
    stream_idle_timeout_s: float = 30.0

    # Optional shared secret for callers of this service. repr=False keeps it
    # out of logs and tracebacks.
    service_api_key: str | None = field(default=None, repr=False)

    @classmethod
    def from_env(cls) -> Settings:
        load_dotenv(override=False)
        return cls(
            router_model=os.getenv("ROUTER_MODEL", cls.router_model),
            answer_model=os.getenv("ANSWER_MODEL", cls.answer_model),
            reasoning_effort=os.getenv("REASONING_EFFORT") or None,
            speculative_routing=_env_bool("SPECULATIVE_ROUTING", cls.speculative_routing),
            llm_timeout_s=_env_float("LLM_TIMEOUT_S", cls.llm_timeout_s),
            llm_max_retries=_env_int("LLM_MAX_RETRIES", cls.llm_max_retries),
            stream_idle_timeout_s=_env_float("STREAM_IDLE_TIMEOUT_S", cls.stream_idle_timeout_s),
            service_api_key=os.getenv("SERVICE_API_KEY") or None,
        )

    def require_provider_keys(self) -> None:
        """Check that every provider we are about to call has its key set.

        Only the variable name is reported. The value is never read into a
        message, log line or exception.
        """
        missing = []
        for spec in {self.router_model, self.answer_model}:
            var = PROVIDER_KEY_VARS.get(provider_of(spec) or "")
            if var and not os.getenv(var):
                missing.append(var)
        if missing:
            names = ", ".join(sorted(set(missing)))
            raise ConfigError(
                f"Missing required environment variable(s): {names}. "
                "Copy .env.example to .env and fill them in, or export them."
            )
