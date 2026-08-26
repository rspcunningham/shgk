from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any

from agents.models.openai_chatcompletions import OpenAIChatCompletionsModel

from .openrouter import get_openrouter_client, openrouter_supports_structured_outputs


def parse_json_payload(value: object) -> object:
    """Parse JSON-only model output while tolerating a single markdown fence."""

    if not isinstance(value, str):
        return value
    text = value.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        if start < 0:
            raise ValueError("model did not return a JSON object") from None
        try:
            payload, _ = json.JSONDecoder().raw_decode(text[start:])
        except json.JSONDecodeError as error:
            raise ValueError(f"model returned malformed JSON: {error}") from error
        return payload


@dataclass(slots=True)
class ProviderModelFactory:
    """Resolve CLI provider/model names into Agents SDK model objects."""

    provider: str

    def __post_init__(self) -> None:
        self.provider = self.provider.lower().strip()
        if self.provider not in {"openai", "openrouter", "anthropic"}:
            raise ValueError(
                f"Unsupported provider {self.provider!r}; "
                "choose openai, anthropic, or openrouter"
            )

    @property
    def api_key_env(self) -> str:
        return {
            "openai": "OPENAI_API_KEY",
            "anthropic": "ANTHROPIC_API_KEY",
            "openrouter": "OPENROUTER_API_KEY",
        }[self.provider]

    def extra_body(self) -> dict[str, Any] | None:
        """Provider extras so OpenRouter only routes to schema-capable endpoints."""

        if self.provider != "openrouter":
            return None
        return {"provider": {"require_parameters": True}}

    def supports_structured_outputs(self, model_name: str) -> bool:
        if self.provider in {"openai", "anthropic"}:
            return True
        return openrouter_supports_structured_outputs(model_name)

    def require_structured_outputs(self, model_name: str) -> None:
        if self.supports_structured_outputs(model_name):
            return
        raise ValueError(
            f"{self.provider}:{model_name} does not support structured outputs. "
            "This tool only runs models that advertise native JSON schema."
        )

    def require_api_key(self) -> None:
        if not os.environ.get(self.api_key_env):
            raise RuntimeError(
                f"{self.api_key_env} is not available in the environment, "
                ".env.local, or .env"
            )

    def model(self, model_name: str) -> str | Any:
        self.require_api_key()
        if self.provider == "anthropic":
            raise ValueError(
                "Anthropic models do not use the Agents SDK; "
                "use shgk.anthropic_provider instead"
            )
        if self.provider == "openai":
            return model_name
        return OpenAIChatCompletionsModel(
            model=model_name,
            openai_client=get_openrouter_client().openai,
        )


def parse_model_spec(spec: str) -> tuple[str, str]:
    """Parse PROVIDER:MODEL while preserving colons inside OpenRouter model IDs."""

    provider, separator, model = spec.partition(":")
    if not separator or not provider.strip() or not model.strip():
        raise ValueError(f"Invalid model specification {spec!r}; use PROVIDER:MODEL")
    provider = provider.lower().strip()
    ProviderModelFactory(provider)  # validate the provider name
    return provider, model.strip()
