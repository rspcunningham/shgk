from __future__ import annotations

import asyncio
import json
import time

import httpx

from shgk.openrouter import (
    OpenRouterClient,
    OpenRouterRateLimiter,
    get_openrouter_client,
    is_free_model,
    load_openrouter_model_catalog,
    model_from_httpx_request,
    reset_openrouter_client,
    set_openrouter_model_catalog,
)
from shgk.providers import ProviderModelFactory


def _request(model: str, method: str = "POST") -> httpx.Request:
    return httpx.Request(
        method,
        "https://openrouter.ai/api/v1/chat/completions",
        content=json.dumps({"model": model, "messages": []}).encode("utf-8"),
    )


def test_identifies_free_models_from_request_bodies() -> None:
    assert is_free_model("nvidia/nemotron-3-ultra-550b-a55b:free")
    assert not is_free_model("nvidia/nemotron-3-ultra-550b-a55b")
    assert (
        model_from_httpx_request(_request("z-ai/glm-5.2:free")) == "z-ai/glm-5.2:free"
    )
    assert model_from_httpx_request(_request("z-ai/glm-5.2")) == "z-ai/glm-5.2"


def test_limiter_spaces_free_posts_and_skips_paid(tmp_path) -> None:
    limiter = OpenRouterRateLimiter(interval=0.05, path=tmp_path / "rate-limit")
    free = _request("nvidia/foo:free")
    paid = _request("nvidia/foo")

    async def run() -> tuple[float, float]:
        started = time.monotonic()
        await limiter.gate_request(free)
        await limiter.gate_request(free)
        free_elapsed = time.monotonic() - started
        started = time.monotonic()
        await limiter.gate_request(paid)
        await limiter.gate_request(paid)
        paid_elapsed = time.monotonic() - started
        return free_elapsed, paid_elapsed

    free_elapsed, paid_elapsed = asyncio.run(run())
    assert free_elapsed >= 0.05
    assert paid_elapsed < 0.05
    assert limiter.should_gate(free)
    assert not limiter.should_gate(paid)
    assert not limiter.should_gate(_request("nvidia/foo:free", method="GET"))


def test_limiter_serializes_concurrent_free_requests(tmp_path) -> None:
    limiter = OpenRouterRateLimiter(interval=0.05, path=tmp_path / "rate-limit")
    free = _request("google/gemma-4-31b-it:free")

    async def run() -> float:
        started = time.monotonic()
        await asyncio.gather(limiter.gate_request(free), limiter.gate_request(free))
        return time.monotonic() - started

    assert asyncio.run(run()) >= 0.05


def test_factory_reuses_process_wide_openrouter_client(monkeypatch) -> None:
    reset_openrouter_client()
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    first = ProviderModelFactory("openrouter").model("nvidia/foo:free")
    second = ProviderModelFactory("openrouter").model("z-ai/glm-5.2:free")
    client = get_openrouter_client()
    try:
        assert first._client is client.openai
        assert second._client is client.openai
        assert get_openrouter_client() is client
        assert isinstance(client, OpenRouterClient)
    finally:
        reset_openrouter_client()


class _CatalogResponse:
    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict[str, object]:
        return {
            "data": [
                {
                    "id": "z-ai/glm-5.2:free",
                    "supported_parameters": ["structured_outputs", "tools"],
                },
                {
                    "id": "google/gemma-4-26b-a4b-it:free",
                    "supported_parameters": ["structured_outputs", "response_format"],
                },
                {
                    "id": "nvidia/nemotron-3-super-120b-a12b:free",
                    "supported_parameters": ["structured_outputs"],
                },
                {
                    "id": "nvidia/nemotron-3-ultra-550b-a55b:free",
                    "supported_parameters": ["tools", "temperature"],
                },
                {
                    "id": "google/gemma-4-31b-it:free",
                    "supported_parameters": ["response_format", "tools"],
                },
            ]
        }


class _CatalogClient:
    def get(self, url: str) -> _CatalogResponse:
        assert url.endswith("/models")
        return _CatalogResponse()


def test_openrouter_catalog_detects_structured_outputs() -> None:
    set_openrouter_model_catalog(None)
    try:
        catalog = load_openrouter_model_catalog(client=_CatalogClient())
        factory = ProviderModelFactory("openrouter")
        factory.require_structured_outputs("z-ai/glm-5.2:free")
        factory.require_structured_outputs("google/gemma-4-26b-a4b-it:free")
        factory.require_structured_outputs("nvidia/nemotron-3-super-120b-a12b:free")
        assert factory.supports_structured_outputs("z-ai/glm-5.2:free")
        assert not factory.supports_structured_outputs(
            "nvidia/nemotron-3-ultra-550b-a55b:free"
        )
        assert not factory.supports_structured_outputs("google/gemma-4-31b-it:free")
        assert factory.extra_body() == {"provider": {"require_parameters": True}}
        assert catalog["z-ai/glm-5.2:free"] == ("structured_outputs", "tools")
    finally:
        set_openrouter_model_catalog(None)


def test_openrouter_rejects_models_without_structured_outputs() -> None:
    set_openrouter_model_catalog(
        {
            "nvidia/nemotron-3-ultra-550b-a55b:free": ("tools",),
        }
    )
    try:
        factory = ProviderModelFactory("openrouter")
        try:
            factory.require_structured_outputs("nvidia/nemotron-3-ultra-550b-a55b:free")
        except ValueError as error:
            assert "does not support structured outputs" in str(error)
        else:
            raise AssertionError("expected ValueError")
        try:
            factory.require_structured_outputs("missing/model:free")
        except ValueError as error:
            assert "Unknown OpenRouter model" in str(error)
        else:
            raise AssertionError("expected ValueError")
    finally:
        set_openrouter_model_catalog(None)


def test_openai_always_supports_structured_outputs() -> None:
    factory = ProviderModelFactory("openai")
    factory.require_structured_outputs("gpt-5.6-sol")
    assert factory.supports_structured_outputs("gpt-5.6-sol")
    assert factory.extra_body() is None

