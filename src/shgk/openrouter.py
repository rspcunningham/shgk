from __future__ import annotations

import asyncio
import fcntl
import json
import os
import threading
import time
from pathlib import Path

import httpx
from openai import AsyncOpenAI, DefaultAsyncHttpxClient


OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
OPENROUTER_MODELS_URL = f"{OPENROUTER_BASE_URL}/models"
FREE_MODEL_INTERVAL = 3.2
_CLIENT_LOCK = threading.Lock()
_CATALOG_LOCK = threading.Lock()
_client: OpenRouterClient | None = None
_model_catalog: dict[str, tuple[str, ...]] | None = None


def is_free_model(model: str) -> bool:
    return model.endswith(":free")


def model_from_httpx_request(request: httpx.Request) -> str:
    try:
        content = request.content
    except Exception:
        return ""
    if not content:
        return ""
    try:
        payload = json.loads(content)
    except (json.JSONDecodeError, UnicodeDecodeError, TypeError):
        return ""
    if not isinstance(payload, dict):
        return ""
    model = payload.get("model")
    return model if isinstance(model, str) else ""


class OpenRouterRateLimiter:
    """Space :free OpenRouter request starts evenly across the process and machine."""

    def __init__(
        self,
        *,
        interval: float | None = None,
        path: str | Path | None = None,
    ):
        if interval is None:
            interval = float(
                os.environ.get("SHGK_OPENROUTER_FREE_INTERVAL", FREE_MODEL_INTERVAL)
            )
        self.interval = max(0.0, interval)
        self.path = Path(
            path
            or os.environ.get(
                "SHGK_OPENROUTER_RATE_LIMIT_FILE",
                "data/.openrouter-rate-limit",
            )
        )
        self._lock = asyncio.Lock()

    def should_gate(self, request: httpx.Request) -> bool:
        if request.method not in {"POST", "PUT", "PATCH"}:
            return False
        return is_free_model(model_from_httpx_request(request))

    def _reserve_slot(self) -> None:
        if self.interval <= 0:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a+", encoding="utf-8") as stream:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
            stream.seek(0)
            try:
                previous = float(stream.read().strip() or 0)
            except ValueError:
                previous = 0
            now = time.time()
            delay = self.interval - (now - previous)
            if 0 < delay <= self.interval:
                time.sleep(delay)
            stream.seek(0)
            stream.truncate()
            stream.write(str(time.time()))
            stream.flush()
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)

    async def gate_request(self, request: httpx.Request) -> None:
        if not self.should_gate(request):
            return
        async with self._lock:
            await asyncio.to_thread(self._reserve_slot)


class OpenRouterClient:
    """Process-wide OpenAI-compatible client with :free RPM gating on the wire."""

    def __init__(
        self,
        *,
        api_key: str,
        limiter: OpenRouterRateLimiter | None = None,
    ):
        if not api_key:
            raise RuntimeError(
                "OPENROUTER_API_KEY is not available in the environment, "
                ".env.local, or .env"
            )
        self.limiter = limiter or OpenRouterRateLimiter()
        headers = {"X-Title": "SHGK translation benchmark"}
        referer = os.environ.get("OPENROUTER_HTTP_REFERER")
        if referer:
            headers["HTTP-Referer"] = referer
        self._http = DefaultAsyncHttpxClient(
            event_hooks={"request": [self.limiter.gate_request]},
        )
        self.openai = AsyncOpenAI(
            api_key=api_key,
            base_url=OPENROUTER_BASE_URL,
            default_headers=headers,
            http_client=self._http,
        )

    @classmethod
    def from_env(cls) -> OpenRouterClient:
        return cls(api_key=os.environ.get("OPENROUTER_API_KEY", ""))


def get_openrouter_client() -> OpenRouterClient:
    global _client
    if _client is None:
        with _CLIENT_LOCK:
            if _client is None:
                _client = OpenRouterClient.from_env()
    return _client


def reset_openrouter_client() -> None:
    global _client
    with _CLIENT_LOCK:
        _client = None


def set_openrouter_model_catalog(
    catalog: dict[str, tuple[str, ...]] | None,
) -> None:
    """Install or clear the in-process OpenRouter /models cache."""

    global _model_catalog
    with _CATALOG_LOCK:
        _model_catalog = catalog


def load_openrouter_model_catalog(
    *,
    client: httpx.Client | None = None,
) -> dict[str, tuple[str, ...]]:
    """Return model id -> supported_parameters from OpenRouter's public catalog."""

    global _model_catalog
    if _model_catalog is not None:
        return _model_catalog
    with _CATALOG_LOCK:
        if _model_catalog is not None:
            return _model_catalog
        owns_client = client is None
        http = client or httpx.Client(timeout=30.0)
        try:
            response = http.get(OPENROUTER_MODELS_URL)
            response.raise_for_status()
            payload = response.json()
        finally:
            if owns_client:
                http.close()
        catalog: dict[str, tuple[str, ...]] = {}
        for item in payload.get("data") or []:
            if not isinstance(item, dict):
                continue
            model_id = item.get("id")
            if not isinstance(model_id, str) or not model_id:
                continue
            params = item.get("supported_parameters") or []
            catalog[model_id] = tuple(params) if isinstance(params, list) else ()
        _model_catalog = catalog
        return _model_catalog


def openrouter_supports_structured_outputs(model: str) -> bool:
    catalog = load_openrouter_model_catalog()
    if model not in catalog:
        raise ValueError(
            f"Unknown OpenRouter model {model!r}; it is not in the OpenRouter catalog"
        )
    return "structured_outputs" in catalog[model]
