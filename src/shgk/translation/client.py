"""The three SDK agents, and what counts as a retryable failure."""

from __future__ import annotations

import asyncio
import json
import time

import httpx
from agents import Agent, ModelSettings, RunConfig, Runner, set_default_openai_client
from openai import (
    APIConnectionError,
    APITimeoutError,
    AsyncOpenAI,
    DefaultAsyncHttpxClient,
    InternalServerError,
    RateLimitError,
)

from .models import (
    AgentCall,
    EnglishEdit,
    TranslationCandidate,
    TranslationCritique,
    TranslationInput,
    UsageTotals,
)
from .policy import TRANSLATION_POLICY_VERSION
from .prompts import CRITIC_INSTRUCTIONS, EDITOR_INSTRUCTIONS, TRANSLATOR_INSTRUCTIONS

TRANSLATOR_MODEL = "gpt-5.6-luna"
CRITIC_MODEL = "gpt-5.6-luna"
EDITOR_MODEL = "gpt-5.6-luna"
REASONING_EFFORT = "max"
TRANSLATION_WORKFLOW_VERSION = 13

_TRANSIENT_API_ERRORS = (
    RateLimitError,
    APIConnectionError,
    APITimeoutError,
    InternalServerError,
    TimeoutError,
)


def is_transient_error(error: BaseException | str) -> bool:
    if isinstance(error, _TRANSIENT_API_ERRORS):
        return True
    text = str(error)
    return any(
        marker in text
        for marker in (
            "RateLimitError:",
            "APIConnectionError:",
            "APITimeoutError:",
            "InternalServerError:",
            "Error code: 408",
            "Error code: 429",
            "Error code: 500",
            "Error code: 502",
            "Error code: 503",
            "Error code: 504",
            "'code': 429",
            "'code': 500",
            "'code': 502",
            "'code': 503",
            "'code': 504",
            "Error code: 402",
            "'code': 402",
            "requires more credits",
            "Upstream idle timeout exceeded",
            "temporarily overloaded",
            "Service temporarily overloaded",
            # Anthropic 529 shape: {"type": "overloaded_error", ...}
            "overloaded_error",
            "'message': 'Overloaded'",
            "Error code: 529",
            "no choices (possible provider error payload)",
        )
    )


_MAX_TOKENS_BY_EFFORT = {
    "none": 2500,
    "low": 4000,
    "medium": 12000,
    "high": 24000,
    "xhigh": 48000,
    "max": 60000,
}


class AgentsTranslationClient:
    """Three independent SDK agents with application-owned orchestration."""

    def __init__(
        self,
        *,
        min_request_interval: float = 0.0,
        transient_retries: int = 6,
    ):
        self.translator_model = TRANSLATOR_MODEL
        self.critic_model = CRITIC_MODEL
        self.editor_model = EDITOR_MODEL
        self.reasoning_effort = REASONING_EFFORT
        self.min_request_interval = min_request_interval
        self.transient_retries = transient_retries
        self._last_request_at = 0.0
        self._request_lock = asyncio.Lock()

        # The Agents SDK stamps a fresh prompt_cache_key on every run, which
        # routes each request to a different cache partition: the shared
        # constitution prefix is written every time and never read back (we
        # measured cache_write on 100% of calls and zero hits). A stable
        # per-agent key restores prefix caching; the version suffix keeps a
        # prompt change from colliding with a stale partition.
        def settings_for(role: str) -> ModelSettings:
            return ModelSettings(
                max_tokens=_MAX_TOKENS_BY_EFFORT[REASONING_EFFORT],
                include_usage=True,
                store=False,
                reasoning={"effort": REASONING_EFFORT},
                extra_args={
                    "prompt_cache_key": (
                        f"shgk-translate-p{TRANSLATION_POLICY_VERSION}"
                        f"-w{TRANSLATION_WORKFLOW_VERSION}-{role}"
                    )
                },
            )

        self.translator = Agent(
            name="ChGK English translator",
            instructions=TRANSLATOR_INSTRUCTIONS,
            model=TRANSLATOR_MODEL,
            model_settings=settings_for("translator"),
            output_type=TranslationCandidate,
        )
        self.critic = Agent(
            name="ChGK translation critic",
            instructions=CRITIC_INSTRUCTIONS,
            model=CRITIC_MODEL,
            model_settings=settings_for("critic"),
            output_type=TranslationCritique,
        )
        self.editor = Agent(
            name="ChGK English copy editor",
            instructions=EDITOR_INSTRUCTIONS,
            model=EDITOR_MODEL,
            model_settings=settings_for("editor"),
            output_type=EnglishEdit,
        )
        self.run_config = RunConfig(
            tracing_disabled=True,
            workflow_name="ChGK translation and critique",
        )


    async def _run(
        self,
        agent: Agent,
        prompt: str,
        output_type: type[TranslationCandidate | TranslationCritique | EnglishEdit],
    ) -> AgentCall:
        result = None
        for retry in range(self.transient_retries + 1):
            if self.min_request_interval > 0:
                async with self._request_lock:
                    delay = self.min_request_interval - (
                        time.monotonic() - self._last_request_at
                    )
                    if delay > 0:
                        await asyncio.sleep(delay)
                    self._last_request_at = time.monotonic()
            try:
                result = await Runner.run(
                    agent,
                    prompt,
                    max_turns=1,
                    run_config=self.run_config,
                )
                break
            except Exception as error:
                if not is_transient_error(error) or retry >= self.transient_retries:
                    raise
                await asyncio.sleep(min(60.0, 5.0 * (2**retry)))
        if result is None:
            raise AssertionError("unreachable")
        output = result.final_output_as(output_type, raise_if_incorrect_type=True)
        usage = result.context_wrapper.usage
        return AgentCall(
            output=output,
            usage=UsageTotals(
                requests=usage.requests,
                input_tokens=usage.input_tokens,
                output_tokens=usage.output_tokens,
                cached_input_tokens=usage.input_tokens_details.cached_tokens,
                cache_write_input_tokens=usage.input_tokens_details.cache_write_tokens,
                reasoning_output_tokens=usage.output_tokens_details.reasoning_tokens,
            ),
        )

    async def propose(
        self,
        source: TranslationInput,
        *,
        previous: TranslationCandidate | None = None,
        feedback: TranslationCritique | None = None,
    ) -> AgentCall:
        task: dict[str, object] = {"original": source.prompt_dict()}
        if previous is not None and feedback is not None:
            task["previous_candidate"] = previous.model_dump()
            task["critic_feedback"] = feedback.model_dump()
            task["instruction"] = "Revise the candidate in response to the critique."
        else:
            task["instruction"] = "Produce the first English candidate."
        return await self._run(
            self.translator,
            json.dumps(task, ensure_ascii=False, indent=2),
            TranslationCandidate,
        )

    async def critique(
        self, source: TranslationInput, candidate: TranslationCandidate
    ) -> AgentCall:
        task = {
            "original": source.prompt_dict(),
            "candidate": candidate.model_dump(),
            "instruction": "Judge this candidate independently under the constitution.",
        }
        return await self._run(
            self.critic,
            json.dumps(task, ensure_ascii=False, indent=2),
            TranslationCritique,
        )

    async def edit(
        self, source: TranslationInput, candidate: TranslationCandidate
    ) -> AgentCall:
        task = {
            "original": source.prompt_dict(),
            "critic_approved_candidate": candidate.model_dump(),
            "instruction": "Copy-edit this candidate under the editor contract.",
        }
        return await self._run(
            self.editor,
            json.dumps(task, ensure_ascii=False, indent=2),
            EnglishEdit,
        )


_POOLED_CLIENT_INSTALLED = False


def install_pooled_openai_client(max_connections: int = 2048) -> None:
    """Widen the shared OpenAI connection pool for high-concurrency runs.

    The SDK's default pool caps at 1000 connections, which silently becomes the
    throughput ceiling long before the account's rate limits do. Token usage is
    what actually binds here (~25k tokens/question against 10M tokens/min), so
    the pool must not be the limiting factor.
    """

    global _POOLED_CLIENT_INSTALLED
    if _POOLED_CLIENT_INSTALLED:
        return
    set_default_openai_client(
        AsyncOpenAI(
            http_client=DefaultAsyncHttpxClient(
                limits=httpx.Limits(
                    max_connections=max_connections,
                    max_keepalive_connections=max_connections // 2,
                ),
                timeout=httpx.Timeout(600.0, connect=15.0),
            )
        )
    )
    _POOLED_CLIENT_INSTALLED = True
