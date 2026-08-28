"""The three SDK agents that translate, critique and copy-edit."""

from __future__ import annotations

import json

from agents import Agent, ModelSettings, RunConfig, Runner, set_default_openai_client
from openai import AsyncOpenAI

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
# Each question in flight holds at most one connection, and the SDK's pool
# allows a thousand; past that requests queue inside httpx regardless. So this
# is not a tuning knob, it is where the transport stops helping.
MAX_IN_FLIGHT = 1000

# The SDK retries on its own: it honours Retry-After and x-should-retry, covers
# 408, 409, 429, 5xx, connection errors and timeouts, and jitters. All this
# raises is how many attempts it gets, since two is thin for a run measured in
# hours against a rate limit.
MAX_RETRIES = 8


TRANSLATION_WORKFLOW_VERSION = 13


def _configure_retries() -> None:
    """Give the shared client a retry budget suited to a long batch run."""
    set_default_openai_client(AsyncOpenAI(max_retries=MAX_RETRIES))


# Reasoning tokens count against max_tokens, so a higher effort needs headroom
# or the model runs out of budget before emitting its structured output.
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

    def __init__(self) -> None:
        _configure_retries()
        self.translator_model = TRANSLATOR_MODEL
        self.critic_model = CRITIC_MODEL
        self.editor_model = EDITOR_MODEL
        self.reasoning_effort = REASONING_EFFORT

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
        result = await Runner.run(
            agent, prompt, max_turns=1, run_config=self.run_config
        )
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


