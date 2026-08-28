"""Propose, critique, revise, then copy-edit -- orchestrated by us, not the SDK."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from .models import (
    ENGLISH_FIELDS,
    REQUIRED_ENGLISH_FIELDS,
    EnglishEdit,
    TranslationCandidate,
    TranslationClient,
    TranslationCritique,
    TranslationInput,
    UsageTotals,
)


@dataclass(slots=True)
class WorkflowResult:
    candidate: TranslationCandidate
    translation_attempts: int
    critic_attempts: int
    editor_attempts: int
    usage: UsageTotals
    history: list[dict[str, Any]]
    pre_editor_candidate: TranslationCandidate | None
    editor_result: EnglishEdit | None
    editor_usage: UsageTotals
    editor_status: Literal["unchanged", "edited", "needs_rework", "skipped"]



def _local_issues(candidate: TranslationCandidate) -> list[str]:
    if candidate.status == "untranslatable":
        return (
            []
            if candidate.untranslatable_reason.strip()
            else ["The untranslatable result has no concrete reason."]
        )
    missing = _missing_required(_english_fields(candidate))
    if not candidate.changes_description.strip():
        missing.append("changes_description")
    return (
        [f"Required translated fields are empty: {', '.join(missing)}."]
        if missing
        else []
    )


def _english_fields(source: TranslationCandidate | EnglishEdit) -> dict[str, str]:
    return {name: getattr(source, name) for name in ENGLISH_FIELDS}


def _missing_required(fields: dict[str, str]) -> list[str]:
    return [name for name in REQUIRED_ENGLISH_FIELDS if not fields[name].strip()]


async def _finalize_with_editor(
    client: TranslationClient,
    source: TranslationInput,
    candidate: TranslationCandidate,
    *,
    translation_attempts: int,
    critic_attempts: int,
    usage: UsageTotals,
    history: list[dict[str, Any]],
) -> WorkflowResult:
    if candidate.status == "untranslatable":
        return WorkflowResult(
            candidate=candidate,
            translation_attempts=translation_attempts,
            critic_attempts=critic_attempts,
            editor_attempts=0,
            usage=usage,
            history=history,
            pre_editor_candidate=None,
            editor_result=None,
            editor_usage=UsageTotals(),
            editor_status="skipped",
        )

    pre_editor = candidate.model_copy(deep=True)
    edited_call = await client.edit(source, pre_editor)
    usage.add(edited_call.usage)
    edit = edited_call.output
    if not isinstance(edit, EnglishEdit):
        raise TypeError("editor returned the wrong structured output")

    original_fields = _english_fields(pre_editor)
    edited_fields = _english_fields(edit)
    missing = _missing_required(edited_fields)
    if edit.decision == "needs_rework" or missing:
        reason = edit.needs_rework_reason.strip()
        if missing:
            reason = f"Editor returned empty required fields: {', '.join(missing)}."
        edit = edit.model_copy(
            update={
                "decision": "needs_rework",
                **original_fields,
                "needs_rework_reason": reason
                or "Safe English copy editing requires substantive puzzle changes.",
            }
        )
        final = pre_editor
        editor_status = "needs_rework"
    else:
        changed = edited_fields != original_fields
        editor_status = "edited" if changed else "unchanged"
        edit = edit.model_copy(
            update={
                "decision": editor_status,
                "needs_rework_reason": "",
            }
        )
        if changed:
            summary = edit.edit_summary.strip() or "Polished the English prose."
            existing = pre_editor.changes_description.rstrip()
            changes_description = f"{existing} English copy edit: {summary}".strip()
            final = pre_editor.model_copy(
                update={**edited_fields, "changes_description": changes_description}
            )
        else:
            final = pre_editor

    history.append({"editor": edit.model_dump()})
    return WorkflowResult(
        candidate=final,
        translation_attempts=translation_attempts,
        critic_attempts=critic_attempts,
        editor_attempts=1,
        usage=usage,
        history=history,
        pre_editor_candidate=pre_editor,
        editor_result=edit,
        editor_usage=edited_call.usage,
        editor_status=editor_status,
    )


async def run_translation_workflow(
    client: TranslationClient,
    source: TranslationInput,
    *,
    max_revisions: int = 2,
) -> WorkflowResult:
    usage = UsageTotals()
    history: list[dict[str, Any]] = []
    previous: TranslationCandidate | None = None
    feedback: TranslationCritique | None = None
    last_playable: TranslationCandidate | None = None

    for attempt in range(max_revisions + 1):
        proposed = await client.propose(source, previous=previous, feedback=feedback)
        usage.add(proposed.usage)
        candidate = proposed.output
        if not isinstance(candidate, TranslationCandidate):
            raise TypeError("translator returned the wrong structured output")

        reviewed = await client.critique(source, candidate)
        usage.add(reviewed.usage)
        critique = reviewed.output
        if not isinstance(critique, TranslationCritique):
            raise TypeError("critic returned the wrong structured output")

        local_issues = _local_issues(candidate)
        if candidate.status != "untranslatable" and not local_issues:
            last_playable = candidate
        if critique.decision == "accept" and (candidate.status == "untranslatable") != (
            critique.accepted_status == "untranslatable"
        ):
            local_issues.append(
                "The critic and writer disagree on whether the question is translatable."
            )
        if local_issues:
            critique = critique.model_copy(
                update={
                    "decision": "revise",
                    "issues": [*critique.issues, *local_issues],
                    "revision_instructions": " ".join(
                        filter(None, [critique.revision_instructions, *local_issues])
                    ),
                }
            )

        history.append(
            {
                "attempt": attempt + 1,
                "candidate": candidate.model_dump(),
                "critique": critique.model_dump(),
            }
        )
        if critique.decision == "accept":
            candidate = candidate.model_copy(
                update={"status": critique.accepted_status}
            )
            return await _finalize_with_editor(
                client,
                source,
                candidate,
                translation_attempts=attempt + 1,
                critic_attempts=attempt + 1,
                usage=usage,
                history=history,
            )

        if attempt == max_revisions:
            if (
                critique.accepted_status != "untranslatable"
                and last_playable is not None
            ):
                salvaged = last_playable.model_copy(
                    update={"status": critique.accepted_status}
                )
                return await _finalize_with_editor(
                    client,
                    source,
                    salvaged,
                    translation_attempts=attempt + 1,
                    critic_attempts=attempt + 1,
                    usage=usage,
                    history=history,
                )
            reason = critique.summary.strip() or critique.revision_instructions.strip()
            exhausted = TranslationCandidate(
                status="untranslatable",
                question_en="",
                answer_en="",
                explanation_en="",
                acceptance_criteria_en="",
                handout_text_en="",
                changes_description=(
                    "No candidate passed independent review within the revision limit."
                ),
                untranslatable_reason=reason
                or "The translation did not pass independent review.",
            )
            return await _finalize_with_editor(
                client,
                source,
                exhausted,
                translation_attempts=attempt + 1,
                critic_attempts=attempt + 1,
                usage=usage,
                history=history,
            )

        previous = candidate
        feedback = critique

    raise AssertionError("unreachable")
