"""Stage 4: translate canonical questions and record the result."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from ..db import DEFAULT_PATH as DEFAULT_DB_PATH
from ..db import connect
from ..progress import Reporter
from .client import MAX_IN_FLIGHT
from .models import TranslationClient, TranslationInput, UsageTotals
from .workflow import WorkflowResult, run_translation_workflow


@dataclass(slots=True)
class RunResult:
    """What a translation run selected and what became of it."""

    selected: int = 0
    completed: int = 0
    errors: int = 0
    translated_ids: list[int] = field(default_factory=list)
    # Summed as the run goes, so reporting spend needs no second query.
    usage: UsageTotals = field(default_factory=UsageTotals)


# One list, used to build the statement and to order the values, so the two can
# never drift apart.
TRANSLATION_COLUMNS = (
    "question_id",
    "content_hash",
    "status",
    "question_en",
    "answer_en",
    "explanation_en",
    "acceptance_criteria_en",
    "handout_text_en",
    "changes_description",
    "untranslatable_reason",
    "editor_status",
    "translation_attempts",
    "critic_attempts",
    "editor_attempts",
    "api_requests",
    "input_tokens",
    "cached_input_tokens",
    "cache_write_input_tokens",
    "output_tokens",
    "reasoning_output_tokens",
    "completed_at",
)

_UPSERT = f"""
INSERT INTO translations ({", ".join(TRANSLATION_COLUMNS)})
VALUES ({", ".join("?" * len(TRANSLATION_COLUMNS))})
ON CONFLICT(question_id) DO UPDATE SET
    {", ".join(
        f"{column} = excluded.{column}"
        for column in TRANSLATION_COLUMNS
        if column != "question_id"
    )}
"""

def prune_translations(connection) -> int:
    """Delete translations whose canonical record is gone or has changed.

    Stage 3 rebuilds its records from scratch, and a record's content_hash
    moves whenever a new printing adds text to translate. A translation that no
    longer matches an existing record is not stale English worth keeping: it
    is a row that describes nothing in the corpus. Removing it here is what
    lets every other query treat "has a translation" as "has a current one".
    """
    return connection.execute(
        """
        DELETE FROM translations
        WHERE NOT EXISTS (
            SELECT 1 FROM questions_canonical AS q
            WHERE q.id = translations.question_id
              AND q.content_hash = translations.content_hash
        )
        """
    ).rowcount


class TranslationPipeline:
    """Stage 4: translate canonical questions that have no translation."""

    def __init__(self, database: str | Path = DEFAULT_DB_PATH):
        self.database = Path(database)

    def _pending_inputs(
        self,
        *,
        limit: int,
        refresh: bool,
    ) -> list[TranslationInput]:
        # Sampled at random rather than taken in order. Question ids follow the
        # packages they came in, so ordering by them translates the corpus
        # oldest-first, and any measurement taken part-way through -- cost,
        # untranslatable rate, quality -- describes the oldest packages rather
        # than the corpus.
        #
        # Stale translations are pruned at build time, so a question either
        # has a current translation or none at all.
        freshness = (
            ""
            if refresh
            else "AND NOT EXISTS (SELECT 1 FROM translations AS t WHERE t.question_id = q.id)"
        )
        with connect(self.database, read_only=True) as connection:
            rows = connection.execute(
                f"""
                SELECT q.id, q.content_hash, q.question, q.answer, q.explanation,
                       q.acceptance_criteria, q.handout_text, p.title AS package_title
                FROM questions_canonical AS q
                JOIN packages AS p ON p.id = q.package_id
                WHERE 1 {freshness}
                ORDER BY RANDOM()
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        return [
            TranslationInput(
                question_id=row["id"],
                content_hash=row["content_hash"],
                question=row["question"],
                answer=row["answer"],
                explanation=row["explanation"],
                acceptance_criteria=row["acceptance_criteria"],
                handout_text=row["handout_text"],
                package_title=row["package_title"],
            )
            for row in rows
        ]

    def _save(self, source: TranslationInput, result: WorkflowResult) -> None:
        candidate = result.candidate
        row = {
            "question_id": source.question_id,
            "content_hash": source.content_hash,
            "editor_status": result.editor_status,
            "translation_attempts": result.translation_attempts,
            "critic_attempts": result.critic_attempts,
            "editor_attempts": result.editor_attempts,
            "api_requests": result.usage.requests,
            "input_tokens": result.usage.input_tokens,
            "cached_input_tokens": result.usage.cached_input_tokens,
            "cache_write_input_tokens": result.usage.cache_write_input_tokens,
            "output_tokens": result.usage.output_tokens,
            "reasoning_output_tokens": result.usage.reasoning_output_tokens,
            "completed_at": datetime.now(UTC).isoformat(),
            # The candidate's own field names are the column names.
            **{k: v for k, v in candidate.model_dump().items() if k in TRANSLATION_COLUMNS},
        }
        with connect(self.database) as connection:
            connection.execute(_UPSERT, [row[column] for column in TRANSLATION_COLUMNS])

    async def run(
        self,
        client: TranslationClient,
        *,
        limit: int = 10,
        max_revisions: int = 2,
        refresh: bool = False,
        fail_fast: bool = False,
        concurrency: int = MAX_IN_FLIGHT,
        progress: Reporter | None = None,
    ) -> RunResult:
        inputs = self._pending_inputs(limit=limit, refresh=refresh)
        result = RunResult(selected=len(inputs))
        # Rate limits are handled where they surface, by backing off the call
        # that was refused. This only keeps a very large run from holding every
        # question, and its prompts, in memory at once.
        semaphore = asyncio.Semaphore(max(1, concurrency))
        finished = 0

        async def translate_one(source: TranslationInput) -> None:
            nonlocal finished
            async with semaphore:
                try:
                    workflow = await run_translation_workflow(
                        client, source, max_revisions=max_revisions
                    )
                except Exception as error:
                    result.errors += 1
                    finished += 1
                    if progress:
                        progress(finished, len(inputs), f"error: {error}")
                    if fail_fast:
                        raise
                    return
                self._save(source, workflow)
                result.translated_ids.append(source.question_id)
                result.usage.add(workflow.usage)
                result.completed += 1
                finished += 1
                if progress:
                    progress(finished, len(inputs), workflow.candidate.status)

        async with asyncio.TaskGroup() as group:
            for source in inputs:
                group.create_task(translate_one(source))
        return result

    def stats(self) -> dict[str, object]:
        with connect(self.database, read_only=True) as connection:
            by_status = {
                row["status"]: row["questions"]
                for row in connection.execute(
                    """
                    SELECT status, COUNT(*) AS questions
                    FROM translations GROUP BY status
                    """
                )
            }
            totals = connection.execute(
                """
                SELECT COUNT(*) AS translations,
                       SUM(api_requests) AS api_requests,
                       SUM(input_tokens) AS input_tokens,
                       SUM(cached_input_tokens) AS cached_input_tokens,
                       SUM(output_tokens) AS output_tokens,
                       SUM(reasoning_output_tokens) AS reasoning_output_tokens
                FROM translations
                """
            ).fetchone()
            pending = connection.execute(
                """
                SELECT COUNT(*) FROM questions_canonical AS q
                WHERE NOT EXISTS (SELECT 1 FROM translations AS t WHERE t.question_id = q.id)
                """
            ).fetchone()[0]
        return {"by_status": by_status, "pending": pending, **dict(totals)}
