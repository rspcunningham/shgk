from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Callable, TextIO

from ..translation import (
    build_translation_client,
    is_transient_error,
    run_translation_workflow,
    workflow_result_dict,
)
from .models import BenchmarkCase
from .models import load_jsonl
from .scoring import BenchmarkScorer, score_raw_file


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _write_jsonl_record(stream: TextIO, record: dict[str, object]) -> None:
    stream.write(json.dumps(record, ensure_ascii=False) + "\n")
    stream.flush()


async def run_benchmark(
    cases: list[BenchmarkCase],
    *,
    provider: str,
    model: str,
    output: str | Path,
    reasoning_effort: str = "low",
    max_revisions: int = 2,
    overwrite: bool = False,
    resume: bool = True,
    concurrency: int = 1,
    progress: Callable[[str], None] | None = None,
) -> dict[str, int]:
    output = Path(output)
    if concurrency < 1:
        raise ValueError("concurrency must be at least 1")
    output.parent.mkdir(parents=True, exist_ok=True)
    reuse_existing = output.exists() and not overwrite
    existing_records = load_jsonl(output) if reuse_existing else []
    if reuse_existing:
        retained = [
            record
            for record in existing_records
            if not (
                record.get("status") == "error"
                and is_transient_error(str(record.get("error") or ""))
            )
        ]
        if len(retained) != len(existing_records):
            with output.open("w", encoding="utf-8") as stream:
                for record in retained:
                    _write_jsonl_record(stream, record)
            existing_records = retained
    existing = {str(record.get("case_id")): record for record in existing_records}
    for record in existing_records:
        if record.get("provider") != provider or record.get("model") != model:
            raise ValueError(f"Existing benchmark output has a different model: {output}")
    client = build_translation_client(
        provider=provider,
        translator_model=model,
        critic_model=model,
        editor_model=model,
        reasoning_effort=reasoning_effort,
        transient_retries=0,
    )
    counts = {
        "cases": len(cases),
        "completed": sum(
            1 for record in existing_records if record.get("status") == "completed"
        ),
        "errors": sum(1 for record in existing_records if record.get("status") == "error"),
    }
    mode = "a" if reuse_existing else "w"
    write_lock = asyncio.Lock()
    semaphore = asyncio.Semaphore(concurrency)

    async def run_case(index: int, case: BenchmarkCase) -> None:
        if case.case_id in existing:
            if progress:
                progress(f"[{index}/{len(cases)}] {case.case_id}: already complete")
            return
        async with semaphore:
            started_at = _utc_now()
            try:
                source = case.translation_input()
                result = await run_translation_workflow(
                    client, source, max_revisions=max_revisions
                )
                record = {
                    "schema_version": 1,
                    "case_id": case.case_id,
                    "provider": provider,
                    "model": model,
                    "status": "completed",
                    "error": "",
                    "started_at": started_at,
                    "completed_at": _utc_now(),
                    "case": case.model_dump(),
                    "translation": workflow_result_dict(source, result),
                }
                counts_key = "completed"
                message = result.candidate.status
            except Exception as error:
                if is_transient_error(error):
                    if progress:
                        progress(
                            f"[{index}/{len(cases)}] {case.case_id}: "
                            f"transient, not stored: "
                            f"{type(error).__name__}: {error}"
                        )
                    return
                record = {
                    "schema_version": 1,
                    "case_id": case.case_id,
                    "provider": provider,
                    "model": model,
                    "status": "error",
                    "error": f"{type(error).__name__}: {error}",
                    "started_at": started_at,
                    "completed_at": _utc_now(),
                    "case": case.model_dump(),
                    "translation": None,
                }
                counts_key = "errors"
                message = record["error"]
            async with write_lock:
                _write_jsonl_record(stream, record)
                counts[counts_key] += 1
            if progress:
                progress(f"[{index}/{len(cases)}] {case.case_id}: {message}")

    with output.open(mode, encoding="utf-8") as stream:
        async with asyncio.TaskGroup() as group:
            for index, case in enumerate(cases, start=1):
                group.create_task(run_case(index, case))
    return counts


def model_slug(provider: str, model: str) -> str:
    value = f"{provider}-{model}".lower()
    return "".join(character if character.isalnum() else "-" for character in value).strip(
        "-"
    )


async def run_parallel_suite(
    cases: list[BenchmarkCase],
    planned: list[tuple[str, str, Path, Path]],
    *,
    make_scorers: Callable[[], list[BenchmarkScorer]],
    reasoning_effort: str = "low",
    max_revisions: int = 2,
    overwrite: bool = False,
    concurrency: int = 1,
    progress: Callable[[str], None] | None = None,
) -> tuple[list[dict[str, object]], list[Path]]:
    """Generate and score every model concurrently. Each model keeps its own files."""

    async def run_one(
        provider: str, model: str, raw_path: Path, scored_path: Path
    ) -> tuple[dict[str, object], Path]:
        label = f"{provider}:{model}"

        def tagged(message: str) -> None:
            if progress:
                progress(f"{label} {message}")

        tagged("starting")
        generation = await run_benchmark(
            cases,
            provider=provider,
            model=model,
            output=raw_path,
            reasoning_effort=reasoning_effort,
            max_revisions=max_revisions,
            overwrite=overwrite,
            concurrency=concurrency,
            progress=tagged,
        )
        scoring = await score_raw_file(
            raw_path,
            scored_path,
            scorers=make_scorers(),
            overwrite=overwrite,
            concurrency=concurrency,
            progress=tagged,
        )
        return (
            {
                "provider": provider,
                "model": model,
                "raw": str(raw_path),
                "scored": str(scored_path),
                "generation": generation,
                "scoring": scoring,
            },
            scored_path,
        )

    pairs = await asyncio.gather(
        *[run_one(provider, model, raw_path, scored_path) for provider, model, raw_path, scored_path in planned]
    )
    return [item[0] for item in pairs], [item[1] for item in pairs]
