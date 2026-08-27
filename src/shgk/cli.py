from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import sys
from collections import Counter
from collections.abc import Iterable
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from datetime import UTC, datetime
from pathlib import Path

from dotenv import load_dotenv

from .benchmarking.models import load_cases, load_rubric
from .benchmarking.report import render_report
from .benchmarking.runner import model_slug, run_benchmark, run_parallel_suite
from .benchmarking.scoring import (
    AgentsRubricScorer,
    DeterministicScorer,
    PanelRubricScorer,
    score_raw_file,
)
from .corpus import CorpusReader
from .database import QuestionDatabase
from .http import HttpClient, PageCache
from .pipeline import BasicFilterPipeline
from .providers import ProviderModelFactory, parse_model_spec
from .sources.gotquestions import (
    BASE_URL as GOTQUESTIONS_BASE_URL,
)
from .sources.gotquestions import (
    discover_pack_ids,
    parse_pack,
)
from .translation import (
    DEFAULT_CRITIC_MODEL,
    DEFAULT_EDITOR_MODEL,
    DEFAULT_REASONING_EFFORT,
    DEFAULT_TRANSLATOR_MODEL,
    TranslationPipeline,
    build_translation_client,
    install_pooled_openai_client,
    workflow_result_dict,
)
from .translation import TRANSLATION_WORKFLOW_VERSION
from .translation_policy import TRANSLATION_POLICY_VERSION

DEFAULT_DB = Path("data/questions.sqlite3")
DEFAULT_CACHE = Path("data/cache")
DEFAULT_PIPELINE_DB = Path("data/pipeline.sqlite3")
DEFAULT_BENCHMARK_RUBRIC = Path("benchmarks/rubric-v2.json")
DEFAULT_BENCHMARK_DIR = Path("benchmark")
DEFAULT_JUDGE_PANEL = ("openai:gpt-5.6-sol", "anthropic:claude-sonnet-5")
DEFAULT_JUDGE_PASSES = 2


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _print_counts(label: str, counts: Counter[str]) -> None:
    details = ", ".join(
        f"{name}={counts.get(name, 0)}" for name in ("inserted", "updated", "unchanged")
    )
    print(f"{label}: {details}")


def _page_url(page: int) -> str:
    return (
        GOTQUESTIONS_BASE_URL if page == 1 else f"{GOTQUESTIONS_BASE_URL}/?page={page}"
    )


def _discover_gotquestions_packs(
    cache: PageCache,
    client: HttpClient,
    *,
    pages: int | None,
    refresh: bool,
) -> list[int]:
    result: list[int] = []
    seen: set[int] = set()
    page = 1
    while pages is None or page <= pages:
        html = cache.get_text(
            "gotquestions", f"index-{page}", _page_url(page), client, refresh=refresh
        )
        page_ids = discover_pack_ids(html)
        new_ids = [pack_id for pack_id in page_ids if pack_id not in seen]
        if not new_ids:
            break
        result.extend(new_ids)
        seen.update(new_ids)
        page += 1
    return result


def _ingest_gotquestions(args: argparse.Namespace) -> int:
    database = QuestionDatabase(args.db)
    cache = PageCache(args.cache)
    total: Counter[str] = Counter()
    failures = 0

    with HttpClient(delay=args.delay, retries=args.retries) as client:
        pack_ids = list(dict.fromkeys(args.pack_id or []))
        if not pack_ids:
            pages = None if args.all else args.pages
            pack_ids = _discover_gotquestions_packs(
                cache, client, pages=pages, refresh=args.refresh
            )
        if args.limit_packages is not None:
            pack_ids = pack_ids[: args.limit_packages]
        print(f"Found {len(pack_ids)} package(s)")

        def fetch_pack(pack_id: int):
            url = f"{GOTQUESTIONS_BASE_URL}/pack/{pack_id}"
            html = cache.get_text(
                "gotquestions",
                f"pack-{pack_id}",
                url,
                client,
                refresh=args.refresh,
            )
            return parse_pack(html, fetched_at=_utc_now())

        completed = 0
        iterator = iter(pack_ids)
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            pending: dict[Future, int] = {}

            def submit_next() -> bool:
                try:
                    pack_id = next(iterator)
                except StopIteration:
                    return False
                pending[executor.submit(fetch_pack, pack_id)] = pack_id
                return True

            for _ in range(min(len(pack_ids), args.workers * 2)):
                submit_next()

            while pending:
                done, _ = wait(pending, return_when=FIRST_COMPLETED)
                for future in done:
                    pack_id = pending.pop(future)
                    completed += 1
                    try:
                        records = future.result()
                        counts = database.upsert(records)
                        total.update(counts)
                        print(
                            f"[{completed}/{len(pack_ids)}] pack {pack_id}: "
                            f"{len(records)} question(s)"
                        )
                    except Exception as error:  # keep a long crawl moving
                        failures += 1
                        print(
                            f"[{completed}/{len(pack_ids)}] pack {pack_id}: ERROR: {error}",
                            file=sys.stderr,
                        )
                        if args.fail_fast:
                            raise
                    submit_next()

    _print_counts("GotQuestions import", total)
    if failures:
        print(f"failures={failures}", file=sys.stderr)
    return 1 if failures else 0


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return parsed


def _nonnegative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be zero or greater")
    return parsed


def _stats(args: argparse.Namespace) -> int:
    rows = QuestionDatabase(args.db).stats()
    if args.json:
        print(json.dumps([dict(row) for row in rows], ensure_ascii=False, indent=2))
        return 0
    if not rows:
        print("Database is empty")
        return 0
    print("source\tgame_kind\tquestions\twith_explanation\twith_media")
    for row in rows:
        print(
            f"{row['source']}\t{row['game_kind']}\t{row['questions']}\t"
            f"{row['with_explanation']}\t{row['with_media']}"
        )
    return 0


def _init(args: argparse.Namespace) -> int:
    QuestionDatabase(args.db).initialize()
    print(f"Initialized {args.db}")
    return 0


def _filter_basic(args: argparse.Namespace) -> int:
    result = BasicFilterPipeline(args.source_db, args.pipeline_db).run()
    print(
        f"Basic filter: total={result['total']}, eligible={result['eligible']}, "
        f"rejected={result['rejected']}"
    )
    return 0


def _pipeline_stats(args: argparse.Namespace) -> int:
    result = BasicFilterPipeline(args.source_db, args.pipeline_db).stats()
    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    print(
        f"total={result['total']} eligible={result['eligible']} "
        f"rejected={result['rejected']}"
    )
    print("source\ttotal\teligible\trejected")
    for row in result["by_source"]:
        print(f"{row['source']}\t{row['total']}\t{row['eligible']}\t{row['rejected']}")
    print("reason\tquestions")
    for row in result["rejection_reasons"]:
        print(f"{row['reason']}\t{row['questions']}")
    return 0


def _load_local_environment() -> None:
    load_dotenv(".env.local", override=False)
    load_dotenv(".env", override=False)


def _translate(args: argparse.Namespace) -> int:
    _load_local_environment()
    factory = ProviderModelFactory(args.provider)
    factory.require_api_key()
    translator_model = args.model or args.translator_model
    critic_model = args.model or args.critic_model
    editor_model = args.model or args.editor_model
    if args.provider == "openai" and args.workers > 8:
        install_pooled_openai_client()
    client = build_translation_client(
        provider=args.provider,
        translator_model=translator_model,
        critic_model=critic_model,
        editor_model=editor_model,
        reasoning_effort=args.reasoning_effort,
    )
    pipeline = TranslationPipeline(args.source_db, args.pipeline_db)
    if args.no_commit and args.output is None:
        raise ValueError("--no-commit requires --output so results are not discarded")

    output_stream = None
    try:
        if args.output is not None:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            output_stream = args.output.open("w", encoding="utf-8")

        def write_result(source, workflow_result) -> None:
            if output_stream is None:
                return
            record = {
                "provider": args.provider,
                "translator_model": translator_model,
                "critic_model": critic_model,
                "editor_model": editor_model,
                "reasoning_effort": args.reasoning_effort,
                **workflow_result_dict(source, workflow_result),
            }
            output_stream.write(json.dumps(record, ensure_ascii=False) + "\n")
            output_stream.flush()

        def report_progress(message: str) -> None:
            print(message, flush=True)

        result = asyncio.run(
            pipeline.run(
                client,
                limit=args.limit,
                offset=args.offset,
                sample_size=args.sample_size,
                seed=args.seed,
                max_revisions=args.max_revisions,
                sources=args.source,
                refresh=args.refresh,
                commit=not args.no_commit,
                fail_fast=args.fail_fast,
                workers=args.workers,
                progress=report_progress,
                on_result=write_result,
            )
        )
    finally:
        if output_stream is not None:
            output_stream.close()
    print(
        f"Translation run ({'not committed' if args.no_commit else 'committed'}): "
        f"selected={result['selected']} "
        f"completed={result['completed']} errors={result['errors']}"
    )
    return 1 if result["errors"] else 0


def _read(args: argparse.Namespace) -> int:
    with CorpusReader(args.source_db, args.pipeline_db) as reader:
        try:
            quads = [reader.read(id) for id in args.id]
        except KeyError as error:
            print(error.args[0], file=sys.stderr)
            return 1
    if args.json:
        payload = [
            {
                "id": quad.id,
                "english_question": quad.english_question,
                "russian_question": quad.russian_question,
                "english_answer": quad.english_answer,
                "russian_answer": quad.russian_answer,
            }
            for quad in quads
        ]
        print(json.dumps(payload if len(payload) > 1 else payload[0],
                         ensure_ascii=False, indent=2))
        return 0
    for index, quad in enumerate(quads):
        if index:
            print()
        print(f"=== Question {quad.id} ===")
        for label, value in (
            ("English question", quad.english_question),
            ("English answer", quad.english_answer),
            ("Russian question", quad.russian_question),
            ("Russian answer", quad.russian_answer),
        ):
            print(f"--- {label} ---")
            print(value if value is not None else "[not translated]")
    return 0


def _serve(args: argparse.Namespace) -> int:
    from .web import serve

    serve(args.source_db, args.pipeline_db, host=args.host, port=args.port)
    return 0


def _translation_stats(args: argparse.Namespace) -> int:
    rows = TranslationPipeline(args.source_db, args.pipeline_db).stats()
    if args.json:
        print(json.dumps(rows, ensure_ascii=False, indent=2))
        return 0
    if not rows:
        print("No translation results")
        return 0
    print(
        "translator\tcritic\teditor\treasoning\tstatus\teditor_status\t"
        "questions\tapi_requests\tinput_tokens\tcached_input_tokens\t"
        "output_tokens\treasoning_output_tokens"
    )
    for row in rows:
        print(
            f"{row['translator_model']}\t{row['critic_model']}\t"
            f"{row['editor_model']}\t{row['reasoning_effort']}\t{row['status']}\t"
            f"{row['editor_status']}\t{row['questions']}\t"
            f"{row['api_requests']}\t{row['input_tokens']}\t"
            f"{row['cached_input_tokens']}\t{row['output_tokens']}\t"
            f"{row['reasoning_output_tokens']}"
        )
    return 0


def _judge_specs(args: argparse.Namespace) -> list[str]:
    return args.judge_model or list(DEFAULT_JUDGE_PANEL)


def _build_scorers(args: argparse.Namespace):
    selected = args.scorer or ["deterministic", "llm-rubric"]
    scorers = []
    if "deterministic" in selected:
        scorers.append(DeterministicScorer())
    if "llm-rubric" in selected:
        rubric = load_rubric(args.rubric)
        members = []
        for spec in _judge_specs(args):
            provider, model = parse_model_spec(spec)
            factory = ProviderModelFactory(provider)
            factory.require_api_key()
            factory.require_structured_outputs(model)
            if provider == "anthropic":
                from .anthropic_provider import AnthropicRubricScorer

                members.append(
                    AnthropicRubricScorer(
                        rubric,
                        model=model,
                        reasoning_effort=args.judge_reasoning_effort,
                    )
                )
            else:
                members.append(
                    AgentsRubricScorer(
                        rubric,
                        provider=provider,
                        model=model,
                        reasoning_effort=args.judge_reasoning_effort,
                    )
                )
        if len(members) == 1 and args.judge_passes == 1:
            scorers.append(members[0])
        else:
            scorers.append(
                PanelRubricScorer(rubric, members=members, passes=args.judge_passes)
            )
    return scorers


def _benchmark(args: argparse.Namespace) -> int:
    _load_local_environment()
    factory = ProviderModelFactory(args.provider)
    factory.require_api_key()
    factory.require_structured_outputs(args.model)
    cases = load_cases(args.cases, limit=args.limit)
    result = asyncio.run(
        run_benchmark(
            cases,
            provider=args.provider,
            model=args.model,
            output=args.output,
            reasoning_effort=args.reasoning_effort,
            max_revisions=args.max_revisions,
            overwrite=args.overwrite,
            concurrency=args.concurrency,
            progress=print,
        )
    )
    incomplete = result["cases"] - result["completed"]
    print(
        f"Benchmark: cases={result['cases']} completed={result['completed']} "
        f"errors={result['errors']} output={args.output}"
    )
    if incomplete:
        # Transient failures are retried and never stored, so a run can end with
        # errors=0 and no data at all; that is a failure, not a success.
        print(f"Benchmark: {incomplete} case(s) produced no record", file=sys.stderr)
    return 1 if result["errors"] or incomplete else 0


def _benchmark_score(args: argparse.Namespace) -> int:
    _load_local_environment()
    scorers = _build_scorers(args)
    result = asyncio.run(
        score_raw_file(
            args.input,
            args.output,
            scorers=scorers,
            overwrite=args.overwrite,
            concurrency=args.concurrency,
            progress=print,
        )
    )
    print(
        f"Benchmark scoring: records={result['records']} scored={result['scored']} "
        f"errors={result['scoring_errors']} output={args.output}"
    )
    return 1 if result["scoring_errors"] else 0


def _benchmark_report(args: argparse.Namespace) -> int:
    render_report(args.input, args.output_dir)
    print(f"Reports written to {args.output_dir}")
    return 0


def _benchmark_suite(args: argparse.Namespace) -> int:
    _load_local_environment()
    cases = load_cases(args.cases, limit=args.limit)
    model_specs = [parse_model_spec(value) for value in args.model]
    for provider, model in model_specs:
        factory = ProviderModelFactory(provider)
        factory.require_api_key()
        factory.require_structured_outputs(model)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_dir / "manifest.json"
    planned = []
    for provider, model in model_specs:
        slug = model_slug(provider, model)
        planned.append(
            (
                provider,
                model,
                args.output_dir / f"{slug}.raw.jsonl",
                args.output_dir / f"{slug}.scored.jsonl",
            )
        )

    results, scored_paths = asyncio.run(
        run_parallel_suite(
            cases,
            planned,
            make_scorers=lambda: _build_scorers(args),
            reasoning_effort=args.reasoning_effort,
            max_revisions=args.max_revisions,
            overwrite=args.overwrite,
            concurrency=args.concurrency,
            progress=print,
        )
    )
    summary = render_report(scored_paths, args.output_dir)
    cases_hash = hashlib.sha256(args.cases.read_bytes()).hexdigest()
    manifest = {
        "schema_version": 1,
        "cases": str(args.cases),
        "cases_sha256": cases_hash,
        "case_count": len(cases),
        "models": [f"{provider}:{model}" for provider, model in model_specs],
        "judge_panel": _judge_specs(args),
        "judge_passes": args.judge_passes,
        "rubric": str(args.rubric),
        "translation_policy_version": TRANSLATION_POLICY_VERSION,
        "translation_workflow_version": TRANSLATION_WORKFLOW_VERSION,
        "max_revisions": args.max_revisions,
        "reasoning_effort": args.reasoning_effort,
        "results": results,
        "summary": summary,
    }
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    errors = sum(
        result["generation"]["errors"] + result["scoring"]["scoring_errors"]
        for result in results
    )
    return 1 if errors else 0


def _add_concurrency_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--concurrency",
        type=_positive_int,
        default=16,
        help="max overlapping cases per model; suite models run in parallel",
    )


def _add_benchmark_scoring_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--scorer",
        action="append",
        choices=("deterministic", "llm-rubric"),
        help="repeat to choose scorers; defaults to both",
    )
    parser.add_argument("--rubric", type=Path, default=DEFAULT_BENCHMARK_RUBRIC)
    parser.add_argument(
        "--judge-model",
        action="append",
        metavar="PROVIDER:MODEL",
        help=(
            "repeat for a cross-family judge panel; defaults to "
            + " + ".join(DEFAULT_JUDGE_PANEL)
        ),
    )
    parser.add_argument(
        "--judge-passes",
        type=_positive_int,
        default=DEFAULT_JUDGE_PASSES,
        help="independent scoring passes per judge; medians reduce judge noise",
    )
    parser.add_argument(
        "--judge-reasoning-effort",
        choices=("none", "low", "medium", "high", "xhigh", "max"),
        default="low",
    )


def _add_storage_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)


def _add_fetch_arguments(parser: argparse.ArgumentParser) -> None:
    _add_storage_arguments(parser)
    parser.add_argument("--refresh", action="store_true", help="ignore cached HTML")
    parser.add_argument(
        "--delay", type=float, default=1.0, help="seconds between requests"
    )
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--workers", type=_positive_int, default=1)
    parser.add_argument("--fail-fast", action="store_true")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="shgk", description="Build a local Russian ChGK question corpus"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    init_parser = subparsers.add_parser("init", help="create the SQLite database")
    init_parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    init_parser.set_defaults(handler=_init)

    gq_parser = subparsers.add_parser(
        "gotquestions", help="fetch sports ChGK packages from GotQuestions"
    )
    _add_fetch_arguments(gq_parser)
    gq_parser.add_argument("--pack-id", type=int, action="append")
    discovery = gq_parser.add_mutually_exclusive_group()
    discovery.add_argument("--pages", type=int, default=1)
    discovery.add_argument(
        "--all", action="store_true", help="crawl all package index pages"
    )
    gq_parser.add_argument("--limit-packages", type=int)
    gq_parser.set_defaults(handler=_ingest_gotquestions)


    stats_parser = subparsers.add_parser("stats", help="summarize the corpus")
    stats_parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    stats_parser.add_argument("--json", action="store_true")
    stats_parser.set_defaults(handler=_stats)

    filter_parser = subparsers.add_parser(
        "filter-basic",
        help="select questions with text, answer, explanation, and no media",
    )
    filter_parser.add_argument("--source-db", type=Path, default=DEFAULT_DB)
    filter_parser.add_argument("--pipeline-db", type=Path, default=DEFAULT_PIPELINE_DB)
    filter_parser.set_defaults(handler=_filter_basic)

    pipeline_stats_parser = subparsers.add_parser(
        "pipeline-stats", help="summarize materialized pipeline layers"
    )
    pipeline_stats_parser.add_argument("--source-db", type=Path, default=DEFAULT_DB)
    pipeline_stats_parser.add_argument(
        "--pipeline-db", type=Path, default=DEFAULT_PIPELINE_DB
    )
    pipeline_stats_parser.add_argument("--json", action="store_true")
    pipeline_stats_parser.set_defaults(handler=_pipeline_stats)

    translate_parser = subparsers.add_parser(
        "translate",
        help="translate eligible questions with writer, critic, and editor agents",
    )
    translate_parser.add_argument("--source-db", type=Path, default=DEFAULT_DB)
    translate_parser.add_argument(
        "--pipeline-db", type=Path, default=DEFAULT_PIPELINE_DB
    )
    translate_parser.add_argument(
        "--provider", choices=("openai", "anthropic", "openrouter"), default="openai"
    )
    translate_parser.add_argument(
        "--model",
        help="use one model for writer, critic, and editor",
    )
    selection = translate_parser.add_mutually_exclusive_group()
    selection.add_argument("--limit", type=_positive_int, default=10)
    selection.add_argument(
        "--sample-size", type=_positive_int, help="select a reproducible random sample"
    )
    translate_parser.add_argument("--seed", type=int, default=0)
    translate_parser.add_argument("--offset", type=_nonnegative_int, default=0)
    translate_parser.add_argument("--max-revisions", type=_nonnegative_int, default=2)
    translate_parser.add_argument("--source", action="append")
    translate_parser.add_argument(
        "--translator-model", default=DEFAULT_TRANSLATOR_MODEL
    )
    translate_parser.add_argument("--critic-model", default=DEFAULT_CRITIC_MODEL)
    translate_parser.add_argument("--editor-model", default=DEFAULT_EDITOR_MODEL)
    translate_parser.add_argument(
        "--reasoning-effort",
        choices=("none", "low", "medium", "high", "xhigh", "max"),
        default=DEFAULT_REASONING_EFFORT,
    )
    translate_parser.add_argument(
        "--workers",
        type=_positive_int,
        default=8,
        help="questions translated concurrently (default: 8)",
    )
    translate_parser.add_argument("--refresh", action="store_true")
    translate_parser.add_argument(
        "--no-commit",
        action="store_true",
        help="run the workflow without writing translations",
    )
    translate_parser.add_argument(
        "--output", type=Path, help="write full workflow results as JSONL"
    )
    translate_parser.add_argument("--fail-fast", action="store_true")
    translate_parser.set_defaults(handler=_translate)

    read_parser = subparsers.add_parser(
        "read", help="read English/Russian question-answer quads by question id"
    )
    read_parser.add_argument("id", type=int, nargs="+")
    read_parser.add_argument("--source-db", type=Path, default=DEFAULT_DB)
    read_parser.add_argument("--pipeline-db", type=Path, default=DEFAULT_PIPELINE_DB)
    read_parser.add_argument("--json", action="store_true")
    read_parser.set_defaults(handler=_read)

    serve_parser = subparsers.add_parser(
        "serve", help="browse English questions in a local web page"
    )
    serve_parser.add_argument("--source-db", type=Path, default=DEFAULT_DB)
    serve_parser.add_argument("--pipeline-db", type=Path, default=DEFAULT_PIPELINE_DB)
    serve_parser.add_argument("--host", default="127.0.0.1")
    serve_parser.add_argument("--port", type=_positive_int, default=8765)
    serve_parser.set_defaults(handler=_serve)

    translation_stats_parser = subparsers.add_parser(
        "translation-stats", help="summarize translation results and token usage"
    )
    translation_stats_parser.add_argument("--source-db", type=Path, default=DEFAULT_DB)
    translation_stats_parser.add_argument(
        "--pipeline-db", type=Path, default=DEFAULT_PIPELINE_DB
    )
    translation_stats_parser.add_argument("--json", action="store_true")
    translation_stats_parser.set_defaults(handler=_translation_stats)

    benchmark_parser = subparsers.add_parser(
        "benchmark", help="run one model through the three-stage workflow"
    )
    benchmark_parser.add_argument(
        "--provider", choices=("openai", "anthropic", "openrouter"), required=True
    )
    benchmark_parser.add_argument("--model", required=True)
    benchmark_parser.add_argument("--cases", type=Path, required=True)
    benchmark_parser.add_argument("--output", type=Path, required=True)
    benchmark_parser.add_argument("--limit", type=_positive_int)
    benchmark_parser.add_argument(
        "--max-revisions", type=_nonnegative_int, default=2
    )
    benchmark_parser.add_argument(
        "--reasoning-effort",
        choices=("none", "low", "medium", "high", "xhigh", "max"),
        default=DEFAULT_REASONING_EFFORT,
    )
    benchmark_parser.add_argument("--overwrite", action="store_true")
    _add_concurrency_argument(benchmark_parser)
    benchmark_parser.set_defaults(handler=_benchmark)

    score_parser = subparsers.add_parser(
        "benchmark-score", help="score an existing raw benchmark JSONL"
    )
    score_parser.add_argument("--input", type=Path, required=True)
    score_parser.add_argument("--output", type=Path, required=True)
    score_parser.add_argument("--overwrite", action="store_true")
    _add_concurrency_argument(score_parser)
    _add_benchmark_scoring_arguments(score_parser)
    score_parser.set_defaults(handler=_benchmark_score)

    report_parser = subparsers.add_parser(
        "benchmark-report", help="compare one or more scored benchmark files"
    )
    report_parser.add_argument("--input", type=Path, action="append", required=True)
    report_parser.add_argument("--output-dir", type=Path, required=True)
    report_parser.set_defaults(handler=_benchmark_report)

    suite_parser = subparsers.add_parser(
        "benchmark-suite", help="run, score, and compare a fixed case set once"
    )
    suite_parser.add_argument("--cases", type=Path, required=True)
    suite_parser.add_argument(
        "--model", action="append", required=True, metavar="PROVIDER:MODEL"
    )
    suite_parser.add_argument("--output-dir", type=Path, default=DEFAULT_BENCHMARK_DIR)
    suite_parser.add_argument("--limit", type=_positive_int)
    suite_parser.add_argument("--max-revisions", type=_nonnegative_int, default=2)
    suite_parser.add_argument(
        "--reasoning-effort",
        choices=("none", "low", "medium", "high", "xhigh", "max"),
        default=DEFAULT_REASONING_EFFORT,
    )
    suite_parser.add_argument("--overwrite", action="store_true")
    _add_concurrency_argument(suite_parser)
    _add_benchmark_scoring_arguments(suite_parser)
    suite_parser.set_defaults(handler=_benchmark_suite)
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.handler(args))
    except KeyboardInterrupt:
        print("Interrupted", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
