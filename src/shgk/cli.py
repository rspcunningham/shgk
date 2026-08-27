from __future__ import annotations

import argparse
import asyncio
import json
import sys
from collections.abc import Iterable
from pathlib import Path

from dotenv import load_dotenv

from . import db
from .curation import rebuild_duplicates, rebuild_exclusions
from .translation import AgentsTranslationClient, TranslationPipeline, install_pooled_openai_client


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


def _build(args: argparse.Namespace) -> int:
    """Stages 1-3. Free, deterministic, and safe to run at any time."""
    db.initialize(args.database)
    with db.connect(args.database) as connection:
        # TODO: stage 1 (scrape new packages) once the source is rewired.
        print("stage 2: exclusions")
        reasons = rebuild_exclusions(connection)
        for reason, count in sorted(reasons.items(), key=lambda item: -item[1]):
            print(f"  {reason:<28} {count:>8,}")
        print("stage 3: duplicates")
        stats = rebuild_duplicates(connection)
        print(f"  groups {stats['groups']:,}, non-canonical {stats['duplicates']:,}")
        connection.commit()
    return 0


def _translate(args: argparse.Namespace) -> int:
    """Stage 4. Costs money, so it is never part of `build`."""
    load_dotenv(".env.local", override=False)
    load_dotenv(".env", override=False)
    if args.workers > 8:
        install_pooled_openai_client()
    client = AgentsTranslationClient()
    result = asyncio.run(
        TranslationPipeline(args.database).run(
            client,
            limit=args.limit,
            offset=args.offset,
            max_revisions=args.max_revisions,
            refresh=args.refresh,
            fail_fast=args.fail_fast,
            workers=args.workers,
            progress=lambda message: print(message, flush=True),
        )
    )
    print(
        f"selected={result['selected']} completed={result['completed']} "
        f"errors={result['errors']}"
    )
    return 1 if result["errors"] else 0


def _stats(args: argparse.Namespace) -> int:
    with db.connect(args.database, read_only=True) as connection:
        def count(table: str) -> int:
            return connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]

        stages = {
            "questions": count("questions"),
            "clean": count("questions_clean"),
            "canonical": count("questions_canonical"),
            "translated": count("questions_translated"),
        }
        exclusions = {
            row["reason"]: row["n"]
            for row in connection.execute(
                "SELECT reason, COUNT(*) AS n FROM question_exclusions "
                "GROUP BY reason ORDER BY n DESC"
            )
        }
    translation = TranslationPipeline(args.database).stats()
    payload = {"stages": stages, "exclusions": exclusions, "translation": translation}
    if args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=2, default=int))
        return 0
    print("stage")
    for name, value in stages.items():
        print(f"  {name:<28}{value:>10,}")
    print("excluded")
    for reason, value in exclusions.items():
        print(f"  {reason:<28}{value:>10,}")
    print("translation")
    print(f"  {'pending':<28}{translation['pending']:>10,}")
    for status, value in (translation["by_status"] or {}).items():
        print(f"  {status:<28}{value:>10,}")
    return 0


def _serve(args: argparse.Namespace) -> int:
    from .web import serve

    serve(args.database, host=args.host, port=args.port)
    return 0


def _add_database_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--database", type=Path, default=db.DEFAULT_PATH)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="shgk", description="Build and read an English ChGK question corpus"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    build = subparsers.add_parser(
        "build", help="fetch new packages and rebuild the curated views"
    )
    _add_database_argument(build)
    build.set_defaults(handler=_build)

    translate = subparsers.add_parser(
        "translate", help="translate canonical questions that have no current translation"
    )
    _add_database_argument(translate)
    translate.add_argument("--limit", type=_positive_int, default=10)
    translate.add_argument("--offset", type=_nonnegative_int, default=0)
    translate.add_argument("--max-revisions", type=_nonnegative_int, default=2)
    translate.add_argument("--workers", type=_positive_int, default=8)
    translate.add_argument(
        "--refresh", action="store_true", help="retranslate rows that are already current"
    )
    translate.add_argument("--fail-fast", action="store_true")
    translate.set_defaults(handler=_translate)

    stats = subparsers.add_parser("stats", help="summarize every stage")
    _add_database_argument(stats)
    stats.add_argument("--json", action="store_true")
    stats.set_defaults(handler=_stats)

    serve = subparsers.add_parser("serve", help="read questions in a local web page")
    _add_database_argument(serve)
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=_positive_int, default=8765)
    serve.set_defaults(handler=_serve)
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return int(args.handler(args))
    except KeyboardInterrupt:
        print("Interrupted", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
