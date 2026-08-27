"""Build data/shgk.sqlite3 from the legacy scrape.

One-time migration off questions.sqlite3 + pipeline.sqlite3 into the single
staged database. Reads both sources read-only.
"""

from __future__ import annotations

import json
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from shgk.curation import (  # noqa: E402
    content_hash,
    detect_kind,
    detect_lang,
    normalized_hash,
    rebuild_duplicates,
    rebuild_exclusions,
    split_host_note,
)
from shgk.schema import SCHEMA, VIEWS  # noqa: E402

LEGACY_CORPUS = Path("data/questions.sqlite3")
LEGACY_PIPELINE = Path("data/pipeline.sqlite3")
TARGET = Path("data/shgk.sqlite3")
PACK_URL = "https://gotquestions.online/pack/{}"


def as_json(value) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def read_only(path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(f"file:{path.resolve().as_posix()}?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    return connection


def load_questions(source: sqlite3.Connection, target: sqlite3.Connection) -> None:
    packages: dict[int, dict] = {}
    rows: list[tuple] = []
    cursor = source.execute(
        """
        SELECT source_question_id, question, answer, explanation,
               acceptance_criteria, handout_text, media_urls_json,
               package_title, played_at, extra_json, fetched_at
        FROM questions WHERE source = 'gotquestions'
        """
    )
    for row in cursor:
        extra = json.loads(row["extra_json"])
        pack = extra.get("pack") or {}
        pack_id = pack.get("id")
        if pack_id is None:
            continue

        entry = packages.get(pack_id)
        if entry is None:
            editors = pack.get("editors") or []
            entry = packages[pack_id] = {
                "id": pack_id,
                "title": row["package_title"] or pack.get("long_title") or "",
                "slug": (pack.get("db_chgk_info_slug") or "").strip(),
                "played_at_start": row["played_at"] or None,
                "played_at_end": pack.get("end_date") or None,
                "editor_ids": as_json([e.get("id") for e in editors]),
                "editor_names": as_json([e.get("name") for e in editors]),
                "url": PACK_URL.format(pack_id),
                "first_seen_at": row["fetched_at"],
                "fetched_at": row["fetched_at"],
                "questions_found": 0,
            }
        entry["questions_found"] += 1
        if row["played_at"] and (
            entry["played_at_start"] is None
            or row["played_at"] < entry["played_at_start"]
        ):
            entry["played_at_start"] = row["played_at"]
        entry["fetched_at"] = max(entry["fetched_at"], row["fetched_at"])
        entry["first_seen_at"] = min(entry["first_seen_at"], row["fetched_at"])

        question, host_note = split_host_note(row["question"])
        answer = row["answer"]
        explanation = row["explanation"] or ""
        criteria = row["acceptance_criteria"] or ""
        handout = row["handout_text"] or ""
        authors = extra.get("authors") or []
        rating = extra.get("rating") or {}
        media = row["media_urls_json"] or "[]"
        rows.append(
            (
                int(row["source_question_id"]),
                pack_id,
                extra.get("question_number"),
                question,
                answer,
                explanation,
                criteria,
                handout,
                host_note,
                detect_kind(row["question"]),
                detect_lang(question),
                1 if media not in ("[]", "") else 0,
                media,
                as_json([a.get("id") for a in authors]),
                as_json([a.get("name") for a in authors]),
                as_json([t.get("id") for t in (rating.get("tournaments") or [])]),
                (extra.get("source_references") or "").strip(),
                1 if extra.get("taken_down") else 0,
                as_json(rating.get("solve_percentages") or []),
                as_json(rating.get("correct_answers") or []),
                content_hash(question, answer, explanation, criteria, handout),
                normalized_hash(question),
            )
        )

    target.executemany(
        """INSERT INTO packages (id,title,slug,played_at_start,played_at_end,
             editor_ids,editor_names,url,status,http_status,page_hash,
             questions_found,first_seen_at,fetched_at,error)
           VALUES (:id,:title,:slug,:played_at_start,:played_at_end,
             :editor_ids,:editor_names,:url,'ok',200,NULL,
             :questions_found,:first_seen_at,:fetched_at,'')""",
        list(packages.values()),
    )
    target.executemany(
        "INSERT INTO questions VALUES (" + ",".join("?" * 22) + ")", rows
    )
    print(f"  loaded {len(rows):,} questions across {len(packages):,} packages")


def load_translations(target: sqlite3.Connection) -> None:
    """Carry existing translations over, re-stamped with the new content_hash.

    The legacy hash covered the whole record including play statistics, so none
    of the old values match; the underlying text is unchanged, so joining on the
    question id and re-stamping preserves the work.
    """
    if not LEGACY_PIPELINE.is_file():
        print("  no legacy pipeline database; skipping translations")
        return
    legacy = read_only(LEGACY_PIPELINE)
    try:
        rows = legacy.execute(
            "SELECT * FROM translations WHERE source = 'gotquestions'"
        ).fetchall()
    except sqlite3.OperationalError:
        print("  legacy pipeline has no translations table; skipping")
        return
    current = {
        question_id: hash_value
        for question_id, hash_value in target.execute(
            "SELECT id, content_hash FROM questions"
        )
    }
    carried = []
    for row in rows:
        question_id = int(row["source_question_id"])
        if question_id not in current:
            continue
        carried.append(
            (
                question_id,
                current[question_id],
                row["status"],
                row["question_en"],
                row["answer_en"],
                row["explanation_en"],
                row["acceptance_criteria_en"],
                row["handout_text_en"],
                row["changes_description"],
                row["untranslatable_reason"],
                row["editor_status"],
                row["translation_attempts"],
                row["critic_attempts"],
                row["editor_attempts"],
                row["api_requests"],
                row["input_tokens"],
                row["cached_input_tokens"],
                row["cache_write_input_tokens"],
                row["output_tokens"],
                row["reasoning_output_tokens"],
                row["completed_at"],
            )
        )
    target.executemany(
        "INSERT OR REPLACE INTO translations VALUES (" + ",".join("?" * 21) + ")",
        carried,
    )
    legacy.close()
    print(f"  carried {len(carried):,} of {len(rows):,} legacy translations")


def main() -> int:
    if not LEGACY_CORPUS.is_file():
        print(f"missing {LEGACY_CORPUS}", file=sys.stderr)
        return 1
    if TARGET.exists():
        print(f"{TARGET} already exists; remove it first", file=sys.stderr)
        return 1

    source = read_only(LEGACY_CORPUS)
    target = sqlite3.connect(TARGET)
    target.executescript(SCHEMA)
    target.executescript(VIEWS)

    print("stage 1: raw corpus")
    load_questions(source, target)
    target.commit()

    print("stage 2: exclusions")
    for reason, count in sorted(
        rebuild_exclusions(target).items(), key=lambda item: -item[1]
    ):
        print(f"  {reason:<28} {count:>8,}")

    print("stage 3: duplicates")
    stats = rebuild_duplicates(target)
    print(f"  groups {stats['groups']:,}, non-canonical rows {stats['duplicates']:,}")

    print("stage 4: translations")
    load_translations(target)

    target.commit()
    target.execute("VACUUM")
    target.close()
    source.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
