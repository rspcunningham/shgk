"""One-time migration: questions.sqlite3 -> corpus.sqlite3.

Builds the normalized two-table corpus. Reads the old database read-only and
writes a new file, so the source is never modified.
"""

from __future__ import annotations

import json
import re
import sqlite3
import sys
import unicodedata
from hashlib import sha256
from pathlib import Path

SOURCE = Path("data/questions.sqlite3")
TARGET = Path("data/corpus.sqlite3")
PACK_URL = "https://gotquestions.online/pack/{}"

SCHEMA = """
CREATE TABLE packages (
    id              INTEGER PRIMARY KEY,
    title           TEXT    NOT NULL DEFAULT '',
    slug            TEXT    NOT NULL DEFAULT '',
    played_at_start TEXT,
    played_at_end   TEXT,
    editor_ids      TEXT    NOT NULL DEFAULT '[]',
    editor_names    TEXT    NOT NULL DEFAULT '[]',
    url             TEXT    NOT NULL,
    status          TEXT    NOT NULL,
    http_status     INTEGER,
    page_hash       TEXT,
    questions_found INTEGER NOT NULL DEFAULT 0,
    first_seen_at   TEXT    NOT NULL,
    fetched_at      TEXT    NOT NULL,
    error           TEXT    NOT NULL DEFAULT ''
);

CREATE TABLE questions (
    id                  INTEGER PRIMARY KEY,
    package_id          INTEGER NOT NULL REFERENCES packages(id),
    question_number     INTEGER,
    question            TEXT    NOT NULL,
    answer              TEXT    NOT NULL,
    explanation         TEXT    NOT NULL DEFAULT '',
    acceptance_criteria TEXT    NOT NULL DEFAULT '',
    handout_text        TEXT    NOT NULL DEFAULT '',
    has_media           INTEGER NOT NULL DEFAULT 0,
    media_urls          TEXT    NOT NULL DEFAULT '[]',
    author_ids          TEXT    NOT NULL DEFAULT '[]',
    author_names        TEXT    NOT NULL DEFAULT '[]',
    tournament_ids      TEXT    NOT NULL DEFAULT '[]',
    source_references   TEXT    NOT NULL DEFAULT '',
    taken_down          INTEGER NOT NULL DEFAULT 0,
    solve_percentages   TEXT    NOT NULL DEFAULT '[]',
    correct_answers     TEXT    NOT NULL DEFAULT '[]',
    content_hash        TEXT    NOT NULL,
    normalized_hash     TEXT    NOT NULL
);
"""

INDEXES = """
CREATE INDEX questions_package_idx ON questions(package_id);
CREATE INDEX questions_norm_idx    ON questions(normalized_hash);
"""

_PUNCT = re.compile(r"[^\w\s]", re.UNICODE)
_SPACE = re.compile(r"\s+")


def normalized_hash(question: str) -> str:
    """Hash of the question text with case, punctuation and spacing folded away.

    Used to find the same question reprinted in different packages.
    """
    text = unicodedata.normalize("NFKC", question).casefold()
    text = _SPACE.sub(" ", _PUNCT.sub(" ", text)).strip()
    return sha256(text.encode("utf-8")).hexdigest()[:16]


def content_hash(*fields: str) -> str:
    """Hash of the text a translation depends on.

    Deliberately excludes play statistics: a question that gets replayed at a new
    tournament has not changed, and must not look stale to the translation layer.
    """
    canonical = "\0".join(fields).encode("utf-8")
    return sha256(canonical).hexdigest()[:16]


def as_json_list(values) -> str:
    return json.dumps(values, ensure_ascii=False, separators=(",", ":"))


def main() -> int:
    if not SOURCE.is_file():
        print(f"missing {SOURCE}", file=sys.stderr)
        return 1
    if TARGET.exists():
        print(f"{TARGET} already exists; remove it first", file=sys.stderr)
        return 1

    src = sqlite3.connect(f"file:{SOURCE.resolve().as_posix()}?mode=ro", uri=True)
    src.row_factory = sqlite3.Row
    dst = sqlite3.connect(TARGET)
    dst.executescript(SCHEMA)

    packages: dict[int, dict] = {}
    questions: list[tuple] = []
    skipped = 0
    total = 0

    cursor = src.execute(
        """
        SELECT source_question_id, question, answer, explanation,
               acceptance_criteria, handout_text, media_urls_json,
               package_title, played_at, extra_json, fetched_at
        FROM questions WHERE source = 'gotquestions'
        """
    )
    for row in cursor:
        total += 1
        extra = json.loads(row["extra_json"])
        pack = extra.get("pack") or {}
        pack_id = pack.get("id")
        if pack_id is None:
            skipped += 1
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
                "editor_ids": as_json_list([e.get("id") for e in editors]),
                "editor_names": as_json_list([e.get("name") for e in editors]),
                "url": PACK_URL.format(pack_id),
                "first_seen_at": row["fetched_at"],
                "fetched_at": row["fetched_at"],
                "questions_found": 0,
            }
        entry["questions_found"] += 1
        if row["played_at"] and (
            entry["played_at_start"] is None or row["played_at"] < entry["played_at_start"]
        ):
            entry["played_at_start"] = row["played_at"]
        if row["fetched_at"] > entry["fetched_at"]:
            entry["fetched_at"] = row["fetched_at"]
        if row["fetched_at"] < entry["first_seen_at"]:
            entry["first_seen_at"] = row["fetched_at"]

        authors = extra.get("authors") or []
        rating = extra.get("rating") or {}
        media = row["media_urls_json"] or "[]"
        question = row["question"]
        questions.append(
            (
                int(row["source_question_id"]),
                pack_id,
                extra.get("question_number"),
                question,
                row["answer"],
                row["explanation"] or "",
                row["acceptance_criteria"] or "",
                row["handout_text"] or "",
                1 if media not in ("[]", "") else 0,
                media,
                as_json_list([a.get("id") for a in authors]),
                as_json_list([a.get("name") for a in authors]),
                as_json_list([t.get("id") for t in (rating.get("tournaments") or [])]),
                (extra.get("source_references") or "").strip(),
                1 if extra.get("taken_down") else 0,
                as_json_list(rating.get("solve_percentages") or []),
                as_json_list(rating.get("correct_answers") or []),
                content_hash(
                    question,
                    row["answer"],
                    row["explanation"] or "",
                    row["acceptance_criteria"] or "",
                    row["handout_text"] or "",
                ),
                normalized_hash(question),
            )
        )

    dst.executemany(
        """INSERT INTO packages (id,title,slug,played_at_start,played_at_end,
             editor_ids,editor_names,url,status,http_status,page_hash,
             questions_found,first_seen_at,fetched_at,error)
           VALUES (:id,:title,:slug,:played_at_start,:played_at_end,
             :editor_ids,:editor_names,:url,'ok',200,NULL,
             :questions_found,:first_seen_at,:fetched_at,'')""",
        list(packages.values()),
    )
    dst.executemany(
        "INSERT INTO questions VALUES (" + ",".join("?" * 19) + ")", questions
    )
    dst.executescript(INDEXES)
    dst.commit()
    dst.execute("VACUUM")
    dst.close()
    src.close()
    print(f"read {total:,} rows; wrote {len(questions):,} questions, "
          f"{len(packages):,} packages; skipped {skipped:,}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
