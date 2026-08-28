"""Stages 2 and 3: decide which rows are usable questions, and which are reprints.

Stage 3 does not pick a winner among reprints; it assembles one record from
all of them. Each supplementary text field is taken from whichever printing has
the longest version, and play statistics are pooled, so a question played at
four tournaments is measured against all four fields and described by the
fullest explanation anyone wrote for it.

Both stages are pure functions of already-stored question text, cheap enough to
rebuild from scratch over the whole corpus, so neither keeps incremental state
-- including the grouping hash, which is recomputed here rather than stored, so
that changing how text is folded takes effect on the next build with nothing to
migrate.
"""

from __future__ import annotations

import json
import re
import sqlite3
import unicodedata
from hashlib import sha256

# Presenter stage directions that the source embeds in the question text. They
# are instructions for reading the question aloud, not part of the puzzle, and
# several are about Russian pronunciation and so cannot survive translation.
# A handful of notes in the source are never closed, or are closed with ")" or
# "}". The bracketed form is tried first so multi-line notes still match; the
# run-to-end-of-line fallback only fires when no "]" follows at all.
HOST_NOTE = re.compile(
    r"\[\s*(?:Ведущему|ведущему|Чтецу|чтецу|Комментарий\s+для\s+ведущего)\b"
    r"(?:[^\]]*\]|[^\n]*(?=\n))"
)

# A pack declares a multi-part question on its own line, optionally after a host
# note. Matching the bare word would also hit "столица", "лицо" and "полиция".
MULTIPART = re.compile(r"(?:^|\n)\s*(?:\[[^\]]*\]\s*)*(Блиц|Дуплет|БЛИЦ|ДУПЛЕТ)\b")


REFERS_TO_OTHER = re.compile(r"предыдущ\w*\s+вопрос|прошл\w*\s+вопрос")
HANDOUT_LOST = re.compile(r"раздаточн\w*[^.\]]{0,60}(?:утерян|утрачен|потерян|не сохран)", re.I)

_PUNCT = re.compile(r"[^\w\s]", re.UNICODE)
_SPACE = re.compile(r"\s+")

MIN_QUESTION_LENGTH = 40


def split_host_note(question: str) -> tuple[str, str]:
    """Separate presenter instructions from the question a player actually hears."""
    notes = [match.group(0) for match in HOST_NOTE.finditer(question)]
    if not notes:
        return question, ""
    stripped = HOST_NOTE.sub("", question)
    # Collapse only horizontal runs and the blank lines the removal leaves
    # behind; newlines are meaningful here, since handout material and
    # multi-line quotations rely on them.
    stripped = re.sub(r"[ \t]+", " ", stripped)
    stripped = re.sub(r"\n{3,}", "\n\n", stripped)
    stripped = "\n".join(line.strip() for line in stripped.split("\n")).strip()
    return stripped, "\n".join(notes)


def detect_kind(question: str) -> str:
    match = MULTIPART.search(question)
    if not match:
        return "normal"
    return "blitz" if match.group(1).lower() == "блиц" else "duplet"



def normalized_hash(question: str) -> str:
    """Hash of the question with case, punctuation, spacing and yo folded away.

    Deliberately conservative: it folds only differences that no reprint of a
    question could meaningfully carry. Writing "e" for "yo" is a typographic
    convention rather than a change to the text, and the source is inconsistent
    about it, so the two spellings must hash alike. Anything looser -- stemming,
    or reaching into the handout for questions whose text alone is generic --
    starts merging questions that only look the same.
    """
    text = unicodedata.normalize("NFKC", question).casefold().replace("\u0451", "\u0435")
    text = _SPACE.sub(" ", _PUNCT.sub(" ", text)).strip()
    return sha256(text.encode("utf-8")).hexdigest()[:16]


def content_hash(*fields: str) -> str:
    """Hash of the text a translation depends on.

    Excludes play statistics on purpose: a question replayed at a new tournament
    has not changed and must not look stale to the translation stage.
    """
    return sha256("\0".join(fields).encode("utf-8")).hexdigest()[:16]


def exclusion_reason(question: str, answer: str) -> str | None:
    """Why this row is not a usable question, or None if it is one.

    Deliberately narrow. Attributes such as has_media and taken_down stay
    queryable on the row instead of removing it, and rules that misfired on real
    data during the audit -- handout references whose material turned out to be
    inline, questions ending in an imperative rather than a question mark -- are
    not applied.
    """
    if not answer.strip():
        return "no_answer"
    if len(question.strip()) < MIN_QUESTION_LENGTH:
        return "not_a_question"
    if REFERS_TO_OTHER.search(question):
        return "refers_to_other_question"
    if HANDOUT_LOST.search(question):
        return "handout_lost"
    return None


def rebuild_exclusions(connection: sqlite3.Connection) -> dict[str, int]:
    """Recompute stage 2 from scratch."""
    connection.execute("DELETE FROM question_exclusions")
    counts: dict[str, int] = {}
    rows = connection.execute("SELECT id, question, answer FROM questions").fetchall()
    excluded = []
    for question_id, question, answer in rows:
        reason = exclusion_reason(question, answer)
        if reason is not None:
            excluded.append((question_id, reason))
            counts[reason] = counts.get(reason, 0) + 1
    connection.executemany(
        "INSERT INTO question_exclusions (question_id, reason) VALUES (?, ?)", excluded
    )
    return counts


# Text fields where a later printing may carry more than the first one did.
# The question itself is not among them: its variants differ only in the
# folding that normalized_hash already ignores.
MERGED_TEXT = ("answer", "explanation", "acceptance_criteria", "handout_text",
               "source_references")

# Columns that stage 3 adds to a canonical record beyond those of ``questions``.
POOLED = ("printings", "playings", "total_teams", "solve_rate")


def _playings(row: dict) -> tuple[list, list]:
    """The paired per-playing arrays, truncated to a common length.

    The source ships them index-paired and equal-length. If that ever stops
    being true, dropping the tail beats aborting a whole-corpus rebuild.
    """
    percentages = json.loads(row["solve_percentages"])
    corrects = json.loads(row["correct_answers"])
    common = min(len(percentages), len(corrects))
    return percentages[:common], corrects[:common]


def _teams(percentage: float, correct: int) -> float:
    """Field size of one playing: the source records the count of teams that
    solved a question and the percentage that did, so the field is the ratio."""
    return correct / (percentage / 100) if percentage else 0.0


def _union(lists: list[str]) -> str:
    seen: dict = {}
    for encoded in lists:
        for item in json.loads(encoded):
            seen.setdefault(json.dumps(item), item)
    return json.dumps(list(seen.values()))


def merge_printings(printings: list[dict]) -> dict:
    """One canonical record from every clean printing of a question.

    Identity and the question text come from the earliest printing. Each field
    in MERGED_TEXT is the longest version any printing carries, which is the
    same thing as the only version when there is one printing.
    """
    printings = sorted(printings, key=lambda row: row["id"])
    record = dict(printings[0])
    for column in MERGED_TEXT:
        record[column] = max(
            (row[column] for row in printings), key=lambda text: len(text.strip())
        )
    record["tournament_ids"] = _union([row["tournament_ids"] for row in printings])

    percentages: list[float] = []
    corrects: list[int] = []
    for row in printings:
        row_percentages, row_corrects = _playings(row)
        percentages += row_percentages
        corrects += row_corrects
    total_teams = sum(map(_teams, percentages, corrects))
    record.update(
        solve_percentages=json.dumps(percentages),
        correct_answers=json.dumps(corrects),
        printings=len(printings),
        playings=len(percentages),
        total_teams=total_teams,
        solve_rate=sum(corrects) / total_teams if total_teams else None,
        content_hash=content_hash(
            record["question"], record["answer"], record["explanation"],
            record["acceptance_criteria"], record["handout_text"],
        ),
    )
    return record


def rebuild_canonical(connection: sqlite3.Connection) -> dict[str, int]:
    """Recompute stage 3 from scratch.

    Groups clean questions by normalized text and writes one merged record per
    group. Rows are streamed: a question with a single printing is written as
    soon as it is read, and only the members of multi-printing groups -- a few
    percent of the corpus -- are held until the end.
    """
    connection.execute("DELETE FROM question_printings")
    connection.execute("DELETE FROM questions_canonical")

    members: dict[str, list[int]] = {}
    for question_id, question in connection.execute(
        "SELECT id, question FROM questions_clean"
    ):
        members.setdefault(normalized_hash(question), []).append(question_id)
    group_of = {
        question_id: hash_value
        for hash_value, ids in members.items()
        if len(ids) > 1
        for question_id in ids
    }

    cursor = connection.execute("SELECT * FROM questions_clean")
    columns = [description[0] for description in cursor.description]
    canonical_columns = columns + list(POOLED)
    insert = (
        f"INSERT INTO questions_canonical ({', '.join(canonical_columns)}) "
        f"VALUES ({', '.join(':' + column for column in canonical_columns)})"
    )

    pending: dict[str, list[dict]] = {}
    records: list[dict] = []
    for values in cursor:
        row = dict(zip(columns, values, strict=True))
        if row["id"] in group_of:
            pending.setdefault(group_of[row["id"]], []).append(row)
        else:
            records.append(merge_printings([row]))
    records.extend(merge_printings(group) for group in pending.values())

    connection.executemany(insert, records)
    connection.executemany(
        "INSERT INTO question_printings (question_id, canonical_id) VALUES (?, ?)",
        [
            (question_id, min(ids))
            for ids in members.values()
            for question_id in ids
        ],
    )
    return {
        "questions": len(records),
        "merged": len(pending),
        "reprints": len(group_of) - len(pending),
    }
