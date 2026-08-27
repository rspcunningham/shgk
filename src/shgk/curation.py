"""Stages 2 and 3: decide which rows are usable questions, and which are reprints.

Both stages are pure functions of already-stored question text, cheap enough to
rebuild from scratch over the whole corpus, so neither keeps incremental state.
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

# Ukrainian and Belarusian questions appear occasionally; these letters do not
# occur in Russian orthography.
UKRAINIAN = re.compile(r"[іїєґ]")
BELARUSIAN = re.compile(r"[ў]")

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


def detect_lang(text: str) -> str:
    """Cheap script-based language guess; the corpus is overwhelmingly Russian.

    None of these letters occur in Russian orthography. "ў" is distinctively
    Belarusian and "ї/є/ґ" distinctively Ukrainian, but both languages use "і"
    heavily, so the distinctive letters decide and a bare run of "і" falls to
    Ukrainian, which is the commoner of the two here.
    """
    belarusian = len(BELARUSIAN.findall(text))
    ukrainian = len(UKRAINIAN.findall(text))
    if belarusian + ukrainian <= 2:
        return "ru"
    if belarusian:
        return "be"
    return "uk"


def normalized_hash(question: str) -> str:
    """Hash of the question with case, punctuation and spacing folded away."""
    text = unicodedata.normalize("NFKC", question).casefold()
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

    Deliberately narrow. Attributes such as has_media, taken_down and lang stay
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


def _total_teams(solve_percentages: str, correct_answers: str) -> float:
    """Teams that played, summed over every recorded playing.

    The source stores the correct-answer count and the percentage that solved
    it, so the field size is their quotient.
    """
    try:
        percentages = json.loads(solve_percentages)
        corrects = json.loads(correct_answers)
    except (TypeError, ValueError):
        return 0.0
    return sum(
        correct / (percentage / 100)
        for percentage, correct in zip(percentages, corrects)
        if percentage
    )


def rebuild_duplicates(connection: sqlite3.Connection) -> dict[str, int]:
    """Recompute stage 3 from scratch.

    Groups clean questions by normalized text and keeps the copy with the
    largest measured field, since that row carries the most reliable difficulty
    signal. Duplicates are recorded rather than deleted so their play statistics
    remain available for merging once the label schema is settled.
    """
    connection.execute("DELETE FROM question_duplicates")
    groups: dict[str, list[tuple[int, float]]] = {}
    for question_id, hash_value, percentages, corrects in connection.execute(
        """
        SELECT id, normalized_hash, solve_percentages, correct_answers
        FROM questions_clean
        """
    ):
        groups.setdefault(hash_value, []).append(
            (question_id, _total_teams(percentages, corrects))
        )

    duplicates = []
    for members in groups.values():
        if len(members) < 2:
            continue
        canonical = max(members, key=lambda member: (member[1], -member[0]))[0]
        duplicates.extend(
            (question_id, canonical)
            for question_id, _ in members
            if question_id != canonical
        )
    connection.executemany(
        "INSERT INTO question_duplicates (question_id, canonical_id) VALUES (?, ?)",
        duplicates,
    )
    return {
        "groups": sum(1 for members in groups.values() if len(members) > 1),
        "duplicates": len(duplicates),
    }
