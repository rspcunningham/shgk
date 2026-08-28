"""Parse GotQuestions package pages into corpus rows.

The site is a Next.js app that ships the whole package as JSON inside its RSC
stream, so parsing means recovering that object rather than scraping markup.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any
from urllib.parse import urljoin

from bs4 import BeautifulSoup

from ..curation import content_hash, detect_kind, normalized_hash, split_host_note

BASE_URL = "https://gotquestions.online"
PACK_URL = BASE_URL + "/pack/{}"

_PACK_LINK_RE = re.compile(r"^/pack/(\d+)(?:[/?#]|$)")
_NEXT_CHUNK_RE = re.compile(r"self\.__next_f\.push\(\[1,(\"(?:\\.|[^\"\\])*\")\]\)")
_BLANK_LINES = re.compile(r"\n{3,}")

MEDIA_FIELDS = {
    "razdatkaPic": "handout",
    "audio": "question_audio",
    "answerPic": "answer",
    "commentPic": "explanation",
    "commentAudio": "explanation_audio",
}


class GotQuestionsParseError(ValueError):
    pass


def clean_text(value: Any) -> str:
    """Normalize source text without flattening meaningful line breaks."""
    if value is None:
        return ""
    text = str(value).replace("\r\n", "\n").replace("\r", "\n")
    lines = [re.sub(r"[ \t]+", " ", line).strip() for line in text.split("\n")]
    return _BLANK_LINES.sub("\n\n", "\n".join(lines)).strip()


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


@dataclass(frozen=True, slots=True)
class ParsedQuestion:
    id: int
    question_number: int | None
    question: str
    answer: str
    explanation: str
    acceptance_criteria: str
    handout_text: str
    host_note: str
    kind: str
    has_media: int
    media_urls: str
    author_ids: str
    author_names: str
    tournament_ids: str
    source_references: str
    taken_down: int
    solve_percentages: str
    correct_answers: str
    content_hash: str
    normalized_hash: str


@dataclass(frozen=True, slots=True)
class ParsedPack:
    id: int
    title: str
    slug: str
    played_at_start: str | None
    played_at_end: str | None
    editor_ids: str
    editor_names: str
    questions: list[ParsedQuestion]


def discover_pack_ids(html: str) -> list[int]:
    soup = BeautifulSoup(html, "lxml")
    result: list[int] = []
    seen: set[int] = set()
    for anchor in soup.find_all("a", href=True):
        match = _PACK_LINK_RE.match(str(anchor["href"]))
        if match and int(match.group(1)) not in seen:
            seen.add(int(match.group(1)))
            result.append(int(match.group(1)))
    return result


def extract_pack_object(html: str) -> dict[str, Any]:
    """Recover the complete package object from the Next.js RSC stream."""
    soup = BeautifulSoup(html, "lxml")
    chunks: list[str] = []
    for script in soup.find_all("script"):
        script_text = script.string or script.get_text()
        for match in _NEXT_CHUNK_RE.finditer(script_text):
            try:
                chunks.append(json.loads(match.group(1)))
            except json.JSONDecodeError as error:
                raise GotQuestionsParseError("Invalid Next.js data chunk") from error

    stream = "".join(chunks)
    marker_index = stream.find('"pack":')
    if marker_index < 0:
        raise GotQuestionsParseError("Could not find embedded package data")
    object_start = stream.find("{", marker_index + len('"pack":'))
    if object_start < 0:
        raise GotQuestionsParseError("Embedded package value is not an object")
    try:
        pack, _ = json.JSONDecoder().raw_decode(stream, object_start)
    except json.JSONDecodeError as error:
        raise GotQuestionsParseError("Embedded package JSON is incomplete") from error
    if not isinstance(pack, dict) or "id" not in pack or "tours" not in pack:
        raise GotQuestionsParseError("Embedded package has an unexpected shape")
    return pack


def _people(items: Any) -> tuple[str, str]:
    if not isinstance(items, list):
        return "[]", "[]"
    people = [item for item in items if isinstance(item, dict)]
    return (
        _json([person.get("id") for person in people]),
        _json([clean_text(person.get("name")) for person in people]),
    )


def _media(question: dict[str, Any]) -> list[dict[str, str]]:
    return [
        {"role": role, "url": urljoin(BASE_URL, question[field].strip())}
        for field, role in MEDIA_FIELDS.items()
        if isinstance(question.get(field), str) and question[field].strip()
    ]


def _question(raw: dict[str, Any]) -> ParsedQuestion:
    text = clean_text(raw.get("text"))
    if not text:
        text = "[Media question; see media URLs]"
    question, host_note = split_host_note(text)
    answer = clean_text(raw.get("answer"))
    explanation = clean_text(raw.get("comment"))
    criteria = clean_text(raw.get("zachet"))
    handout = clean_text(raw.get("razdatkaText"))
    author_ids, author_names = _people(raw.get("authors"))
    media = _media(raw)
    tournaments = raw.get("tournaments") or []
    return ParsedQuestion(
        id=int(raw["id"]),
        question_number=raw.get("number"),
        question=question,
        answer=answer,
        explanation=explanation,
        acceptance_criteria=criteria,
        handout_text=handout,
        host_note=host_note,
        # Detected on the original text: a pack can declare the multi-part
        # marker inside a host note.
        kind=detect_kind(text),
        has_media=1 if media else 0,
        media_urls=_json(media),
        author_ids=author_ids,
        author_names=author_names,
        tournament_ids=_json(
            [t.get("id") for t in tournaments if isinstance(t, dict)]
        ),
        source_references=(raw.get("source") or "").strip(),
        taken_down=1 if raw.get("takenDown") else 0,
        solve_percentages=_json(raw.get("complexity") or []),
        correct_answers=_json(raw.get("correct_answers") or []),
        content_hash=content_hash(question, answer, explanation, criteria, handout),
        normalized_hash=normalized_hash(question),
    )


def parse_pack(html: str) -> ParsedPack:
    pack = extract_pack_object(html)
    editor_ids, editor_names = _people(pack.get("editors"))
    questions = [
        _question(raw)
        for tour in (pack.get("tours") or [])
        if isinstance(tour, dict)
        for raw in (tour.get("questions") or [])
        if isinstance(raw, dict) and raw.get("id") is not None
    ]
    declared = pack.get("questions")
    if isinstance(declared, int) and declared != len(questions):
        raise GotQuestionsParseError(
            f"Package {pack['id']} declared {declared} questions "
            f"but parsed {len(questions)}"
        )
    return ParsedPack(
        id=int(pack["id"]),
        title=clean_text(pack.get("title")) or clean_text(pack.get("longTitle")),
        slug=clean_text(pack.get("dbchgkinfoslug")),
        played_at_start=clean_text(pack.get("startDate")) or None,
        played_at_end=clean_text(pack.get("endDate")) or None,
        editor_ids=editor_ids,
        editor_names=editor_names,
        questions=questions,
    )
