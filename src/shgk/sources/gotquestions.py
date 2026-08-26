from __future__ import annotations

from datetime import UTC, datetime
import json
import re
from typing import Any
from urllib.parse import urljoin

from bs4 import BeautifulSoup

from ..models import QuestionRecord, clean_text


BASE_URL = "https://gotquestions.online"
_PACK_LINK_RE = re.compile(r"^/pack/(\d+)(?:[/?#]|$)")
_NEXT_CHUNK_RE = re.compile(
    r"self\.__next_f\.push\(\[1,(\"(?:\\.|[^\"\\])*\")\]\)"
)


class GotQuestionsParseError(ValueError):
    pass


def discover_pack_ids(html: str) -> list[int]:
    soup = BeautifulSoup(html, "lxml")
    result: list[int] = []
    seen: set[int] = set()
    for anchor in soup.find_all("a", href=True):
        match = _PACK_LINK_RE.match(anchor["href"])
        if not match:
            continue
        pack_id = int(match.group(1))
        if pack_id not in seen:
            seen.add(pack_id)
            result.append(pack_id)
    return result


def extract_pack_object(html: str) -> dict[str, Any]:
    """Extract the complete package object from the Next.js RSC stream."""
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
    marker = '"pack":'
    marker_index = stream.find(marker)
    if marker_index < 0:
        raise GotQuestionsParseError("Could not find embedded package data")
    object_start = stream.find("{", marker_index + len(marker))
    if object_start < 0:
        raise GotQuestionsParseError("Embedded package value is not an object")
    try:
        pack, _ = json.JSONDecoder().raw_decode(stream, object_start)
    except json.JSONDecodeError as error:
        raise GotQuestionsParseError("Embedded package JSON is incomplete") from error
    if not isinstance(pack, dict) or "id" not in pack or "tours" not in pack:
        raise GotQuestionsParseError("Embedded package has an unexpected shape")
    return pack


def _people(items: Any) -> list[dict[str, Any]]:
    if not isinstance(items, list):
        return []
    return [
        {"id": item.get("id"), "name": clean_text(item.get("name"))}
        for item in items
        if isinstance(item, dict)
    ]


def _media(question: dict[str, Any]) -> list[dict[str, str]]:
    fields = {
        "razdatkaPic": "handout",
        "audio": "question_audio",
        "answerPic": "answer",
        "commentPic": "explanation",
        "commentAudio": "explanation_audio",
    }
    result = []
    for field, role in fields.items():
        value = question.get(field)
        if isinstance(value, str) and value.strip():
            result.append({"role": role, "url": urljoin(BASE_URL, value.strip())})
    return result


def parse_pack(html: str, *, fetched_at: str | None = None) -> list[QuestionRecord]:
    pack = extract_pack_object(html)
    fetched_at = fetched_at or datetime.now(UTC).isoformat()
    pack_id = str(pack["id"])
    package_title = clean_text(pack.get("title"))
    played_at = clean_text(pack.get("startDate"))
    pack_editors = _people(pack.get("editors"))
    db_slug = clean_text(pack.get("dbchgkinfoslug"))
    records: list[QuestionRecord] = []

    for tour in pack.get("tours") or []:
        if not isinstance(tour, dict):
            continue
        for question in tour.get("questions") or []:
            if not isinstance(question, dict) or question.get("id") is None:
                continue
            question_id = str(question["id"])
            media = _media(question)
            extra = {
                "pack": {
                    "id": pack.get("id"),
                    "long_title": pack.get("longTitle"),
                    "end_date": pack.get("endDate"),
                    "publication_date": pack.get("pubDate"),
                    "editors": pack_editors,
                    "info": pack.get("info"),
                    "discussion_url": pack.get("discussionURL"),
                    "db_chgk_info_slug": db_slug or None,
                },
                "round": {
                    "id": tour.get("id"),
                    "number": tour.get("number"),
                    "title": tour.get("title"),
                    "info": tour.get("info"),
                    "editors": _people(tour.get("editors")),
                },
                "question_number": question.get("number"),
                "authors": _people(question.get("authors")),
                "source_references": question.get("source"),
                "rejected_answers": question.get("nezachet"),
                "notes": question.get("note"),
                "tags": question.get("tags") or [],
                "rating": {
                    "teams": question.get("teams") or [],
                    "solve_percentages": question.get("complexity") or [],
                    "correct_answers": question.get("correct_answers") or [],
                    "tournaments": question.get("tournaments") or [],
                },
                "taken_down": bool(question.get("takenDown")),
            }
            record = QuestionRecord(
                source="gotquestions",
                source_question_id=question_id,
                source_url=f"{BASE_URL}/question/{question_id}",
                game_kind="sport_chgk",
                question=question.get("text") or "",
                answer=question.get("answer") or "",
                explanation=question.get("comment") or "",
                acceptance_criteria=question.get("zachet") or "",
                handout_text=question.get("razdatkaText") or "",
                media_urls_json=json.dumps(
                    media, ensure_ascii=False, sort_keys=True, separators=(",", ":")
                ),
                package_title=package_title,
                played_at=played_at,
                extra_json=json.dumps(
                    extra, ensure_ascii=False, sort_keys=True, separators=(",", ":")
                ),
                fetched_at=fetched_at,
            )
            records.append(record.finalize())

    expected = pack.get("questions")
    if isinstance(expected, int) and expected != len(records):
        raise GotQuestionsParseError(
            f"Package {pack_id} declared {expected} questions but parsed {len(records)}"
        )
    return records

