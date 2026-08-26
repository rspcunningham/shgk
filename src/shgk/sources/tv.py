from __future__ import annotations

from collections import Counter
from datetime import UTC, datetime
import json
import re
from typing import Iterable
from urllib.parse import urljoin, urlsplit

from bs4 import BeautifulSoup, NavigableString, Tag

from ..models import QuestionRecord, clean_text


BASE_URL = "https://чгк-инфо.рф"
ROUND_ID_RE = re.compile(r"^rau\d+(?:_\d+)?$")
GAME_PATH_RE = re.compile(r"^/igry-\d{4}-yh/sezon-\d{4}/[^/?#]+$")
DATE_IN_PATH_RE = re.compile(r"/(\d{2})(\d{2})(\d{4})-")


class TvParseError(ValueError):
    pass


def season_url(year: int) -> str:
    decade = year - year % 10
    return f"{BASE_URL}/igry-{decade}-yh/sezon-{year}"


def discover_game_urls(html: str, *, year: int | None = None) -> list[str]:
    soup = BeautifulSoup(html, "lxml")
    result: list[str] = []
    seen: set[str] = set()
    for anchor in soup.find_all("a", href=True):
        href = anchor["href"]
        path = urlsplit(urljoin(BASE_URL, href)).path
        if not GAME_PATH_RE.match(path):
            continue
        if year is not None:
            expected_prefix = urlsplit(season_url(year)).path.rstrip("/") + "/"
            # A season can finish in January of the next calendar year. The
            # season path is authoritative; the date in the slug is not.
            if not path.startswith(expected_prefix):
                continue
        url = urljoin(BASE_URL, path)
        if url not in seen:
            seen.add(url)
            result.append(url)
    return result


def _following_value(label: Tag) -> str:
    pieces: list[str] = []
    for sibling in label.next_siblings:
        if isinstance(sibling, Tag) and sibling.name == "strong":
            break
        if isinstance(sibling, NavigableString):
            pieces.append(str(sibling))
        elif isinstance(sibling, Tag):
            pieces.append(sibling.get_text(" ", strip=True))
    return clean_text(" ".join(pieces))


def _media(container: Tag | None, role: str) -> list[dict[str, str]]:
    if container is None:
        return []
    result: list[dict[str, str]] = []
    for element in container.find_all(["img", "video", "audio", "source"]):
        value = element.get("src")
        if value:
            result.append({"role": role, "url": urljoin(BASE_URL, value)})
    return result


def _container_for_label(row: Tag, expected: str) -> Tag | None:
    for strong in row.find_all("strong"):
        normalized = clean_text(strong.get_text(" ", strip=True)).upper().rstrip(":")
        if re.fullmatch(rf"{re.escape(expected)}(?:\s+\d+)?(?:\s*\([^)]*\))?", normalized):
            return strong.parent if isinstance(strong.parent, Tag) else None
    return None


def _without_label(container: Tag | None, label: str) -> str:
    if container is None:
        return ""
    text = clean_text(container.get_text("\n", strip=True))
    return re.sub(
        rf"^{re.escape(label)}(?:\s+\d+)?(?:\s*\([^)]*\))?\s*:?\s*",
        "",
        text,
        flags=re.IGNORECASE,
    )


def _answers(answer_container: Tag | None) -> tuple[str, str, str, str]:
    if answer_container is None:
        return "", "", "", ""

    official = ""
    team_answer = ""
    correct_given = ""
    for strong in answer_container.find_all("strong"):
        label = clean_text(strong.get_text(" ", strip=True)).lower().rstrip(":")
        value = _following_value(strong)
        if label == "правильный ответ" and value:
            official = value
        elif label == "дан правильный ответ" and value:
            correct_given = value
        elif re.fullmatch(r"ответ(?:\s+\d+)?", label) and value:
            team_answer = value

    explanation = ""
    for element in answer_container.find_all(["p", "li"]):
        text = clean_text(element.get_text(" ", strip=True))
        match = re.search(r"(?:^|\s)Правильный ответ:\s*(.+)", text, re.IGNORECASE)
        if match and clean_text(match.group(1)):
            explanation = clean_text(match.group(1))
            break

    if not official:
        official = correct_given
    if not official and explanation:
        official = explanation
    if not explanation:
        explanation = official
    transcript = _without_label(answer_container, "ОТВЕТ")
    return official, explanation, team_answer, transcript


def _played_at(url: str) -> str:
    match = DATE_IN_PATH_RE.search(urlsplit(url).path)
    if not match:
        return ""
    day, month, year = match.groups()
    return f"{year}-{month}-{day}"


def parse_game(
    html: str,
    game_url: str,
    *,
    fetched_at: str | None = None,
) -> list[QuestionRecord]:
    soup = BeautifulSoup(html, "lxml")
    fetched_at = fetched_at or datetime.now(UTC).isoformat()
    heading = soup.find("h1")
    package_title = clean_text(heading.get_text(" ", strip=True) if heading else "")
    game_path = urlsplit(game_url).path.rstrip("/")
    records: list[QuestionRecord] = []
    round_occurrences: Counter[str] = Counter()

    round_nodes = soup.find_all("div", id=ROUND_ID_RE)
    for round_node in round_nodes:
        if round_node.find("div", id=ROUND_ID_RE) is not None:
            continue
        classes = round_node.get("class") or []
        row = round_node if {"bliz", "vopit"}.intersection(classes) else round_node.find(
            "div", class_=lambda value: value and "vopit" in value.split()
        )
        if row is None:
            continue
        question_container = _container_for_label(row, "ВОПРОС")
        answer_container = _container_for_label(row, "ОТВЕТ")
        if question_container is None:
            continue

        round_id = str(round_node.get("id"))
        round_occurrences[round_id] += 1
        occurrence = round_occurrences[round_id]
        unique_round_id = round_id if occurrence == 1 else f"{round_id}~{occurrence}"
        question_text = _without_label(question_container, "ВОПРОС")
        official, explanation, team_answer, transcript = _answers(answer_container)
        media = _media(question_container, "question") + _media(answer_container, "answer")
        round_heading = round_node.find(["h2", "h3", "h4"])
        extra = {
            "round": clean_text(
                round_heading.get_text(" ", strip=True) if round_heading else round_id
            ),
            "source_round_id": round_id,
            "source_round_occurrence": occurrence,
            "team_answer": team_answer,
            "answer_transcript": transcript,
        }
        records.append(
            QuestionRecord(
                source="chgk_info_tv",
                source_question_id=f"{game_path}#{unique_round_id}",
                source_url=f"{game_url}#{round_id}",
                game_kind="tv_chgk",
                question=question_text,
                answer=official,
                explanation=explanation,
                media_urls_json=json.dumps(
                    media, ensure_ascii=False, sort_keys=True, separators=(",", ":")
                ),
                package_title=package_title,
                played_at=_played_at(game_url),
                extra_json=json.dumps(
                    extra, ensure_ascii=False, sort_keys=True, separators=(",", ":")
                ),
                fetched_at=fetched_at,
            ).finalize()
        )

    if not records:
        raise TvParseError(f"No television questions found at {game_url}")
    return records
