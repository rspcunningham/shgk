from __future__ import annotations

import json

import pytest

from shgk.sources.gotquestions import (
    GotQuestionsParseError,
    discover_pack_ids,
    parse_pack,
)


def _pack_html(pack: dict) -> str:
    stream = "prefix" + json.dumps(
        {"pack": pack}, ensure_ascii=False, separators=(",", ":")
    )
    script_argument = json.dumps(stream, ensure_ascii=False)
    return f"<html><script>self.__next_f.push([1,{script_argument}])</script></html>"


def test_discovers_unique_pack_ids() -> None:
    html = """
    <a href="/pack/42">one</a><a href="/pack/42?tab=1">duplicate</a>
    <a href="/pack/7">two</a><a href="/question/99">not a pack</a>
    """
    assert discover_pack_ids(html) == [42, 7]


def test_parses_embedded_pack() -> None:
    pack = {
        "id": 42,
        "title": "Тестовый пакет",
        "startDate": "2025-01-02T00:00:00",
        "questions": 1,
        "dbchgkinfoslug": "test-pack",
        "editors": [{"id": 2, "name": " Редактор "}],
        "tours": [
            {
                "id": 8,
                "number": 1,
                "title": "Раунд",
                "questions": [
                    {
                        "id": 9001,
                        "number": 3,
                        "text": "Строка 1\n  Строка 2",
                        "answer": "Ответ",
                        "comment": "Комментарий",
                        "zachet": "Зачёт",
                        "razdatkaText": "Раздатка",
                        "razdatkaPic": "/media/image.jpg",
                        "authors": [{"id": 3, "name": "Автор"}],
                    }
                ],
            }
        ],
    }

    parsed = parse_pack(_pack_html(pack))

    assert parsed.id == 42
    assert parsed.title == "Тестовый пакет"
    assert parsed.slug == "test-pack"
    assert parsed.played_at_start == "2025-01-02T00:00:00"
    assert json.loads(parsed.editor_ids) == [2]
    assert json.loads(parsed.editor_names) == ["Редактор"]

    assert len(parsed.questions) == 1
    question = parsed.questions[0]
    assert question.id == 9001
    assert question.question_number == 3
    assert question.question == "Строка 1\nСтрока 2"
    assert question.answer == "Ответ"
    assert question.explanation == "Комментарий"
    assert question.acceptance_criteria == "Зачёт"
    assert question.handout_text == "Раздатка"
    assert json.loads(question.author_ids) == [3]
    assert json.loads(question.author_names) == ["Автор"]
    assert question.has_media == 1
    assert json.loads(question.media_urls) == [
        {"role": "handout", "url": "https://gotquestions.online/media/image.jpg"}
    ]
    assert question.content_hash


def test_declared_question_count_is_enforced() -> None:
    """A mismatch means the RSC stream was truncated, not that the pack is small."""
    pack = {
        "id": 42, "title": "T", "questions": 5,
        "tours": [{"id": 8, "questions": [
            {"id": 1, "text": "Вопрос", "answer": "Ответ"}
        ]}],
    }
    with pytest.raises(GotQuestionsParseError):
        parse_pack(_pack_html(pack))


def test_host_note_is_split_out_during_parsing() -> None:
    pack = {
        "id": 42, "title": "T", "questions": 1,
        "tours": [{"id": 8, "questions": [{
            "id": 1,
            "text": "[Ведущему: пауза]\nНазовите предмет для чая в поезде.",
            "answer": "Стакан",
        }]}],
    }
    question = parse_pack(_pack_html(pack)).questions[0]
    assert question.question == "Назовите предмет для чая в поезде."
    assert question.host_note == "[Ведущему: пауза]"


def test_missing_question_text_becomes_a_media_placeholder() -> None:
    pack = {
        "id": 42, "title": "T", "questions": 1,
        "tours": [{"id": 8, "questions": [{
            "id": 1, "text": "", "answer": "Ответ", "answerPic": "/a.jpg",
        }]}],
    }
    question = parse_pack(_pack_html(pack)).questions[0]
    assert question.question == "[Media question; see media URLs]"
    assert question.has_media == 1
