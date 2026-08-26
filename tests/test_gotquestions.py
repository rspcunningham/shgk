from __future__ import annotations

import json

from shgk.sources.gotquestions import discover_pack_ids, parse_pack


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

    records = parse_pack(_pack_html(pack), fetched_at="2025-01-03T00:00:00+00:00")

    assert len(records) == 1
    record = records[0]
    assert record.source_question_id == "9001"
    assert record.question == "Строка 1\nСтрока 2"
    assert record.answer == "Ответ"
    assert record.explanation == "Комментарий"
    assert json.loads(record.media_urls_json) == [
        {"role": "handout", "url": "https://gotquestions.online/media/image.jpg"}
    ]
    assert json.loads(record.extra_json)["pack"]["db_chgk_info_slug"] == "test-pack"
    assert record.content_hash
