from __future__ import annotations

import json

from shgk.sources.tv import discover_game_urls, parse_game, season_url


def test_season_and_game_discovery() -> None:
    assert season_url(2025).endswith("/igry-2020-yh/sezon-2025")
    html = """
    <a href="/igry-2020-yh/sezon-2025/30032025-test">game</a>
    <a href="/igry-2020-yh/sezon-2025/30032025-test">duplicate</a>
    <a href="/igry-2020-yh/sezon-2025/02012026-final">cross-year final</a>
    <a href="/igry-2020-yh/sezon-2025">season</a>
    <a href="/igry-2010-yh/sezon-2014/27092014-old-game">related game</a>
    """
    assert discover_game_urls(html, year=2025) == [
        "https://чгк-инфо.рф/igry-2020-yh/sezon-2025/30032025-test",
        "https://чгк-инфо.рф/igry-2020-yh/sezon-2025/02012026-final",
    ]


def test_parses_tv_round() -> None:
    html = """
    <html><h1>30.03.2025 Первая игра</h1>
      <div id="rau1">
        <h3>Раунд 1 (Телезритель)</h3>
        <div class="row vopit">
          <div class="col-xs-12"><strong>ВОПРОС</strong>
            <p>Что находится в чёрном ящике?</p>
            <img src="/images/question.jpg">
          </div>
          <div class="col-xs-12"><strong>ОТВЕТ</strong>
            <p>Отвечает игрок: яблоко.</p>
            <p><strong>Правильный ответ:</strong> Апельсин</p>
            <p>Он оранжевый.</p>
          </div>
        </div>
      </div>
    </html>
    """
    url = "https://чгк-инфо.рф/igry-2020-yh/sezon-2025/30032025-test"

    records = parse_game(html, url, fetched_at="2025-04-01T00:00:00+00:00")

    assert len(records) == 1
    record = records[0]
    assert record.question == "Что находится в чёрном ящике?"
    assert record.answer == "Апельсин"
    assert record.played_at == "2025-03-30"
    assert record.source_question_id.endswith("#rau1")
    assert json.loads(record.media_urls_json) == [
        {"role": "question", "url": "https://чгк-инфо.рф/images/question.jpg"}
    ]


def test_parses_numbered_blitz_subquestions() -> None:
    html = """
    <html><h1>Блиц</h1><div id="rau3"><div class="row vopit">
      <div class="row bliz" id="rau3_1">
        <div><strong>ВОПРОС 1 (Автор, Город)</strong><p>Первый вопрос?</p></div>
        <div><strong>ОТВЕТ 1</strong><p>Правильный ответ: Первый ответ.</p></div>
      </div>
      <div class="row bliz" id="rau3_2">
        <div><strong>ВОПРОС 2</strong><p>Второй вопрос?</p></div>
        <div><strong>ОТВЕТ 2</strong><p>Правильный ответ: Второй ответ.</p></div>
      </div>
    </div></div></html>
    """
    url = "https://чгк-инфо.рф/igry-2020-yh/sezon-2025/02012026-final"

    records = parse_game(html, url)

    assert [record.source_question_id.rsplit("#", 1)[-1] for record in records] == [
        "rau3_1",
        "rau3_2",
    ]
    assert [record.question for record in records] == ["Первый вопрос?", "Второй вопрос?"]
    assert [record.answer for record in records] == ["Первый ответ.", "Второй ответ."]


def test_parses_self_wrapped_and_duplicate_round_ids() -> None:
    html = """
    <html><h1>Особые раунды</h1>
      <div class="row vopit" id="rau0">
        <div><strong>ВОПРОС</strong><p>Предварительный?</p></div>
        <div><strong>ОТВЕТ</strong><p>Правильный ответ: Да.</p></div>
      </div>
      <div class="row bliz" id="rau0">
        <div><strong>ВОПРОС 1</strong><p>Повторный?</p></div>
        <div><strong>ОТВЕТ 1</strong><p>Правильный ответ: Тоже да.</p></div>
      </div>
    </html>
    """

    records = parse_game(
        html, "https://чгк-инфо.рф/igry-2020-yh/sezon-2025/02012026-final"
    )

    assert [record.source_question_id.rsplit("#", 1)[-1] for record in records] == [
        "rau0",
        "rau0~2",
    ]
