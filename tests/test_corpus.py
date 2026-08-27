from __future__ import annotations

import asyncio
import sqlite3

import pytest

from shgk.corpus import CorpusReader
from shgk.translation import TranslationPipeline

from test_translation import FakeClient, _candidate, _critique, _seed


def _translated(tmp_path, count: int = 2):
    path = _seed(tmp_path, count)
    client = FakeClient([_candidate()] * count, [_critique()] * count)
    asyncio.run(TranslationPipeline(path).run(client, limit=count))
    return path


def test_read_returns_a_bilingual_quad_with_explanations(tmp_path) -> None:
    path = _translated(tmp_path, 1)
    with CorpusReader(path) as reader:
        quad = reader.read(1)
    assert quad.id == 1
    assert quad.english_question == "Question"
    assert quad.russian_question.startswith("Вопрос")
    assert quad.english_answer == "Answer\n\nExplanation"
    assert quad.russian_answer == "Ответ\n\nОбъяснение"


def test_untranslated_questions_are_not_readable(tmp_path) -> None:
    """questions_translated only exposes rows that have usable English."""
    path = _seed(tmp_path, 1)
    with CorpusReader(path) as reader:
        with pytest.raises(KeyError):
            reader.read(1)


def test_untranslatable_results_are_not_exposed(tmp_path) -> None:
    path = _seed(tmp_path, 1)
    client = FakeClient(
        [_candidate(status="untranslatable")],
        [_critique(status="untranslatable")],
    )
    asyncio.run(TranslationPipeline(path).run(client, limit=1))
    with CorpusReader(path) as reader:
        with pytest.raises(LookupError):
            reader.random()


def test_random_returns_a_translated_question(tmp_path) -> None:
    path = _translated(tmp_path, 3)
    with CorpusReader(path) as reader:
        assert reader.random().id in {1, 2, 3}


def test_reader_is_read_only(tmp_path) -> None:
    path = _translated(tmp_path, 1)
    with CorpusReader(path) as reader:
        with pytest.raises(sqlite3.OperationalError):
            reader._connection.execute("DELETE FROM questions")
