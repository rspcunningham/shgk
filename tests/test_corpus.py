from __future__ import annotations

import asyncio
import sqlite3

import pytest

from shgk.corpus import CorpusReader
from shgk.database import QuestionDatabase
from shgk.pipeline import BasicFilterPipeline
from shgk.translation import TranslationPipeline

from test_translation import FakeClient, _candidate, _critique, _record


def _build_corpus(tmp_path):
    source_path = tmp_path / "questions.sqlite3"
    pipeline_path = tmp_path / "pipeline.sqlite3"
    QuestionDatabase(source_path).upsert(
        [_record("a-translated"), _record("b-pending")]
    )
    BasicFilterPipeline(source_path, pipeline_path).run()
    client = FakeClient([_candidate()], [_critique()])
    asyncio.run(
        TranslationPipeline(source_path, pipeline_path).run(client, limit=1)
    )
    ids = {}
    with sqlite3.connect(source_path) as connection:
        for identifier, id in connection.execute(
            "SELECT source_question_id, id FROM questions"
        ):
            ids[identifier] = id
    return source_path, pipeline_path, ids


def test_read_returns_bilingual_quad_with_explanations(tmp_path) -> None:
    source_path, pipeline_path, ids = _build_corpus(tmp_path)

    with CorpusReader(source_path, pipeline_path) as reader:
        quad = reader.read(ids["a-translated"])
        pending = reader.read(ids["b-pending"])

    assert quad.id == ids["a-translated"]
    assert quad.english_question == "Question"
    assert quad.russian_question == "Вопрос"
    assert quad.english_answer == "Answer\n\nExplanation"
    assert quad.russian_answer == "Ответ\n\nОбъяснение"

    assert pending.english_question is None
    assert pending.english_answer is None
    assert pending.russian_question == "Вопрос"


def test_read_unknown_id_raises(tmp_path) -> None:
    source_path, pipeline_path, _ = _build_corpus(tmp_path)
    with CorpusReader(source_path, pipeline_path) as reader:
        with pytest.raises(KeyError):
            reader.read(999999)


def test_reader_is_read_only(tmp_path) -> None:
    source_path, pipeline_path, _ = _build_corpus(tmp_path)
    with CorpusReader(source_path, pipeline_path) as reader:
        with pytest.raises(sqlite3.OperationalError):
            reader._connection.execute("DELETE FROM questions")
