from __future__ import annotations

import asyncio

from shgk import corpus, db
from shgk.translation import TranslationPipeline
from test_ingest import _site
from test_translation import FakeClient, _candidate, _critique


def test_build_runs_every_stage_and_reports_the_corpus(tmp_path) -> None:
    database = tmp_path / "shgk.sqlite3"
    built = corpus.build(_site([1, 2], questions=3), database=database, workers=2)

    assert built.fetched["new"] == 2
    assert built.fetched["questions"] == 6
    assert built.stats.questions == 6
    assert built.stats.clean == 6
    assert built.stats.canonical == 6
    assert built.stats.translated == 0
    assert built.stats.awaiting_translation == 6


def test_build_creates_the_database_if_it_is_missing(tmp_path) -> None:
    database = tmp_path / "nested" / "shgk.sqlite3"
    corpus.build(_site([1]), database=database, workers=1)
    assert database.is_file()


def test_build_is_idempotent(tmp_path) -> None:
    database = tmp_path / "shgk.sqlite3"
    first = corpus.build(_site([1, 2], questions=2), database=database, workers=2)
    second = corpus.build(_site([1, 2], questions=2), database=database, workers=2)
    assert second.fetched["new"] == 0
    assert second.stats == first.stats


def test_build_counts_exclusions_and_duplicates(tmp_path) -> None:
    database = tmp_path / "shgk.sqlite3"
    corpus.build(_site([1]), database=database, workers=1)
    with db.connect(database) as connection:
        # One row that is not a question, and one reprint of an existing one.
        connection.execute(
            "INSERT INTO questions (id,package_id,question,answer,content_hash) "
            "VALUES (900,1,'$1a','Ответ','h')"
        )
        original = connection.execute(
            "SELECT question FROM questions WHERE id != 900 ORDER BY id LIMIT 1"
        ).fetchone()
        connection.execute(
            "INSERT INTO questions (id,package_id,question,answer,content_hash) "
            "VALUES (901,1,?,'Ответ','h2')",
            (original["question"],),
        )
        connection.commit()

    built = corpus.build(_site([1]), database=database, workers=1)
    assert built.exclusions == {"not_a_question": 1}
    assert built.canonical["merged"] == 1
    assert built.canonical["reprints"] == 1
    assert built.stats.clean == built.stats.questions - 1
    assert built.stats.canonical == built.stats.clean - 1


def test_translated_questions_stop_counting_as_awaiting(tmp_path) -> None:
    database = tmp_path / "shgk.sqlite3"
    corpus.build(_site([1], questions=2), database=database, workers=1)
    asyncio.run(
        TranslationPipeline(database).run(
            FakeClient([_candidate()], [_critique()]), limit=1
        )
    )
    with db.connect(database, read_only=True) as connection:
        stats = corpus.stats(connection)
    assert stats.translated == 1
    assert stats.awaiting_translation == 1


def test_run_result_sums_usage_across_questions(tmp_path) -> None:
    """Reporting spend must not need a second pass over the database."""
    database = tmp_path / "shgk.sqlite3"
    corpus.build(_site([1], questions=3), database=database, workers=1)
    client = FakeClient([_candidate()] * 3, [_critique()] * 3)
    result = asyncio.run(TranslationPipeline(database).run(client, limit=3))

    assert result.completed == 3
    with db.connect(database, read_only=True) as connection:
        stored = connection.execute(
            "SELECT SUM(input_tokens) i, SUM(output_tokens) o FROM translations"
        ).fetchone()
    assert result.usage.input_tokens == stored["i"]
    assert result.usage.output_tokens == stored["o"]
