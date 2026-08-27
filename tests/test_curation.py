from __future__ import annotations

import sqlite3

import pytest

from shgk.curation import (
    content_hash,
    detect_kind,
    detect_lang,
    exclusion_reason,
    normalized_hash,
    rebuild_duplicates,
    rebuild_exclusions,
    split_host_note,
)
from shgk.schema import SCHEMA, VIEWS


def test_host_note_is_separated_from_the_question():
    question = "[Ведущему: сделать паузу]\nНазовите этот предмет из трёх букв."
    text, note = split_host_note(question)
    assert text == "Назовите этот предмет из трёх букв."
    assert note == "[Ведущему: сделать паузу]"


def test_host_note_extraction_preserves_line_structure():
    """Handout material and quotations rely on newlines, so they must survive."""
    question = "[Чтецу: медленно]\nРаздаточный материал\nлев   тростник\nПеред вами перевод."
    text, _ = split_host_note(question)
    assert text == "Раздаточный материал\nлев тростник\nПеред вами перевод."


def test_host_note_without_a_closing_bracket_stops_at_the_line_end():
    question = "[Ведущему: читать чётко\nНазовите предмет."
    text, note = split_host_note(question)
    assert note == "[Ведущему: читать чётко"
    assert text == "Назовите предмет."


def test_question_without_a_host_note_is_untouched():
    question = "В слове «return» [ретёрн] дизайнер обыграл значение."
    text, note = split_host_note(question)
    assert (text, note) == (question, "")


@pytest.mark.parametrize(
    "question, expected",
    [
        ("Блиц. Три вопроса.\n1. а\n2. б", "blitz"),
        ("Дуплет. Два вопроса.\n1. а\n2. б", "duplet"),
        ("[Ведущему: пауза]\nДуплет. Два вопроса.", "duplet"),
        # The bare substring appears inside ordinary words.
        ("В столице лицо полиции изменилось.", "normal"),
        ("Назовите блицкриг одним словом.", "normal"),
    ],
)
def test_detect_kind(question, expected):
    assert detect_kind(question) == expected


@pytest.mark.parametrize(
    "text, expected",
    [
        ("У своїй автобіографії розповідає про вчителя", "uk"),
        ("У пастаноўцы непаразуменне паміж тутэйшымі ў горадзе", "be"),
        ("Русский текст, в котором один раз мелькнуло і.", "ru"),
        ("Назовите этот предмет одним словом.", "ru"),
    ],
)
def test_detect_lang(text, expected):
    assert detect_lang(text) == expected


def test_content_hash_ignores_play_statistics():
    """A question replayed at a new tournament has not changed."""
    first = content_hash("вопрос", "ответ", "комментарий", "", "")
    second = content_hash("вопрос", "ответ", "комментарий", "", "")
    assert first == second
    assert first != content_hash("вопрос", "другой ответ", "комментарий", "", "")


def test_normalized_hash_folds_case_punctuation_and_spacing():
    assert normalized_hash("Назовите  предмет!") == normalized_hash("назовите предмет")
    assert normalized_hash("Назовите предмет") != normalized_hash("Назовите город")


@pytest.mark.parametrize(
    "question, answer, expected",
    [
        ("$1a", "1. Абхазский.", "not_a_question"),
        ("---", "Ответ", "not_a_question"),
        ("Назовите предмет, использующийся для чая в поезде.", "", "no_answer"),
        (
            "В предыдущем вопросе мы говорили о нём. Назовите его снова, пожалуйста.",
            "Стакан",
            "refers_to_other_question",
        ),
        (
            "Этот вопрос сопровождал раздаточный материал, но он утерян. Назовите предмет.",
            "Стакан",
            "handout_lost",
        ),
        ("Назовите предмет, использующийся для чая в поезде.", "Стакан", None),
    ],
)
def test_exclusion_reason(question, answer, expected):
    assert exclusion_reason(question, answer) == expected


def test_exclusion_keeps_questions_that_only_look_wrong():
    """Rules that misfired during the corpus audit must stay out."""
    imperative = "Учёные назвали программу в честь этого объекта. Назовите его."
    describes_image = "На эмблеме общества изображена летучая мышь. Назовите учёного."
    inline_handout = "Раздаточный материал\nлев тростник\nПеред вами перевод. Назовите ЕЁ."
    for question in (imperative, describes_image, inline_handout):
        assert exclusion_reason(question, "ответ") is None


@pytest.fixture
def database():
    connection = sqlite3.connect(":memory:")
    connection.executescript(SCHEMA)
    connection.executescript(VIEWS)
    connection.execute(
        "INSERT INTO packages (id,url,status,first_seen_at,fetched_at) "
        "VALUES (1,'u','ok','t','t')"
    )
    return connection


def _insert(connection, question_id, question, answer="ответ", **overrides):
    values = {
        "solve_percentages": "[]",
        "correct_answers": "[]",
        "normalized_hash": normalized_hash(question),
        **overrides,
    }
    connection.execute(
        """INSERT INTO questions (id,package_id,question,answer,solve_percentages,
             correct_answers,content_hash,normalized_hash)
           VALUES (?,1,?,?,?,?,'h',?)""",
        (
            question_id,
            question,
            answer,
            values["solve_percentages"],
            values["correct_answers"],
            values["normalized_hash"],
        ),
    )


def test_rebuild_exclusions_is_idempotent(database):
    _insert(database, 1, "Назовите предмет, использующийся для чая в поезде.")
    _insert(database, 2, "$1a")
    for _ in range(2):
        counts = rebuild_exclusions(database)
        assert counts == {"not_a_question": 1}
        assert database.execute("SELECT COUNT(*) FROM questions_clean").fetchone()[0] == 1


def test_rebuild_duplicates_keeps_the_best_measured_copy(database):
    text = "Назовите предмет, использующийся для чая в поезде."
    # 5 of 10 teams solved it, versus 90 of 100 -- the second is better evidence.
    _insert(database, 1, text, solve_percentages="[50.0]", correct_answers="[5]")
    _insert(database, 2, text, solve_percentages="[90.0]", correct_answers="[90]")
    rebuild_exclusions(database)
    stats = rebuild_duplicates(database)
    assert stats == {"groups": 1, "duplicates": 1}
    survivor = database.execute("SELECT id FROM questions_canonical").fetchall()
    assert survivor == [(2,)]


def test_excluded_questions_are_not_chosen_as_canonical(database):
    """A duplicate group must never be represented by a row stage 2 threw out."""
    text = "Назовите предмет, использующийся для чая в поезде."
    _insert(database, 1, text, solve_percentages="[90.0]", correct_answers="[90]")
    _insert(database, 2, "$1a")
    rebuild_exclusions(database)
    rebuild_duplicates(database)
    assert database.execute(
        "SELECT COUNT(*) FROM question_duplicates d "
        "JOIN question_exclusions x ON x.question_id = d.canonical_id"
    ).fetchone()[0] == 0
