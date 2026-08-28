from __future__ import annotations

import sqlite3

import pytest

from shgk.curation import (
    content_hash,
    detect_kind,
    exclusion_reason,
    merge_printings,
    normalized_hash,
    rebuild_canonical,
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


def test_content_hash_ignores_play_statistics():
    """A question replayed at a new tournament has not changed."""
    first = content_hash("вопрос", "ответ", "комментарий", "", "")
    second = content_hash("вопрос", "ответ", "комментарий", "", "")
    assert first == second
    assert first != content_hash("вопрос", "другой ответ", "комментарий", "", "")


def test_normalized_hash_folds_case_punctuation_spacing_and_yo():
    assert normalized_hash("Назовите  предмет!") == normalized_hash("назовите предмет")
    assert normalized_hash("Всё о ёлках") == normalized_hash("Все о елках")
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
        "explanation": "",
        "handout_text": "",
        **overrides,
    }
    connection.execute(
        """INSERT INTO questions (id,package_id,question,answer,solve_percentages,
             correct_answers,explanation,handout_text,content_hash)
           VALUES (?,1,?,?,?,?,?,?,'h')""",
        (
            question_id,
            question,
            answer,
            values["solve_percentages"],
            values["correct_answers"],
            values["explanation"],
            values["handout_text"],
        ),
    )


def _canonical(connection, question_id):
    row = connection.execute(
        "SELECT printings, playings, total_teams, solve_rate, solve_percentages, "
        "correct_answers FROM questions_canonical WHERE id = ?",
        (question_id,),
    ).fetchone()
    return tuple(row) if row else None


TEXT = "Назовите предмет, использующийся для чая в поезде."


def test_rebuild_canonical_keeps_one_record_per_question(database):
    _insert(database, 1, TEXT)
    _insert(database, 2, "Назовите город, в котором родился Пушкин, пожалуйста.")
    rebuild_exclusions(database)
    assert rebuild_canonical(database) == {"questions": 2, "merged": 0, "reprints": 0}
    _insert(database, 3, "назовите  предмет, использующийся для чая в поезде")
    _insert(database, 4, "Назовите предмет, использующийся для чая в поезде!")
    assert rebuild_canonical(database) == {"questions": 2, "merged": 1, "reprints": 2}
    assert database.execute(
        "SELECT question_id, canonical_id FROM question_printings ORDER BY 1"
    ).fetchall() == [(1, 1), (2, 2), (3, 1), (4, 1)]


def test_rebuild_canonical_takes_the_longest_of_each_text_field(database):
    # Neither printing has everything; the record should have all of it.
    _insert(database, 5, TEXT, answer="Подстаканник.", explanation="Держит стакан.")
    _insert(database, 2, TEXT, answer="Подстаканник.\nЗачёт: стакан.", handout_text="рисунок")
    rebuild_exclusions(database)
    rebuild_canonical(database)
    record = database.execute(
        "SELECT id, answer, explanation, handout_text, content_hash "
        "FROM questions_canonical"
    ).fetchone()
    assert record[:4] == (2, "Подстаканник.\nЗачёт: стакан.", "Держит стакан.", "рисунок")
    assert record[4] == content_hash(
        TEXT, "Подстаканник.\nЗачёт: стакан.", "Держит стакан.", "", "рисунок"
    )


def test_rebuild_canonical_pools_play_data(database):
    # 5 of 10 teams, then 90 of 100: pooled, 95 of 110.
    _insert(database, 1, TEXT, solve_percentages="[50.0]", correct_answers="[5]")
    _insert(database, 2, TEXT, solve_percentages="[90.0]", correct_answers="[90]")
    rebuild_exclusions(database)
    rebuild_canonical(database)
    printings, playings, teams, rate, percentages, corrects = _canonical(database, 1)
    assert (printings, playings) == (2, 2)
    assert teams == pytest.approx(110)
    assert rate == pytest.approx(95 / 110)
    assert (percentages, corrects) == ("[50.0, 90.0]", "[5, 90]")
    assert _canonical(database, 2) is None


def test_singletons_get_play_stats_too(database):
    _insert(database, 1, TEXT, solve_percentages="[25.0]", correct_answers="[5]")
    _insert(database, 2, "Назовите город, в котором родился Пушкин, пожалуйста.")
    rebuild_exclusions(database)
    rebuild_canonical(database)
    assert _canonical(database, 1)[:4] == (1, 1, pytest.approx(20), pytest.approx(0.25))
    assert _canonical(database, 2)[:4] == (1, 0, 0, None)


def test_merge_unions_tournaments_and_keeps_the_earliest_identity():
    base = {
        "id": 9, "package_id": 1, "question": TEXT, "answer": "a",
        "explanation": "", "acceptance_criteria": "", "handout_text": "",
        "source_references": "", "tournament_ids": "[2, 3]",
        "solve_percentages": "[]", "correct_answers": "[]",
    }
    later = {**base, "id": 4, "package_id": 7, "tournament_ids": "[3, 1]"}
    record = merge_printings([base, later])
    assert (record["id"], record["package_id"]) == (4, 7)
    assert record["tournament_ids"] == "[3, 1, 2]"


def test_rebuild_canonical_is_idempotent(database):
    _insert(database, 1, TEXT, solve_percentages="[50.0]", correct_answers="[5]")
    _insert(database, 2, TEXT)
    rebuild_exclusions(database)
    for _ in range(2):
        rebuild_canonical(database)
        assert database.execute("SELECT COUNT(*) FROM questions_canonical").fetchone()[0] == 1
        assert _canonical(database, 1)[:3] == (2, 1, pytest.approx(10))


def test_excluded_questions_never_reach_stage_3(database):
    _insert(database, 1, TEXT, solve_percentages="[90.0]", correct_answers="[90]")
    _insert(database, 2, "$1a")
    rebuild_exclusions(database)
    rebuild_canonical(database)
    assert database.execute("SELECT id FROM questions_canonical").fetchall() == [(1,)]
    assert database.execute("SELECT question_id FROM question_printings").fetchall() == [(1,)]
