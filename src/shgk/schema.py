"""Schema for the single-file corpus and pipeline database.

Four stages live here, each idempotent and separately rebuildable:

1. ``questions`` / ``packages`` -- the raw scrape target.
2. ``question_exclusions``      -- rows that are not a usable question at all.
3. ``question_duplicates``      -- non-canonical reprints of another question.
4. ``translations``            -- the expensive stage, keyed on content_hash.

Stages 2 and 3 are pure functions of the question text and rebuild from scratch
in seconds, so they carry no incremental bookkeeping. Only stage 4 costs money,
which is why it alone records the content_hash it was produced from.
"""

from __future__ import annotations

SCHEMA = """
CREATE TABLE IF NOT EXISTS packages (
    id              INTEGER PRIMARY KEY,
    title           TEXT    NOT NULL DEFAULT '',
    slug            TEXT    NOT NULL DEFAULT '',
    played_at_start TEXT,
    played_at_end   TEXT,
    editor_ids      TEXT    NOT NULL DEFAULT '[]',
    editor_names    TEXT    NOT NULL DEFAULT '[]',
    url             TEXT    NOT NULL,
    status          TEXT    NOT NULL,
    http_status     INTEGER,
    page_hash       TEXT,
    questions_found INTEGER NOT NULL DEFAULT 0,
    first_seen_at   TEXT    NOT NULL,
    fetched_at      TEXT    NOT NULL,
    error           TEXT    NOT NULL DEFAULT ''
);

CREATE TABLE IF NOT EXISTS questions (
    id                  INTEGER PRIMARY KEY,
    package_id          INTEGER NOT NULL REFERENCES packages(id),
    question_number     INTEGER,
    question            TEXT    NOT NULL,
    answer              TEXT    NOT NULL,
    explanation         TEXT    NOT NULL DEFAULT '',
    acceptance_criteria TEXT    NOT NULL DEFAULT '',
    handout_text        TEXT    NOT NULL DEFAULT '',
    host_note           TEXT    NOT NULL DEFAULT '',
    kind                TEXT    NOT NULL DEFAULT 'normal',
    has_media           INTEGER NOT NULL DEFAULT 0,
    media_urls          TEXT    NOT NULL DEFAULT '[]',
    author_ids          TEXT    NOT NULL DEFAULT '[]',
    author_names        TEXT    NOT NULL DEFAULT '[]',
    tournament_ids      TEXT    NOT NULL DEFAULT '[]',
    source_references   TEXT    NOT NULL DEFAULT '',
    taken_down          INTEGER NOT NULL DEFAULT 0,
    solve_percentages   TEXT    NOT NULL DEFAULT '[]',
    correct_answers     TEXT    NOT NULL DEFAULT '[]',
    content_hash        TEXT    NOT NULL,
    normalized_hash     TEXT    NOT NULL
);

CREATE TABLE IF NOT EXISTS question_exclusions (
    question_id INTEGER PRIMARY KEY REFERENCES questions(id),
    reason      TEXT NOT NULL
);

-- Only non-canonical rows are listed; the canonical row is absent from this
-- table, so questions_canonical is a plain anti-join.
CREATE TABLE IF NOT EXISTS question_duplicates (
    question_id  INTEGER PRIMARY KEY REFERENCES questions(id),
    canonical_id INTEGER NOT NULL REFERENCES questions(id)
);

CREATE TABLE IF NOT EXISTS translations (
    question_id            INTEGER PRIMARY KEY REFERENCES questions(id),
    content_hash           TEXT    NOT NULL,
    status                 TEXT    NOT NULL
                           CHECK (status IN ('translated','adapted','untranslatable')),
    question_en            TEXT    NOT NULL DEFAULT '',
    answer_en              TEXT    NOT NULL DEFAULT '',
    explanation_en         TEXT    NOT NULL DEFAULT '',
    acceptance_criteria_en TEXT    NOT NULL DEFAULT '',
    handout_text_en        TEXT    NOT NULL DEFAULT '',
    changes_description    TEXT    NOT NULL DEFAULT '',
    untranslatable_reason  TEXT    NOT NULL DEFAULT '',
    editor_status          TEXT    NOT NULL
                           CHECK (editor_status IN
                                  ('unchanged','edited','needs_rework','skipped')),
    translation_attempts   INTEGER NOT NULL DEFAULT 0,
    critic_attempts        INTEGER NOT NULL DEFAULT 0,
    editor_attempts        INTEGER NOT NULL DEFAULT 0,
    api_requests           INTEGER NOT NULL DEFAULT 0,
    input_tokens           INTEGER NOT NULL DEFAULT 0,
    cached_input_tokens    INTEGER NOT NULL DEFAULT 0,
    cache_write_input_tokens INTEGER NOT NULL DEFAULT 0,
    output_tokens          INTEGER NOT NULL DEFAULT 0,
    reasoning_output_tokens  INTEGER NOT NULL DEFAULT 0,
    completed_at           TEXT    NOT NULL
);

CREATE INDEX IF NOT EXISTS questions_package_idx ON questions(package_id);
CREATE INDEX IF NOT EXISTS questions_norm_idx    ON questions(normalized_hash);
CREATE INDEX IF NOT EXISTS duplicates_canonical_idx
    ON question_duplicates(canonical_id);
CREATE INDEX IF NOT EXISTS translations_status_idx ON translations(status);
"""

VIEWS = """
DROP VIEW IF EXISTS questions_clean;
DROP VIEW IF EXISTS questions_canonical;
DROP VIEW IF EXISTS questions_translated;

-- Stage 2: everything that is a usable, self-contained question.
CREATE VIEW questions_clean AS
    SELECT q.* FROM questions AS q
    WHERE NOT EXISTS (
        SELECT 1 FROM question_exclusions AS x WHERE x.question_id = q.id
    );

-- Stage 3: one row per distinct question.
CREATE VIEW questions_canonical AS
    SELECT q.* FROM questions_clean AS q
    WHERE NOT EXISTS (
        SELECT 1 FROM question_duplicates AS d WHERE d.question_id = q.id
    );

-- Stage 4: canonical questions that have usable English text.
CREATE VIEW questions_translated AS
    SELECT q.*, t.question_en, t.answer_en, t.explanation_en,
           t.acceptance_criteria_en, t.handout_text_en, t.status AS translation_status
    FROM questions_canonical AS q
    JOIN translations AS t
      ON t.question_id = q.id AND t.content_hash = q.content_hash
    WHERE t.status IN ('translated', 'adapted');
"""
