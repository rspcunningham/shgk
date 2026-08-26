from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
import re
from typing import Any


_BLANK_LINES = re.compile(r"\n{3,}")


def clean_text(value: Any) -> str:
    """Normalize source text without flattening meaningful line breaks."""
    if value is None:
        return ""
    text = str(value).replace("\r\n", "\n").replace("\r", "\n")
    lines = [re.sub(r"[ \t]+", " ", line).strip() for line in text.split("\n")]
    return _BLANK_LINES.sub("\n\n", "\n".join(lines)).strip()


@dataclass(slots=True)
class QuestionRecord:
    source: str
    source_question_id: str
    source_url: str
    game_kind: str
    question: str
    answer: str
    explanation: str = ""
    acceptance_criteria: str = ""
    handout_text: str = ""
    media_urls_json: str = "[]"
    package_title: str = ""
    played_at: str = ""
    extra_json: str = "{}"
    content_hash: str = ""
    fetched_at: str = ""

    def finalize(self) -> QuestionRecord:
        for field in (
            "source",
            "source_question_id",
            "source_url",
            "game_kind",
            "question",
            "answer",
            "explanation",
            "acceptance_criteria",
            "handout_text",
            "package_title",
            "played_at",
        ):
            setattr(self, field, clean_text(getattr(self, field)))

        if not self.question:
            self.question = "[Media question; see media URLs]"

        payload = asdict(self)
        payload.pop("content_hash", None)
        payload.pop("fetched_at", None)
        canonical = json.dumps(
            payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        self.content_hash = sha256(canonical).hexdigest()
        return self
