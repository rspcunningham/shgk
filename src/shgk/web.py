"""A minimal local reader for the English corpus.

Serves one question at a time: read it, reveal the answer, optionally show the
Russian source. Standard library only — no framework, no build step.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from functools import partial
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from .corpus import CorpusReader

PAGE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>ChGK — English corpus</title>
<style>
  :root {
    color-scheme: light dark;
    --surface: #fcfcfb; --plane: #f4f4f1;
    --ink: #0b0b0b; --ink-2: #52514e; --muted: #898781;
    --rule: #e1e0d9; --accent: #2a78d6;
  }
  @media (prefers-color-scheme: dark) {
    :root {
      --surface: #1a1a19; --plane: #0d0d0d;
      --ink: #ffffff; --ink-2: #c3c2b7; --muted: #898781;
      --rule: #2c2c2a; --accent: #3987e5;
    }
  }
  * { box-sizing: border-box; }
  body {
    margin: 0; min-height: 100vh; background: var(--plane); color: var(--ink);
    font: 16px/1.6 system-ui, -apple-system, "Segoe UI", sans-serif;
    display: flex; flex-direction: column; align-items: center;
  }
  header {
    width: 100%; border-bottom: 1px solid var(--rule); background: var(--surface);
    padding: 14px 24px; display: flex; gap: 16px; align-items: baseline;
  }
  header h1 { font-size: 15px; font-weight: 600; margin: 0; letter-spacing: .01em; }
  header .meta { color: var(--muted); font-size: 13px; margin-left: auto; }
  main {
    width: min(46rem, 92vw); background: var(--surface); margin: 32px 0 64px;
    border: 1px solid var(--rule); border-radius: 10px; padding: 40px;
  }
  .label {
    font-size: 11px; letter-spacing: .09em; text-transform: uppercase;
    color: var(--muted); margin-bottom: 10px;
  }
  .question { font-size: 21px; line-height: 1.62; white-space: pre-wrap; }
  .answer { font-size: 20px; font-weight: 600; white-space: pre-wrap; }
  .explanation {
    font-size: 16px; color: var(--ink-2); white-space: pre-wrap; margin-top: 12px;
  }
  section { border-top: 1px solid var(--rule); margin-top: 28px; padding-top: 24px; }
  section[hidden] { display: none; }
  .ru { font-size: 16px; color: var(--ink-2); white-space: pre-wrap; }
  .ru + .ru { margin-top: 14px; }
  .controls { display: flex; gap: 10px; margin-top: 32px; flex-wrap: wrap; }
  button {
    font: inherit; font-size: 14px; padding: 9px 18px; border-radius: 7px;
    border: 1px solid var(--rule); background: var(--plane); color: var(--ink);
    cursor: pointer;
  }
  button:hover { border-color: var(--muted); }
  button.primary { background: var(--accent); border-color: var(--accent); color: #fff; }
  kbd {
    font: inherit; font-size: 11px; color: var(--muted); border: 1px solid var(--rule);
    border-radius: 4px; padding: 1px 5px; margin-left: 7px;
  }
  .hint { color: var(--muted); font-size: 13px; margin-top: 18px; }
</style>
</head>
<body>
<header>
  <h1>What? Where? When? — English corpus</h1>
  <span class="meta" id="meta"></span>
</header>
<main>
  <div class="label">Question</div>
  <div class="question" id="question">Loading…</div>

  <section id="answer-block" hidden>
    <div class="label">Answer</div>
    <div class="answer" id="answer"></div>
    <div class="explanation" id="explanation"></div>
  </section>

  <section id="russian-block" hidden>
    <div class="label">Russian source</div>
    <div class="ru" id="ru-question"></div>
    <div class="ru" id="ru-answer"></div>
  </section>

  <div class="controls">
    <button class="primary" id="reveal">Reveal answer<kbd>space</kbd></button>
    <button id="next">Next question<kbd>N</kbd></button>
    <button id="toggle-ru">Show Russian<kbd>R</kbd></button>
  </div>
  <div class="hint" id="hint"></div>
</main>
<script>
const $ = (id) => document.getElementById(id);
let showRussian = false;

// The stored answer is "answer\\n\\nexplanation"; split for presentation only.
function splitAnswer(text) {
  const i = (text || "").indexOf("\\n\\n");
  return i === -1 ? [text || "", ""] : [text.slice(0, i), text.slice(i + 2)];
}

async function load() {
  $("answer-block").hidden = true;
  $("question").textContent = "Loading…";
  const q = await (await fetch("/api/question")).json();
  const [answer, explanation] = splitAnswer(q.english_answer);
  $("question").textContent = q.english_question;
  $("answer").textContent = answer;
  $("explanation").textContent = explanation;
  $("ru-question").textContent = q.russian_question;
  $("ru-answer").textContent = q.russian_answer;
  $("meta").textContent = "#" + q.id;
  $("hint").textContent = "";
  applyRussian();
}

function applyRussian() {
  $("russian-block").hidden = !showRussian;
  $("toggle-ru").textContent = showRussian ? "Hide Russian" : "Show Russian";
  $("toggle-ru").insertAdjacentHTML("beforeend", '<kbd>R</kbd>');
}

$("reveal").onclick = () => { $("answer-block").hidden = false; };
$("next").onclick = load;
$("toggle-ru").onclick = () => { showRussian = !showRussian; applyRussian(); };

document.addEventListener("keydown", (e) => {
  if (e.target.tagName === "INPUT") return;
  if (e.code === "Space") { e.preventDefault(); $("answer-block").hidden = false; }
  else if (e.key.toLowerCase() === "n") load();
  else if (e.key.toLowerCase() === "r") { showRussian = !showRussian; applyRussian(); }
});

load();
</script>
</body>
</html>
"""


class _Handler(BaseHTTPRequestHandler):
    def __init__(self, *args, source_db: Path, pipeline_db: Path, **kwargs):
        # A reader is opened per request rather than shared: SQLite connections
        # are bound to the thread that created them, and each request here runs
        # on its own thread. Opening a connection is cheap.
        self._source_db = source_db
        self._pipeline_db = pipeline_db
        super().__init__(*args, **kwargs)

    def _send(self, status: int, body: bytes, content_type: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802 - stdlib naming
        if self.path in ("/", "/index.html"):
            self._send(200, PAGE.encode("utf-8"), "text/html; charset=utf-8")
        elif self.path == "/api/question":
            try:
                with CorpusReader(self._source_db, self._pipeline_db) as reader:
                    quad = reader.random_quad()
            except LookupError as error:
                self._send(
                    503,
                    json.dumps({"error": str(error)}).encode("utf-8"),
                    "application/json",
                )
                return
            payload = json.dumps(asdict(quad), ensure_ascii=False).encode("utf-8")
            self._send(200, payload, "application/json; charset=utf-8")
        else:
            self._send(404, b"not found", "text/plain")

    def log_message(self, *args) -> None:
        """Silence per-request logging; the CLI prints what matters."""


def serve(
    source_db: str | Path,
    pipeline_db: str | Path,
    *,
    host: str = "127.0.0.1",
    port: int = 8765,
) -> None:
    source_db, pipeline_db = Path(source_db), Path(pipeline_db)
    # Fail loudly at startup rather than on the first request.
    CorpusReader(source_db, pipeline_db).close()
    handler = partial(_Handler, source_db=source_db, pipeline_db=pipeline_db)
    server = ThreadingHTTPServer((host, port), handler)
    print(f"Reading questions at http://{host}:{port}  (Ctrl-C to stop)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")
    finally:
        server.server_close()
