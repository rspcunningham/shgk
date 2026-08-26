from __future__ import annotations

TRANSLATION_POLICY_VERSION = 6

TRANSLATION_CONSTITUTION = """
Preserve the original path from clues to answer. Repair language-dependent
mechanics, but do not create a different puzzle.

Permitted adaptations:
- Write English that sounds as though it was originally written by a competent
  English-language quiz editor. Reorder and recast phrasing without changing the
  information available to the player.
- Translate discourse markers, presenter formulas, and introductions by their
  function rather than word for word. For example, Russian «Внимание» introducing
  a displayed item normally becomes “Take a look at...”, “Here is...”, or is
  omitted; it is not normally the English exclamation “Attention!”
- Use standard English names, titles, quotations, and transliterations.
- Translate a proverb, maxim, title, or quotation with its standard English form
  when one exists, or with a clear natural rendering of the same meaning. It need
  not already be a common English saying unless its exact form is itself a clue.
- Change stated word-count or letter-count requirements when they merely
  describe the form of the English answer.
- Repair grammatical gender, case, pronoun, or inflection clues by rephrasing
  them so the same inference remains available in English.
- Make grammatically implicit Russian information explicit only when English
  needs it and it does not add a new clue.
- Reproduce a pun, rhyme, or sound relationship differently only when it uses
  the same facts, has the same answer, and preserves essentially the same
  reasoning route and difficulty.
- Remove a language-specific instruction only when it is irrelevant to solving
  the question and is not an essential clue.

Forbidden adaptations:
- Do not change the answer.
- Do not replace a Russian person, work, place, quotation, or cultural reference
  with an English-language analogue.
- Do not add facts that appear only in the explanation or outside the source.
- Do not move explanation material into the question.
- When the exact letters, script, spelling, typography, or wording is part of the
  clue, preserve it exactly. Otherwise use standard English transliteration and
  typography, and document any meaningful change.
- Do not remove an essential clue, invent an unrelated mechanism, turn an
  inference puzzle into simple recall, resolve deliberate ambiguity, or
  materially alter the difficulty or reasoning route.

Usually untranslatable:
- An essential acrostic or exact letter-position mechanism cannot survive.
- An essential Russian-only homophone, rhyme, spelling, or name meaning has no
  fair English rendering.
- An essential grammatical distinction has no reasonable English equivalent.
- The solving mechanism depends on a Russian word, spelling, name, or technical
  term having a relationship that its English counterpart does not have.
- Interacting word or letter constraints cannot all be repaired.
- The English text is readable but no longer gives a fair route to the answer.
- Solving requires directly inspecting, reading, hearing, or manipulating an
  image, object, sound, inscription, or handout whose needed evidence is absent
  from the supplied question and handout text. Merely describing an object in a
  self-contained literary or factual clue does not make it a missing artifact.

Audit every reference to a presented item. Phrases such as “this,” “these,” “here
is,” “take a look,” “which one,” and Russian «Внимание, ...» may indicate that the
original host showed a prop, image, list, or set of choices that is absent from the
record. Do not write as though an unsupplied item is present. If its visual,
audible, spatial, or textual details are needed to solve the question, mark it
untranslatable. If the item is genuinely generic and no unstated feature matters,
make the English self-contained without pretending that it has been supplied.

The dividing line is local repair versus replacement: local repair is allowed;
writing a replacement puzzle is not. If no permitted adaptation preserves a fair
route to the original answer, mark the question untranslatable.

Naturalness is an acceptance criterion, not optional polish. Every final English
field must be grammatical, idiomatic, concise, and natural when read aloud. It
must be free of literal Russian calques, awkward source-language syntax, and
translated presenter boilerplate that an English-language quiz editor would
immediately rewrite. Several equally natural phrasings may be acceptable; do not
cycle among them merely as a matter of taste.

The English must also be semantically accurate to the source: preserve its facts,
qualifications, clue relationships, answer, and intended uncertainty. Do not add,
strengthen, or silently alter claims merely to make the prose smoother.

Source explanations may contain presenter feedback, stage directions, false
starts, or transcript corrections. Treat them as audit context, not as essential
solving mechanics unless the question actually depends on them.
""".strip()
