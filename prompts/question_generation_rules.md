# Question generation rules

These are the instructions given to the Claude Opus 5.5 agents that drafted the candidate questions of SejongQA. The same rules applied to all question types. This file summarizes the instructions; it is not a verbatim prompt.

Candidates drafted under these rules then went through two verification stages: a separate LLM verification (`independent_solving_protocol.md`) and a human review in which two reviewers checked every question, answer and quotation against the source articles. The human review excluded 25 questions, leaving the final 975.

## 1. Material
- The facts used in a question come only from the wiki.
- Evidence is checked against the source articles cited by the wiki (the 4,273 articles of the retrieval corpus). Every `source_id` must be one of these articles, and every evidence quotation must appear verbatim in the cited article.

## 2. Question rules
- Length limit (characters, including spaces): 50 for Identity, Temporal and 2- and 3-document chains; 70 for 5-document chains; 90 for 7-document chains.
- Questions are kept short. Their difficulty must come from the question structure (contrastive pairs, temporal expressions, chains), not from added modifiers. Each question is a single sentence in written style.
- A question must not contain dates, era names, sexagenary cycle names or Western years; the intermediate entities of a chain; or the answer string.
- The answer is a short single span with exactly one correct value. A title that could refer to several people is restricted by the event described in the question.

## 3. Formats by type
- **Identity (contrastive pairs).** A pair is two questions sharing a `pair_id`; it counts as correct only if both questions are answered correctly.
  - Same-person pair (Alias): the two questions ask about the same fact of the same person under different names (personal name, title or appellation). Both answers are the same.
  - Different-person pair (Confusable): the two questions use the same question frame for different people with identical or easily confused names. The answers differ.
- **Temporal (contrastive pairs).** The two questions share the same anchor event, and the temporal expression of one is replaced by its opposite or by a neighboring expression, so that the two answers differ. Both answers must satisfy the stated relation according to the dates recorded in the source articles. Temporal expressions are defined as calendar intervals relative to the anchor article and are grouped into day, month, year and order categories.
- **Relation Chain.** A chain passes through N different wiki pages in order (N = 2, 3, 5, 7). Shortcuts that skip an intermediate page are not allowed.

## 4. Draft record format (one JSON object per line)
```json
{"id": "PJ-001a", "pair_id": "PJ-001", "category": "인물판정", "sub": "같은 사람",
 "question": "…", "answer": "…", "answer_aliases": [],
 "source_ids": ["YY-MM-DD[NN]"], "source_pages": ["wiki page"],
 "evidence": [{"source_id": "YY-MM-DD[NN]", "quote": "verbatim quotation from the article"}]}
```
- Temporal items add `relation`, `anchor` {source_id, event} and `target` {source_id, event}.
- Relation Chain items add `hops` and `chain` [{step, from, relation, to, page, source_ids, quote}] of length N−1.
- In the released data these fields are split across `data/questions.jsonl`, `data/answers.jsonl` and `data/evidence.jsonl`. The wiki page names (`source_pages`, `chain[].page`) are not released because the wiki itself is not released.

## 5. Automatic checks
Every draft had to pass the following checks with no violation of a required rule: schema; unique IDs; length limits; no date leakage; no answer leakage; `source_ids` within the 4,273 articles; verbatim quotations; exactly two questions per pair; equal answers within same-person pairs and different answers within different-person and Temporal pairs; the temporal relation holds when computed from the recorded dates; chain length, number of pages and connectivity.
