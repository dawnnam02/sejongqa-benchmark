# Independent solving verification

This is the protocol for the separate LLM verification stage, the first of the two verification stages applied to candidate questions. Every drafted question was solved by Claude Opus 5.5 agents that had not taken part in drafting it. The second stage, the human review, is described at the end of this file.

## Purpose
The goal is to confirm that each question can be answered from its gold articles and has exactly one correct answer, so that errors in the wiki do not carry over into the questions. The automatic checks in `question_generation_rules.md` do not test whether a quotation actually supports the answer; this step does.

## What the solver receives
- Only `{id, question, evidence: [{source_id, text = full article}]}`.
- The solver is not given the answer, answer aliases, quotations, chains, pair IDs, subtypes or notes.

## What the solver returns
`{id, answer, confidence, alternatives, note}`: the answer, a confidence level, any alternative answers that the articles also support, and a note quoting the supporting sentence.

## Independence
- The solver is a different agent from the one that drafted the question.
- Questions are shuffled and given masked IDs.
- The two questions of a contrastive pair are assigned to different solvers.
- One solver handles at most 40 questions.

## Judging and handling
- Each solution is judged as a match, mismatch, multiple answers, partial match or unsolved.
- A question is re-examined against the source articles if its solution was not judged a match, or if it was accepted only because one answer contained the other. The question is then revised or discarded; nothing is discarded automatically. The decision and its reason are recorded.
- A revised question must pass the automatic checks again and is solved again by a new solver.
- If one question of a pair is discarded, the whole pair is discarded.
- Questions discarded at this stage were candidates and were never part of the benchmark; they are not included in the 25 questions excluded during human review.

## Human review
After the LLM verification, two reviewers checked the question, answer and supporting quotations of every question against the source articles. 25 questions were excluded in this review, leaving the final 975 questions.
