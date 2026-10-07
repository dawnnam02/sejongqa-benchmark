# License

## 1. Data and annotations: CC BY-NC 4.0
- Covers: `data/` (questions, answers, evidence identifiers, article identifiers and Identity references; see §4 for quoted text), `runs/`, `prompts/`, and `graphrag/settings.yaml`.
- License: Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0), https://creativecommons.org/licenses/by-nc/4.0/.

## 2. Code: MIT License
- Covers: `scripts/`.

Copyright (c) 2026 Hyonam Kim and Seungyun Choi

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

## 3. GraphRAG prompts: MIT License (Microsoft)
- `graphrag/prompts/` contains prompts adapted from Microsoft GraphRAG (https://github.com/microsoft/graphrag), Copyright (c) Microsoft Corporation, released under the MIT License. The adapted files remain under the MIT License, and the Microsoft copyright notice applies to them together with the permission notice in §2.
- The two example articles in `graphrag/prompts/extract_graph.txt` are masked; see §5.

## 4. Quoted source text
- `data/evidence.jsonl` contains short quotations from the new modern Korean translation of the Sejong Sillok (신역 조선왕조실록; © Institute for the Translation of Korean Classics, 한국고전번역원, 2022–; provided in the Korean Classics DB, https://db.itkc.or.kr, under its terms of use). They are included only as evidence for the answers.
- Rights in the translation remain with their holders; the licenses above do not cover the quoted text. If the rights holder objects, the `quote` fields will be removed and the `source_id` fields kept.

## 5. Full article texts
- Full article texts are not redistributed. Obtain them from the provider (Korean Classics DB, https://db.itkc.or.kr) under its terms of use and rebuild the corpus with `scripts/rebuild_corpus.py`. The masked example articles in `graphrag/prompts/extract_graph.txt` are restored the same way.

## 6. Model outputs
- The answers in `runs/` were generated with gpt-4.1-mini and gpt-4.1 (OpenAI). They are released for reproducibility, subject to the model provider's terms.
