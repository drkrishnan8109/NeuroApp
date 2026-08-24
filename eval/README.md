# neuro_qa_100.jsonl — 100 grounded neurology Q&A

One JSON object per line, shaped for direct upload to LangSmith:

```json
{"inputs":  {"question": "..."},
 "outputs": {"answer": "...", "answerable": true},
 "metadata": {"source_file": "...", "page": 412, "gold_content_key": "sha1..."}}
```

## How it was built

Every question comes from a real corpus chunk, not from model recall. The 21,030 chunks
(1500/300) were filtered to 14,104 substantive prose passages — reference lists, tables,
contents pages and figure captions were dropped by digit density, citation markers and
line-break ratio — then sampled across all four books and passed to `qwen2.5:14b` with
instructions to write one specialist-level question answerable entirely from that passage.

128 passages were attempted; 100 survived filtering for length, self-reference and
near-duplicate questions. All 100 questions are unique and each maps to a distinct chunk
on a distinct page.

| source | questions |
| --- | --- |
| Bradley's *Neurology in Clinical Practice* | 48 |
| Harrison's *Neurology in Clinical Medicine* | 21 |
| *Clinical Neurophysiology* | 17 |
| *Handbook of Emergency Neurology* | 14 |

## `gold_content_key` is the useful part

It is the SHA-1 of the normalised chunk text — the same key `src/keyword_search.content_key`
computes — so retrieval can be scored WITHOUT a generator or a judge:

```python
hits = retriever.retrieve(q, k=6, mode="hybrid")["hits"]
hit  = gold in {h["content_key"] for h in hits}      # recall@6
```

That runs in seconds. The LLM-judged metrics take roughly an hour per sweep, which is why
retrieval parameters (`rrf_k`, `candidate_k`, chunk size, the weighted-fusion alpha) have
never been tuned. This file makes them tunable.

## Two limitations to keep in mind

**The reference answers are model-written.** They are extracted from the source passage
rather than invented, but they have not been reviewed by a clinician. A wrong reference
answer becomes a silent error in every correctness score computed against it. Review a
sample before treating this as ground truth.

**Questions written from a chunk are easier to retrieve than real ones**, because they
inherit that chunk's vocabulary. Recall@k here will read higher than on questions a
clinician would actually type. Useful for comparing configurations against each other,
optimistic as an absolute number.

**No refusal cases.** Every row is `answerable: true`. To exercise the abstention gate,
add out-of-corpus questions with `answerable: false`, as in the 14-question set.

## The gold keys are tied to 1500/300 chunking

`gold_content_key` is the hash of a **1500/300** chunk. Recall can therefore only be scored
against an index built with the same split:

| backend | chunking | recall scoreable? |
| --- | --- | --- |
| Pinecone (BGE-M3) | 1500/300 | yes |
| BM25 via `build_bm25_local(1500, 300)` | 1500/300 | yes |
| Chroma (MiniLM) | 1000/200 | **no** — different chunk boundaries, so every key misses |

Measured on a 12-question sample against the BM25 arm at 1500/300: **recall@6 = 10/12**.
