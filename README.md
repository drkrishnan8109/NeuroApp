# NeuroApp

Hybrid retrieval over four neurology reference textbooks, with an abstention gate that
declines questions the corpus does not cover.

`notebooks/RAGApp_hybrid.ipynb` runs the whole pipeline: ingest → embed → index →
retrieve → generate → evaluate.

## What is here

| Path | Tracked | Notes |
| --- | --- | --- |
| `notebooks/RAGApp_hybrid.ipynb` | yes | The pipeline and its evaluation |
| `src/` | yes | Loader, embeddings, Pinecone + Chroma stores, BM25, guardrails, serving pipeline |
| `requirements.txt` | yes | Python dependencies |
| `.env.example` | yes | Key names, no values |
| `.env` | **no** | Live API keys — see below |
| `pdfs/` | **no** | Licensed textbooks, 162 MB |
| `data/` | **no** | Parsed-page cache and Chroma store, 362 MB, both regenerable |

The untracked paths are present in this working copy, so the notebook runs as-is. They are
excluded from git deliberately, not by accident:

- **`.env`** holds eight live keys. Anything committed to git stays in its history even
  after deletion, so a single push would mean rotating all eight.
- **`pdfs/`** is licensed material. Bradley's alone is 117 MB, above GitHub's 100 MB
  per-file limit, so it would need LFS even if redistribution were settled.
- **`data/`** is derived from `pdfs/` and rebuilds by running the notebook.

## Running it

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env      # then fill in your keys
jupyter lab notebooks/RAGApp_hybrid.ipynb
```

Also needs [Ollama](https://ollama.com) with the two local models:

```bash
ollama pull qwen2.5:14b     # generation
ollama pull gemma4:12b      # judging
```

## The corpus

Four English references, 5,314 pages, split at 1,500 characters with 300 overlap:

- Bradley's *Neurology in Clinical Practice*, 8th ed. (3,078 pp)
- *Clinical Neurophysiology*, 3rd ed. (915 pp)
- Harrison's *Neurology in Clinical Medicine*, 3E (896 pp)
- *Handbook of Emergency Neurology*, 2023 (425 pp)

## Retrieval

Dense (BGE-M3, 1024-d, cosine) fused with sparse (BM25, lucene, k1=1.5 b=0.75) by
reciprocal rank, `rrf_k=60`, 20 candidates per arm, top 6 to the generator.

Two vector backends are supported and the pipeline falls back between them. They are not
interchangeable piecemeal — the query embedder, the chunk size, and the abstention floor
are each calibrated to a particular store:

| Backend | Embedder | Vectors | Chunking | Floor |
| --- | --- | --- | --- | --- |
| Pinecone | BGE-M3, 1024-d | 21,030 | 1500/300 | cosine ≥ 0.53 |
| Chroma (local, in `data/`) | all-MiniLM-L6-v2, 384-d | 30,478 | 1000/200 | cosine ≥ 0.43 |

Each floor sits midway between that store's measured in-corpus minimum and its
out-of-corpus maximum. Carrying one model's floor across to the other refuses nearly
everything or nothing.

## Not medical advice

Decision support over reference texts, not a diagnosis, and no substitute for clinical
judgement or current local guidelines. Verify against the cited page before acting.
