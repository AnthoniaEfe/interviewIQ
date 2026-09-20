# Architecture

InterviewIQ separates **deterministic research algorithms** from **LLM language tasks**.

```
CV/JD text
  → Luna structured extraction (planned)
  → normalization (planned)
  → TF-IDF + BM25 + Sentence-Transformers
  → profile-gap matrix + competency weights
  → Terra question pool (planned)
  → question embeddings + redundancy
  → competency-weighted MMR
  → final question set
```

## Stack

- Backend: FastAPI, Pydantic v2, OpenAI SDK
- Retrieval: scikit-learn TF-IDF, rank-bm25
- Embeddings: Sentence-Transformers (`all-MiniLM-L6-v2`), CPU or CUDA
- Persistence: file-based experiment results first; MongoDB only when interactive demo storage is required
- Frontend: not started (Phase 20)

## Modules

| Package | Responsibility |
| --- | --- |
| `app.config` | Typed settings and centralized pricing |
| `app.services.llm` | Provider, router, retries, usage |
| `app.services.retrieval` | TF-IDF, BM25 |
| `app.services.embeddings` | EmbeddingService abstraction |
| `app.services.gap_analysis` | Gap labels and weights |
| `app.services.mmr` | Competency-weighted MMR |

Ablation flags in settings can disable TF-IDF, BM25, semantic similarity, gap analysis, weighting, MMR, and redundancy filtering independently.

## What was preserved

`interviewiq_baseline/` remains unchanged. It is a flat-prompt generator with hard-coded `N_QS=15` and default `gpt-4o`. The application baseline mode will reimplement that methodology behind the shared LLM router so model identity stays a controlled variable.
