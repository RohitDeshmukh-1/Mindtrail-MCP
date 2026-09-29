# Retrieval benchmark

This measures one thing: **given a question, does Mindtrail return the right stored memory,
and nothing it shouldn't?** It does not measure end-to-end agent task success. That needs
agent-in-the-loop experiments, which are on the roadmap.

Reproduce everything here with:

```bash
mindtrail bench --embedder hashing --failures      # default install
mindtrail bench --embedder fastembed --failures    # with mindtrail[semantic]
```

## Datasets

Both datasets ship inside the package (`src/mindtrail/evaluation/data/`).

| Dataset | Memories | Queries | Domains | Role |
|---|---:|---:|---|---|
| `dev` (mindtrail-retrieval v1) | 57 | 92 | web shop, ML pipeline, Rust CLI, personal | Default thresholds were tuned on this |
| `holdout` (mindtrail-retrieval-holdout v1) | 30 | 39 | mobile app, data platform, infrastructure, personal | Written **after** tuning and never used to tune |

Each query belongs to one of five categories:

- **lexical:** the question shares wording with the memory.
- **paraphrase:** the same meaning in different words ("how do we ship a new version?" should find
  "deployments go through GitHub Actions").
- **temporal:** the fact changed. The current memory must be found, and the replaced or expired
  one must *not* appear.
- **isolation:** the answer exists only in *another* project, which must never be returned.
- **negative:** nothing relevant is stored anywhere, so the correct answer is an empty result.

Memories are stored at simulated ages, so recency, supersession and expiry behave as they
would in real use.

## Metrics

- **recall@1 / recall@5:** share of answerable queries whose target is ranked first / in the
  top 5.
- **MRR, nDCG@5:** rank-sensitive quality over answerable queries.
- **abstention:** share of queries with *no* in-scope answer that correctly returned
  nothing. Returning unrelated memories wastes the agent's context and can mislead it.
- **stale/foreign leaks:** share of queries that returned a replaced, expired, or other-project
  memory. **This must be 0%**, and CI enforces it.
- **p50 / p95 ms:** search latency per query, including query embedding.

## Results

Measured on 2026-09-29 on a laptop with a 13th-gen Intel CPU (Raptor Lake) running Windows 11 with Python 3.13,
fastembed 0.8.1 and onnxruntime 1.30.0, single-threaded. Full per-query output is in
[`results/`](results).

### Holdout (never tuned on)

| Embedder | recall@1 | recall@5 | MRR | paraphrase recall@5 | abstention | leaks | p50 |
|---|---:|---:|---:|---:|---:|---:|---:|
| hashing (default, offline) | 64% | 75% | 0.696 | 46% | 82% | **0%** | 0.8 ms |
| **bge-small via fastembed** (`[semantic]`) | **79%** | **89%** | **0.839** | **77%** | **91%** | **0%** | 9.4 ms |

### Dev (thresholds tuned here)

| Embedder | recall@1 | recall@5 | MRR | paraphrase recall@5 | abstention | leaks | p50 |
|---|---:|---:|---:|---:|---:|---:|---:|
| hashing (default, offline) | 63% | 67% | 0.651 | 27% | 68% | **0%** | 1.0 ms |
| **bge-small via fastembed** (`[semantic]`) | **74%** | **89%** | **0.804** | **73%** | **89%** | **0%** | 10.3 ms |

What this shows:

- **Isolation and staleness held on every query:** all 29 queries with a replaced, expired
  or other-project memory to avoid returned zero leaks. These guarantees are enforced
  structurally (SQL scope filters, supersession), not by ranking luck.
- **Wording-matched queries are solved** by both embedders (100% recall@5).
- **Paraphrase is where the neural model earns its download:** recall@5 roughly doubles
  (46% → 77% on holdout).
- **Holdout scores match dev**, so the tuned thresholds did not overfit.

## Calibration

Three candidate gates control what counts as relevant (`RankingWeights` in
`src/mindtrail/memory/retrieval.py`):

1. keyword hits below 40% of the best BM25 score are dropped;
2. keyword hits whose vector similarity is below a support floor are dropped as incidental
   word overlap (hashing 0.10, bge-small 0.60);
3. vector-only hits need a minimum cosine similarity (hashing 0.30, bge-small 0.65).

These were chosen by grid search on `dev`, trading recall against abstention. The main change
from the uncalibrated defaults was a large gain in abstention: bge-small went from 32% to 89%
on dev at the cost of 8 points of recall@5. A 0.05 step in the bge-small thresholds moves
abstention sharply, so if you change the embedding model, recalibrate it with
`mindtrail bench`.

## Limitations

- **Small and synthetic:** 131 hand-written queries by the Mindtrail authors. Treat the numbers
  as a regression baseline, not a claim about your data.
- **English only**, and focused on software-project memory.
- **No end-to-end agent evaluation yet.** Better retrieval does not by itself prove agents
  perform better; that experiment is next on the roadmap.
- **Latency comes from brute-force vector search** over a small corpus on one laptop. It says
  nothing about large stores.

Contributions of new datasets (especially real, anonymized memory logs) are very welcome. See
the dataset schema in `src/mindtrail/evaluation/dataset.py`.
