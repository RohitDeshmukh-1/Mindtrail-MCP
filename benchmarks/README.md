# Retrieval benchmark

This measures two things:

1. **Retrieval:** given a question, does Mindtrail return the right stored memory, and nothing
   it shouldn't? (`mindtrail bench`, below.)
2. **Agent in the loop:** does a real coding agent store facts on its own and use them in a
   later session? ([Agent evaluation](#agent-evaluation).)

Reproduce everything here with:

```bash
mindtrail bench --embedder hashing --failures      # default install
mindtrail bench --embedder fastembed --failures    # with mindtrail[semantic] (bge-base)
MINDTRAIL_EMBEDDING_MODEL=BAAI/bge-small-en-v1.5 mindtrail bench --embedder fastembed
mindtrail bench --dataset holdout-v2 locomo:test   # other sets
```

## Datasets

The first three datasets ship inside the package (`src/mindtrail/evaluation/data/`).

| Dataset | Memories | Queries | Domains | Role |
|---|---:|---:|---|---|
| `dev` (mindtrail-retrieval v1) | 57 | 92 | web shop, ML pipeline, Rust CLI, personal | Default thresholds were tuned on this |
| `holdout` (mindtrail-retrieval-holdout v1) | 30 | 39 | mobile app, data platform, infrastructure, personal | Written **after** tuning and never used to tune |
| `holdout-v2` (mindtrail-retrieval-holdout v2) | 32 | 43 | game, payments, lab, docs, personal | Sealed in git before the model and reranker work, evaluated only at the end |
| `locomo:test` ([LoCoMo](https://github.com/snap-research/locomo), CC BY-NC 4.0) | 4431 turns | 1152 | long personal conversations | External check; downloaded on demand, not tuned on |

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

Measured on 2026-09-30 on a laptop with a 13th-gen Intel CPU (Raptor Lake) running Windows 11
with Python 3.13, fastembed 0.8.1 and onnxruntime 1.30.0. Full per-query output is in
[`results/`](results) (`semantic*` = bge-base, `semantic-small*` = bge-small).

### Held-out sets (never tuned on)

| Embedder | set | recall@1 | recall@5 | MRR | paraphrase recall@5 | abstention | leaks |
|---|---|---:|---:|---:|---:|---:|---:|
| hashing (default, offline) | holdout | 64% | 75% | 0.696 | 46% | 82% | **0%** |
| bge-small (`MINDTRAIL_EMBEDDING_MODEL`) | holdout | 75% | 82% | 0.786 | 62% | 100% | **0%** |
| **bge-base** (`[semantic]` default) | holdout | **82%** | **93%** | **0.875** | **85%** | **100%** | **0%** |
| hashing (default, offline) | holdout-v2 | 58% | 68% | 0.618 | 44% | 100% | **0%** |
| bge-small (`MINDTRAIL_EMBEDDING_MODEL`) | holdout-v2 | 81% | 90% | 0.855 | 83% | 100% | **0%** |
| **bge-base** (`[semantic]` default) | holdout-v2 | **90%** | **94%** | **0.919** | **89%** | **100%** | **0%** |

### Dev (thresholds tuned here)

| Embedder | recall@1 | recall@5 | MRR | paraphrase recall@5 | abstention | leaks |
|---|---:|---:|---:|---:|---:|---:|
| hashing (default, offline) | 63% | 67% | 0.651 | 27% | 68% | **0%** |
| bge-small | 75% | 84% | 0.788 | 63% | 100% | **0%** |
| **bge-base** | **85%** | **96%** | **0.902** | **90%** | 95% | **0%** |

### LoCoMo test split (1152 questions)

| Embedder | recall@5 | recall@10 | hit@1 | hit@5 |
|---|---:|---:|---:|---:|
| hashing | 50.6% | 60.1% | 31.9% | 56.9% |
| bge-small | 55.5% | 64.4% | 34.9% | 63.3% |
| bge-base | 55.7% | 63.9% | 37.0% | 63.4% |

Search latency with bge-base is about 30 ms p50 per query on the bundled sets, most of it query
embedding. Hashing is about 1 ms.

What this shows:

- **Isolation and staleness held on every query:** no replaced, expired or other-project memory
  was returned on any set. These guarantees are enforced structurally (SQL scope filters,
  supersession), not by ranking luck.
- **Wording-matched queries are solved** by every embedder (100% recall@5).
- **Paraphrase is where the neural model earns its download,** and bge-base is clearly ahead of
  bge-small on developer memory (85% vs 62% paraphrase recall@5 on holdout).
- **On LoCoMo the two bge models are tied** within the confidence interval. Long chat logs
  about people's lives are a different domain, and multi-hop questions need more than one
  retrieval step.
- **Held-out scores match or beat dev**, so the tuned thresholds did not overfit.

## Calibration

Three candidate gates control what counts as relevant (`RankingWeights` in
`src/mindtrail/memory/retrieval.py`, per-model values in
`src/mindtrail/embeddings/fastembed_provider.py`):

1. keyword hits below 40% of the best BM25 score are dropped;
2. keyword hits whose vector similarity is below a support floor are dropped as incidental
   word overlap (hashing 0.10, bge-small 0.57, bge-base 0.50);
3. vector-only hits need a minimum cosine similarity (hashing 0.30, bge-small 0.62,
   bge-base 0.53).

These were chosen by grid search on `dev` only, trading recall against abstention. For
bge-base, `min_similarity` is flat between 0.51 and 0.54, but the keyword support floor
matters: at 0.48 abstention drops to 89%, and above 0.505 recall@5 starts to fall. 0.50 is the
middle of that plateau. If you change the embedding model, recalibrate with `mindtrail bench`.

## Rerankers

A cross-encoder stage exists (`MINDTRAIL_RERANKER=<fastembed model id>`) but is **off by
default**. On the dev set, every small reranker available through fastembed (MS MARCO
MiniLM-L-6/L-12 and Jina v1 tiny/turbo) ranked developer memories worse than the embeddings
alone. The MS MARCO models reward word overlap over meaning on short, technical text.

## Agent evaluation

`benchmarks/agent_eval.py` runs real Claude Code sessions (`claude -p`). Each session is a
separate process, so nothing carries over except what Mindtrail stored.

1. **Teach:** the user mentions a project fact in passing ("FYI, this project uses pnpm").
   The agent is never told to use Mindtrail.
2. **Ask:** a new session gets a task that depends on the fact, worded differently ("How do I
   add lodash? Just give me the command").
3. **Control:** the same ask sessions with no Mindtrail server.

An answer counts as correct only if it *applies* the fact: the command starts with `pnpm`, the
commit message starts with `fix:`, the type hint is `Optional[str]` on a Python 3.9 project.
Fact set 2 was written before any results were seen, as a held-out check on the instruction
change below.

Results with Claude Sonnet, 12 facts. The current-instruction row combines two runs on
2026-09-30 and one on 2026-10-01. The 2026-10-01 run's per-session answers and tool calls
are in [`results/agent-eval-facts1.json`](results/agent-eval-facts1.json) and
[`results/agent-eval-facts2.json`](results/agent-eval-facts2.json).

| | stored the fact unprompted | correct in a later session | called `recall` |
|---|---:|---:|---:|
| No Mindtrail (2026-10-01) | – | 0/12 | – |
| Mindtrail, original server instructions | 12/12 | 10/12 | 10/12 |
| **Mindtrail, current server instructions** | **36/36** | **36/36** | **36/36** |

The control row was 1/12 before a scorer fix. For the timestamp fact, the control agent
recommended ISO 8601 and listed epoch milliseconds only as a fallback, and the first scorer
counted any mention. An answer now counts only if it recommends epoch milliseconds ahead of
ISO 8601.

Both misses with the original instructions happened when the agent answered a quick
question from habit, without calling `recall` (`npm install lodash`, and `str | None` for a
Python 3.9 project). The server instructions now tell agents to recall before any
project-specific answer, even a one-line one. That fixed both held-out misses as well.

What this does **not** show: the repository is empty, the facts are stated explicitly, and
the sample is small (one model, 12 facts). Longer sessions with real code, facts mixed into
unrelated work, and other agents (Cursor, Codex) are untested.

```bash
python benchmarks/agent_eval.py --facts 1 --control   # 18 agent sessions
python benchmarks/agent_eval.py --facts 2
```

## Limitations

- **Small and synthetic:** 174 hand-written queries by the Mindtrail authors. Treat the numbers
  as a regression baseline, not a claim about your data.
- **English only**, and focused on software-project memory.
- **The agent evaluation is small** (see above). It shows agents use the tools and apply what
  they recall. It does not yet measure task success on real codebases.
- **Latency comes from brute-force vector search** over a small corpus on one laptop. It says
  nothing about large stores.

Contributions of new datasets (especially real, anonymized memory logs) are very welcome. See
the dataset schema in `src/mindtrail/evaluation/dataset.py`.
