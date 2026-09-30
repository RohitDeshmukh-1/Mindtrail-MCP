**mindtrail-retrieval v1**, embedder `hashing-v1-512`, reranker `none`, k=5

| category | queries | accuracy | recall@1 | recall@5 | MRR | nDCG@5 | abstention | stale/foreign leaks | p50 ms | p95 ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| isolation | 10 | 60% | 67% | 67% | 0.667 | 0.667 | 57% | 0% | 0.7 | 1.5 |
| lexical | 30 | 93% | 93% | 100% | 0.967 | 0.975 | - | - | 1.1 | 1.8 |
| negative | 12 | 75% | - | - | - | - | 75% | - | 0.8 | 1.9 |
| paraphrase | 30 | 23% | 23% | 27% | 0.250 | 0.254 | - | 0% | 1.0 | 1.6 |
| temporal | 10 | 90% | 90% | 90% | 0.900 | 0.900 | - | 0% | 1.1 | 1.3 |
| **overall** | 92 | 64% | 63% | 67% | 0.651 | 0.656 | 68% | 0% | 1.0 | 1.7 |

95% bootstrap confidence intervals (overall): accuracy 54-74%, recall@1 52-74%, recall@5 56-78%, abstention 47-89%

**mindtrail-retrieval-holdout v1**, embedder `hashing-v1-512`, reranker `none`, k=5

| category | queries | accuracy | recall@1 | recall@5 | MRR | nDCG@5 | abstention | stale/foreign leaks | p50 ms | p95 ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| isolation | 4 | 75% | - | - | - | - | 75% | 0% | 1.1 | 1.4 |
| lexical | 12 | 92% | 92% | 100% | 0.958 | 0.969 | - | - | 0.9 | 3.3 |
| negative | 6 | 83% | - | - | - | - | 83% | - | 0.7 | 1.4 |
| paraphrase | 13 | 31% | 31% | 46% | 0.385 | 0.405 | - | - | 1.0 | 1.5 |
| temporal | 4 | 100% | 100% | 100% | 1.000 | 1.000 | 100% | 0% | 0.8 | 1.4 |
| **overall** | 39 | 69% | 64% | 75% | 0.696 | 0.710 | 82% | 0% | 0.9 | 1.6 |

95% bootstrap confidence intervals (overall): accuracy 54-82%, recall@1 46-82%, recall@5 57-89%, abstention 55-100%

**mindtrail-retrieval-holdout v2**, embedder `hashing-v1-512`, reranker `none`, k=5

| category | queries | accuracy | recall@1 | recall@5 | MRR | nDCG@5 | abstention | stale/foreign leaks | p50 ms | p95 ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| isolation | 4 | 100% | - | - | - | - | 100% | 0% | 0.6 | 0.7 |
| lexical | 10 | 100% | 100% | 100% | 1.000 | 1.000 | - | - | 0.8 | 1.5 |
| negative | 7 | 100% | - | - | - | - | 100% | - | 0.5 | 1.0 |
| paraphrase | 18 | 28% | 28% | 44% | 0.343 | 0.368 | - | - | 1.1 | 1.7 |
| temporal | 4 | 100% | 100% | 100% | 1.000 | 1.000 | 100% | 0% | 0.7 | 1.0 |
| **overall** | 43 | 70% | 58% | 68% | 0.618 | 0.633 | 100% | 0% | 0.9 | 1.6 |

95% bootstrap confidence intervals (overall): accuracy 56-84%, recall@1 39-77%, recall@5 52-84%, abstention 100-100%

**LoCoMo (test split)**: 4431 turns, 1152 questions, embedder `hashing-v1-512`, reranker `none`

| category | questions | recall@5 | recall@10 | hit@1 | hit@5 | p50 ms |
|---|---:|---:|---:|---:|---:|---:|
| multi-hop | 208 | 20.2% | 30.4% | 19.2% | 40.9% | 14.8 |
| open-domain | 73 | 26.7% | 32.5% | 13.7% | 37.0% | 15.2 |
| single-hop | 641 | 62.0% | 71.4% | 36.7% | 63.8% | 15.6 |
| temporal | 230 | 54.0% | 64.0% | 35.7% | 58.3% | 14.8 |
| **overall** | 1152 | 50.6% | 60.1% | 31.9% | 56.9% | 15.2 |

95% bootstrap CI (overall): recall@5 47.9%-53.3%, recall@10 57.3%-62.7%, hit@1 29.3%-34.5%, hit@5 54.0%-59.7%
