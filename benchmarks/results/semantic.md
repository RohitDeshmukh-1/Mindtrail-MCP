**mindtrail-retrieval v1**, embedder `fastembed:BAAI/bge-base-en-v1.5`, reranker `none`, k=5

| category | queries | accuracy | recall@1 | recall@5 | MRR | nDCG@5 | abstention | stale/foreign leaks | p50 ms | p95 ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| isolation | 10 | 90% | 100% | 100% | 1.000 | 1.000 | 86% | 0% | 27.4 | 33.5 |
| lexical | 30 | 93% | 93% | 100% | 0.967 | 0.975 | - | - | 31.1 | 35.6 |
| negative | 12 | 100% | - | - | - | - | 100% | - | 32.5 | 36.1 |
| paraphrase | 30 | 70% | 70% | 90% | 0.794 | 0.822 | - | 0% | 32.1 | 36.4 |
| temporal | 10 | 100% | 100% | 100% | 1.000 | 1.000 | - | 0% | 30.8 | 34.0 |
| **overall** | 92 | 87% | 85% | 96% | 0.902 | 0.917 | 95% | 0% | 31.1 | 36.1 |

95% bootstrap confidence intervals (overall): accuracy 79-93%, recall@1 77-93%, recall@5 90-100%, abstention 84-100%


**mindtrail-retrieval-holdout v1**, embedder `fastembed:BAAI/bge-base-en-v1.5`, reranker `none`, k=5

| category | queries | accuracy | recall@1 | recall@5 | MRR | nDCG@5 | abstention | stale/foreign leaks | p50 ms | p95 ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| isolation | 4 | 100% | - | - | - | - | 100% | 0% | 29.1 | 35.9 |
| lexical | 12 | 92% | 92% | 100% | 0.958 | 0.969 | - | - | 29.9 | 36.1 |
| negative | 6 | 100% | - | - | - | - | 100% | - | 32.0 | 33.9 |
| paraphrase | 13 | 69% | 69% | 85% | 0.769 | 0.789 | - | - | 33.6 | 35.9 |
| temporal | 4 | 100% | 100% | 100% | 1.000 | 1.000 | 100% | 0% | 34.6 | 38.0 |
| **overall** | 39 | 87% | 82% | 93% | 0.875 | 0.889 | 100% | 0% | 32.0 | 36.1 |

95% bootstrap confidence intervals (overall): accuracy 77-97%, recall@1 68-96%, recall@5 82-100%, abstention 100-100%


**mindtrail-retrieval-holdout v2**, embedder `fastembed:BAAI/bge-base-en-v1.5`, reranker `none`, k=5

| category | queries | accuracy | recall@1 | recall@5 | MRR | nDCG@5 | abstention | stale/foreign leaks | p50 ms | p95 ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| isolation | 4 | 100% | - | - | - | - | 100% | 0% | 29.2 | 32.9 |
| lexical | 10 | 100% | 100% | 100% | 1.000 | 1.000 | - | - | 31.0 | 53.0 |
| negative | 7 | 100% | - | - | - | - | 100% | - | 28.7 | 33.1 |
| paraphrase | 18 | 83% | 83% | 89% | 0.861 | 0.868 | - | - | 33.7 | 47.1 |
| temporal | 4 | 100% | 100% | 100% | 1.000 | 1.000 | 100% | 0% | 26.9 | 29.6 |
| **overall** | 43 | 93% | 90% | 94% | 0.919 | 0.924 | 100% | 0% | 31.0 | 42.2 |

95% bootstrap confidence intervals (overall): accuracy 84-100%, recall@1 77-100%, recall@5 84-100%, abstention 100-100%


**LoCoMo (test split)**: 4431 turns, 1152 questions, embedder `fastembed:BAAI/bge-base-en-v1.5`, reranker `none`

| category | questions | recall@5 | recall@10 | hit@1 | hit@5 | p50 ms |
|---|---:|---:|---:|---:|---:|---:|
| multi-hop | 208 | 30.8% | 38.2% | 28.8% | 58.7% | 49.0 |
| open-domain | 73 | 25.5% | 32.6% | 15.1% | 37.0% | 50.6 |
| single-hop | 641 | 65.8% | 75.4% | 40.9% | 67.7% | 52.3 |
| temporal | 230 | 59.9% | 65.0% | 40.4% | 63.9% | 50.5 |
| **overall** | 1152 | 55.7% | 63.9% | 37.0% | 63.4% | 51.2 |

95% bootstrap CI (overall): recall@5 53.0%-58.4%, recall@10 61.3%-66.5%, hit@1 34.1%-39.9%, hit@5 60.7%-66.1%

