**mindtrail-retrieval v1**, embedder `fastembed:BAAI/bge-small-en-v1.5`, reranker `none`, k=5

| category | queries | accuracy | recall@1 | recall@5 | MRR | nDCG@5 | abstention | stale/foreign leaks | p50 ms | p95 ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| isolation | 10 | 100% | 100% | 100% | 1.000 | 1.000 | 100% | 0% | 115.0 | 183.0 |
| lexical | 30 | 93% | 93% | 100% | 0.967 | 0.975 | - | - | 113.5 | 244.5 |
| negative | 12 | 100% | - | - | - | - | 100% | - | 133.3 | 263.7 |
| paraphrase | 30 | 50% | 50% | 63% | 0.550 | 0.571 | - | 0% | 105.9 | 238.4 |
| temporal | 10 | 90% | 90% | 90% | 0.900 | 0.900 | - | 0% | 113.5 | 189.8 |
| **overall** | 92 | 80% | 75% | 84% | 0.788 | 0.800 | 100% | 0% | 116.7 | 238.4 |

95% bootstrap confidence intervals (overall): accuracy 72-88%, recall@1 64-85%, recall@5 74-92%, abstention 100-100%

**mindtrail-retrieval-holdout v1**, embedder `fastembed:BAAI/bge-small-en-v1.5`, reranker `none`, k=5

| category | queries | accuracy | recall@1 | recall@5 | MRR | nDCG@5 | abstention | stale/foreign leaks | p50 ms | p95 ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| isolation | 4 | 100% | - | - | - | - | 100% | 0% | 78.3 | 172.6 |
| lexical | 12 | 92% | 92% | 100% | 0.958 | 0.969 | - | - | 100.0 | 184.0 |
| negative | 6 | 100% | - | - | - | - | 100% | - | 96.1 | 256.1 |
| paraphrase | 13 | 54% | 54% | 62% | 0.577 | 0.587 | - | - | 91.5 | 198.6 |
| temporal | 4 | 100% | 100% | 100% | 1.000 | 1.000 | 100% | 0% | 75.7 | 170.3 |
| **overall** | 39 | 82% | 75% | 82% | 0.786 | 0.795 | 100% | 0% | 100.0 | 251.6 |

95% bootstrap confidence intervals (overall): accuracy 69-95%, recall@1 57-89%, recall@5 68-96%, abstention 100-100%

**mindtrail-retrieval-holdout v2**, embedder `fastembed:BAAI/bge-small-en-v1.5`, reranker `none`, k=5

| category | queries | accuracy | recall@1 | recall@5 | MRR | nDCG@5 | abstention | stale/foreign leaks | p50 ms | p95 ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| isolation | 4 | 100% | - | - | - | - | 100% | 0% | 72.1 | 261.2 |
| lexical | 10 | 100% | 100% | 100% | 1.000 | 1.000 | - | - | 65.4 | 221.9 |
| negative | 7 | 100% | - | - | - | - | 100% | - | 96.6 | 167.7 |
| paraphrase | 18 | 67% | 67% | 83% | 0.750 | 0.772 | - | - | 111.3 | 232.1 |
| temporal | 4 | 100% | 100% | 100% | 1.000 | 1.000 | 100% | 0% | 67.4 | 480.5 |
| **overall** | 43 | 86% | 81% | 90% | 0.855 | 0.868 | 100% | 0% | 105.0 | 232.1 |

95% bootstrap confidence intervals (overall): accuracy 74-95%, recall@1 65-94%, recall@5 81-100%, abstention 100-100%

**LoCoMo (test split)**: 4431 turns, 1152 questions, embedder `fastembed:BAAI/bge-small-en-v1.5`, reranker `none`

| category | questions | recall@5 | recall@10 | hit@1 | hit@5 | p50 ms |
|---|---:|---:|---:|---:|---:|---:|
| multi-hop | 208 | 30.4% | 39.5% | 26.4% | 60.1% | 34.3 |
| open-domain | 73 | 26.5% | 35.9% | 16.4% | 39.7% | 35.1 |
| single-hop | 641 | 66.0% | 75.1% | 37.8% | 67.6% | 32.1 |
| temporal | 230 | 58.2% | 66.3% | 40.4% | 61.7% | 32.5 |
| **overall** | 1152 | 55.5% | 64.4% | 34.9% | 63.3% | 32.4 |

95% bootstrap CI (overall): recall@5 52.8%-58.1%, recall@10 61.8%-66.9%, hit@1 32.0%-37.6%, hit@5 60.5%-66.0%
