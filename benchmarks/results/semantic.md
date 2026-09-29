**mindtrail-retrieval v1**, embedder `fastembed:BAAI/bge-small-en-v1.5`, k=5

| category | queries | recall@1 | recall@5 | MRR | nDCG@5 | abstention | stale/foreign leaks | p50 ms | p95 ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| isolation | 10 | 100% | 100% | 1.000 | 1.000 | 71% | 0% | 9.2 | 9.8 |
| lexical | 30 | 90% | 100% | 0.950 | 0.963 | - | - | 10.0 | 11.1 |
| negative | 12 | - | - | - | - | 100% | - | 10.8 | 12.8 |
| paraphrase | 30 | 47% | 73% | 0.572 | 0.613 | - | 0% | 10.7 | 12.8 |
| temporal | 10 | 100% | 100% | 1.000 | 1.000 | - | 0% | 10.2 | 14.2 |
| **overall** | 92 | 74% | 89% | 0.804 | 0.826 | 89% | 0% | 10.3 | 12.8 |

**mindtrail-retrieval-holdout v1**, embedder `fastembed:BAAI/bge-small-en-v1.5`, k=5

| category | queries | recall@1 | recall@5 | MRR | nDCG@5 | abstention | stale/foreign leaks | p50 ms | p95 ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| isolation | 4 | - | - | - | - | 75% | 0% | 9.1 | 9.9 |
| lexical | 12 | 92% | 100% | 0.958 | 0.969 | - | - | 9.4 | 11.8 |
| negative | 6 | - | - | - | - | 100% | - | 9.1 | 9.8 |
| paraphrase | 13 | 62% | 77% | 0.692 | 0.712 | - | - | 9.7 | 10.6 |
| temporal | 4 | 100% | 100% | 1.000 | 1.000 | 100% | 0% | 9.3 | 14.4 |
| **overall** | 39 | 79% | 89% | 0.839 | 0.853 | 91% | 0% | 9.4 | 11.8 |

