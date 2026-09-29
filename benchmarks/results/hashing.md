**mindtrail-retrieval v1**, embedder `hashing-v1-512`, k=5

| category | queries | recall@1 | recall@5 | MRR | nDCG@5 | abstention | stale/foreign leaks | p50 ms | p95 ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| isolation | 10 | 67% | 67% | 0.667 | 0.667 | 57% | 0% | 1.5 | 2.5 |
| lexical | 30 | 93% | 100% | 0.967 | 0.975 | - | - | 1.0 | 1.6 |
| negative | 12 | - | - | - | - | 75% | - | 0.9 | 2.0 |
| paraphrase | 30 | 23% | 27% | 0.250 | 0.254 | - | 0% | 0.9 | 2.0 |
| temporal | 10 | 90% | 90% | 0.900 | 0.900 | - | 0% | 1.3 | 1.9 |
| **overall** | 92 | 63% | 67% | 0.651 | 0.656 | 68% | 0% | 1.0 | 2.0 |

**mindtrail-retrieval-holdout v1**, embedder `hashing-v1-512`, k=5

| category | queries | recall@1 | recall@5 | MRR | nDCG@5 | abstention | stale/foreign leaks | p50 ms | p95 ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| isolation | 4 | - | - | - | - | 75% | 0% | 0.6 | 1.3 |
| lexical | 12 | 92% | 100% | 0.958 | 0.969 | - | - | 0.8 | 1.4 |
| negative | 6 | - | - | - | - | 83% | - | 0.6 | 0.9 |
| paraphrase | 13 | 31% | 46% | 0.385 | 0.405 | - | - | 0.8 | 1.4 |
| temporal | 4 | 100% | 100% | 1.000 | 1.000 | 100% | 0% | 0.7 | 1.0 |
| **overall** | 39 | 64% | 75% | 0.696 | 0.710 | 82% | 0% | 0.8 | 1.4 |

