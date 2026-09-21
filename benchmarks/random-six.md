# Six public classification benchmarks

These results describe the 8.42 GB pre-quantization nyx reference, not the 3.29 GB release GGUF. The compact file was selected using the separate fidelity evaluation reported below.

Frozen paired cohort: 2,277 cases; case SHA-256 `046a38c24747fd724003aca7e5f1d1b6e148d599092222361c0717ecff84ecad`; every failure counts as wrong.

| Dataset | Cases | Qwen3.5-4B base | nyx | Jev 1.13 |
|---|---:|---:|---:|---:|
| AG News | 400 | 87.75% (351/400) | 85.75% (343/400) | 86.50% (346/400) |
| TREC coarse | 500 | 86.60% (433/500) | 84.20% (421/500) | 93.00% (465/500) |
| CoLA | 400 | 77.75% (311/400) | 81.50% (326/400) | 77.75% (311/400) |
| RTE | 277 | 84.12% (233/277) | 88.09% (244/277) | 91.34% (253/277) |
| Emotion | 300 | 47.67% (143/300) | 46.33% (139/300) | 47.00% (141/300) |
| IMDb | 400 | 95.75% (383/400) | 95.75% (383/400) | 97.00% (388/400) |
| Dataset-macro accuracy | 2,277 | 79.94% | 80.27% | 82.10% |
| Dataset-macro F1 | 2,277 | 79.83% | 80.48% | 81.85% |
| Pooled micro accuracy | 2,277 | 81.42% (1,854/2,277) | 81.51% (1,856/2,277) | 83.62% (1,904/2,277) |

Jev-only versus nyx-only correct cases were 121 to 73; the exact two-sided McNemar p-value is `0.0006995`. nyx-only versus Qwen-only correct cases were 112 to 110 (`p=0.9465`).

## Runtime

The release profile uses one llama.cpp execution slot because it was the fastest measured configuration for this latency-oriented workload.

| Measurement | Result |
|---|---:|
| Reference model file | 8,424,393,056 bytes |
| Measured VRAM | 9,927,208,960 bytes |
| Round trip, median | 55.56 ms |
| Round trip, p95 | 72.77 ms |
| Round trip, mean | 57.92 ms |
| Full-run wall time | 132,780.73 ms |
| Sustained throughput | 17.15 requests/s |

The nyx timing is direct loopback HTTP to llama.cpp on one AMD Instinct MI325X, with full GPU offload, Flash Attention, Q8 KV cache, 4,096 tokens per slot, and client/server concurrency 1. It excludes the authenticated gateway and external network. The Qwen base used the identical local profile and measured 54.85 ms median, 67.96 ms p95, and 17.44 requests/s.

The previously observed Jev remote-HTTPS run used client concurrency 8 and reported 182.33 ms median, 245.12 ms p95, and 42.43 requests/s. Different hardware, network topology, and concurrency make that timing descriptive rather than a controlled speed comparison.

## Batching check

The first 120 frozen cases were repeated at three concurrency levels. The server had the same number of 4,096-token slots as the client concurrency.

| Concurrency | Correct | Median / p95 | Throughput |
|---:|---:|---:|---:|
| 1 | 94/120 | 57.20 / 89.01 ms | 15.79 requests/s |
| 2 | 94/120 | 162.80 / 219.25 ms | 11.77 requests/s |
| 8 | 94/120 | 606.22 / 887.68 ms | 11.61 requests/s |

All 120 class decisions were identical across the three runs. Batching was rejected because it reduced throughput and increased latency.

## Protocol and limits

- Dataset selection, sampled examples, request order, and labels were frozen before provider scoring.
- AG News, CoLA, Emotion, and IMDb are fixed class-stratified samples; TREC test and RTE validation are complete.
- Each request contained one Choice question. Both systems completed 2,277/2,277 cases with zero errors.
- Public-benchmark pretraining overlap is unknown. This finite suite does not establish universal classification quality.
- The NVIDIA RTX 4000 Ada target still requires an on-device latency and class-decision parity run before production rollout.

## Compact release fidelity

The released importance-matrix IQ4_XS candidate was evaluated against the reference on a frozen 1,024-case selection spanning Banking77, BoolQ, CLINC150, code-defect annotation, commit type, DBpedia14, email spam, SNLI, structured rules, and tweet sentiment.

| Metric | Reference | Released GGUF |
|---|---:|---:|
| File size | 8,424,393,056 bytes | 3,288,582,784 bytes |
| Correct | 873/1,024 | 874/1,024 |
| Accuracy | 85.25% | 85.35% |
| Decision agreement | — | 98.73% (1,011/1,024) |
| Decision disagreements | — | 13 |

The released candidate gained seven decisions and lost six relative to the reference. This is fidelity evidence, not a fresh public-benchmark run. The candidate did not meet the experiment's original 99.5% agreement target and was released only after that threshold was explicitly waived.
