# Six public classification benchmarks

Frozen paired cohort: 2,277 cases; case SHA-256 `046a38c24747fd724003aca7e5f1d1b6e148d599092222361c0717ecff84ecad`; client concurrency 8; every failure counts as wrong.

| Dataset | Cases | nyx INT4+FP8 | Qwen/Qwen3.5-35B-A3B | Jev 1.13 |
|---|---:|---:|---:|---:|
| AG News | 400 | 84.75% (339/400) | 85.75% (343/400) | 86.50% (346/400) |
| TREC coarse | 500 | 92.00% (460/500) | 94.00% (470/500) | 93.00% (465/500) |
| CoLA | 400 | 79.00% (316/400) | 79.75% (319/400) | 77.75% (311/400) |
| RTE | 277 | 93.14% (258/277) | 93.86% (260/277) | 91.34% (253/277) |
| Emotion | 300 | 43.33% (130/300) | 44.67% (134/300) | 47.00% (141/300) |
| IMDb | 400 | 96.50% (386/400) | 96.25% (385/400) | 97.00% (388/400) |

| Aggregate | nyx INT4+FP8 | Qwen/Qwen3.5-35B-A3B | Jev 1.13 |
|---|---:|---:|---:|
| Dataset-macro accuracy | 81.45% | 82.38% | 82.10% |
| Dataset-macro F1 | 81.06% | 82.02% | 81.85% |
| Pooled micro accuracy | 82.96% | 83.93% | 83.62% |

## Latency and throughput

| Measurement | nyx INT4+FP8 | Qwen/Qwen3.5-35B-A3B | Jev 1.13 |
|---|---:|---:|---:|
| Backend inference, median | 62.97 ms | 74.45 ms | not exposed |
| Backend inference, p95 | 85.30 ms | 97.38 ms | not exposed |
| API handling, median | 63.89 ms | 75.70 ms | not exposed |
| Round trip, mean | 70.79 ms | 83.49 ms | 188.13 ms |
| Round trip, median | 66.25 ms | 78.23 ms | 182.33 ms |
| Round trip, p95 | 89.22 ms | 101.82 ms | 245.12 ms |
| Round trip, p99 | 128.93 ms | 160.57 ms | 310.42 ms |
| Full-run wall time | 20,209.23 ms | 23,824.15 ms | 53,665.76 ms |
| Sustained throughput | 112.67 req/s | 95.58 req/s | 42.43 req/s |

Backend inference and API handling are parsed from the local `Server-Timing` response header. Round trip is client-observed request-to-body completion. Jev did not expose server timing during this run.

## Protocol

- Dataset selection, sampled examples, request order, and labels were frozen before any provider scoring.
- AG News, CoLA, Emotion, and IMDb are fixed class-stratified samples; TREC test and RTE validation are complete.
- Each HTTP request contained one Choice question. The three providers received identical JSON bodies in identical order over persistent HTTP/1.1 connections.
- Eight asynchronous client workers issued requests. The local SGLang runtime used continuous batching with `max-running-requests=64`; because client concurrency was eight, this run did not exercise batches above eight active requests.
- The retry policy allowed three attempts for retryable HTTP statuses. All three runs completed 2,277/2,277 with zero errors and zero retries.
- nyx and the original Qwen model ran sequentially on the same AMD Instinct MI325X and local API stack. Jev 1.13 ran over remote HTTPS on different, undisclosed infrastructure. Cross-service speed figures are descriptive, not a controlled kernel or hardware comparison.
- Public-benchmark pretraining contamination is unknown for all three models. This finite suite does not establish universal classification quality.

## Paired comparison

Exact two-sided McNemar tests use identical case-level predictions. Qwen-only versus nyx-only correct cases were 63 to 41 (`p=0.03896`); Jev-only versus nyx-only were 76 to 61 (`p=0.23154`); Qwen-only versus Jev-only were 66 to 59 (`p=0.59168`). None crosses the Bonferroni-adjusted threshold of `0.0167` for three comparisons.
