# Pre-quantization H1 comparison

This record isolates the exact intersection shared by three completed evaluations: the official `Qwen/Qwen3.5-35B-A3B` base in BF16, the pre-quantization `nyx` BF16 checkpoint, and Jev 1.13.0.

## Results

| Task | Cases | Qwen base BF16 | `nyx` BF16 | Jev 1.13.0 |
|---|---:|---:|---:|---:|
| Comment triage | 500 | 295 (59.00%) | 462 (92.40%) | 455 (91.00%) |
| Report completeness | 500 | 439 (87.80%) | 488 (97.60%) | 468 (93.60%) |
| Broad assets, exact set | 500 | 285 (57.00%) | 298 (59.60%) | 284 (56.80%) |
| **Overall** | **1,500** | **1,019 (67.93%)** | **1,248 (83.20%)** | **1,207 (80.47%)** |

## Protocol

- The intersection contains comment fixtures 501–1000 plus all 500 authored report fixtures, scored once for completeness and once for broad-asset exact-set classification.
- Case IDs and expected labels were matched across the three result sets. All 1,500 expected labels matched exactly; no case was excluded after matching.
- Qwen and `nyx` used BF16 before any FP8 or INT4 conversion, the same AMD Instinct MI325X, the same SGLang/API stack, and server concurrency 8.
- The base was the official `Qwen/Qwen3.5-35B-A3B` revision `59d61f3ce65a6d9863b86d2e96597125219dc754`.
- Jev was evaluated as `jev-1.13.0` through its remote API with the same typed requests, prompts, thresholds, and reference labels. Because the serving infrastructure differs, this record makes no speed comparison.

## Scope

These are synthetic, authored H1 development fixtures with known scenario-family overlap. They are useful for measuring task adaptation on this workload, but they are not an independent public holdout and do not establish general model superiority.
