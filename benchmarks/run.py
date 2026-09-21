#!/usr/bin/env python3
"""Prepare, run, and compare the frozen nyx/Jev classification benchmark."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import math
import os
import random
import re
import statistics
import time
import urllib.request
from collections import defaultdict
from pathlib import Path
from typing import Any

import httpx


HERE = Path(__file__).resolve().parent
TASKS_PATH = HERE / "tasks.json"
DEFAULT_OUTPUT = HERE / "runs"
RETRYABLE = {429, 500, 502, 503, 504, 529}


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def task_config() -> dict[str, Any]:
    return json.loads(TASKS_PATH.read_text())


def state_for(dataset: str, item: dict[str, Any]) -> str:
    if dataset in {"ag_news", "emotion", "imdb"}:
        return str(item["text"])
    if dataset == "cola":
        return "Sentence: " + item["sentence"]
    if dataset == "rte":
        return "Premise:\n" + item["sentence1"] + "\n\nHypothesis:\n" + item["sentence2"]
    raise ValueError(f"Unknown dataset: {dataset}")


def load_huggingface_rows(dataset: str, spec: dict[str, Any]) -> list[dict[str, Any]]:
    from datasets import load_dataset

    loaded = load_dataset(
        spec["repo"],
        spec["config"],
        split=spec["split"],
        revision=spec["revision"],
    )
    labels = spec["labels"]
    rows = []
    for source_index, item in enumerate(loaded):
        raw_state = state_for(dataset, item)
        rows.append({
            "source_index": source_index,
            "state": raw_state[:16_000],
            "label": labels[int(item["label"])],
            "original_chars": len(raw_state),
            "truncated": len(raw_state) > 16_000,
        }
        )
    return rows


def load_trec_rows(spec: dict[str, Any]) -> list[dict[str, Any]]:
    with urllib.request.urlopen(spec["url"], timeout=30) as response:
        content = response.read()
    digest = sha256_bytes(content)
    if digest != spec["source_sha256"]:
        raise ValueError(f"TREC source hash changed: {digest}")
    label_map = {
        "ABBR": "abbreviation",
        "ENTY": "entity",
        "DESC": "description",
        "HUM": "human",
        "LOC": "location",
        "NUM": "numeric",
    }
    rows = []
    for source_index, raw in enumerate(content.splitlines()):
        decoded = raw.replace(b"\xf0", b" ").decode().strip()
        fine_label, separator, question = decoded.partition(" ")
        if not separator:
            raise ValueError(f"Malformed TREC row: {source_index}")
        rows.append(
            {
                "source_index": source_index,
                "state": question,
                "label": label_map[fine_label.split(":", 1)[0]],
                "original_chars": len(question),
                "truncated": False,
            }
        )
    return rows


def select_rows(
    dataset: str,
    rows: list[dict[str, Any]],
    labels: list[str],
    per_class: int | None,
    seed: int,
) -> list[dict[str, Any]]:
    if per_class is None:
        return rows
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["label"]].append(row)
    selected = []
    for label in labels:
        candidates = grouped[label]
        if len(candidates) < per_class:
            raise ValueError(f"{dataset}/{label}: expected {per_class}, found {len(candidates)}")
        random.Random(f"{seed}:{dataset}:{label}").shuffle(candidates)
        selected.extend(candidates[:per_class])
    random.Random(f"{seed}:{dataset}:order").shuffle(selected)
    return selected


def prepare(output: Path) -> None:
    config = task_config()
    seed = int(config["seed"])
    cases = []
    for dataset, spec in config["tasks"].items():
        rows = load_trec_rows(spec) if spec["source"] == "url" else load_huggingface_rows(dataset, spec)
        selected = select_rows(dataset, rows, spec["labels"], spec["per_class"], seed)
        for row in selected:
            identity = sha256_bytes(
                f'{dataset}\0{row["source_index"]}\0{row["state"]}\0{row["label"]}'.encode()
            )[:16]
            cases.append(
                {
                    "id": f'{dataset}-{row["source_index"]:05d}-{identity}',
                    "dataset": dataset,
                    "source_index": row["source_index"],
                    "state": row["state"],
                    "questions": {
                        "label": {
                            "type": "choice",
                            "instructions": spec["instructions"],
                            "criteria": spec["criteria"],
                        }
                    },
                    "label": row["label"],
                    "truncated": row["truncated"],
                    "original_chars": row["original_chars"],
                }
            )
    random.Random(f"{seed}:global-order").shuffle(cases)
    output.mkdir(parents=True, exist_ok=True)
    cases_path = output / "cases.jsonl"
    content = "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in cases)
    cases_path.write_text(content)
    digest = sha256(cases_path)
    if len(cases) != config["expected_cases"] or digest != config["expected_sha256"]:
        cases_path.unlink()
        raise ValueError(f"Frozen-set mismatch: cases={len(cases)}, sha256={digest}")
    print(json.dumps({"cases": len(cases), "path": str(cases_path), "sha256": digest}, indent=2))


def load_cases(output: Path) -> list[dict[str, Any]]:
    path = output / "cases.jsonl"
    if not path.exists():
        raise FileNotFoundError("Run the prepare command first")
    config = task_config()
    if sha256(path) != config["expected_sha256"]:
        raise ValueError("Case hash does not match tasks.json")
    with path.open() as stream:
        cases = [json.loads(line) for line in stream]
    if len(cases) != config["expected_cases"]:
        raise ValueError("Case count does not match tasks.json")
    return cases


def validate_answer(body: dict[str, Any], criteria: dict[str, str]) -> str:
    answer = body["answers"]["label"]
    if answer.get("type") != "choice":
        raise ValueError("Expected a Choice answer")
    prediction = answer["choice"]
    probabilities = answer["probabilities"]
    if prediction not in criteria or set(probabilities) != set(criteria):
        raise ValueError("Response choices do not match the request")
    values = list(probabilities.values())
    if not all(isinstance(value, (int, float)) and math.isfinite(value) and value >= 0 for value in values):
        raise ValueError("Invalid probabilities")
    if not math.isclose(sum(values), 1.0, abs_tol=0.011):
        raise ValueError("Probabilities do not sum to one")
    return prediction


async def run_provider(args: argparse.Namespace) -> None:
    if not re.fullmatch(r"[a-zA-Z0-9._-]+", args.name):
        raise ValueError("name may contain only letters, numbers, dot, underscore, and dash")
    cases = load_cases(args.output)
    key = os.environ.get(args.api_key_env, "")
    if not key:
        raise ValueError(f"Set {args.api_key_env}")
    queue: asyncio.Queue[tuple[int, dict[str, Any]]] = asyncio.Queue()
    for sequence, case in enumerate(cases):
        queue.put_nowait((sequence, case))
    records: list[dict[str, Any] | None] = [None] * len(cases)
    started = time.perf_counter()
    limits = httpx.Limits(max_connections=args.concurrency, max_keepalive_connections=args.concurrency)
    async with httpx.AsyncClient(timeout=args.timeout, limits=limits, trust_env=False) as client:

        async def worker() -> None:
            while True:
                try:
                    sequence, case = queue.get_nowait()
                except asyncio.QueueEmpty:
                    return
                payload = {"model": args.model, "state": case["state"], "questions": case["questions"]}
                attempts = []
                prediction = None
                error = None
                latency_ms = 0.0
                for attempt in range(1, 4):
                    tick = time.perf_counter()
                    try:
                        response = await client.post(
                            args.url,
                            headers={"Authorization": f"Bearer {key}"},
                            json=payload,
                        )
                        latency_ms += (time.perf_counter() - tick) * 1000
                        attempts.append(response.status_code)
                        if response.status_code == 200:
                            prediction = validate_answer(response.json(), case["questions"]["label"]["criteria"])
                            break
                        error = f"HTTP {response.status_code}: {response.text[:200]}"
                        if response.status_code not in RETRYABLE:
                            break
                    except Exception as exc:
                        latency_ms += (time.perf_counter() - tick) * 1000
                        attempts.append(None)
                        error = f"{type(exc).__name__}: {exc}"
                    await asyncio.sleep(0.5 * attempt)
                records[sequence] = {
                    "id": case["id"],
                    "dataset": case["dataset"],
                    "label": case["label"],
                    "prediction": prediction,
                    "correct": prediction == case["label"],
                    "latency_ms": latency_ms,
                    "attempts": attempts,
                    "error": error if prediction is None else None,
                }
                queue.task_done()

        await asyncio.gather(*(worker() for _ in range(args.concurrency)))
    wall_ms = (time.perf_counter() - started) * 1000
    final = [record for record in records if record is not None]
    result = summarize(args.name, args.url, args.model, args.concurrency, wall_ms, final)
    path = args.output / f"result-{args.name}.json"
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: value for key, value in result.items() if key != "records"}, indent=2))


def summarize(
    name: str,
    url: str,
    model: str,
    concurrency: int,
    wall_ms: float,
    records: list[dict[str, Any]],
) -> dict[str, Any]:
    per_dataset = {}
    for dataset in task_config()["tasks"]:
        selected = [row for row in records if row["dataset"] == dataset]
        correct = sum(row["correct"] for row in selected)
        per_dataset[dataset] = {
            "cases": len(selected),
            "correct": correct,
            "accuracy": correct / len(selected),
            "errors": sum(row["prediction"] is None for row in selected),
        }
    latencies = [row["latency_ms"] for row in records]
    return {
        "name": name,
        "url": url,
        "model": model,
        "concurrency": concurrency,
        "case_sha256": task_config()["expected_sha256"],
        "cases": len(records),
        "correct": sum(row["correct"] for row in records),
        "pooled_accuracy": sum(row["correct"] for row in records) / len(records),
        "dataset_macro_accuracy": statistics.fmean(row["accuracy"] for row in per_dataset.values()),
        "errors": sum(row["prediction"] is None for row in records),
        "wall_ms": wall_ms,
        "throughput_per_second": len(records) / (wall_ms / 1000),
        "latency_ms": {
            "median": statistics.median(latencies),
            "p95": percentile(latencies, 0.95),
        },
        "per_dataset": per_dataset,
        "records": records,
    }


def compare(output: Path, left: str, right: str) -> None:
    results = {}
    for name in (left, right):
        path = output / f"result-{name}.json"
        results[name] = json.loads(path.read_text())
        if results[name]["case_sha256"] != task_config()["expected_sha256"]:
            raise ValueError(f"{name}: case hash mismatch")
    left_ids = [row["id"] for row in results[left]["records"]]
    right_ids = [row["id"] for row in results[right]["records"]]
    if left_ids != right_ids:
        raise ValueError("Providers did not run the same ordered case set")
    lines = [
        f"| Benchmark | {left} | {right} |",
        "|---|---:|---:|",
    ]
    for dataset in task_config()["tasks"]:
        lines.append(
            f"| {dataset} | {results[left]['per_dataset'][dataset]['accuracy']:.2%} | "
            f"{results[right]['per_dataset'][dataset]['accuracy']:.2%} |"
        )
    lines.extend(
        [
            f"| Dataset-macro accuracy | {results[left]['dataset_macro_accuracy']:.2%} | "
            f"{results[right]['dataset_macro_accuracy']:.2%} |",
            f"| Pooled accuracy | {results[left]['pooled_accuracy']:.2%} | "
            f"{results[right]['pooled_accuracy']:.2%} |",
            f"| Median latency | {results[left]['latency_ms']['median']:.2f} ms | "
            f"{results[right]['latency_ms']['median']:.2f} ms |",
            f"| p95 latency | {results[left]['latency_ms']['p95']:.2f} ms | "
            f"{results[right]['latency_ms']['p95']:.2f} ms |",
            f"| Throughput | {results[left]['throughput_per_second']:.2f} req/s | "
            f"{results[right]['throughput_per_second']:.2f} req/s |",
        ]
    )
    paired = {
        "both_correct": 0,
        "left_only": 0,
        "right_only": 0,
        "both_wrong": 0,
    }
    for a, b in zip(results[left]["records"], results[right]["records"]):
        key = "both_correct" if a["correct"] and b["correct"] else "left_only" if a["correct"] else "right_only" if b["correct"] else "both_wrong"
        paired[key] += 1
    comparison = {
        "case_sha256": task_config()["expected_sha256"],
        "left": left,
        "right": right,
        "paired": paired,
        "markdown": "\n".join(lines),
    }
    (output / f"comparison-{left}-vs-{right}.json").write_text(
        json.dumps(comparison, indent=2, sort_keys=True) + "\n"
    )
    print(comparison["markdown"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("prepare")
    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("--name", required=True)
    run_parser.add_argument("--url", required=True)
    run_parser.add_argument("--model", required=True)
    run_parser.add_argument("--api-key-env", required=True)
    run_parser.add_argument("--concurrency", type=int, default=1)
    run_parser.add_argument("--timeout", type=float, default=120)
    compare_parser = subparsers.add_parser("compare")
    compare_parser.add_argument("--left", required=True)
    compare_parser.add_argument("--right", required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        prepare(args.output)
    elif args.command == "run":
        if not 1 <= args.concurrency <= 32:
            raise ValueError("concurrency must be between 1 and 32")
        asyncio.run(run_provider(args))
    else:
        compare(args.output, args.left, args.right)


if __name__ == "__main__":
    main()
