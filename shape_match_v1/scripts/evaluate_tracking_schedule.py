#!/usr/bin/env python3
"""Evaluate the predeclared three-pair scheduling gate; never tune it from results."""
import argparse
import csv
import json
import math
import statistics
from pathlib import Path


def percentile(values, fraction):
    return sorted(values)[math.ceil(len(values) * fraction) - 1]


def evaluate(root):
    runs = {}
    for kind in ("static", "dynamic"):
        for i in range(1, 4):
            name = f"{kind}-{i}"
            run = json.loads((root / f"{name}.json").read_text())
            if (run["iterations"], run["warmup"], run["threads"]) != (500, 20, 4):
                raise ValueError(f"{name}: unexpected benchmark configuration")
            with (root / f"{name}.json.csv").open() as stream:
                rows = list(csv.DictReader(stream))
            if len(rows) != 500 or len(run["samples_ms"]) != 500:
                raise ValueError(f"{name}: incomplete samples")
            run["tracking"] = {
                "median_ms": statistics.median(float(r["tracking_ms"]) for r in rows),
                "p95_ms": percentile([float(r["tracking_ms"]) for r in rows], .95),
                "p99_ms": percentile([float(r["tracking_ms"]) for r in rows], .99),
            }
            run["evaluated_poses"] = sorted({int(r["evaluated_poses"]) for r in rows})
            runs[name] = run
    median_ok = all(runs[f"dynamic-{i}"]["median_ms"] <= 1.02 * runs[f"static-{i}"]["median_ms"]
                    for i in range(1, 4))
    improved = sum(runs[f"dynamic-{i}"]["p95_ms"] <= .95 * runs[f"static-{i}"]["p95_ms"]
                   for i in range(1, 4))
    pooled = {kind: percentile([v for i in range(1, 4) for v in runs[f"{kind}-{i}"]["samples_ms"]], .95)
              for kind in ("static", "dynamic")}
    counts_ok = (len({tuple(r["evaluated_poses"]) for r in runs.values()}) == 1 and
                 all(len(r["evaluated_poses"]) == 1 for r in runs.values()))
    identical = ((root / "static-results/matching_results.csv").read_bytes() ==
                 (root / "dynamic-results/matching_results.csv").read_bytes())
    accepted = median_ok and improved >= 2 and pooled["dynamic"] <= .95 * pooled["static"]
    result = {"runs": runs, "csv_byte_identical": identical, "evaluations_identical": counts_ok,
              "gate": {"median_all_within_2_percent": median_ok,
                       "pairs_p95_improved_at_least_5_percent": improved,
                       "pooled_p95_ms": pooled,
                       "accept_dynamic": accepted and identical and counts_ok,
                       "dynamic_all_p95_le_20ms": all(runs[f"dynamic-{i}"]["p95_ms"] <= 20
                                                     for i in range(1, 4))}}
    trace_path = root / "trace-final.json.tracking.csv"
    if not trace_path.exists():
        trace_path = root / "trace.json.tracking.csv"
    if trace_path.exists():
        with trace_path.open() as stream:
            rows = list(csv.DictReader(stream))
        batches = {}
        for row in rows:
            batches.setdefault((row["frame"], row["batch"]), []).append(row)
        metrics = {"start_spread_ms": [], "finish_spread_ms": [], "max_cpu_over_mean": [],
                   "max_evaluations_over_mean": [], "wall_minus_cpu_ms": []}
        for batch in batches.values():
            if not sum(int(w["candidates"]) for w in batch):
                continue
            for key, column in (("start_spread_ms", "start_ms"), ("finish_spread_ms", "end_ms")):
                values = [float(w[column]) for w in batch]
                metrics[key].append(max(values) - min(values))
            for key, column in (("max_cpu_over_mean", "cpu_ms"),
                                ("max_evaluations_over_mean", "evaluations")):
                values = [float(w[column]) for w in batch]
                if min(values) >= 0 and sum(values) > 0:
                    metrics[key].append(max(values) / statistics.mean(values))
            for w in batch:
                if float(w["cpu_ms"]) >= 0:
                    metrics["wall_minus_cpu_ms"].append(
                        float(w["end_ms"]) - float(w["start_ms"]) - float(w["cpu_ms"]))
        result["diagnostic_only"] = {
            "batches": len(batches),
            "metrics": {k: {"median": statistics.median(v), "p95": percentile(v, .95)}
                        for k, v in metrics.items() if v}}
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    print(json.dumps(evaluate(args.directory), separators=(",", ":"), allow_nan=False))
