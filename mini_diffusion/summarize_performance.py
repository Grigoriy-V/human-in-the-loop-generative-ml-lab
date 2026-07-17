from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path

import yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", default="reports/performance_experiments.jsonl")
    parser.add_argument("--decisions", default="reports/performance_decisions.yaml")
    parser.add_argument("--csv", default="reports/performance_experiments.csv")
    parser.add_argument("--final-output", default="reports/cifar10_optimized_benchmark.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records = [
        json.loads(line)
        for line in Path(args.log).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    decisions = yaml.safe_load(Path(args.decisions).read_text(encoding="utf-8"))
    columns = [
        "experiment_id",
        "timestamp",
        "git_commit",
        "overrides",
        "batch_size",
        "run_index",
        "iterations_per_second",
        "images_per_second",
        "wall_step_seconds_mean",
        "step_seconds_median",
        "step_seconds_p95",
        "data_seconds_mean",
        "transfer_seconds_mean",
        "optimizer_seconds_mean",
        "ema_seconds_mean",
        "cuda_peak_allocated_gb",
        "cuda_peak_reserved_gb",
        "gpu_utilization_mean",
        "finite_loss",
        "result",
        "keep_reject",
        "notes",
    ]
    csv_path = Path(args.csv)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for record in records:
            decision = decisions.get(record["experiment_id"], {})
            row = {column: record.get(column) for column in columns}
            row["overrides"] = ";".join(record.get("overrides", []))
            row["keep_reject"] = decision.get("decision", "pending")
            row["notes"] = decision.get("notes", record.get("notes", ""))
            writer.writerow(row)

    grouped = defaultdict(list)
    for record in records:
        grouped[record["experiment_id"]].append(record)
    final_runs = [
        record
        for record in grouped["FINAL_OPTIMIZED"]
        if record.get("result") == "measured"
    ]
    images = [record["images_per_second"] for record in final_runs]
    iterations = [record["iterations_per_second"] for record in final_runs]
    steps = [record["wall_step_seconds_mean"] for record in final_runs]
    historical = 1787.5728452494807
    a0 = statistics.median(record["images_per_second"] for record in grouped["A0"])
    summary = {
        "experiment_id": "FINAL_OPTIMIZED",
        "runs": len(final_runs),
        "warmup_steps_per_run": final_runs[0]["warmup_steps"],
        "measured_steps_per_run": final_runs[0]["measured_steps"],
        "images_per_second": {
            "median": statistics.median(images),
            "mean": statistics.fmean(images),
            "min": min(images),
            "max": max(images),
        },
        "iterations_per_second_median": statistics.median(iterations),
        "wall_step_seconds": {
            "median": statistics.median(steps),
            "mean": statistics.fmean(steps),
        },
        "gpu_step_seconds_median": statistics.median(
            record["step_seconds_median"] for record in final_runs
        ),
        "gpu_step_seconds_p95_median": statistics.median(
            record["step_seconds_p95"] for record in final_runs
        ),
        "data_seconds_mean_median": statistics.median(
            record["data_seconds_mean"] for record in final_runs
        ),
        "transfer_seconds_mean_median": statistics.median(
            record["transfer_seconds_mean"] for record in final_runs
        ),
        "gpu_utilization_mean_median": statistics.median(
            record["gpu_utilization_mean"] for record in final_runs
        ),
        "historical_baseline_images_per_second": historical,
        "repeated_a0_images_per_second_median": a0,
        "improvement_vs_historical_percent": (statistics.median(images) / historical - 1) * 100,
        "improvement_vs_a0_percent": (statistics.median(images) / a0 - 1) * 100,
        "batch_size": final_runs[0]["batch_size"],
        "effective_batch_size": final_runs[0]["effective_batch_size"],
        "dtype": final_runs[0]["dtype"],
        "device_name": final_runs[0]["device_name"],
        "trainable_parameters": final_runs[0]["trainable_parameters"],
        "cuda_peak_allocated_gb": statistics.median(
            record["cuda_peak_allocated_gb"] for record in final_runs
        ),
        "cuda_peak_reserved_gb": statistics.median(
            record["cuda_peak_reserved_gb"] for record in final_runs
        ),
        "finite_loss_all_runs": all(record["finite_loss"] for record in final_runs),
        "source_log": args.log,
    }
    output = Path(args.final_output)
    output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"csv_written: {csv_path}")
    print(f"summary_written: {output}")


if __name__ == "__main__":
    main()
