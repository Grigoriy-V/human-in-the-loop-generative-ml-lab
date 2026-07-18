from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
import yaml
from PIL import Image, ImageDraw
from torchvision.utils import save_image

from mini_diffusion.evaluate_afhq_cat import (
    duplicate_diagnostics,
    generated_images,
    inception,
    load_config_from_checkpoint,
    reference_images,
    write_result,
)
from mini_diffusion.evaluator import _batched_features, fid, kid, pixel_diagnostics, precision_recall, sha256


def load_protocol(path: str | Path) -> dict[str, Any]:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def config_sha256(path: str | Path) -> str:
    return sha256(path)


def finite_state(state: dict[str, torch.Tensor]) -> bool:
    return all(torch.isfinite(value).all() for value in state.values() if isinstance(value, torch.Tensor))


def inspect_variant(variant: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any], str]:
    checkpoint_path = Path(variant["checkpoint"])
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Missing checkpoint for {variant['id']}: {checkpoint_path}")
    checkpoint, model_cfg = load_config_from_checkpoint(checkpoint_path)
    if int(checkpoint["global_step"]) != int(variant["expected_step"]):
        raise ValueError(f"{variant['id']} has step {checkpoint['global_step']}, expected {variant['expected_step']}")
    if variant["weights"] not in {"raw", "ema"}:
        raise ValueError(f"Unsupported weights for {variant['id']}: {variant['weights']}")
    if not finite_state(checkpoint["model"]) or not finite_state(checkpoint["ema"]["shadow"]):
        raise ValueError(f"{variant['id']} contains non-finite raw or EMA weights")
    if variant["experiment"] == "repa" and "repa_projector" not in checkpoint:
        raise ValueError(f"{variant['id']} is expected to be a REPA checkpoint")
    if variant["experiment"] == "baseline" and "repa_projector" in checkpoint:
        raise ValueError(f"{variant['id']} is expected to be a baseline checkpoint")
    return checkpoint, model_cfg, sha256(checkpoint_path)


def compatible(reference: dict[str, Any], candidate: dict[str, Any]) -> bool:
    paths = (
        ("data", "dataset"), ("data", "resolution"), ("data", "latent_resolution"), ("data", "num_classes"),
        ("vae", "model_id"), ("model", "patch_size"), ("model", "hidden_size"), ("model", "depth"),
        ("model", "num_heads"), ("model", "mlp_ratio"),
    )
    return all(reference[first][second] == candidate[first][second] for first, second in paths)


def labeled_grid(images: torch.Tensor, output: Path, title: str, *, nrow: int = 10) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".body.png")
    save_image(images, temporary, nrow=nrow)
    with Image.open(temporary).convert("RGB") as grid:
        header = 26
        canvas = Image.new("RGB", (grid.width, grid.height + header), "white")
        canvas.paste(grid, (0, header))
        ImageDraw.Draw(canvas).text((6, 7), title, fill="black")
        canvas.save(output)
    temporary.unlink()


def paired_grid(left: torch.Tensor, right: torch.Tensor, output: Path, title: str) -> None:
    if left.shape != right.shape:
        raise ValueError("Comparison images must have identical shapes")
    pairs = torch.stack([image for index in range(len(left)) for image in (left[index], right[index])])
    labeled_grid(pairs, output, title, nrow=10)


def score(images: torch.Tensor, reference_features: torch.Tensor, feature_model, preprocess, device: torch.device, threshold: float) -> tuple[torch.Tensor, dict[str, Any]]:
    generated_features = _batched_features(feature_model, preprocess, images, device, 64)
    precision, recall = precision_recall(reference_features, generated_features)
    metrics = {
        "fid": fid(reference_features, generated_features, device),
        "kid": kid(reference_features, generated_features),
        "precision": precision,
        "recall": recall,
        "pixel": pixel_diagnostics(images),
        **duplicate_diagnostics(generated_features, threshold),
    }
    return generated_features, metrics


def failure_count(metrics: dict[str, Any]) -> int:
    pixel = metrics["pixel"]
    return int(pixel["finite_failures"] + pixel["black_white_failures"])


def delta(left: dict[str, Any], right: dict[str, Any]) -> dict[str, dict[str, float | None]]:
    result: dict[str, dict[str, float | None]] = {}
    for field in ("fid", "kid", "precision", "recall"):
        before, after = float(left["metrics"][field]), float(right["metrics"][field])
        result[field] = {"absolute": after - before, "percent": None if before == 0 else (after - before) / before * 100}
    return result


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    fields = ["variant", "experiment", "step", "weights", "fid", "kid", "precision", "recall", "failures", "duplicates", "low_detail_warnings", "sampling_seconds", "images_per_second", "peak_allocated_gb", "peak_reserved_gb", "checkpoint_sha256", "grid_sha256"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)


def write_markdown(results: dict[str, Any], output: Path) -> None:
    rows = results["rows"]
    lines = [
        "# AFHQ Cats Baseline vs REPA Quick Comparison", "",
        "Fixed protocol: held-out official AFHQ Cats test split, 200 images, seeds 1000-1199, class 0, Heun-50, CFG 1.0, shared VAE and Inception-v3 features.", "",
        "| Variant | Step | Weights | FID | KID | Precision | Recall | Failures | Duplicates | img/s | Peak VRAM (GB) |",
        "| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(f"| {row['variant']} | {row['step']} | {row['weights']} | {row['fid']:.3f} | {row['kid']:.5f} | {row['precision']:.3f} | {row['recall']:.3f} | {row['failures']} | {row['duplicates']} | {row['images_per_second']:.2f} | {row['peak_allocated_gb']:.2f} |")
    lines.extend(["", "## Matched Changes", ""])
    for comparison_id, values in results["changes"].items():
        lines.append(f"### {comparison_id}")
        for metric, change in values.items():
            percent = "n/a" if change["percent"] is None else f"{change['percent']:+.2f}%"
            lines.append(f"- {metric}: {change['absolute']:+.6f} ({percent})")
        lines.append("")
    lines.extend([
        "## Interpretation", "",
        "Loss values are intentionally excluded: baseline and REPA optimize different objectives. Read matched checkpoint comparisons together with FID, KID, precision, recall, failures, duplicate counts, and paired fixed-seed grids; do not select a winner from FID alone.", "",
        "- Baseline dynamics: compare `baseline_10k_vs_20k`.",
        "- REPA dynamics and current raw checkpoint: compare `repa_10k_vs_20k`.",
        "- Equal-budget REPA evidence: compare `baseline_vs_repa_10k` and `baseline_vs_repa_20k` only.",
        "- EMA is diagnostic only: compare `repa_raw_vs_ema_20k`.", "",
        f"All checkpoint SHA-256 values were unchanged after evaluation: `{results['checkpoint_unchanged']}`. Repeated fixed-seed one-image probes were bitwise identical for every variant: `{results['deterministic_sampling']}`.",
        "Full-1000 evaluation, sampler ablation, CFG sweep, and training were not run.",
    ])
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def append_ledger(config_path: Path, config_hash: str, output: Path, results: dict[str, Any]) -> str:
    base_event_id = "afhq-cats-baseline-vs-repa-quick-10k-20k-20260718"
    ledger = ROOT / "reports" / "experiment_ledger.jsonl"
    existing_ids = {json.loads(line)["event_id"] for line in ledger.read_text(encoding="utf-8").splitlines() if line}
    suffix = 1
    event_id = base_event_id
    while event_id in existing_ids:
        suffix += 1
        event_id = f"{base_event_id}-{suffix}"
    event = {
        "schema_version": "1.0", "event_id": event_id, "timestamp_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "experiment_id": "afhq-cats-baseline-vs-repa", "event_type": "evaluation", "status": "completed",
        "git_commit": results["git_commit"], "config_path": str(config_path).replace("\\", "/"), "config_sha256": config_hash,
        "dataset_fingerprint": results["dataset_fingerprint"], "checkpoint_path": None, "checkpoint_step": None, "checkpoint_sha256": None,
        "exact_command": f".\\.venv\\Scripts\\python.exe mini_diffusion\\evaluate_comparison.py --config {str(config_path).replace('/', '\\\\')}",
        "runtime": {"device": results["device"], "gpu": results["gpu"], "dtype": "bf16", "batch": results["protocol"]["sample_batch_size"], "effective_batch": results["protocol"]["sample_batch_size"], "duration_seconds": results["duration_seconds"]},
        "metrics": {"protocol": results["protocol"], "variants": {key: value["metrics"] for key, value in results["variants"].items()}, "changes": results["changes"]},
        "artifacts": {"report": str(output / "report.md").replace("\\", "/"), "comparison": str(output / "comparison.json").replace("\\", "/"), "metrics_csv": str(output / "metrics.csv").replace("\\", "/")},
        "decision": "continue", "decision_reason": "The unified fixed-protocol quick comparison completed; use its matched metrics and paired grids to decide the next experiment phase.",
        "notes": "No training, full-1000 evaluation, sampler ablation, or CFG sweep was run.",
    }
    with ledger.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(event, sort_keys=True) + "\n")
    return event_id


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the fixed AFHQ Cats baseline vs REPA quick comparison protocol.")
    parser.add_argument("--config", required=True)
    args = parser.parse_args(); config_path = Path(args.config); config = load_protocol(config_path)
    protocol, output = config["protocol"], Path(config["output_dir"])
    if protocol["samples"] != 200 or protocol["sampler"] != "heun" or protocol["steps"] != 50 or float(protocol["guidance_scale"]) != 1.0 or int(protocol["class_id"]) != 0:
        raise ValueError("This comparison CLI supports only the canonical AFHQ quick-200 Heun-50 CFG-1.0 protocol")
    inspected = {variant["id"]: (*inspect_variant(variant), variant) for variant in config["variants"]}
    reference_cfg = inspected["baseline_raw_10k"][1]
    if any(not compatible(reference_cfg, item[1]) for item in inspected.values()):
        raise ValueError("Checkpoint architectures, resolution, latent shape, VAE, or class count are incompatible")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    feature_model, preprocess = inception(device)
    test_images, test_paths = reference_images(reference_cfg, "test")
    if any("train/" in path.replace("\\", "/") for path in test_paths):
        raise ValueError("Held-out test reference set contains training paths")
    reference_features = _batched_features(feature_model, preprocess, test_images, device, 64)
    started = time.perf_counter(); output.mkdir(parents=True, exist_ok=True)
    variants: dict[str, Any] = {}; rows: list[dict[str, Any]] = []
    for variant_id, (checkpoint, model_cfg, digest, variant) in inspected.items():
        checkpoint_path = Path(variant["checkpoint"])
        if device.type == "cuda": torch.cuda.reset_peak_memory_stats(device)
        sampling_started = time.perf_counter()
        images = generated_images(checkpoint, model_cfg, weights=variant["weights"], samples=int(protocol["samples"]), seed_start=int(protocol["seed_start"]), steps=int(protocol["steps"]), guidance_scale=float(protocol["guidance_scale"]), device=device, batch_size=int(protocol["sample_batch_size"]))
        sampling_seconds = time.perf_counter() - sampling_started
        if not torch.isfinite(images).all() or tuple(images.shape) != (int(protocol["samples"]), 3, 128, 128):
            raise RuntimeError(f"Invalid sampled images for {variant_id}")
        deterministic_probe = generated_images(checkpoint, model_cfg, weights=variant["weights"], samples=1, seed_start=int(protocol["seed_start"]), steps=int(protocol["steps"]), guidance_scale=float(protocol["guidance_scale"]), device=device, batch_size=1)
        deterministic_repeat = generated_images(checkpoint, model_cfg, weights=variant["weights"], samples=1, seed_start=int(protocol["seed_start"]), steps=int(protocol["steps"]), guidance_scale=float(protocol["guidance_scale"]), device=device, batch_size=1)
        if not torch.equal(deterministic_probe, deterministic_repeat):
            raise RuntimeError(f"Fixed-seed sampling is not deterministic for {variant_id}")
        features, metrics = score(images, reference_features, feature_model, preprocess, device, float(protocol["duplicate_threshold"]))
        variant_output = output / "variants" / variant_id
        metadata = {"mode": "comparison_sampling", "variant": variant_id, "experiment": variant["experiment"], "weights": variant["weights"], "sampler": "heun", "steps": protocol["steps"], "guidance_scale": protocol["guidance_scale"], "seed_start": protocol["seed_start"], "samples": len(images), "test_split_paths": test_paths, "sampling_seconds": sampling_seconds, "device": str(device), "vae_model_id": model_cfg["vae"]["model_id"]}
        write_result(variant_output, checkpoint_path, images, features, test_images, test_paths, reference_features, metadata, metrics)
        grid_path = output / "grids" / f"{variant_id}.png"; labeled_grid(images, grid_path, f"{variant["experiment"]} | step {checkpoint['global_step']} | {variant['weights']} | seeds 1000-1199")
        nearest_dir = output / "nearest_neighbors"; nearest_dir.mkdir(parents=True, exist_ok=True)
        for name in ("nearest_real.png", "outlier_pairs.png", "nearest_pairs.json"):
            source, destination = variant_output / name, nearest_dir / f"{variant_id}_{name}"
            destination.write_bytes(source.read_bytes())
        after_digest = sha256(checkpoint_path)
        if after_digest != digest:
            raise RuntimeError(f"Checkpoint changed during evaluation: {checkpoint_path}")
        peak_allocated = torch.cuda.max_memory_allocated(device) / 1024**3 if device.type == "cuda" else 0.0
        peak_reserved = torch.cuda.max_memory_reserved(device) / 1024**3 if device.type == "cuda" else 0.0
        record = {"id": variant_id, "experiment": variant["experiment"], "step": int(checkpoint["global_step"]), "weights": variant["weights"], "checkpoint": str(checkpoint_path).replace("\\", "/"), "checkpoint_sha256": digest, "grid": str(grid_path).replace("\\", "/"), "grid_sha256": sha256(grid_path), "metrics": metrics, "sampling_seconds": sampling_seconds, "images_per_second": len(images) / sampling_seconds, "peak_allocated_gb": peak_allocated, "peak_reserved_gb": peak_reserved, "deterministic_probe": True, "images": images}
        variants[variant_id] = record
        rows.append({"variant": variant_id, "experiment": variant["experiment"], "step": record["step"], "weights": record["weights"], "fid": metrics["fid"], "kid": metrics["kid"], "precision": metrics["precision"], "recall": metrics["recall"], "failures": failure_count(metrics), "duplicates": metrics["duplicate_pairs"], "low_detail_warnings": metrics["pixel"]["low_detail_warnings"], "sampling_seconds": sampling_seconds, "images_per_second": record["images_per_second"], "peak_allocated_gb": peak_allocated, "peak_reserved_gb": peak_reserved, "checkpoint_sha256": digest, "grid_sha256": record["grid_sha256"]})
    for comparison in config["comparisons"]:
        left, right = variants[comparison["left"]], variants[comparison["right"]]
        paired_grid(left["images"], right["images"], output / "comparisons" / f"{comparison['id']}.png", f"{comparison['id']} | left then right within each pair | seeds 1000-1199")
    changes = {comparison["id"]: delta(variants[comparison["left"]], variants[comparison["right"]]) for comparison in config["comparisons"][:4]}
    for record in variants.values(): record.pop("images")
    git_commit = __import__("subprocess").check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    results = {"name": config["name"], "protocol": protocol, "dataset_fingerprint": None, "device": str(device), "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None, "git_commit": git_commit, "duration_seconds": time.perf_counter() - started, "variants": variants, "rows": rows, "changes": changes, "checkpoint_unchanged": all(sha256(Path(value["checkpoint"])) == value["checkpoint_sha256"] for value in variants.values()), "deterministic_sampling": all(value["deterministic_probe"] for value in variants.values())}
    results["dataset_fingerprint"] = inspected["baseline_raw_10k"][0]["cache_fingerprint"]
    (output / "comparison.json").write_text(json.dumps(results, indent=2, sort_keys=True), encoding="utf-8")
    (output / "run_manifest.json").write_text(json.dumps({"config": str(config_path).replace("\\", "/"), "config_sha256": config_sha256(config_path), "protocol": protocol, "checkpoint_sha256": {key: value["checkpoint_sha256"] for key, value in variants.items()}}, indent=2, sort_keys=True), encoding="utf-8")
    write_csv(rows, output / "metrics.csv"); write_markdown(results, output / "report.md")
    event_id = append_ledger(config_path, config_sha256(config_path), output, results)
    print(json.dumps({"output": str(output), "ledger_event_id": event_id, "checkpoint_unchanged": results["checkpoint_unchanged"], "rows": rows}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
