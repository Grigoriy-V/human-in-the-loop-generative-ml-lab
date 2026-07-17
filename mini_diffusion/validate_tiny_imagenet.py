from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
from torch.utils.data import DataLoader

from mini_diffusion.data import build_tiny_imagenet


EXPECTED_CLASSES = 200
EXPECTED_TRAIN_IMAGES = 100_000
EXPECTED_VAL_IMAGES = 10_000
EXPECTED_TRAIN_PER_CLASS = 500
EXPECTED_VAL_PER_CLASS = 50


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate an extracted Tiny ImageNet archive before training."
    )
    parser.add_argument("--root", default="datasets/tiny-imagenet-200")
    parser.add_argument("--resolution", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    return parser.parse_args()


def _check_distribution(
    counts: Counter[int], expected_classes: int, expected_per_class: int, split: str
) -> list[str]:
    issues = []
    if len(counts) != expected_classes:
        issues.append(f"{split}: expected {expected_classes} classes, found {len(counts)}")
    invalid = {label: count for label, count in counts.items() if count != expected_per_class}
    if invalid:
        preview = dict(list(sorted(invalid.items()))[:5])
        issues.append(
            f"{split}: expected {expected_per_class} images per class; mismatches {preview}"
        )
    return issues


def validate_dataset(
    root: str | Path, resolution: int = 64, batch_size: int = 8, num_workers: int = 0
) -> dict:
    train = build_tiny_imagenet(str(root), split="train", resolution=resolution)
    val = build_tiny_imagenet(str(root), split="val", resolution=resolution)
    issues = []
    if len(train.wnids) != EXPECTED_CLASSES:
        issues.append(
            f"wnids.txt: expected {EXPECTED_CLASSES} classes, found {len(train.wnids)}"
        )
    if len(set(train.wnids)) != len(train.wnids):
        issues.append("wnids.txt contains duplicate class IDs")
    if len(train) != EXPECTED_TRAIN_IMAGES:
        issues.append(f"train: expected {EXPECTED_TRAIN_IMAGES} images, found {len(train)}")
    if len(val) != EXPECTED_VAL_IMAGES:
        issues.append(f"val: expected {EXPECTED_VAL_IMAGES} images, found {len(val)}")
    issues.extend(
        _check_distribution(
            Counter(label for _, label in train.samples),
            EXPECTED_CLASSES,
            EXPECTED_TRAIN_PER_CLASS,
            "train",
        )
    )
    issues.extend(
        _check_distribution(
            Counter(label for _, label in val.samples),
            EXPECTED_CLASSES,
            EXPECTED_VAL_PER_CLASS,
            "val",
        )
    )
    if issues:
        raise RuntimeError("Tiny ImageNet validation failed:\n- " + "\n- ".join(issues))

    loader = DataLoader(
        train,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )
    images, labels = next(iter(loader))
    expected_shape = (batch_size, 3, resolution, resolution)
    if tuple(images.shape) != expected_shape:
        raise RuntimeError(
            f"Loader batch shape mismatch: expected {expected_shape}, got {tuple(images.shape)}"
        )
    if not torch.isfinite(images).all():
        raise RuntimeError("Loader produced NaN or Inf image values")
    if images.min() < -1.0001 or images.max() > 1.0001:
        raise RuntimeError("Loader normalization is outside expected [-1, 1] range")
    if labels.min() < 0 or labels.max() >= EXPECTED_CLASSES:
        raise RuntimeError("Loader produced a class index outside [0, 199]")

    return {
        "status": "ok",
        "root": str(Path(root).as_posix()),
        "classes": len(train.wnids),
        "train_images": len(train),
        "val_images": len(val),
        "batch_shape": list(images.shape),
        "batch_dtype": str(images.dtype),
        "batch_min": float(images.min()),
        "batch_max": float(images.max()),
        "num_workers": num_workers,
    }


def main() -> None:
    args = parse_args()
    try:
        report = validate_dataset(
            args.root,
            resolution=args.resolution,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
        )
    except (FileNotFoundError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2) from None
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
