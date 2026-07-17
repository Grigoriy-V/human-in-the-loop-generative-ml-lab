from pathlib import Path

import pytest
import torch
from PIL import Image

from mini_diffusion.data.tiny_imagenet import TinyImageNet, build_tiny_imagenet


def _write_image(path: Path, color: tuple[int, int, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (64, 64), color).save(path, format="JPEG")


def _make_tiny_layout(root: Path) -> None:
    root.mkdir()
    (root / "wnids.txt").write_text("n0001\nn0002\n", encoding="utf-8")
    (root / "words.txt").write_text(
        "n0001\tclass one\nn0002\tclass two\n", encoding="utf-8"
    )
    _write_image(root / "train/n0001/images/n0001_0.JPEG", (255, 0, 0))
    _write_image(root / "train/n0002/images/n0002_0.JPEG", (0, 255, 0))
    _write_image(root / "val/images/val_0.JPEG", (0, 0, 255))
    _write_image(root / "val/images/val_1.JPEG", (255, 255, 255))
    (root / "val/val_annotations.txt").write_text(
        "val_0.JPEG\tn0001\t0\t0\t64\t64\n"
        "val_1.JPEG\tn0002\t0\t0\t64\t64\n",
        encoding="utf-8",
    )


def test_tiny_imagenet_train_and_val_layout(tmp_path):
    root = tmp_path / "tiny-imagenet-200"
    _make_tiny_layout(root)

    train = TinyImageNet(str(root), split="train", resolution=64)
    val = TinyImageNet(str(root), split="val", resolution=64)

    assert len(train) == 2
    assert len(val) == 2
    image, label = val[0]
    assert image.shape == (3, 64, 64)
    assert image.dtype == torch.float32
    assert torch.isfinite(image).all()
    assert -1.0 <= float(image.min()) <= float(image.max()) <= 1.0
    assert label == 0
    assert train.class_info(1) == {"index": 1, "wnid": "n0002", "name": "class two"}


def test_tiny_imagenet_missing_root_has_download_link(tmp_path):
    with pytest.raises(FileNotFoundError, match="zenodo.org"):
        TinyImageNet(str(tmp_path / "missing"))


def test_fake_tiny_imagenet_does_not_require_archive(tmp_path):
    dataset = build_tiny_imagenet(
        str(tmp_path / "missing"), fake_data=True, fake_size=4, num_classes=200
    )
    image, label = dataset[0]
    assert len(dataset) == 4
    assert image.shape == (3, 64, 64)
    assert 0 <= label < 200
