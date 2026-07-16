from __future__ import annotations

from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class TinyImageNet(Dataset):
    def __init__(self, root: str, split: str = "train", resolution: int = 64):
        self.root = Path(root)
        self.split = split
        self.resolution = resolution
        if not self.root.exists():
            raise FileNotFoundError(
                f"Tiny ImageNet not found at {self.root}. Download it manually into "
                "datasets/tiny-imagenet-200/ as described in README.md."
            )
        self.wnids = self._read_lines("wnids.txt")
        self.wnid_to_idx = {wnid: i for i, wnid in enumerate(self.wnids)}
        self.words = self._read_words()
        self.samples = self._collect_samples()
        if not self.samples:
            raise RuntimeError(f"No Tiny ImageNet samples found for split={split} at {self.root}")
        self.transform = transforms.Compose(
            [
                transforms.Resize((resolution, resolution)),
                transforms.RandomHorizontalFlip() if split == "train" else transforms.Lambda(lambda x: x),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )

    def _read_lines(self, name: str) -> list[str]:
        path = self.root / name
        if not path.exists():
            raise FileNotFoundError(f"Missing Tiny ImageNet file: {path}")
        return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]

    def _read_words(self) -> dict[str, str]:
        path = self.root / "words.txt"
        if not path.exists():
            return {}
        words = {}
        for line in path.read_text(encoding="utf-8").splitlines():
            parts = line.split("\t", 1)
            if len(parts) == 2:
                words[parts[0]] = parts[1]
        return words

    def _collect_samples(self) -> list[tuple[Path, int]]:
        if self.split == "train":
            samples = []
            for wnid in self.wnids:
                for path in sorted((self.root / "train" / wnid / "images").glob("*.JPEG")):
                    samples.append((path, self.wnid_to_idx[wnid]))
            return samples
        if self.split == "val":
            ann = self.root / "val" / "val_annotations.txt"
            if not ann.exists():
                raise FileNotFoundError(f"Missing Tiny ImageNet validation annotations: {ann}")
            labels = {}
            for line in ann.read_text(encoding="utf-8").splitlines():
                cols = line.split("\t")
                if len(cols) >= 2:
                    labels[cols[0]] = self.wnid_to_idx[cols[1]]
            return [
                (self.root / "val" / "images" / name, idx)
                for name, idx in labels.items()
                if (self.root / "val" / "images" / name).exists()
            ]
        raise ValueError(f"Unsupported Tiny ImageNet split: {self.split}")

    def class_info(self, index: int) -> dict[str, str | int]:
        wnid = self.wnids[index]
        return {"index": index, "wnid": wnid, "name": self.words.get(wnid, wnid)}

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        path, label = self.samples[index]
        image = Image.open(path).convert("RGB")
        return self.transform(image), label


def build_tiny_imagenet(root: str, split: str = "train", resolution: int = 64):
    return TinyImageNet(root=root, split=split, resolution=resolution)
