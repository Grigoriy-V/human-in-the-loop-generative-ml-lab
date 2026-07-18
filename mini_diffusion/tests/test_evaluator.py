from __future__ import annotations

import pytest
import torch

from mini_diffusion.evaluator import (
    IMAGENETTE_IMAGENET_INDEX, fid, fixed_protocol, kid, precision_recall,
    protocol_noise, reference_cache_key, validate_feature_inputs,
)


def test_fixed_protocol_is_complete_and_deterministic() -> None:
    first = fixed_protocol(10, 2, 100)
    second = fixed_protocol(10, 2, 100)
    assert first == second and len(first) == 20
    assert [spec.class_id for spec in first] == [item for item in range(10) for _ in range(2)]
    assert torch.equal(protocol_noise(first, (4, 2, 2)), protocol_noise(second, (4, 2, 2)))


def test_imagenette_class_mapping_has_all_targets() -> None:
    assert len(IMAGENETTE_IMAGENET_INDEX) == 10
    assert IMAGENETTE_IMAGENET_INDEX["n01440764"] == 0
    assert IMAGENETTE_IMAGENET_INDEX["n03888257"] == 701


def test_metric_validation_and_self_comparison() -> None:
    features = torch.randn(8, 4)
    validate_feature_inputs(features, features.clone())
    assert fid(features, features.clone(), torch.device("cpu")) == pytest.approx(0.0, abs=1e-7)
    assert kid(features, features.clone(), subsets=2, subset_size=6) == pytest.approx(0.0, abs=1e-6)
    assert precision_recall(features, features.clone(), k=2) == (1.0, 1.0)
    with pytest.raises(ValueError):
        validate_feature_inputs(torch.randn(1, 4), torch.randn(2, 4))
    with pytest.raises(ValueError):
        validate_feature_inputs(torch.randn(2, 4), torch.randn(2, 5))


def test_reference_cache_key_changes_with_files(tmp_path) -> None:
    root = tmp_path / "imagenette2-160" / "val" / "n01440764"
    root.mkdir(parents=True)
    image = root / "one.jpg"; image.write_bytes(b"one")
    classes = {"n01440764": 0}
    first = reference_cache_key(root.parents[1], "val", classes)
    image.write_bytes(b"two")
    assert reference_cache_key(root.parents[1], "val", classes) != first
