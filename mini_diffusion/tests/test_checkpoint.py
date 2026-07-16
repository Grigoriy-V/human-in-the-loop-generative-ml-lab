import copy
import random

import numpy as np
import torch

from mini_diffusion.diffusion import EMA
from mini_diffusion.train import build_model, load_checkpoint, save_checkpoint


def test_checkpoint_round_trip(tmp_path):
    cfg = {
        "name": "test",
        "output_dir": str(tmp_path),
        "data": {"resolution": 32, "num_classes": 10},
        "model": {
            "base_channels": 8,
            "channel_mults": [1, 2],
            "num_res_blocks": 1,
            "attention_resolutions": [16],
            "dropout": 0.0,
            "class_cond": True,
            "cond_drop_prob": 0.0,
            "num_heads": 1,
        },
        "train": {"ema_decay": 0.99},
    }
    model = build_model(cfg)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    ema = EMA(model, decay=0.99)
    loss = sum(parameter.square().mean() for parameter in model.parameters())
    loss.backward()
    optimizer.step()
    ema.update(model)
    expected_model = {name: value.detach().clone() for name, value in model.state_dict().items()}
    expected_ema = {name: value.detach().clone() for name, value in ema.shadow.items()}
    expected_optimizer = copy.deepcopy(optimizer.state_dict())
    path = tmp_path / "ckpt.pt"
    save_checkpoint(path, model, optimizer, ema, cfg, global_step=3)
    saved = torch.load(path, map_location="cpu", weights_only=False)
    torch.set_rng_state(saved["torch_rng_state"])
    expected_torch_random = torch.rand(4)
    np.random.set_state(saved["numpy_rng_state"])
    expected_numpy_random = np.random.rand(4)
    random.setstate(saved["python_rng_state"])
    expected_python_random = [random.random() for _ in range(4)]
    torch.manual_seed(999)
    np.random.seed(999)
    random.seed(999)
    with torch.no_grad():
        for p in model.parameters():
            p.add_(1.0)
        for value in ema.shadow.values():
            value.zero_()
        for state in optimizer.state.values():
            state["exp_avg"].zero_()
    loaded_cfg, step = load_checkpoint(path, model, optimizer, ema)
    assert loaded_cfg["name"] == "test"
    assert step == 3
    for name, value in model.state_dict().items():
        assert torch.equal(value, expected_model[name])
    for name, value in ema.shadow.items():
        assert torch.equal(value, expected_ema[name])
    loaded_state = next(iter(optimizer.state.values()))["exp_avg"]
    expected_state = next(iter(expected_optimizer["state"].values()))["exp_avg"]
    assert torch.equal(loaded_state, expected_state)
    assert torch.equal(torch.rand(4), expected_torch_random)
    assert np.array_equal(np.random.rand(4), expected_numpy_random)
    assert [random.random() for _ in range(4)] == expected_python_random
