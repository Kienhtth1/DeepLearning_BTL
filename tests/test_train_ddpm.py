"""Kiểm thử huấn luyện DDPM và EMA - Phase 2."""
from pathlib import Path
import pytest
import torch
import torch.nn as nn

from src.models.ddpm import DDPM
from src.training.train_ddpm import EMA, ema_scope, train_ddpm


def test_ema_update_and_restore():
    """Kiểm tra EMA tính toán moving average chính xác và khôi phục trọng số."""
    linear = nn.Linear(4, 4, bias=False)
    nn.init.constant_(linear.weight, 1.0)

    ema = EMA(linear, decay=0.9)
    assert torch.allclose(ema.shadow["weight"], torch.ones_like(linear.weight))

    # Thay đổi trọng số model: 1.0 -> 2.0
    with torch.no_grad():
        linear.weight.fill_(2.0)

    ema.update(linear)
    # shadow = 0.9 * 1.0 + 0.1 * 2.0 = 1.1
    expected = 0.9 * 1.0 + 0.1 * 2.0
    assert torch.allclose(ema.shadow["weight"], torch.full_like(linear.weight, expected))

    # Kiểm tra scope tạm thời
    with ema_scope(linear, ema):
        assert torch.allclose(linear.weight, torch.full_like(linear.weight, expected))

    # Sau scope phải về lại trọng số gốc 2.0
    assert torch.allclose(linear.weight, torch.full_like(linear.weight, 2.0))


def test_train_ddpm_smoke(tmp_path):
    """Smoke test quá trình train_ddpm trên subset nhỏ (1 epoch)."""
    cfg = {
        "seed": 42,
        "data": {
            "name": "fashion_mnist",
            "image_size": 32,
            "batch_size": 16,
            "num_workers": 0,
            "subset_size": 32,  # Chỉ lấy 32 ảnh để chạy trong vài giây
        },
        "model": {
            "name": "ddpm",
            "timesteps": 10,
            "beta_schedule": "linear",
            "base_channels": 16,
            "channel_mults": [1, 2],
            "num_res_blocks": 1,
        },
        "train": {
            "epochs": 1,
            "lr": 1e-3,
            "ema_decay": 0.99,
            "sample_every": 1,
            "n_preview": 4,
        },
        "output": {
            "checkpoint_dir": str(tmp_path / "checkpoints"),
            "sample_dir": str(tmp_path / "samples"),
            "log_dir": str(tmp_path / "outputs"),
        },
    }

    result = train_ddpm(cfg, device="cpu", checkpoint_dir=tmp_path / "checkpoints", sample_dir=tmp_path / "samples")

    best_ckpt = Path(result["best_ckpt"])
    last_ckpt = Path(result["last_ckpt"])
    assert best_ckpt.exists(), "Chưa tạo ddpm_best.pth"
    assert last_ckpt.exists(), "Chưa tạo ddpm_last.pth"

    # Nạp và kiểm tra nội dung checkpoint
    ckpt = torch.load(best_ckpt, map_location="cpu")
    assert "model_state_dict" in ckpt
    assert "optimizer_state_dict" in ckpt
    assert "ema_state_dict" in ckpt
    assert "model_config" in ckpt
    assert ckpt["model_config"]["channels"] == 1
    assert ckpt["model_config"]["image_size"] == 32
