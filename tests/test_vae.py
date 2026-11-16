"""Kiểm thử VAE - Phase 2 (TV1).   pytest -q tests/test_vae.py"""
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from src.models.base import GenerativeModel
from src.models.vae import VAE
from src.training.train_vae import load_vae, train_vae


@pytest.mark.parametrize("channels,size,hidden", [(1, 32, [32, 64, 128]), (3, 64, [32, 64, 128, 256])])
def test_shapes_and_range(channels, size, hidden):
    m = VAE(channels, size, latent_dim=16, hidden_dims=hidden)
    assert isinstance(m, GenerativeModel)
    x = torch.rand(4, channels, size, size) * 2 - 1
    x_hat, mu, logvar = m(x)
    assert x_hat.shape == x.shape and mu.shape == (4, 16) and logvar.shape == (4, 16)
    s = m.sample(5)
    assert s.shape == (5, channels, size, size)
    assert s.min() >= -1 and s.max() <= 1


def test_loss_parts_and_backward():
    m = VAE(1, 32, latent_dim=8)
    x = torch.rand(6, 1, 32, 32) * 2 - 1
    out = m.training_loss(x)
    assert set(out) >= {"loss", "recon", "kl"}
    assert torch.isfinite(out["loss"]) and out["kl"] >= 0
    out["loss"].backward()
    assert all(p.grad is not None for p in m.parameters() if p.requires_grad)
    # beta = 0 -> loss chỉ còn recon (ngẫu nhiên do reparam nên so gần đúng theo cấu trúc)
    out0 = m.training_loss(x, beta=0.0)
    assert torch.allclose(out0["loss"].detach(), out0["recon"])


def test_kl_zero_when_posterior_is_prior():
    m = VAE(1, 32, latent_dim=8)
    mu, logvar = torch.zeros(3, 8), torch.zeros(3, 8)
    kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    assert kl.item() == 0.0


def test_conditional_and_interpolation():
    m = VAE(1, 32, num_classes=10, latent_dim=8, conditional=True)
    x = torch.rand(4, 1, 32, 32) * 2 - 1
    y = torch.tensor([0, 3, 5, 9])
    assert torch.isfinite(m.training_loss(x, y)["loss"])
    assert m.sample(10).shape == (10, 1, 32, 32)          # y mặc định 0..9
    assert m.interpolate(x[0], x[1], steps=7, y=y[0]).shape == (7, 1, 32, 32)
    with pytest.raises(ValueError):
        VAE(1, 32, num_classes=None, conditional=True)


def test_same_z_gives_same_image():
    m = VAE(1, 32, latent_dim=8)
    z = torch.randn(2, 8)
    assert torch.allclose(m.sample(2, z=z), m.sample(2, z=z))


def test_train_vae_smoke(tmp_path: Path):
    """Train 2 epoch trên bộ ảnh giả nhỏ, kiểm tra checkpoint / log / ảnh mẫu."""
    root = tmp_path / "data" / "custom" / "fake"
    for c in ("a", "b"):
        (root / c).mkdir(parents=True)
        for i in range(24):
            Image.fromarray(np.random.randint(0, 256, (40, 40, 3), dtype=np.uint8)).save(root / c / f"{i}.png")

    cfg = {
        "seed": 0,
        "data": {"name": "custom", "custom_subdir": "fake", "data_root": str(tmp_path / "data"),
                 "image_size": 32, "batch_size": 8, "num_workers": 0},
        "model": {"name": "vae", "latent_dim": 8, "hidden_dims": [16, 32], "conditional": True},
        "train": {"epochs": 2, "lr": 1e-3, "kl_warmup_epochs": 1, "sample_every": 1, "n_preview": 4},
        "output": {"log_dir": str(tmp_path / "outputs")},
    }
    res = train_vae(cfg, device="cpu", checkpoint_dir=tmp_path / "ckpt", sample_dir=tmp_path / "samples")

    assert Path(res["best_ckpt"]).exists() and Path(res["last_ckpt"]).exists()
    assert len(res["history"]) == 2
    assert (tmp_path / "outputs" / "vae_loss_curve.png").exists()
    assert (tmp_path / "outputs" / "vae_train_log.json").exists()
    assert (tmp_path / "samples" / "vae" / "sample_epoch_002.png").exists()
    assert (tmp_path / "samples" / "vae" / "recon_epoch_002.png").exists()

    model, ckpt = load_vae(res["best_ckpt"])
    assert ckpt["model_config"]["channels"] == 3 and model.conditional
    assert model.sample(4).shape == (4, 3, 32, 32)
