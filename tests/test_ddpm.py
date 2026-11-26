"""Unified unit and smoke tests for the TV3 DDPM Phase 2 implementation."""

from __future__ import annotations

import importlib
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.models import DDPM, GenerativeModel


train_module = importlib.import_module("src.training.train_ddpm")


def _small_ddpm(**overrides) -> DDPM:
    """Create a small DDPM configuration suitable for CPU tests."""
    config = {
        "channels": 1,
        "image_size": 32,
        "timesteps": 20,
        "base_channels": 16,
        "channel_mults": (1, 2),
        "num_res_blocks": 1,
    }
    config.update(overrides)
    return DDPM(**config)


def test_ddpm_is_subclass_of_generative_model():
    assert issubclass(DDPM, GenerativeModel)


def test_ddpm_initializes_for_fashion_mnist():
    model = _small_ddpm(num_classes=10)
    assert model.channels == 1
    assert model.image_size == 32
    assert model.num_classes == 10
    assert model.num_params() > 0


def test_ddpm_initializes_for_celeba():
    model = _small_ddpm(channels=3, image_size=64, num_classes=None)
    assert model.channels == 3
    assert model.image_size == 64
    assert model.num_classes is None
    assert model.num_params() > 0


def test_unet_forward_preserves_image_shape():
    model = _small_ddpm()
    x = torch.randn(4, 1, 32, 32)
    timesteps = torch.randint(0, model.timesteps, (4,), dtype=torch.long)
    prediction = model.unet(x, timesteps)
    assert prediction.shape == x.shape
    assert torch.isfinite(prediction).all()


def test_q_sample_matches_forward_diffusion_formula():
    model = _small_ddpm()
    x_start = torch.randn(4, 1, 32, 32)
    noise = torch.randn_like(x_start)
    timesteps = torch.tensor([0, 3, 10, 19], dtype=torch.long)

    actual = model.q_sample(x_start, timesteps, noise=noise)
    alpha_bar = model.alphas_cumprod.gather(0, timesteps).view(4, 1, 1, 1)
    expected = alpha_bar.sqrt() * x_start + (1.0 - alpha_bar).sqrt() * noise

    assert actual.shape == x_start.shape
    assert torch.allclose(actual, expected, atol=1e-6)
    assert torch.isfinite(actual).all()


def test_training_loss_is_scalar_finite_and_backpropagates():
    model = _small_ddpm()
    images = torch.rand(4, 1, 32, 32) * 2.0 - 1.0

    losses = model.training_loss(images)
    assert isinstance(losses, dict)
    assert "loss" in losses
    loss = losses["loss"]
    assert loss.ndim == 0
    assert loss.requires_grad
    assert torch.isfinite(loss)

    loss.backward()
    assert any(
        parameter.grad is not None and torch.count_nonzero(parameter.grad).item() > 0
        for parameter in model.parameters()
    )


def test_cosine_beta_schedule_is_valid():
    model = _small_ddpm(beta_schedule="cosine")
    assert torch.isfinite(model.betas).all()
    assert torch.all(model.betas > 0)
    assert torch.all(model.betas < 1)
    assert torch.all(model.alphas_cumprod[1:] <= model.alphas_cumprod[:-1])


def test_reverse_diffusion_step_is_finite():
    model = _small_ddpm()
    noisy = torch.randn(2, 1, 32, 32)
    result = model.p_sample(noisy, t=5)
    assert result.shape == noisy.shape
    assert torch.isfinite(result).all()


def test_sample_returns_valid_images():
    model = _small_ddpm(timesteps=5)
    samples = model.sample(n=2, device="cpu")
    assert samples.shape == (2, 1, 32, 32)
    assert samples.min().item() >= -1.0
    assert samples.max().item() <= 1.0
    assert torch.isfinite(samples).all()


def test_state_dict_round_trip_preserves_parameters_and_buffers():
    model_a = _small_ddpm(timesteps=10)
    model_b = _small_ddpm(timesteps=10)
    model_b.load_state_dict(model_a.state_dict())

    for param_a, param_b in zip(model_a.parameters(), model_b.parameters()):
        assert torch.allclose(param_a, param_b)
    for buffer_a, buffer_b in zip(model_a.buffers(), model_b.buffers()):
        assert torch.allclose(buffer_a, buffer_b)


def test_ema_update_scope_and_state_round_trip():
    linear = nn.Linear(4, 4, bias=False)
    nn.init.constant_(linear.weight, 1.0)
    ema = train_module.EMA(linear, decay=0.9)

    with torch.no_grad():
        linear.weight.fill_(2.0)
    ema.update(linear)
    expected = torch.full_like(linear.weight, 1.1)
    assert torch.allclose(ema.shadow["weight"], expected)

    with train_module.ema_scope(linear, ema):
        assert torch.allclose(linear.weight, expected)
    assert torch.allclose(linear.weight, torch.full_like(linear.weight, 2.0))

    restored_ema = train_module.EMA(linear, decay=0.5)
    restored_ema.load_state_dict(ema.state_dict())
    assert restored_ema.decay == 0.9
    assert torch.allclose(restored_ema.shadow["weight"], expected)


def test_train_ddpm_smoke_with_synthetic_data(tmp_path, monkeypatch):
    """Smoke-test trainer without downloading Fashion-MNIST or requiring network access."""
    batch_size = 8
    train_x = torch.rand(16, 1, 32, 32) * 2.0 - 1.0
    train_y = torch.zeros(16, dtype=torch.long)
    val_x = torch.rand(8, 1, 32, 32) * 2.0 - 1.0
    val_y = torch.zeros(8, dtype=torch.long)

    info = SimpleNamespace(
        name="synthetic_fashion_mnist",
        channels=1,
        image_size=32,
        num_classes=10,
        num_samples=16,
    )

    def fake_get_dataloader_from_config(cfg, split="train"):
        if split == "train":
            dataset = TensorDataset(train_x, train_y)
        else:
            dataset = TensorDataset(val_x, val_y)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=(split == "train"))
        split_info = SimpleNamespace(**vars(info))
        split_info.num_samples = len(dataset)
        return loader, split_info

    monkeypatch.setattr(train_module, "get_dataloader_from_config", fake_get_dataloader_from_config)

    config = {
        "seed": 42,
        "data": {"batch_size": batch_size, "num_workers": 0},
        "model": {
            "name": "ddpm",
            "timesteps": 5,
            "beta_schedule": "linear",
            "base_channels": 16,
            "channel_mults": [1, 2],
            "num_res_blocks": 1,
        },
        "train": {
            "epochs": 1,
            "lr": 1e-3,
            "ema_decay": 0.9,
            "sample_every": 1,
            "n_preview": 4,
        },
        "output": {
            "checkpoint_dir": str(tmp_path / "checkpoints"),
            "sample_dir": str(tmp_path / "samples"),
            "log_dir": str(tmp_path / "outputs"),
        },
    }

    result = train_module.train_ddpm(
        config,
        device="cpu",
        checkpoint_dir=tmp_path / "checkpoints",
        sample_dir=tmp_path / "samples",
    )

    best_checkpoint = Path(result["best_ckpt"])
    last_checkpoint = Path(result["last_ckpt"])
    assert best_checkpoint.is_file()
    assert last_checkpoint.is_file()
    assert (tmp_path / "outputs" / "ddpm_loss_curve.png").is_file()
    assert (tmp_path / "outputs" / "ddpm_train_log.json").is_file()
    assert (tmp_path / "samples" / "ddpm" / "sample_epoch_001.png").is_file()

    checkpoint = torch.load(best_checkpoint, map_location="cpu")
    assert "model_state_dict" in checkpoint
    assert "optimizer_state_dict" in checkpoint
    assert "ema_state_dict" in checkpoint
    assert "model_config" in checkpoint
    assert checkpoint["model_config"]["channels"] == 1
    assert checkpoint["model_config"]["image_size"] == 32
