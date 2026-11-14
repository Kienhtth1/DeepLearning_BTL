"""Kiểm thử toàn diện mô hình DDPM - Phase 2."""
import pytest
import torch

from src.models import DDPM, GenerativeModel


def test_ddpm_is_subclass():
    """Kiểm tra DDPM kế thừa đúng GenerativeModel."""
    assert issubclass(DDPM, GenerativeModel)


def test_ddpm_init_fashion_mnist():
    """Kiểm tra khởi tạo DDPM cho Fashion-MNIST (1 channel, 32x32)."""
    model = DDPM(
        channels=1,
        image_size=32,
        num_classes=10,
        timesteps=100,
        base_channels=32,
        channel_mults=(1, 2),
        num_res_blocks=1,
    )
    assert model.channels == 1
    assert model.image_size == 32
    assert model.num_classes == 10
    assert model.num_params() > 0


def test_ddpm_init_celeba():
    """Kiểm tra khởi tạo DDPM cho CelebA (3 channels, 64x64)."""
    model = DDPM(
        channels=3,
        image_size=64,
        num_classes=None,
        timesteps=100,
        base_channels=32,
        channel_mults=(1, 2),
        num_res_blocks=1,
    )
    assert model.channels == 3
    assert model.image_size == 64
    assert model.num_classes is None
    assert model.num_params() > 0


def test_unet_forward_shape():
    """Kiểm tra forward của U-Net giữ nguyên kích thước (B, C, H, W)."""
    model = DDPM(
        channels=1,
        image_size=32,
        timesteps=50,
        base_channels=32,
        channel_mults=(1, 2),
        num_res_blocks=1,
    )
    b, c, h, w = 4, 1, 32, 32
    x = torch.randn(b, c, h, w)
    t = torch.randint(0, 50, (b,), dtype=torch.long)
    out = model.unet(x, t)
    assert out.shape == (b, c, h, w)


def test_q_sample_forward_diffusion():
    """Kiểm tra forward diffusion q_sample tạo shape đúng."""
    model = DDPM(
        channels=1,
        image_size=32,
        timesteps=50,
        base_channels=32,
        channel_mults=(1, 2),
        num_res_blocks=1,
    )
    x = torch.randn(4, 1, 32, 32)
    t = torch.tensor([0, 10, 25, 49], dtype=torch.long)
    x_t = model.q_sample(x, t)
    assert x_t.shape == x.shape
    assert not torch.isnan(x_t).any()
    assert not torch.isinf(x_t).any()


def test_training_loss():
    """Kiểm tra training_loss trả về dict có key 'loss' và có grad."""
    model = DDPM(
        channels=1,
        image_size=32,
        timesteps=50,
        base_channels=32,
        channel_mults=(1, 2),
        num_res_blocks=1,
    )
    x = torch.randn(4, 1, 32, 32)
    loss_dict = model.training_loss(x)
    assert isinstance(loss_dict, dict)
    assert "loss" in loss_dict
    loss = loss_dict["loss"]
    assert loss.dim() == 0  # scalar
    assert loss.requires_grad
    assert not torch.isnan(loss).any()

    # Thử backward
    loss.backward()
    has_grad = any(p.grad is not None and torch.norm(p.grad) > 0 for p in model.parameters())
    assert has_grad


def test_sample_output():
    """Kiểm tra hàm sample sinh ảnh đúng shape (n, C, H, W) và trong khoảng [-1, 1]."""
    model = DDPM(
        channels=1,
        image_size=32,
        timesteps=5,  # số bước nhỏ để test nhanh
        base_channels=16,
        channel_mults=(1, 2),
        num_res_blocks=1,
    )
    samples = model.sample(n=2, device="cpu")
    assert samples.shape == (2, 1, 32, 32)
    assert samples.min() >= -1.0
    assert samples.max() <= 1.0
    assert not torch.isnan(samples).any()
    assert not torch.isinf(samples).any()


def test_state_dict_save_load():
    """Kiểm tra lưu và tải state_dict đầy đủ buffers."""
    model1 = DDPM(
        channels=1,
        image_size=32,
        timesteps=10,
        base_channels=16,
        channel_mults=(1, 2),
        num_res_blocks=1,
    )
    model2 = DDPM(
        channels=1,
        image_size=32,
        timesteps=10,
        base_channels=16,
        channel_mults=(1, 2),
        num_res_blocks=1,
    )

    state = model1.state_dict()
    model2.load_state_dict(state)

    for p1, p2 in zip(model1.parameters(), model2.parameters()):
        assert torch.allclose(p1, p2)

    for b1, b2 in zip(model1.buffers(), model2.buffers()):
        assert torch.allclose(b1, b2)
