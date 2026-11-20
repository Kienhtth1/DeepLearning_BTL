"""Kiểm thử toàn diện mô hình DCGAN - Phase 2."""
import pytest
import torch

from src.models.base import GenerativeModel
from src.models.dcgan import DCGAN


def test_dcgan_is_subclass():
    """Kiểm tra DCGAN kế thừa đúng GenerativeModel."""
    assert issubclass(DCGAN, GenerativeModel)


def test_dcgan_init_fashion_mnist():
    """Kiểm tra khởi tạo DCGAN cho Fashion-MNIST (1 channel, 32x32)."""
    model = DCGAN(
        channels=1,
        image_size=32,
        num_classes=10,
        latent_dim=64,
        ngf=32,
        ndf=32,
    )
    assert model.channels == 1
    assert model.image_size == 32
    assert model.num_classes == 10
    assert model.latent_dim == 64
    assert model.num_params() > 0


def test_dcgan_init_celeba():
    """Kiểm tra khởi tạo DCGAN cho CelebA (3 channels, 64x64)."""
    model = DCGAN(
        channels=3,
        image_size=64,
        num_classes=None,
        latent_dim=100,
        ngf=32,
        ndf=32,
    )
    assert model.channels == 3
    assert model.image_size == 64
    assert model.num_classes is None
    assert model.num_params() > 0


@pytest.mark.parametrize(
    "channels,image_size",
    [
        (1, 32),  # Fashion-MNIST
        (3, 64),  # CelebA
    ],
)
def test_generator_and_discriminator_forward_shape(channels, image_size):
    """Kiểm tra forward của Generator và Discriminator đúng shape ở cả 32x32 và 64x64."""
    latent_dim = 64
    model = DCGAN(
        channels=channels,
        image_size=image_size,
        latent_dim=latent_dim,
        ngf=16,
        ndf=16,
    )
    b = 4
    z = torch.randn(b, latent_dim)
    fake_imgs = model.generator(z)
    assert fake_imgs.shape == (b, channels, image_size, image_size)
    assert fake_imgs.min() >= -1.0
    assert fake_imgs.max() <= 1.0

    logits = model.discriminator(fake_imgs)
    assert logits.shape == (b,)
    assert not torch.isnan(logits).any()
    assert not torch.isinf(logits).any()


def test_discriminator_and_generator_loss_gradients():
    """Kiểm tra discriminator_loss và generator_loss cô lập gradient đúng cách."""
    model = DCGAN(
        channels=1,
        image_size=32,
        latent_dim=32,
        ngf=16,
        ndf=16,
        label_smoothing=0.1,
    )
    x = torch.randn(4, 1, 32, 32)

    # 1. Backward D loss: chỉ Discriminator có grad, Generator không nhận grad nhờ .detach()
    model.zero_grad()
    d_loss, d_metrics = model.discriminator_loss(x)
    assert d_loss.dim() == 0
    assert "d_loss" in d_metrics and "d_real" in d_metrics and "d_fake" in d_metrics
    d_loss.backward()

    d_has_grad = any(p.grad is not None and torch.norm(p.grad) > 0 for p in model.discriminator.parameters())
    g_has_grad = any(p.grad is not None and torch.norm(p.grad) > 0 for p in model.generator.parameters())
    assert d_has_grad
    assert not g_has_grad

    # 2. Backward G loss: Generator nhận grad
    model.zero_grad()
    g_loss, g_metrics = model.generator_loss(batch_size=4, device=x.device)
    assert g_loss.dim() == 0
    assert "g_loss" in g_metrics and "d_fake_g" in g_metrics
    g_loss.backward()

    g_has_grad_after = any(p.grad is not None and torch.norm(p.grad) > 0 for p in model.generator.parameters())
    assert g_has_grad_after


def test_training_loss():
    """Kiểm tra training_loss trả về dict chuẩn với các key 'loss', 'd_loss', 'g_loss'."""
    model = DCGAN(
        channels=1,
        image_size=32,
        latent_dim=32,
        ngf=16,
        ndf=16,
    )
    x = torch.randn(4, 1, 32, 32)
    loss_dict = model.training_loss(x)
    assert isinstance(loss_dict, dict)
    assert "loss" in loss_dict
    assert "d_loss" in loss_dict
    assert "g_loss" in loss_dict

    loss = loss_dict["loss"]
    assert loss.dim() == 0
    assert loss.requires_grad
    assert not torch.isnan(loss).any()

    loss.backward()
    has_grad = any(p.grad is not None and torch.norm(p.grad) > 0 for p in model.parameters())
    assert has_grad


def test_sample_output():
    """Kiểm tra hàm sample sinh ảnh đúng shape (n, C, H, W) và trong khoảng [-1, 1]."""
    model = DCGAN(
        channels=1,
        image_size=32,
        latent_dim=32,
        ngf=16,
        ndf=16,
    )
    samples = model.sample(n=2, device="cpu")
    assert samples.shape == (2, 1, 32, 32)
    assert samples.min() >= -1.0
    assert samples.max() <= 1.0
    assert not torch.isnan(samples).any()
    assert not torch.isinf(samples).any()


def test_state_dict_save_load():
    """Kiểm tra lưu và tải state_dict đầy đủ parameters và buffers (BatchNorm)."""
    model1 = DCGAN(
        channels=1,
        image_size=32,
        latent_dim=32,
        ngf=16,
        ndf=16,
    )
    model2 = DCGAN(
        channels=1,
        image_size=32,
        latent_dim=32,
        ngf=16,
        ndf=16,
    )

    # Chạy 1 lượt forward ở chế độ train để cập nhật running_mean / running_var của BatchNorm
    model1.train()
    _ = model1.training_loss(torch.randn(4, 1, 32, 32))

    state = model1.state_dict()
    model2.load_state_dict(state)

    for p1, p2 in zip(model1.parameters(), model2.parameters()):
        assert torch.allclose(p1, p2)

    for b1, b2 in zip(model1.buffers(), model2.buffers()):
        assert torch.allclose(b1, b2)