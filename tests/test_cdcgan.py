"""Kiểm thử toàn diện mô hình Conditional DCGAN (cDCGAN) - Phase 3."""
import pytest
import torch

from src.models.base import GenerativeModel
from src.models.cdcgan import ConditionalDCGAN


def test_cdcgan_is_subclass():
    """Kiểm tra ConditionalDCGAN kế thừa đúng GenerativeModel."""
    assert issubclass(ConditionalDCGAN, GenerativeModel)


def test_cdcgan_init_valid_and_invalid():
    """Kiểm tra khởi tạo cDCGAN hợp lệ và báo lỗi khi thiếu num_classes."""
    model = ConditionalDCGAN(
        channels=1,
        image_size=32,
        num_classes=10,
        latent_dim=64,
        embed_dim=32,
        ngf=32,
        ndf=32,
    )
    assert model.channels == 1
    assert model.image_size == 32
    assert model.num_classes == 10
    assert model.latent_dim == 64
    assert model.embed_dim == 32
    assert model.num_params() > 0

    with pytest.raises(ValueError):
        ConditionalDCGAN(channels=1, image_size=32, num_classes=None)

    with pytest.raises(ValueError):
        ConditionalDCGAN(channels=1, image_size=32, num_classes=0)


@pytest.mark.parametrize(
    "channels,image_size",
    [
        (1, 32),  # Fashion-MNIST
        (3, 64),  # Ảnh màu 64x64
    ],
)
def test_generator_and_discriminator_forward_shape(channels, image_size):
    """Kiểm tra forward của Generator và Discriminator đúng shape ở cả 32x32 và 64x64."""
    latent_dim = 64
    num_classes = 10
    model = ConditionalDCGAN(
        channels=channels,
        image_size=image_size,
        num_classes=num_classes,
        latent_dim=latent_dim,
        embed_dim=16,
        ngf=16,
        ndf=16,
    )
    b = 4
    z = torch.randn(b, latent_dim)
    y = torch.randint(0, num_classes, (b,))

    fake_imgs = model.generator(z, y)
    assert fake_imgs.shape == (b, channels, image_size, image_size)
    assert fake_imgs.min() >= -1.0
    assert fake_imgs.max() <= 1.0

    logits = model.discriminator(fake_imgs, y)
    assert logits.shape == (b,)
    assert not torch.isnan(logits).any()
    assert not torch.isinf(logits).any()


def test_label_conditioning_effect():
    """Kiểm tra nhãn lớp y thực sự tác động đến đầu ra của cả Generator và Discriminator."""
    model = ConditionalDCGAN(
        channels=1,
        image_size=32,
        num_classes=10,
        latent_dim=32,
        embed_dim=16,
        ngf=16,
        ndf=16,
    )
    model.eval()

    # Cùng 1 vector nhiễu z nhưng 2 nhãn khác nhau phải cho ra 2 ảnh khác nhau
    z = torch.randn(2, 32)
    y0 = torch.tensor([0, 0], dtype=torch.long)
    y7 = torch.tensor([7, 7], dtype=torch.long)
    img_y0 = model.generator(z, y0)
    img_y7 = model.generator(z, y7)
    assert not torch.allclose(img_y0, img_y7)

    # Cùng 1 ảnh x nhưng ghép với 2 nhãn khác nhau phải cho ra logit khác nhau ở Discriminator
    logit_y0 = model.discriminator(img_y0, y0)
    logit_y7 = model.discriminator(img_y0, y7)
    assert not torch.allclose(logit_y0, logit_y7)


def test_discriminator_and_generator_loss_gradients():
    """Kiểm tra discriminator_loss và generator_loss cô lập gradient đúng cách."""
    model = ConditionalDCGAN(
        channels=1,
        image_size=32,
        num_classes=10,
        latent_dim=32,
        embed_dim=16,
        ngf=16,
        ndf=16,
        label_smoothing=0.1,
    )
    x = torch.randn(4, 1, 32, 32)
    y = torch.tensor([0, 2, 5, 9], dtype=torch.long)

    # 1. Backward D loss: chỉ Discriminator (kể cả label_emb của D) có grad, Generator không nhận grad
    model.zero_grad()
    d_loss, d_metrics = model.discriminator_loss(x, y)
    assert d_loss.dim() == 0
    assert set(d_metrics.keys()) == {"d_loss", "d_real", "d_fake"}
    d_loss.backward()

    d_has_grad = any(p.grad is not None and torch.norm(p.grad) > 0 for p in model.discriminator.parameters())
    g_has_grad = any(p.grad is not None and torch.norm(p.grad) > 0 for p in model.generator.parameters())
    assert d_has_grad
    assert not g_has_grad

    # 2. Backward G loss: Generator (kể cả label_emb của G) nhận grad
    model.zero_grad()
    g_loss, g_metrics = model.generator_loss(batch_size=4, labels=y, device=x.device)
    assert g_loss.dim() == 0
    assert set(g_metrics.keys()) == {"g_loss", "d_fake_g"}
    g_loss.backward()

    g_has_grad_after = any(p.grad is not None and torch.norm(p.grad) > 0 for p in model.generator.parameters())
    assert g_has_grad_after


def test_training_loss():
    """Kiểm tra training_loss hoạt động cả khi truyền y và khi y=None."""
    model = ConditionalDCGAN(
        channels=1,
        image_size=32,
        num_classes=10,
        latent_dim=32,
        embed_dim=16,
        ngf=16,
        ndf=16,
    )
    x = torch.randn(4, 1, 32, 32)
    y = torch.tensor([1, 3, 5, 7], dtype=torch.long)

    for labels in (y, None):
        model.zero_grad()
        loss_dict = model.training_loss(x, labels)
        assert set(loss_dict.keys()) == {"loss", "d_loss", "g_loss"}

        loss = loss_dict["loss"]
        assert loss.dim() == 0
        assert loss.requires_grad
        assert not torch.isnan(loss).any()

        loss.backward()
        has_grad = any(p.grad is not None and torch.norm(p.grad) > 0 for p in model.parameters())
        assert has_grad


def test_sample_output_modes():
    """Kiểm tra hàm sample ở cả 3 chế độ: y=None, y=1 nhãn đơn và y=vector nhãn."""
    model = ConditionalDCGAN(
        channels=1,
        image_size=32,
        num_classes=10,
        latent_dim=32,
        embed_dim=16,
        ngf=16,
        ndf=16,
    )

    # Chế độ 1: y=None (tự chia đều qua các lớp)
    s_default = model.sample(n=10, y=None, device="cpu")
    assert s_default.shape == (10, 1, 32, 32)
    assert s_default.min() >= -1.0 and s_default.max() <= 1.0

    # Chế độ 2: Truyền 1 nhãn duy nhất để sinh n ảnh cùng lớp đó
    s_single_class = model.sample(n=4, y=torch.tensor([7]), device="cpu")
    assert s_single_class.shape == (4, 1, 32, 32)
    assert s_single_class.min() >= -1.0 and s_single_class.max() <= 1.0

    # Chế độ 3: Truyền vector nhãn riêng cho từng ảnh
    s_custom = model.sample(n=3, y=torch.tensor([0, 4, 9]), device="cpu")
    assert s_custom.shape == (3, 1, 32, 32)
    assert not torch.isnan(s_custom).any()
    assert not torch.isinf(s_custom).any()


def test_state_dict_save_load():
    """Kiểm tra lưu và tải state_dict đầy đủ parameters (gồm Embedding) và buffers (BatchNorm)."""
    model1 = ConditionalDCGAN(
        channels=1,
        image_size=32,
        num_classes=10,
        latent_dim=32,
        embed_dim=16,
        ngf=16,
        ndf=16,
    )
    model2 = ConditionalDCGAN(
        channels=1,
        image_size=32,
        num_classes=10,
        latent_dim=32,
        embed_dim=16,
        ngf=16,
        ndf=16,
    )

    model1.train()
    x = torch.randn(4, 1, 32, 32)
    y = torch.tensor([0, 1, 2, 3], dtype=torch.long)
    _ = model1.training_loss(x, y)

    state = model1.state_dict()
    model2.load_state_dict(state)

    for p1, p2 in zip(model1.parameters(), model2.parameters()):
        assert torch.allclose(p1, p2)

    for b1, b2 in zip(model1.buffers(), model2.buffers()):
        assert torch.allclose(b1, b2)