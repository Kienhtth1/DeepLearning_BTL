"""[Phase 3 - TV2] Conditional DCGAN (cDCGAN - Mirza & Osindero, 2014 + Radford et al., 2015).

Kế thừa src.models.base.GenerativeModel. Sinh ảnh có điều kiện theo nhãn lớp y (ví dụ 10 lớp Fashion-MNIST).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.base import GenerativeModel
from src.models.dcgan import DCGANDiscriminator, DCGANGenerator, weights_init_dcgan


class ConditionalDCGANGenerator(nn.Module):
    """Generator có điều kiện: ghép vector nhiễu z với embedding của nhãn lớp y."""

    def __init__(
        self,
        latent_dim: int = 100,
        num_classes: int = 10,
        embed_dim: int = 32,
        channels: int = 1,
        image_size: int = 32,
        ngf: int = 64,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_classes = num_classes
        self.embed_dim = embed_dim

        self.label_emb = nn.Embedding(num_classes, embed_dim)
        self.net = DCGANGenerator(
            latent_dim=latent_dim + embed_dim,
            channels=channels,
            image_size=image_size,
            ngf=ngf,
        )

    def forward(self, z: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        if z.ndim == 4:
            z = z.view(z.size(0), -1)
        y_emb = self.label_emb(y.long())
        zy = torch.cat([z, y_emb], dim=1)
        return self.net(zy)


class ConditionalDCGANDiscriminator(nn.Module):
    """Discriminator có điều kiện: ghép ảnh x với bản đồ không gian của nhãn lớp y."""

    def __init__(
        self,
        channels: int = 1,
        image_size: int = 32,
        num_classes: int = 10,
        ndf: int = 64,
    ):
        super().__init__()
        self.image_size = image_size
        self.num_classes = num_classes

        self.label_emb = nn.Embedding(num_classes, image_size * image_size)
        self.net = DCGANDiscriminator(
            channels=channels + 1,
            image_size=image_size,
            ndf=ndf,
        )

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        b = x.size(0)
        y_map = self.label_emb(y.long()).view(b, 1, self.image_size, self.image_size)
        xy = torch.cat([x, y_map], dim=1)
        return self.net(xy)


class ConditionalDCGAN(GenerativeModel):
    """Conditional Deep Convolutional GAN (cDCGAN)."""

    def __init__(
        self,
        channels: int,
        image_size: int,
        num_classes: int | None = 10,
        latent_dim: int = 100,
        embed_dim: int = 32,
        ngf: int = 64,
        ndf: int = 64,
        label_smoothing: float = 0.0,
    ):
        if num_classes is None or num_classes <= 0:
            raise ValueError(f"ConditionalDCGAN yêu cầu num_classes > 0, nhận được: {num_classes}")

        super().__init__(channels=channels, image_size=image_size, num_classes=num_classes)

        self.latent_dim = latent_dim
        self.embed_dim = embed_dim
        self.ngf = ngf
        self.ndf = ndf
        self.label_smoothing = float(label_smoothing)

        self.generator = ConditionalDCGANGenerator(
            latent_dim=latent_dim,
            num_classes=num_classes,
            embed_dim=embed_dim,
            channels=channels,
            image_size=image_size,
            ngf=ngf,
        )
        self.discriminator = ConditionalDCGANDiscriminator(
            channels=channels,
            image_size=image_size,
            num_classes=num_classes,
            ndf=ndf,
        )

        self.generator.apply(weights_init_dcgan)
        self.discriminator.apply(weights_init_dcgan)
        nn.init.normal_(self.generator.label_emb.weight, 0.0, 0.02)
        nn.init.normal_(self.discriminator.label_emb.weight, 0.0, 0.02)

    def discriminator_loss(
        self, real_images: torch.Tensor, labels: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, float]]:
        b = real_images.size(0)
        device = real_images.device

        real_targets = torch.full((b,), 1.0 - self.label_smoothing, device=device, dtype=torch.float32)
        fake_targets = torch.zeros(b, device=device, dtype=torch.float32)

        real_logits = self.discriminator(real_images, labels)
        loss_real = F.binary_cross_entropy_with_logits(real_logits, real_targets)

        z = torch.randn(b, self.latent_dim, device=device)
        fake_images = self.generator(z, labels).detach()
        fake_logits = self.discriminator(fake_images, labels)
        loss_fake = F.binary_cross_entropy_with_logits(fake_logits, fake_targets)

        d_loss = loss_real + loss_fake
        metrics = {
            "d_loss": d_loss.item(),
            "d_real": torch.sigmoid(real_logits).mean().item(),
            "d_fake": torch.sigmoid(fake_logits).mean().item(),
        }
        return d_loss, metrics

    def generator_loss(
        self, batch_size: int, labels: torch.Tensor, device: torch.device
    ) -> tuple[torch.Tensor, dict[str, float]]:
        real_targets = torch.ones(batch_size, device=device, dtype=torch.float32)

        z = torch.randn(batch_size, self.latent_dim, device=device)
        fake_images = self.generator(z, labels)
        fake_logits = self.discriminator(fake_images, labels)

        g_loss = F.binary_cross_entropy_with_logits(fake_logits, real_targets)
        metrics = {
            "g_loss": g_loss.item(),
            "d_fake_g": torch.sigmoid(fake_logits).mean().item(),
        }
        return g_loss, metrics

    def training_loss(self, x: torch.Tensor, y: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        if y is None:
            y = torch.randint(0, self.num_classes, (x.size(0),), device=x.device)
        d_loss, _ = self.discriminator_loss(x, y)
        g_loss, _ = self.generator_loss(x.size(0), y, x.device)
        return {
            "loss": d_loss + g_loss,
            "d_loss": d_loss,
            "g_loss": g_loss,
        }

    @torch.no_grad()
    def sample(
        self,
        n: int,
        y: torch.Tensor | None = None,
        device: str | torch.device = "cpu",
    ) -> torch.Tensor:
        """Sinh n ảnh theo nhãn lớp y. Nếu y=None, tự chia đều qua các lớp 0..num_classes-1."""
        self.eval()
        device = torch.device(device)
        z = torch.randn((n, self.latent_dim), device=device)
        if y is None:
            y = torch.arange(n, device=device) % self.num_classes
        else:
            y = y.to(device)
            if y.numel() == 1 and n > 1:
                y = y.expand(n)
        return self.generator(z, y).clamp(-1.0, 1.0)