"""[Phase 2 - TV2] DCGAN (Radford et al., 2015).

Kế thừa src.models.base.GenerativeModel. Gợi ý: Generator ConvTranspose+BN+ReLU, output Tanh;
Discriminator Conv+BN+LeakyReLU(0.2); Adam lr=2e-4, betas=(0.5, 0.999).
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.base import GenerativeModel


def weights_init_dcgan(module: nn.Module) -> None:
    """Khởi tạo trọng số chuẩn DCGAN (Radford et al., 2015) an toàn theo kiểu lớp."""
    if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
        nn.init.normal_(module.weight, mean=0.0, std=0.02)
        if module.bias is not None:
            nn.init.constant_(module.bias, 0.0)
    elif isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d)):
        if module.weight is not None:
            nn.init.normal_(module.weight, mean=1.0, std=0.02)
        if module.bias is not None:
            nn.init.constant_(module.bias, 0.0)


class DCGANGenerator(nn.Module):
    """Mạng sinh (Generator) của DCGAN ánh xạ z (B, latent_dim) -> ảnh (B, C, H, W) trong [-1, 1]."""

    def __init__(
        self,
        latent_dim: int = 100,
        channels: int = 1,
        image_size: int = 32,
        ngf: int = 64,
    ):
        super().__init__()
        if image_size < 16 or (image_size & (image_size - 1)) != 0:
            raise ValueError(f"image_size phải là lũy thừa của 2 và >= 16, nhận được: {image_size}")

        self.latent_dim = latent_dim
        self.channels = channels
        self.image_size = image_size
        self.ngf = ngf

        # Số lần nhân đôi kích thước từ 4x4 lên image_size (32 -> 3 bước, 64 -> 4 bước)
        num_upsamples = int(math.log2(image_size)) - 2
        curr_ch = ngf * (2 ** (num_upsamples - 1))

        layers: list[nn.Module] = [
            # (B, latent_dim, 1, 1) -> (B, curr_ch, 4, 4)
            nn.ConvTranspose2d(latent_dim, curr_ch, kernel_size=4, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(curr_ch),
            nn.ReLU(inplace=True),
        ]

        # Các tầng upsample trung gian: 4x4 -> 8x8 -> 16x16 (và -> 32x32 nếu image_size=64)
        for _ in range(num_upsamples - 1):
            next_ch = curr_ch // 2
            layers.extend([
                nn.ConvTranspose2d(curr_ch, next_ch, kernel_size=4, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(next_ch),
                nn.ReLU(inplace=True),
            ])
            curr_ch = next_ch

        # Tầng cuối cùng xuất ảnh trong khoảng [-1, 1] bằng Tanh
        layers.extend([
            nn.ConvTranspose2d(curr_ch, channels, kernel_size=4, stride=2, padding=1, bias=False),
            nn.Tanh(),
        ])

        self.net = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        if z.ndim == 2:
            z = z[:, :, None, None]
        return self.net(z)


class DCGANDiscriminator(nn.Module):
    """Mạng phân biệt (Discriminator) của DCGAN ánh xạ ảnh (B, C, H, W) -> logits (B,)."""

    def __init__(
        self,
        channels: int = 1,
        image_size: int = 32,
        ndf: int = 64,
    ):
        super().__init__()
        if image_size < 16 or (image_size & (image_size - 1)) != 0:
            raise ValueError(f"image_size phải là lũy thừa của 2 và >= 16, nhận được: {image_size}")

        self.channels = channels
        self.image_size = image_size
        self.ndf = ndf

        num_downsamples = int(math.log2(image_size)) - 2
        curr_ch = ndf

        # Tầng đầu tiên không dùng BatchNorm theo thiết kế DCGAN gốc
        layers: list[nn.Module] = [
            nn.Conv2d(channels, curr_ch, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
        ]

        # Các tầng downsample trung gian đưa về kích thước không gian 4x4
        for _ in range(num_downsamples - 1):
            next_ch = curr_ch * 2
            layers.extend([
                nn.Conv2d(curr_ch, next_ch, kernel_size=4, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(next_ch),
                nn.LeakyReLU(0.2, inplace=True),
            ])
            curr_ch = next_ch

        # Tầng cuối: (B, curr_ch, 4, 4) -> (B, 1, 1, 1) -> logits (B,)
        layers.append(nn.Conv2d(curr_ch, 1, kernel_size=4, stride=1, padding=0, bias=False))

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).view(-1)


class DCGAN(GenerativeModel):
    """Deep Convolutional GAN (Radford et al., 2015).

    Kế thừa `GenerativeModel`. Hỗ trợ:
    - Fashion-MNIST (1 channel, 32x32)
    - CelebA (3 channels, 64x64)
    - Custom dataset theo DatasetInfo
    """

    def __init__(
        self,
        channels: int,
        image_size: int,
        num_classes: int | None = None,
        latent_dim: int = 100,
        ngf: int = 64,
        ndf: int = 64,
        label_smoothing: float = 0.0,
    ):
        super().__init__(channels=channels, image_size=image_size, num_classes=num_classes)

        self.latent_dim = latent_dim
        self.ngf = ngf
        self.ndf = ndf
        self.label_smoothing = float(label_smoothing)

        self.generator = DCGANGenerator(
            latent_dim=latent_dim,
            channels=channels,
            image_size=image_size,
            ngf=ngf,
        )
        self.discriminator = DCGANDiscriminator(
            channels=channels,
            image_size=image_size,
            ndf=ndf,
        )

        # Áp dụng khởi tạo trọng số chuẩn DCGAN
        self.generator.apply(weights_init_dcgan)
        self.discriminator.apply(weights_init_dcgan)

    def discriminator_loss(
        self, real_images: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Tính loss cho Discriminator: tối đa hóa log D(x) + log(1 - D(G(z)))."""
        b = real_images.size(0)
        device = real_images.device

        real_targets = torch.full((b,), 1.0 - self.label_smoothing, device=device, dtype=torch.float32)
        fake_targets = torch.zeros(b, device=device, dtype=torch.float32)

        # Đánh giá trên ảnh thật
        real_logits = self.discriminator(real_images)
        loss_real = F.binary_cross_entropy_with_logits(real_logits, real_targets)

        # Đánh giá trên ảnh giả (detach để không truyền gradient ngược về Generator)
        z = torch.randn(b, self.latent_dim, device=device)
        fake_images = self.generator(z).detach()
        fake_logits = self.discriminator(fake_images)
        loss_fake = F.binary_cross_entropy_with_logits(fake_logits, fake_targets)

        d_loss = loss_real + loss_fake
        metrics = {
            "d_loss": d_loss.item(),
            "d_real": torch.sigmoid(real_logits).mean().item(),
            "d_fake": torch.sigmoid(fake_logits).mean().item(),
        }
        return d_loss, metrics

    def generator_loss(
        self, batch_size: int, device: torch.device
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Tính loss cho Generator (Non-saturating loss): tối đa hóa log D(G(z))."""
        real_targets = torch.ones(batch_size, device=device, dtype=torch.float32)

        z = torch.randn(batch_size, self.latent_dim, device=device)
        fake_images = self.generator(z)
        fake_logits = self.discriminator(fake_images)

        g_loss = F.binary_cross_entropy_with_logits(fake_logits, real_targets)
        metrics = {
            "g_loss": g_loss.item(),
            "d_fake_g": torch.sigmoid(fake_logits).mean().item(),
        }
        return g_loss, metrics

    def training_loss(self, x: torch.Tensor, y: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        """Giao diện loss chuẩn kế thừa từ GenerativeModel."""
        b, device = x.size(0), x.device
        d_loss, _ = self.discriminator_loss(x)
        g_loss, _ = self.generator_loss(b, device)
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
        """Sinh n ảnh mới từ vector nhiễu Gaussian chuẩn z ~ N(0, I).
        
        Trả về tensor shape (n, channels, image_size, image_size) với giá trị trong [-1, 1].
        """
        self.eval()
        device = torch.device(device)
        z = torch.randn((n, self.latent_dim), device=device)
        imgs = self.generator(z)
        return imgs.clamp(-1.0, 1.0)