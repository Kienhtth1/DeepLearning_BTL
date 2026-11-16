"""[Phase 2 - TV1] Variational Autoencoder (Kingma & Welling, 2013).

Kiến trúc Conv-VAE dùng chung cho Fashion-MNIST (1x32x32) và CelebA / custom (3x64x64):

    Encoder:  [Conv(4,s2) -> BN -> LeakyReLU] x len(hidden_dims)  ->  flatten  ->  fc_mu, fc_logvar
    Decoder:  fc -> reshape -> [ConvT(4,s2) -> BN -> LeakyReLU] x (len-1) -> ConvT(4,s2) -> Tanh

Ảnh vào/ra trong [-1, 1] (quy ước chung của nhóm) nên dùng loss tái tạo MSE thay cho BCE:

    loss = recon + beta * KL
    recon = sum_pixel (x - x_hat)^2           (trung bình theo batch)
    KL    = -1/2 * sum_latent (1 + logvar - mu^2 - exp(logvar))

Hỗ trợ sẵn cho Phase 3:
- `conditional=True`  -> CVAE: nối one-hot nhãn lớp vào encoder và decoder.
- `interpolate()`     -> nội suy tuyến tính trong không gian ẩn giữa 2 ảnh.
"""
from __future__ import annotations

from typing import Sequence

import torch
import torch.nn.functional as F
from torch import nn

from .base import GenerativeModel


class VAE(GenerativeModel):
    def __init__(
        self,
        channels: int,
        image_size: int,
        num_classes: int | None = None,
        latent_dim: int = 128,
        hidden_dims: Sequence[int] = (32, 64, 128),
        beta: float = 1.0,
        conditional: bool = False,
    ):
        super().__init__(channels=channels, image_size=image_size, num_classes=num_classes)
        n_down = len(hidden_dims)
        if image_size % (2**n_down) != 0:
            raise ValueError(f"image_size={image_size} phải chia hết cho 2^{n_down} (số tầng hidden_dims)")
        if conditional and not num_classes:
            raise ValueError("conditional=True cần dataset có nhãn lớp (num_classes)")

        self.latent_dim = latent_dim
        self.hidden_dims = list(hidden_dims)
        self.beta = beta
        self.conditional = conditional
        self.n_cond = num_classes if conditional else 0
        self.feat_size = image_size // (2**n_down)
        feat_dim = self.hidden_dims[-1] * self.feat_size**2

        # ---------------- Encoder ----------------
        enc, in_ch = [], channels + self.n_cond   # CVAE: nhãn được "vẽ" thành n_cond kênh hằng
        for h in self.hidden_dims:
            enc += [nn.Conv2d(in_ch, h, 4, 2, 1, bias=False), nn.BatchNorm2d(h), nn.LeakyReLU(0.2, inplace=True)]
            in_ch = h
        self.encoder = nn.Sequential(*enc)
        self.fc_mu = nn.Linear(feat_dim, latent_dim)
        self.fc_logvar = nn.Linear(feat_dim, latent_dim)

        # ---------------- Decoder ----------------
        self.fc_dec = nn.Linear(latent_dim + self.n_cond, feat_dim)
        rev = self.hidden_dims[::-1]
        dec = []
        for h_in, h_out in zip(rev[:-1], rev[1:]):
            dec += [nn.ConvTranspose2d(h_in, h_out, 4, 2, 1, bias=False), nn.BatchNorm2d(h_out),
                    nn.LeakyReLU(0.2, inplace=True)]
        dec += [nn.ConvTranspose2d(rev[-1], channels, 4, 2, 1), nn.Tanh()]
        self.decoder = nn.Sequential(*dec)

    # ------------------------------------------------------------------ #
    def _onehot(self, y: torch.Tensor | None, n: int, device) -> torch.Tensor | None:
        if not self.conditional:
            return None
        if y is None:
            raise ValueError("Mô hình conditional cần truyền nhãn y")
        return F.one_hot(y.to(device).long(), self.n_cond).float()

    def encode(self, x: torch.Tensor, y: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        c = self._onehot(y, x.size(0), x.device)
        if c is not None:
            x = torch.cat([x, c[:, :, None, None].expand(-1, -1, x.size(2), x.size(3))], dim=1)
        h = self.encoder(x).flatten(1)
        return self.fc_mu(h), self.fc_logvar(h).clamp(-30.0, 20.0)

    @staticmethod
    def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        return mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)

    def decode(self, z: torch.Tensor, y: torch.Tensor | None = None) -> torch.Tensor:
        c = self._onehot(y, z.size(0), z.device)
        if c is not None:
            z = torch.cat([z, c], dim=1)
        h = self.fc_dec(z).view(-1, self.hidden_dims[-1], self.feat_size, self.feat_size)
        return self.decoder(h)

    def forward(self, x: torch.Tensor, y: torch.Tensor | None = None):
        mu, logvar = self.encode(x, y)
        z = self.reparameterize(mu, logvar)
        return self.decode(z, y), mu, logvar

    # ------------------------------------------------------------------ #
    def training_loss(
        self, x: torch.Tensor, y: torch.Tensor | None = None, beta: float | None = None
    ) -> dict[str, torch.Tensor]:
        """`beta` truyền vào để làm KL warm-up; mặc định dùng self.beta."""
        beta = self.beta if beta is None else beta
        x_hat, mu, logvar = self(x, y)
        b = x.size(0)
        recon = F.mse_loss(x_hat, x, reduction="sum") / b
        kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / b
        return {"loss": recon + beta * kl, "recon": recon.detach(), "kl": kl.detach()}

    @torch.no_grad()
    def sample(
        self,
        n: int,
        y: torch.Tensor | None = None,
        device: str | torch.device = "cpu",
        z: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Sinh n ảnh trong [-1, 1] từ z ~ N(0, I). CVAE: y mặc định lặp 0..num_classes-1."""
        self.eval()
        device = torch.device(device)
        if z is None:
            z = torch.randn(n, self.latent_dim, device=device)
        if self.conditional and y is None:
            y = torch.arange(n, device=device) % self.n_cond
        return self.decode(z.to(device), y).clamp(-1.0, 1.0)

    @torch.no_grad()
    def reconstruct(self, x: torch.Tensor, y: torch.Tensor | None = None) -> torch.Tensor:
        """Tái tạo bằng mu (không lấy mẫu) - dùng để xem chất lượng encoder/decoder."""
        self.eval()
        mu, _ = self.encode(x, y)
        return self.decode(mu, y).clamp(-1.0, 1.0)

    @torch.no_grad()
    def interpolate(
        self, x1: torch.Tensor, x2: torch.Tensor, steps: int = 10, y: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Nội suy tuyến tính giữa 2 ảnh (shape (C,H,W)) trong không gian ẩn -> (steps, C, H, W)."""
        self.eval()
        yy = None if y is None else y.view(1).expand(2)
        mu, _ = self.encode(torch.stack([x1, x2]), yy)
        alphas = torch.linspace(0, 1, steps, device=mu.device)[:, None]
        z = (1 - alphas) * mu[0] + alphas * mu[1]
        y_rep = None if y is None else y.view(1).expand(steps)
        return self.decode(z, y_rep).clamp(-1.0, 1.0)
