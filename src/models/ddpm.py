"""[Phase 2 - TV3] Denoising Diffusion Probabilistic Models (Ho et al., 2020).

Kế thừa src.models.base.GenerativeModel.
Bao gồm:
- Sinusoidal Timestep Embedding + MLP projection.
- Residual Block có GroupNorm + SiLU + time projection + skip connection.
- U-Net nhỏ gọn hỗ trợ ảnh 32x32 (Fashion-MNIST) và 64x64 (CelebA / Custom).
- Linear / Cosine Beta Scheduler lưu dưới dạng register_buffer.
- Forward diffusion q_sample.
- Reverse diffusion sampling p_sample.
- training_loss() trả về {"loss": tensor}.
- sample(n, y, device) trả về tensor ảnh trong [-1, 1].
"""
from __future__ import annotations

import math
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.base import GenerativeModel


def get_group_norm(channels: int, max_groups: int = 32) -> nn.GroupNorm:
    """GroupNorm với số nhóm chia hết cho số channels (tối đa max_groups)."""
    for g in (max_groups, 16, 8, 4, 2, 1):
        if channels % g == 0:
            return nn.GroupNorm(num_groups=g, num_channels=channels)
    return nn.GroupNorm(num_groups=1, num_channels=channels)


class SinusoidalPositionEmbeddings(nn.Module):
    """Mã hoá timestep theo hàm sin và cos (Vaswani et al., Ho et al.)."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, time: torch.Tensor) -> torch.Tensor:
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1) if half_dim > 1 else torch.tensor(0.0, device=device)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None].float() * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        if self.dim % 2 == 1:
            embeddings = F.pad(embeddings, (0, 1))
        return embeddings


class ResidualBlock(nn.Module):
    """Residual Block gồm 2 Conv3x3, GroupNorm, SiLU và cộng timestep embedding projection."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        time_emb_dim: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.norm1 = get_group_norm(in_channels)
        self.act1 = nn.SiLU()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, out_channels),
        )

        self.norm2 = get_group_norm(out_channels)
        self.act2 = nn.SiLU()
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

        if in_channels != out_channels:
            self.residual_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.residual_conv = nn.Identity()

    def forward(self, x: torch.Tensor, time_emb: torch.Tensor) -> torch.Tensor:
        h = self.conv1(self.act1(self.norm1(x)))
        # time_mlp(time_emb) có shape (B, out_channels) -> broadcast sang (B, out_channels, 1, 1)
        time_proj = self.time_mlp(time_emb)[:, :, None, None]
        h = h + time_proj
        h = self.conv2(self.dropout(self.act2(self.norm2(h))))
        return h + self.residual_conv(x)


class AttentionBlock(nn.Module):
    """Self-Attention không gian 2D cho feature maps độ phân giải thấp (e.g. 16x16, 8x8)."""

    def __init__(self, channels: int, num_heads: int = 4):
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads
        self.norm = get_group_norm(channels)
        self.qkv = nn.Conv2d(channels, channels * 3, kernel_size=1, bias=False)
        self.proj_out = nn.Conv2d(channels, channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        h_ = self.norm(x)
        qkv = self.qkv(h_)
        q, k, v = torch.chunk(qkv, 3, dim=1)

        # reshape sang (B, num_heads, H*W, head_dim)
        head_dim = c // self.num_heads
        q = q.reshape(b, self.num_heads, head_dim, h * w).transpose(-1, -2)
        k = k.reshape(b, self.num_heads, head_dim, h * w).transpose(-1, -2)
        v = v.reshape(b, self.num_heads, head_dim, h * w).transpose(-1, -2)

        out = F.scaled_dot_product_attention(q, k, v)
        out = out.transpose(-1, -2).reshape(b, c, h, w)
        return x + self.proj_out(out)


class Downsample(nn.Module):
    """Downsample giảm kích thước ảnh một nửa bằng Conv stride 2."""

    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Upsample(nn.Module):
    """Upsample nhân đôi kích thước ảnh bằng Nearest + Conv3x3 để tránh checkerboard."""

    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2.0, mode="nearest")
        return self.conv(x)


class SmallUNet(nn.Module):
    """U-Net tinh gọn cho DDPM."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        base_channels: int = 64,
        channel_mults: Sequence[int] = (1, 2, 2),
        num_res_blocks: int = 2,
        time_emb_dim: int = 256,
        attention_resolutions: Sequence[int] = (16, 8),
        dropout: float = 0.0,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.base_channels = base_channels
        self.channel_mults = list(channel_mults)
        self.num_res_blocks = num_res_blocks
        self.attention_resolutions = set(attention_resolutions)

        # Timestep MLP
        sinusoidal_dim = base_channels
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(sinusoidal_dim),
            nn.Linear(sinusoidal_dim, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim),
        )

        # Input convolution
        self.conv_in = nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1)

        # Xây dựng Downsampling path
        self.down_blocks = nn.ModuleList()
        ch_history = [base_channels]
        curr_ch = base_channels

        for level, mult in enumerate(self.channel_mults):
            out_ch = base_channels * mult
            for _ in range(num_res_blocks):
                block = ResidualBlock(curr_ch, out_ch, time_emb_dim, dropout=dropout)
                self.down_blocks.append(block)
                curr_ch = out_ch
                ch_history.append(curr_ch)

            if level != len(self.channel_mults) - 1:
                self.down_blocks.append(Downsample(curr_ch))
                ch_history.append(curr_ch)

        # Bottleneck (Mid block)
        self.mid_block1 = ResidualBlock(curr_ch, curr_ch, time_emb_dim, dropout=dropout)
        self.mid_attn = AttentionBlock(curr_ch)
        self.mid_block2 = ResidualBlock(curr_ch, curr_ch, time_emb_dim, dropout=dropout)

        # Xây dựng Upsampling path
        self.up_blocks = nn.ModuleList()
        for level in reversed(range(len(self.channel_mults))):
            out_ch = base_channels * self.channel_mults[level]
            for _ in range(num_res_blocks + 1):
                skip_ch = ch_history.pop()
                block = ResidualBlock(curr_ch + skip_ch, out_ch, time_emb_dim, dropout=dropout)
                self.up_blocks.append(block)
                curr_ch = out_ch

            if level != 0:
                self.up_blocks.append(Upsample(curr_ch))

        assert len(ch_history) == 0, f"ch_history chưa dùng hết: {ch_history}"

        # Output block
        self.norm_out = get_group_norm(curr_ch)
        self.act_out = nn.SiLU()
        self.conv_out = nn.Conv2d(curr_ch, out_channels, kernel_size=3, padding=1)
        nn.init.zeros_(self.conv_out.weight)
        nn.init.zeros_(self.conv_out.bias)

    def forward(self, x: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
        time_emb = self.time_mlp(time)

        h = self.conv_in(x)
        skips = [h]

        down_idx = 0
        for level in range(len(self.channel_mults)):
            for _ in range(self.num_res_blocks):
                h = self.down_blocks[down_idx](h, time_emb)
                down_idx += 1
                skips.append(h)

            if level != len(self.channel_mults) - 1:
                h = self.down_blocks[down_idx](h)
                down_idx += 1
                skips.append(h)

        h = self.mid_block1(h, time_emb)
        h = self.mid_attn(h)
        h = self.mid_block2(h, time_emb)

        up_idx = 0
        for level in reversed(range(len(self.channel_mults))):
            for _ in range(self.num_res_blocks + 1):
                skip = skips.pop()
                h = torch.cat([h, skip], dim=1)
                h = self.up_blocks[up_idx](h, time_emb)
                up_idx += 1

            if level != 0:
                h = self.up_blocks[up_idx](h)
                up_idx += 1

        assert len(skips) == 0, f"Skips còn sót: {len(skips)}"

        return self.conv_out(self.act_out(self.norm_out(h)))


def _extract(a: torch.Tensor, t: torch.Tensor, x_shape: torch.Size) -> torch.Tensor:
    """Trích xuất tensor tại chỉ số t và reshape để broadcast với x."""
    b = t.shape[0]
    out = a.gather(0, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


class DDPM(GenerativeModel):
    """Denoising Diffusion Probabilistic Model (Ho et al., 2020).

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
        timesteps: int = 1000,
        beta_start: float = 1e-4,
        beta_end: float = 0.02,
        beta_schedule: str = "linear",
        base_channels: int = 64,
        channel_mults: Sequence[int] = (1, 2, 2),
        num_res_blocks: int = 2,
        time_emb_dim: int | None = None,
        dropout: float = 0.0,
    ):
        super().__init__(channels=channels, image_size=image_size, num_classes=num_classes)

        self.timesteps = timesteps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.beta_schedule = beta_schedule
        self.base_channels = base_channels
        self.channel_mults = tuple(channel_mults)
        self.num_res_blocks = num_res_blocks
        self.time_emb_dim = time_emb_dim or (base_channels * 4)
        self.dropout = dropout

        # Thiết lập schedule betas
        if beta_schedule == "linear":
            betas = torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float32)
        elif beta_schedule == "cosine":
            # Nichol & Dhariwal, 2021
            s = 0.008
            steps = timesteps + 1
            x = torch.linspace(0, timesteps, steps, dtype=torch.float32)
            alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
            alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
            betas = 1.0 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
            betas = torch.clip(betas, 0.0001, 0.9999)
        else:
            raise ValueError(f"beta_schedule '{beta_schedule}' không được hỗ trợ (chỉ chọn linear | cosine)")

        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)

        # Tính toán trước các hệ số cho Forward & Reverse diffusion
        sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
        sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)
        sqrt_recip_alphas_cumprod = torch.sqrt(1.0 / alphas_cumprod)
        sqrt_recipm1_alphas_cumprod = torch.sqrt(1.0 / alphas_cumprod - 1.0)

        # Posterior variance q(x_{t-1} | x_t, x_0)
        posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        posterior_variance_clipped = posterior_variance.clone()
        posterior_variance_clipped[0] = posterior_variance[1] if timesteps > 1 else 1e-4

        # Hệ số tính posterior mean
        posterior_mean_coef1 = betas * torch.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        posterior_mean_coef2 = (1.0 - alphas_cumprod_prev) * torch.sqrt(alphas) / (1.0 - alphas_cumprod)

        # Đăng ký buffers để PyTorch quản lý thiết bị (CPU/GPU) và lưu vào state_dict
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("alphas_cumprod_prev", alphas_cumprod_prev)
        self.register_buffer("sqrt_alphas_cumprod", sqrt_alphas_cumprod)
        self.register_buffer("sqrt_one_minus_alphas_cumprod", sqrt_one_minus_alphas_cumprod)
        self.register_buffer("sqrt_recip_alphas_cumprod", sqrt_recip_alphas_cumprod)
        self.register_buffer("sqrt_recipm1_alphas_cumprod", sqrt_recipm1_alphas_cumprod)
        self.register_buffer("posterior_variance", posterior_variance)
        self.register_buffer("posterior_variance_clipped", posterior_variance_clipped)
        self.register_buffer("posterior_mean_coef1", posterior_mean_coef1)
        self.register_buffer("posterior_mean_coef2", posterior_mean_coef2)

        # Khởi tạo UNet
        self.unet = SmallUNet(
            in_channels=channels,
            out_channels=channels,
            base_channels=base_channels,
            channel_mults=channel_mults,
            num_res_blocks=num_res_blocks,
            time_emb_dim=self.time_emb_dim,
            dropout=dropout,
        )

    def q_sample(self, x_start: torch.Tensor, t: torch.Tensor, noise: torch.Tensor | None = None) -> torch.Tensor:
        """Forward diffusion: thêm nhiễu vào x_0 tại timestep t.

        x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * epsilon
        """
        if noise is None:
            noise = torch.randn_like(x_start)

        sqrt_alpha = _extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus_alpha = _extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)
        return sqrt_alpha * x_start + sqrt_one_minus_alpha * noise

    def training_loss(self, x: torch.Tensor, y: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        """Tính loss huấn luyện theo Ho et al. (2020).

        1. Sinh timestep ngẫu nhiên cho từng ảnh trong batch.
        2. Sinh Gaussian noise chuẩn.
        3. Tạo x_t bằng q_sample.
        4. Dự đoán noise bằng UNet.
        5. So sánh MSE giữa predicted noise và noise thật.
        """
        b, device = x.shape[0], x.device
        t = torch.randint(0, self.timesteps, (b,), device=device, dtype=torch.long)
        noise = torch.randn_like(x)

        x_noisy = self.q_sample(x, t, noise=noise)
        predicted_noise = self.unet(x_noisy, t)

        loss = F.mse_loss(predicted_noise, noise)
        return {"loss": loss}

    def p_sample(self, x: torch.Tensor, t: int, clip_denoised: bool = True) -> torch.Tensor:
        """Một bước reverse diffusion: p(x_{t-1} | x_t)."""
        b = x.shape[0]
        device = x.device
        t_tensor = torch.full((b,), t, device=device, dtype=torch.long)

        pred_noise = self.unet(x, t_tensor)

        # Dự đoán x_0 từ pred_noise
        sqrt_recip_alphas_cumprod_t = _extract(self.sqrt_recip_alphas_cumprod, t_tensor, x.shape)
        sqrt_recipm1_alphas_cumprod_t = _extract(self.sqrt_recipm1_alphas_cumprod, t_tensor, x.shape)
        pred_x0 = sqrt_recip_alphas_cumprod_t * x - sqrt_recipm1_alphas_cumprod_t * pred_noise

        if clip_denoised:
            pred_x0 = pred_x0.clamp(-1.0, 1.0)

        # Tính mean của posterior q(x_{t-1} | x_t, x_0)
        coef1 = _extract(self.posterior_mean_coef1, t_tensor, x.shape)
        coef2 = _extract(self.posterior_mean_coef2, t_tensor, x.shape)
        mean = coef1 * pred_x0 + coef2 * x

        if t == 0:
            return mean
        else:
            variance = _extract(self.posterior_variance_clipped, t_tensor, x.shape)
            noise = torch.randn_like(x)
            return mean + torch.sqrt(variance) * noise

    @torch.no_grad()
    def sample(
        self,
        n: int,
        y: torch.Tensor | None = None,
        device: str | torch.device = "cpu",
    ) -> torch.Tensor:
        """Sinh n ảnh mới từ Gaussian noise theo quy trình reverse diffusion.

        Trả về tensor shape (n, channels, image_size, image_size) với giá trị trong [-1, 1].
        """
        self.eval()
        device = torch.device(device)
        img = torch.randn((n, self.channels, self.image_size, self.image_size), device=device)

        for t in reversed(range(self.timesteps)):
            img = self.p_sample(img, t, clip_denoised=True)

        return img.clamp(-1.0, 1.0)
