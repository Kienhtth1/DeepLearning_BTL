"""Tiền xử lý ảnh dùng chung cho cả 3 mô hình (VAE, DCGAN, DDPM).

Quy ước của cả nhóm:
- Ảnh đầu vào mô hình luôn là tensor float, shape (C, H, W), giá trị trong [-1, 1].
- Kích thước H = W = image_size (mặc định 32 cho Fashion-MNIST, 64 cho CelebA).
- Muốn hiển thị / lưu ảnh / tính FID thì gọi `denormalize` để đưa về [0, 1].

Giữ chung một quy ước giúp phần đánh giá (Phase 3-4) so sánh công bằng giữa 3 mô hình.
"""
from __future__ import annotations

import torch
from torchvision import transforms as T


def build_transform(
    image_size: int,
    channels: int,
    center_crop: int | None = None,
    hflip: bool = False,
) -> T.Compose:
    """Tạo pipeline transform chuẩn.

    Args:
        image_size: cạnh ảnh đầu ra (vuông).
        channels: 1 (ảnh xám) hoặc 3 (RGB). Ảnh sẽ được ép về đúng số kênh.
        center_crop: cắt giữa trước khi resize (CelebA dùng 140 để bỏ nền).
        hflip: lật ngang ngẫu nhiên - chỉ bật cho tập train, KHÔNG bật cho Fashion-MNIST.
    """
    ops: list = []
    ops.append(T.Grayscale(num_output_channels=1) if channels == 1 else T.Lambda(_to_rgb))
    if center_crop:
        ops.append(T.CenterCrop(center_crop))
    ops += [
        T.Resize(image_size, interpolation=T.InterpolationMode.BILINEAR, antialias=True),
        T.CenterCrop(image_size),
    ]
    if hflip:
        ops.append(T.RandomHorizontalFlip(p=0.5))
    ops += [
        T.ToTensor(),                                  # [0, 1]
        T.Normalize([0.5] * channels, [0.5] * channels),  # [-1, 1]
    ]
    return T.Compose(ops)


def _to_rgb(img):
    return img.convert("RGB") if img.mode != "RGB" else img


def denormalize(x: torch.Tensor) -> torch.Tensor:
    """[-1, 1] -> [0, 1], dùng trước khi lưu ảnh / vẽ / tính FID."""
    return ((x + 1.0) / 2.0).clamp(0.0, 1.0)


def to_uint8(x: torch.Tensor) -> torch.Tensor:
    """[-1, 1] -> uint8 [0, 255] (định dạng mà torchmetrics FID cần)."""
    return (denormalize(x) * 255).round().to(torch.uint8)
