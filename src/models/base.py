"""Giao diện chung cho 3 mô hình - để Phase 4 đánh giá bằng cùng một đoạn code.

TV1 (VAE), TV2 (DCGAN), TV3 (DDPM) kế thừa class này trong Phase 2.
Chỉ cần đảm bảo `sample()` trả ảnh trong [-1, 1], shape (n, C, H, W).
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import torch
from torch import nn


class GenerativeModel(nn.Module, ABC):
    def __init__(self, channels: int, image_size: int, num_classes: int | None = None):
        super().__init__()
        self.channels = channels
        self.image_size = image_size
        self.num_classes = num_classes

    @abstractmethod
    def training_loss(self, x: torch.Tensor, y: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        """Trả về dict loss, bắt buộc có key "loss" (DCGAN có thể trả loss_g / loss_d)."""

    @abstractmethod
    @torch.no_grad()
    def sample(self, n: int, y: torch.Tensor | None = None, device: str | torch.device = "cpu") -> torch.Tensor:
        """Sinh n ảnh trong [-1, 1]. `y` là nhãn lớp nếu làm conditional (Phase 3)."""

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())
