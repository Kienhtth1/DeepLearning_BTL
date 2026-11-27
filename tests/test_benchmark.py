"""Unit tests cho src/evaluation/benchmark.py (chạy offline với model giả lập và mock LPIPS)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
import torch.nn as nn

from src.evaluation import (
    benchmark_model,
    compute_lpips_diversity,
    measure_generation_speed,
    measure_model_cost,
)
from src.models.base import GenerativeModel


class DummyGenerativeModel(GenerativeModel):
    """Mô hình giả lập kế thừa GenerativeModel để test benchmark nhanh trên CPU."""

    def __init__(self, channels: int = 1, image_size: int = 32, num_classes: int | None = 10):
        super().__init__(channels=channels, image_size=image_size, num_classes=num_classes)
        self.net = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.sample_calls: list[int] = []

    def training_loss(self, x: torch.Tensor, y: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        pred = self.net(x)
        return {"loss": torch.mean((pred - x) ** 2)}

    @torch.no_grad()
    def sample(
        self,
        n: int,
        y: torch.Tensor | None = None,
        device: str | torch.device = "cpu",
    ) -> torch.Tensor:
        self.sample_calls.append(n)
        return torch.rand((n, self.channels, self.image_size, self.image_size), device=device) * 2.0 - 1.0


class FakeLPIPS(nn.Module):
    """Mock LPIPS đo khoảng cách L1 trung bình trên từng cặp ảnh đã chuẩn hoá 3 kênh."""

    def __init__(self) -> None:
        super().__init__()
        self.input_shapes: list[tuple[int, ...]] = []
        self.value_ranges: list[tuple[float, float]] = []

    def forward(self, img0: torch.Tensor, img1: torch.Tensor) -> torch.Tensor:
        assert img0.shape == img1.shape
        self.input_shapes.append(tuple(img0.shape))
        self.value_ranges.append((float(img0.min().item()), float(img0.max().item())))
        diff = torch.abs(img0 - img1).mean(dim=(1, 2, 3))
        return diff.view(-1, 1, 1, 1)


def test_measure_generation_speed_counts_warmup_and_metrics():
    model = DummyGenerativeModel(channels=1, image_size=32)
    metrics = measure_generation_speed(
        model=model,
        n_images=10,
        batch_size=4,
        warmup_batches=2,
        device="cpu",
    )

    # 2 lần warm-up (4 ảnh mỗi lần) + 3 batch đo thực tế (4 + 4 + 2)
    assert model.sample_calls == [4, 4, 4, 4, 2]
    assert metrics["device"] == "cpu"
    assert metrics["speed_n_images"] == 10
    assert metrics["batch_size"] == 4
    assert metrics["total_gen_time_sec"] > 0
    assert metrics["images_per_sec"] > 0
    assert metrics["sec_per_image"] > 0
    assert metrics["peak_vram_mb"] == 0.0


def test_measure_model_cost_reads_checkpoint_and_train_log(tmp_path: Path):
    model = DummyGenerativeModel(channels=1, image_size=32)
    ckpt_file = tmp_path / "dummy_best.pth"
    ckpt_file.write_bytes(b"0" * (1024 * 512))  # 0.5 MB

    log_file = tmp_path / "dummy_train_log.json"
    log_payload = {
        "model": "dummy",
        "epochs": 2,
        "completed_epochs": 2,
        "total_time_sec": 30.0,
        "history": [
            {"epoch": 1, "duration_sec": 14.0},
            {"epoch": 2, "duration_sec": 16.0},
        ],
    }
    log_file.write_text(json.dumps(log_payload), encoding="utf-8")

    cost = measure_model_cost(
        model=model,
        model_name="dummy",
        checkpoint_path=ckpt_file,
        train_log_path=log_file,
    )

    assert cost["num_params"] == model.num_params()
    assert cost["checkpoint_size_mb"] == pytest.approx(0.5, abs=1e-3)
    assert cost["completed_epochs"] == 2
    assert cost["total_time_sec"] == 30.0
    assert cost["avg_epoch_sec"] == 15.0


def test_compute_lpips_diversity_expands_grayscale_and_resizes():
    fake_lpips = FakeLPIPS()
    grayscale_images = torch.rand(12, 1, 32, 32) * 2.0 - 1.0

    result = compute_lpips_diversity(
        images=grayscale_images,
        n_pairs=20,
        batch_size=8,
        min_size=64,
        seed=42,
        device="cpu",
        lpips_fn=fake_lpips,
    )

    assert result["lpips_n_images"] == 12
    assert result["lpips_n_pairs"] == 20
    assert result["lpips_diversity"] > 0.0
    assert result["lpips_std"] >= 0.0
    # 20 cặp chia batch_size=8 -> 3 batch (8, 8, 4), tất cả đều được ép 3 kênh và resize >= 64x64
    assert fake_lpips.input_shapes == [(8, 3, 64, 64), (8, 3, 64, 64), (4, 3, 64, 64)]
    for vmin, vmax in fake_lpips.value_ranges:
        assert vmin >= -1.0 and vmax <= 1.0


def test_compute_lpips_diversity_requires_at_least_two_images():
    with pytest.raises(ValueError):
        compute_lpips_diversity(images=torch.zeros(1, 3, 64, 64), lpips_fn=FakeLPIPS())


def test_benchmark_model_reuses_generated_images_and_saves_json(tmp_path: Path):
    model = DummyGenerativeModel(channels=3, image_size=64)
    fake_lpips = FakeLPIPS()

    ckpt_file = tmp_path / "dummy_best.pth"
    torch.save({"model_state_dict": model.state_dict()}, ckpt_file)

    log_file = tmp_path / "dummy_train_log.json"
    log_file.write_text(
        json.dumps({"completed_epochs": 1, "total_time_sec": 12.5, "history": [{"epoch": 1, "duration_sec": 12.5}]}),
        encoding="utf-8",
    )
    out_file = tmp_path / "outputs" / "dummy_benchmark.json"

    pre_generated = torch.rand(10, 3, 64, 64) * 2.0 - 1.0
    result = benchmark_model(
        model=model,
        model_name="dummy",
        checkpoint_path=ckpt_file,
        train_log_path=log_file,
        generated_images=pre_generated,
        n_speed_images=8,
        n_lpips_pairs=16,
        batch_size=4,
        warmup_batches=1,
        seed=42,
        device="cpu",
        lpips_fn=fake_lpips,
        output_path=out_file,
    )

    # Khi truyền sẵn generated_images, model.sample chỉ được gọi trong measure_generation_speed (1 warmup + 2 batch)
    assert model.sample_calls == [4, 4, 4]
    assert out_file.is_file()

    saved = json.loads(out_file.read_text(encoding="utf-8"))
    assert saved["model"] == "dummy"
    assert saved["num_params"] == model.num_params()
    assert saved["total_time_sec"] == 12.5
    assert saved["avg_epoch_sec"] == 12.5
    assert saved["speed_n_images"] == 8
    assert saved["lpips_n_images"] == 10
    assert saved["lpips_n_pairs"] == 16
    assert saved["images_per_sec"] > 0
    assert result["lpips_diversity"] == saved["lpips_diversity"]
