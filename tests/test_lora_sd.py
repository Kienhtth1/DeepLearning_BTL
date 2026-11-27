"""Unit tests offline cho module LoRA Stable Diffusion (không tải model từ HuggingFace)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from src.training.train_lora_sd import (
    DongHoStyleDataset,
    create_comparison_grid,
    plot_lora_loss_curve,
)
from src.utils import ROOT_DIR, load_config


def _save_dummy_image(path: Path, size: tuple[int, int] = (128, 96)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.random.randint(0, 256, (size[1], size[0], 3), dtype=np.uint8)
    Image.fromarray(arr).save(path)


def test_lora_sd_config_loads_and_has_required_fields():
    cfg = load_config(ROOT_DIR / "configs" / "lora_sd.yaml")

    assert cfg["seed"] == 42
    assert cfg["model"]["name"] == "lora_sd"
    assert cfg["model"]["pretrained_model"] == "stable-diffusion-v1-5/stable-diffusion-v1-5"
    assert cfg["data"]["name"] == "custom"
    assert cfg["data"]["custom_subdir"] == "tranh_dan_gian"
    assert cfg["data"]["resolution"] == 512

    assert cfg["lora"]["rank"] == 8
    assert cfg["lora"]["alpha"] == 8
    assert cfg["lora"]["target_modules"] == ["to_q", "to_k", "to_v", "to_out.0"]

    assert cfg["train"]["mixed_precision"] == "fp16"
    assert cfg["train"]["max_train_steps"] == 400
    assert len(cfg["prompt"]["eval_prompts"]) == 4
    assert cfg["inference"]["num_inference_steps"] == 30


def test_dong_ho_dataset_supports_subfolder_and_filters_invalid_files(tmp_path: Path):
    data_root = tmp_path / "tranh_dan_gian"
    for i in range(4):
        _save_dummy_image(data_root / "dong_ho" / f"tranh_{i}.jpg", size=(160, 120))

    # Thêm file lỗi / không phải ảnh / quá nhỏ để kiểm tra lọc an toàn.
    (data_root / "dong_ho" / "broken.jpg").write_bytes(b"not an image")
    (data_root / "dong_ho" / "notes.txt").write_text("metadata", encoding="utf-8")
    _save_dummy_image(data_root / "dong_ho" / "tiny.png", size=(16, 16))

    ds = DongHoStyleDataset(
        root=data_root,
        prompt="dongho style test prompt",
        resolution=512,
        center_crop=True,
        random_flip=True,
    )

    assert len(ds) == 4
    sample = ds[0]
    tensor = sample["pixel_values"]
    assert sample["prompt"] == "dongho style test prompt"
    assert tensor.shape == (3, 512, 512)
    assert tensor.min().item() >= -1.0001
    assert tensor.max().item() <= 1.0001


def test_dong_ho_dataset_supports_flat_directory(tmp_path: Path):
    flat_root = tmp_path / "dong-ho-flat"
    for i in range(3):
        _save_dummy_image(flat_root / f"img_{i}.png", size=(100, 140))
    (flat_root / "corrupt.png").write_bytes(b"corrupt")
    (flat_root / "readme.md").write_text("ignore", encoding="utf-8")

    ds = DongHoStyleDataset(
        root=flat_root,
        prompt="dongho flat prompt",
        resolution=512,
        center_crop=True,
        random_flip=False,
    )

    assert len(ds) == 3
    item = ds[1]
    assert item["pixel_values"].shape == (3, 512, 512)
    assert item["pixel_values"].min().item() >= -1.0001
    assert item["pixel_values"].max().item() <= 1.0001


def test_plot_lora_loss_curve_and_comparison_grid_create_valid_images(tmp_path: Path):
    history = [
        {"step": s, "loss": 0.25 / (1.0 + 0.05 * s), "lr": 1e-4, "elapsed_sec": float(s)}
        for s in range(1, 31)
    ]
    curve_path = plot_lora_loss_curve(history, tmp_path / "outputs" / "lora_sd_loss_curve.png", ma_window=10)
    assert curve_path.is_file()
    with Image.open(curve_path) as im:
        assert im.size[0] > 0 and im.size[1] > 0

    base_imgs = [Image.new("RGB", (128, 128), color=(180, 180, 180)) for _ in range(2)]
    lora_imgs = [Image.new("RGB", (128, 128), color=(210, 160, 90)) for _ in range(2)]
    prompts = [
        "dongho style, a rooster and hens in a village yard",
        "dongho style, a fat cat sitting near lotus flowers",
    ]
    grid_path = create_comparison_grid(
        base_images=base_imgs,
        lora_images=lora_imgs,
        prompts=prompts,
        save_path=tmp_path / "outputs" / "samples" / "lora_sd" / "base_vs_lora.png",
    )
    assert grid_path.is_file()
    with Image.open(grid_path) as im:
        assert im.size[0] > 0 and im.size[1] > 0
