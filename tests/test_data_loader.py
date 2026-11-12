"""Kiểm thử Phase 1:  pytest -q"""
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from src.data import build_transform, denormalize, get_dataloader, get_dataset
from src.data.data_loader import ROOT_DIR


@pytest.fixture
def fake_custom(tmp_path: Path) -> Path:
    """Tạo bộ ảnh giả 2 lớp + 1 ảnh hỏng + 1 ảnh quá nhỏ."""
    root = tmp_path / "custom" / "fake"
    for cls in ("dong_ho", "hang_trong"):
        (root / cls).mkdir(parents=True)
        for i in range(20):
            arr = np.random.randint(0, 256, (80, 100, 3), dtype=np.uint8)
            Image.fromarray(arr).save(root / cls / f"{i}.jpg")
    (root / "dong_ho" / "broken.jpg").write_bytes(b"not an image")
    Image.fromarray(np.zeros((10, 10, 3), dtype=np.uint8)).save(root / "dong_ho" / "tiny.png")
    return tmp_path


def test_transform_range_and_shape():
    img = Image.fromarray(np.random.randint(0, 256, (50, 70, 3), dtype=np.uint8))
    for ch in (1, 3):
        x = build_transform(32, ch)(img)
        assert x.shape == (ch, 32, 32)
        assert x.min() >= -1 and x.max() <= 1
    assert torch.allclose(denormalize(torch.tensor([-1.0, 0.0, 1.0])), torch.tensor([0.0, 0.5, 1.0]))


def test_custom_dataset_and_split(fake_custom):
    ds_all, info = get_dataset("custom", "all", data_root=fake_custom, custom_subdir="fake")
    assert info.num_classes == 2 and info.class_names == ["dong_ho", "hang_trong"]
    assert len(ds_all) == 40  # ảnh hỏng & quá nhỏ đã bị loại
    tr, _ = get_dataset("custom", "train", data_root=fake_custom, custom_subdir="fake")
    va, _ = get_dataset("custom", "val", data_root=fake_custom, custom_subdir="fake")
    assert len(tr) + len(va) == 40
    assert not set(tr.indices) & set(va.indices)  # train/val không trùng


def test_dataloader_batch(fake_custom):
    loader, info = get_dataloader("custom", "train", batch_size=8, num_workers=0,
                                  data_root=fake_custom, custom_subdir="fake")
    x, y = next(iter(loader))
    assert x.shape == (8, 3, 64, 64) and y.shape == (8,)


def test_split_is_deterministic(fake_custom):
    a, _ = get_dataset("custom", "train", data_root=fake_custom, custom_subdir="fake")
    b, _ = get_dataset("custom", "train", data_root=fake_custom, custom_subdir="fake")
    assert a.indices == b.indices


@pytest.mark.skipif(not (ROOT_DIR / "data/raw/FashionMNIST").exists(),
                    reason="Chưa tải Fashion-MNIST (chạy: python main.py data)")
def test_fashion_mnist():
    loader, info = get_dataloader("fashion_mnist", "train", batch_size=16, num_workers=0, download=False)
    x, y = next(iter(loader))
    assert x.shape == (16, 1, 32, 32) and info.num_classes == 10
    assert info.num_samples == 54000
