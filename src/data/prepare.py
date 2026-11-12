"""Phase 1 - chuẩn bị & kiểm tra dữ liệu (chạy 1 lần trước khi train).

    python main.py data                                  # Fashion-MNIST 32x32
    python main.py data --config configs/celeba.yaml     # CelebA 64x64
    python main.py data --set data.name=custom data.custom_subdir=tranh_dan_gian

Việc script làm:
1. Tải dữ liệu (Fashion-MNIST tự tải) và tạo DataLoader train / val / test.
2. Kiểm tra shape, khoảng giá trị [-1, 1], phân bố lớp -> in ra + lưu JSON.
3. Lưu lưới ảnh mẫu: outputs/data_samples/<dataset>_<size>.png
4. Export ảnh thật của tập test ra PNG để Phase 3 tính FID:
   data/processed/<dataset>_<size>/fid_real/
"""
from __future__ import annotations

import json
import time
from collections import Counter
from pathlib import Path

import torch
from torchvision.utils import save_image
from tqdm import tqdm

from src.utils import ROOT_DIR, save_image_grid, set_seed

from .data_loader import get_dataloader
from .transforms import denormalize


def run(cfg: dict, n_fid: int = 10_000, skip_fid: bool = False) -> dict:
    set_seed(cfg.get("seed", 42))
    d = dict(cfg["data"])
    name = d.pop("name")
    tag = f"{name}{'_' + d['custom_subdir'] if name == 'custom' and d.get('custom_subdir') else ''}_{d['image_size']}"
    out_dir = ROOT_DIR / "data" / "processed" / tag
    out_dir.mkdir(parents=True, exist_ok=True)

    report: dict = {"dataset": name, "tag": tag, "splits": {}}
    loaders = {}
    for split in ("train", "val", "test"):
        loader, info = get_dataloader(name, split=split, drop_last=False, **d)
        loaders[split] = loader
        report["splits"][split] = info.num_samples
    report["info"] = info.to_dict()
    report["info"].pop("num_samples")
    _print_header(report)

    # ---- 1 batch: kiểm tra shape & khoảng giá trị ----
    t0 = time.time()
    x, y = next(iter(loaders["train"]))
    report["batch_shape"] = list(x.shape)
    report["batch_load_sec"] = round(time.time() - t0, 3)
    report["value_range"] = [round(x.min().item(), 4), round(x.max().item(), 4)]
    assert x.shape[1:] == (info.channels, info.image_size, info.image_size), "Sai shape!"
    assert x.min() >= -1.0001 and x.max() <= 1.0001, "Ảnh chưa được chuẩn hoá về [-1, 1]!"
    print(f"Batch shape        : {tuple(x.shape)}   (load {report['batch_load_sec']}s)")
    print(f"Khoảng giá trị     : {report['value_range']}  (kỳ vọng ≈ [-1, 1])")

    # ---- mean/std theo kênh (ước lượng trên tối đa 50 batch) ----
    s, s2, n = 0.0, 0.0, 0
    labels = Counter()
    for i, (xb, yb) in enumerate(loaders["train"]):
        if i < 50:
            xb = denormalize(xb)
            s += xb.sum(dim=(0, 2, 3))
            s2 += (xb**2).sum(dim=(0, 2, 3))
            n += xb.numel() // xb.shape[1]
        labels.update(torch.as_tensor(yb).tolist())
        if info.num_classes is None and i >= 50:
            break
    mean = s / n
    std = (s2 / n - mean**2).sqrt()
    report["pixel_mean_[0,1]"] = [round(v, 4) for v in mean.tolist()]
    report["pixel_std_[0,1]"] = [round(v, 4) for v in std.tolist()]
    print(f"Mean / std pixel   : {report['pixel_mean_[0,1]']} / {report['pixel_std_[0,1]']}")
    if info.num_classes:
        names = info.class_names or list(range(info.num_classes))
        report["class_distribution_train"] = {str(names[k]): v for k, v in sorted(labels.items())}
        print("Phân bố lớp (train):")
        for k, v in report["class_distribution_train"].items():
            print(f"   {k:<14} {v}")

    # ---- lưới ảnh mẫu ----
    grid_path = save_image_grid(x[:64], ROOT_DIR / "outputs" / "data_samples" / f"{tag}.png")
    report["sample_grid"] = str(grid_path.relative_to(ROOT_DIR))
    print(f"Lưới ảnh mẫu       : {report['sample_grid']}")

    # ---- export ảnh thật cho FID ----
    if not skip_fid:
        report["fid_real_dir"], report["fid_real_count"] = _export_real(loaders["test"], out_dir / "fid_real", n_fid)
        print(f"Ảnh thật cho FID   : {report['fid_real_count']} ảnh -> {report['fid_real_dir']}")

    with open(out_dir / "dataset_info.json", "w", encoding="utf-8") as fh:
        json.dump(report, fh, ensure_ascii=False, indent=2)
    print(f"Báo cáo dữ liệu    : {(out_dir / 'dataset_info.json').relative_to(ROOT_DIR)}")
    print("Phase 1 OK ✔")
    return report


def _export_real(loader, out: Path, n_max: int) -> tuple[str, int]:
    out.mkdir(parents=True, exist_ok=True)
    existing = len(list(out.glob("*.png")))
    target = min(n_max, len(loader.dataset))
    if existing >= target:
        return str(out.relative_to(ROOT_DIR)), existing
    count = 0
    with tqdm(total=target, desc="Export FID real") as bar:
        for xb, _ in loader:
            for img in xb:
                if count >= target:
                    break
                save_image(denormalize(img), out / f"{count:05d}.png")
                count += 1
                bar.update(1)
            if count >= target:
                break
    return str(out.relative_to(ROOT_DIR)), count


def _print_header(r: dict) -> None:
    i = r["info"]
    print("=" * 60)
    print(f"Dataset            : {r['dataset']}  ({i['channels']}x{i['image_size']}x{i['image_size']})")
    print(f"Số ảnh             : train={r['splits']['train']}  val={r['splits']['val']}  test={r['splits']['test']}")
    print(f"Số lớp             : {i['num_classes'] or 'không có (unconditional)'}")
    print("=" * 60)
