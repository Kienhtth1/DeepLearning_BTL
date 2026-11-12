"""Data loader dùng chung cho cả nhóm - Phase 1.

Mọi mô hình (VAE / DCGAN / DDPM) đều lấy dữ liệu qua MỘT hàm duy nhất:

    from src.data import get_dataloader
    loader, info = get_dataloader("fashion_mnist", split="train", batch_size=128)
    for x, y in loader:   # x: (B, C, H, W) trong [-1, 1], y: nhãn lớp (int)
        ...

`info` cho biết channels / image_size / num_classes để khởi tạo mô hình
mà không phải hard-code.

Bộ dữ liệu hỗ trợ:
- fashion_mnist : tự tải qua torchvision, 1 kênh, resize 28 -> 32 (để U-Net / DCGAN chia 2 đẹp).
- celeba        : 3 kênh, crop giữa 140 rồi resize 64 (chuẩn DCGAN). Tải thủ công, xem data/README.md.
- custom        : ảnh tự thu thập trong data/custom/<subdir>/, 3 kênh, 64x64.
"""
from __future__ import annotations

import csv
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets

from .custom_dataset import FolderImageDataset
from .transforms import build_transform

ROOT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_DATA_ROOT = ROOT_DIR / "data"


# --------------------------------------------------------------------------- #
# Thông số mặc định cho từng bộ dữ liệu
# --------------------------------------------------------------------------- #
@dataclass
class DatasetInfo:
    name: str
    channels: int
    image_size: int
    num_classes: int | None          # None = không có nhãn lớp (unconditional)
    class_names: list[str] | None = None
    center_crop: int | None = None
    hflip: bool = False
    num_samples: int | None = None   # được điền sau khi load

    def to_dict(self) -> dict:
        return asdict(self)


FASHION_CLASSES = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot",
]

PRESETS: dict[str, DatasetInfo] = {
    "fashion_mnist": DatasetInfo("fashion_mnist", 1, 32, 10, FASHION_CLASSES),
    "celeba": DatasetInfo("celeba", 3, 64, None, None, center_crop=140, hflip=True),
    "custom": DatasetInfo("custom", 3, 64, None, None, hflip=True),
}


# --------------------------------------------------------------------------- #
# CelebA
# --------------------------------------------------------------------------- #
class CelebADataset(Dataset):
    """Đọc CelebA từ thư mục đã tải sẵn (bản Kaggle hoặc bản gốc).

    Cần có:  data/raw/celeba/img_align_celeba/*.jpg
    Tuỳ chọn: list_eval_partition.(txt|csv)  -> chia train/val/test chuẩn
              list_attr_celeba.(txt|csv)     -> dùng 1 thuộc tính làm nhãn (vd "Smiling")
    """

    SPLIT_ID = {"train": 0, "val": 1, "test": 2}

    def __init__(self, root: Path, split: str = "train", transform=None, label_attr: str | None = None):
        self.root = Path(root)
        img_dir = self.root / "img_align_celeba"
        # Bản Kaggle bị lồng 2 lớp: img_align_celeba/img_align_celeba/
        if (img_dir / "img_align_celeba").is_dir():
            img_dir = img_dir / "img_align_celeba"
        if not img_dir.is_dir():
            raise FileNotFoundError(
                f"Chưa có CelebA tại {img_dir}. Xem hướng dẫn tải trong data/README.md"
            )
        self.img_dir = img_dir
        self.transform = transform

        files = sorted(p.name for p in img_dir.glob("*.jpg"))
        partition = _read_celeba_table(self.root, "list_eval_partition")
        if partition and split != "all":
            sid = self.SPLIT_ID[split]
            files = [f for f in files if int(partition.get(f, ["-1"])[0]) == sid]
        elif split != "all":
            files = _random_split(files, split)
        self.files = files

        self.labels = None
        if label_attr:
            attrs = _read_celeba_table(self.root, "list_attr_celeba", header=True)
            if not attrs:
                raise FileNotFoundError("Cần list_attr_celeba để dùng label_attr")
            header = attrs.pop("__header__")
            j = header.index(label_attr)
            self.labels = [int(int(attrs[f][j]) > 0) for f in self.files]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img = Image.open(self.img_dir / self.files[idx])
        img.load()
        if self.transform:
            img = self.transform(img)
        return img, (self.labels[idx] if self.labels is not None else 0)


def _read_celeba_table(root: Path, stem: str, header: bool = False) -> dict:
    """Đọc file txt (bản gốc, cách nhau bởi khoảng trắng) hoặc csv (bản Kaggle)."""
    out: dict = {}
    for path in [root / f"{stem}.csv", root / f"{stem}.txt"]:
        if not path.exists():
            continue
        with open(path) as fh:
            if path.suffix == ".csv":
                rows = list(csv.reader(fh))
                if header:
                    out["__header__"] = rows[0][1:]
                body = rows[1:]
            else:
                lines = [ln.split() for ln in fh.read().strip().splitlines()]
                if header:  # dòng 1 = số lượng, dòng 2 = tên thuộc tính
                    out["__header__"] = lines[1]
                    lines = lines[2:]
                body = lines
        for r in body:
            out[r[0]] = r[1:]
        return out
    return out


# --------------------------------------------------------------------------- #
# API chính
# --------------------------------------------------------------------------- #
def get_dataset(
    name: str,
    split: str = "train",
    data_root: str | Path = DEFAULT_DATA_ROOT,
    image_size: int | None = None,
    custom_subdir: str = "",
    label_attr: str | None = None,
    augment: bool | None = None,
    download: bool = True,
) -> tuple[Dataset, DatasetInfo]:
    """Trả về (dataset, info). split ∈ {"train", "val", "test", "all"}."""
    if name not in PRESETS:
        raise ValueError(f"Dataset '{name}' không hỗ trợ. Chọn một trong {list(PRESETS)}")
    if split not in {"train", "val", "test", "all"}:
        raise ValueError("split phải là train / val / test / all")

    base = PRESETS[name]
    info = DatasetInfo(**base.to_dict())
    if image_size:
        info.image_size = image_size
    use_flip = info.hflip if augment is None else augment
    tf = build_transform(info.image_size, info.channels, info.center_crop, hflip=use_flip and split == "train")

    data_root = Path(data_root)
    if name == "fashion_mnist":
        raw = data_root / "raw"
        if split == "all":
            ds = torch.utils.data.ConcatDataset([
                datasets.FashionMNIST(raw, train=True, transform=tf, download=download),
                datasets.FashionMNIST(raw, train=False, transform=tf, download=download),
            ])
        elif split == "test":
            ds = datasets.FashionMNIST(raw, train=False, transform=tf, download=download)
        else:
            # train gốc 60k -> chia 55k train / 5k val (cố định seed)
            full = datasets.FashionMNIST(raw, train=True, transform=tf, download=download)
            ds = Subset(full, _random_split(list(range(len(full))), split))

    elif name == "celeba":
        ds = CelebADataset(data_root / "raw" / "celeba", split, tf, label_attr)
        if label_attr:
            info.num_classes, info.class_names = 2, [f"not_{label_attr}", label_attr]

    else:  # custom
        full = FolderImageDataset(data_root / "custom" / custom_subdir, transform=tf)
        if len(full.classes) > 1:
            info.num_classes, info.class_names = len(full.classes), full.classes
        idx = list(range(len(full)))
        ds = full if split == "all" else Subset(full, _random_split(idx, split))

    info.num_samples = len(ds)
    return ds, info


def get_dataloader(
    name: str,
    split: str = "train",
    batch_size: int = 128,
    num_workers: int = 2,
    subset_size: int | None = None,
    seed: int = 42,
    shuffle: bool | None = None,
    drop_last: bool | None = None,
    **dataset_kwargs,
) -> tuple[DataLoader, DatasetInfo]:
    """Tạo DataLoader chuẩn cho cả nhóm.

    subset_size: chỉ lấy N ảnh đầu (đã xáo cố định) - dùng để debug nhanh trên CPU/Colab.
    """
    ds, info = get_dataset(name, split=split, **dataset_kwargs)
    if subset_size and subset_size < len(ds):
        g = torch.Generator().manual_seed(seed)
        idx = torch.randperm(len(ds), generator=g)[:subset_size].tolist()
        ds = Subset(ds, idx)
        info.num_samples = len(ds)

    is_train = split == "train"
    g = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=is_train if shuffle is None else shuffle,
        drop_last=is_train if drop_last is None else drop_last,   # GAN/BatchNorm thích batch đủ size
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
        worker_init_fn=_seed_worker,
        generator=g,
    )
    return loader, info


def get_dataloader_from_config(cfg: dict, split: str = "train") -> tuple[DataLoader, DatasetInfo]:
    """Đọc mục `data:` trong configs/*.yaml."""
    d = dict(cfg["data"])
    name = d.pop("name")
    return get_dataloader(name, split=split, **d)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
VAL_RATIO = 0.1
SPLIT_SEED = 1234   # KHÔNG đổi - để cả nhóm có cùng tập train/val


def _random_split(items: list, split: str) -> list:
    if split == "all":
        return items
    rng = random.Random(SPLIT_SEED)
    items = list(items)
    rng.shuffle(items)
    n_val = max(1, int(len(items) * VAL_RATIO))
    if split == "train":
        return sorted(items[n_val:])
    return sorted(items[:n_val])  # val và test dùng chung phần giữ lại


def _seed_worker(worker_id: int) -> None:
    s = torch.initial_seed() % 2**32
    np.random.seed(s)
    random.seed(s)
