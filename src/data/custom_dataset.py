"""Dataset cho ảnh tự thu thập (tranh dân gian, trang phục truyền thống...).

Cấu trúc thư mục hỗ trợ:

    data/custom/<ten_bo_anh>/
        dong_ho/      <- mỗi thư mục con là 1 lớp (dùng cho conditional generation)
            001.jpg
        hang_trong/
            ...

Nếu không có thư mục con, toàn bộ ảnh được gán nhãn 0.
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable

from PIL import Image, UnidentifiedImageError
from torch.utils.data import Dataset

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


class FolderImageDataset(Dataset):
    def __init__(self, root: str | Path, transform: Callable | None = None, min_size: int = 32):
        self.root = Path(root)
        if not self.root.exists():
            raise FileNotFoundError(
                f"Không tìm thấy {self.root}. Hãy đặt ảnh vào data/custom/<ten_bo_anh>/ "
                "(xem data/README.md)."
            )
        self.transform = transform

        subdirs = sorted(p for p in self.root.iterdir() if p.is_dir())
        self.classes = [p.name for p in subdirs] or ["all"]
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}

        files: list[tuple[Path, int]] = []
        if subdirs:
            for d in subdirs:
                files += [(f, self.class_to_idx[d.name]) for f in _list_images(d)]
        else:
            files = [(f, 0) for f in _list_images(self.root)]

        # Loại ảnh hỏng / quá nhỏ ngay từ đầu để không lỗi giữa lúc train
        self.samples, self.skipped = [], []
        for f, y in files:
            try:
                with Image.open(f) as im:
                    if min(im.size) < min_size:
                        self.skipped.append((f, "too_small"))
                        continue
            except (UnidentifiedImageError, OSError):
                self.skipped.append((f, "corrupted"))
                continue
            self.samples.append((f, y))

        if not self.samples:
            raise RuntimeError(f"Không có ảnh hợp lệ trong {self.root}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, label = self.samples[idx]
        img = Image.open(path)
        img.load()
        if self.transform is not None:
            img = self.transform(img)
        return img, label


def _list_images(d: Path) -> list[Path]:
    return sorted(f for f in d.rglob("*") if f.suffix.lower() in IMG_EXTS)
