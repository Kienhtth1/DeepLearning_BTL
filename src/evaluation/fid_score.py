"""[Phase 3/4 - TV2] Tính toán điểm FID (Fréchet Inception Distance - Heusel et al., 2017).

Hỗ trợ:
- Tính FID giữa thư mục ảnh sinh (fake_dir) và thư mục ảnh thật (real_dir: data/processed/<dataset>_<size>/fid_real/).
- Hàm tiện ích `evaluate_model_fid(...)` sinh trực tiếp N ảnh (mặc định 10,000) từ mô hình kế thừa `GenerativeModel` và tính FID.
- Tự động chuẩn hóa ảnh 1 kênh (Fashion-MNIST) -> 3 kênh RGB và resize về 299x299 cho Inception-v3.
"""
from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
from PIL import Image
from scipy import linalg
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from tqdm import tqdm

from src.models.base import GenerativeModel

IMG_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


class ImageFolderFlatDataset(Dataset):
    """Dataset đọc toàn bộ file ảnh trong một thư mục (không cần chia thư mục con theo class)."""

    def __init__(self, folder: str | Path, max_images: int | None = None):
        self.folder = Path(folder)
        if not self.folder.exists():
            raise FileNotFoundError(f"Không tìm thấy thư mục ảnh: {self.folder}")

        paths = sorted(
            p for p in self.folder.rglob("*") if p.is_file() and p.suffix.lower() in IMG_EXTENSIONS
        )
        if not paths:
            raise ValueError(f"Thư mục {self.folder} không chứa file ảnh hợp lệ nào.")

        if max_images is not None and max_images > 0:
            paths = paths[:max_images]
        self.paths = paths

        self.transform = transforms.Compose([
            transforms.Resize((299, 299), interpolation=transforms.InterpolationMode.BILINEAR, antialias=True),
            transforms.ToTensor(),  # [0, 1], shape (3, 299, 299)
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> torch.Tensor:
        with Image.open(self.paths[idx]) as img:
            # Tự động chuyển ảnh xám (L) hoặc RGBA sang RGB 3 kênh
            img = img.convert("RGB")
            return self.transform(img)


class InceptionV3FeatureExtractor(nn.Module):
    """Trích xuất vector đặc trưng 2048 chiều từ tầng pool3 của Inception-v3."""

    def __init__(self):
        super().__init__()
        weights = models.Inception_V3_Weights.DEFAULT
        net = models.inception_v3(weights=weights, transform_input=False)
        net.fc = nn.Identity()  # Bỏ lớp phân loại 1000 classes, giữ lại embedding 2048-D
        net.eval()
        self.net = net

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: tensor chuẩn hóa ImageNet, shape (B, 3, 299, 299)
        return self.net(x)


def preprocess_tensor_for_inception(images: torch.Tensor) -> torch.Tensor:
    """Chuyển batch tensor ảnh [-1, 1] (1 hoặc 3 kênh) sang tensor chuẩn hóa cho Inception-v3 (B, 3, 299, 299)."""
    # Đưa từ [-1, 1] về [0, 1]
    x = (images.clamp(-1.0, 1.0) + 1.0) * 0.5
    if x.size(1) == 1:
        x = x.repeat(1, 3, 1, 1)
    elif x.size(1) != 3:
        raise ValueError(f"Số kênh ảnh không hợp lệ cho Inception-v3: {x.size(1)}")

    if x.shape[-2:] != (299, 299):
        x = F.interpolate(x, size=(299, 299), mode="bilinear", align_corners=False)

    mean = torch.tensor([0.485, 0.456, 0.406], device=x.device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=x.device).view(1, 3, 1, 1)
    return (x - mean) / std


def calculate_frechet_distance(
    mu1: np.ndarray,
    sigma1: np.ndarray,
    mu2: np.ndarray,
    sigma2: np.ndarray,
    eps: float = 1e-6,
) -> float:
    """Tính khoảng cách Fréchet giữa 2 phân phối chuẩn đa biến N(mu1, sigma1) và N(mu2, sigma2).

    Công thức:
        FID = ||mu1 - mu2||^2 + Tr(sigma1 + sigma2 - 2 * sqrt(sigma1 @ sigma2))
    """
    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)
    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    diff = mu1 - mu2

    # Căn bậc hai ma trận tích hiệp phương sai
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Loại bỏ phần ảo cực nhỏ do sai số làm tròn dấu phẩy động
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError(f"Phần ảo quá lớn khi tính sqrtm ma trận hiệp phương sai: {m}")
        covmean = covmean.real

    tr_covmean = np.trace(covmean)
    fid = float(diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2.0 * tr_covmean)
    return max(0.0, fid)


@torch.no_grad()
def extract_features_from_folder(
    folder: str | Path,
    extractor: InceptionV3FeatureExtractor,
    device: torch.device,
    batch_size: int = 128,
    num_workers: int = 2,
    max_images: int | None = None,
    desc: str = "Trích xuất đặc trưng",
) -> tuple[np.ndarray, np.ndarray]:
    """Trích xuất toàn bộ vector 2048-D từ thư mục ảnh và trả về (mu, sigma)."""
    dataset = ImageFolderFlatDataset(folder, max_images=max_images)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
    )

    feats_list: list[np.ndarray] = []
    for batch in tqdm(loader, desc=desc, leave=False):
        batch = batch.to(device, non_blocking=True)
        feats = extractor(batch)
        feats_list.append(feats.cpu().numpy())

    all_feats = np.concatenate(feats_list, axis=0)
    mu = np.mean(all_feats, axis=0)
    sigma = np.cov(all_feats, rowvar=False)
    return mu, sigma


def compute_fid_from_folders(
    real_dir: str | Path,
    fake_dir: str | Path,
    device: str | torch.device | None = None,
    batch_size: int = 128,
    num_workers: int = 2,
    max_images: int | None = None,
    cache_real_stats: bool = True,
) -> float:
    """Tính điểm FID giữa thư mục ảnh thật (fid_real/) và thư mục ảnh sinh.

    Tự động cache file `fid_real_stats.npz` cạnh thư mục `real_dir` để các lần đo tiếp theo
    cho VAE / DCGAN / cDCGAN / DDPM chạy nhanh gấp đôi.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)

    real_path = Path(real_dir)
    fake_path = Path(fake_dir)

    extractor = InceptionV3FeatureExtractor().to(device)
    extractor.eval()

    # Kiểm tra cache thống kê của tập ảnh thật
    cache_file = real_path.parent / f"{real_path.name}_stats_{max_images or 'all'}.npz"
    if cache_real_stats and cache_file.exists():
        cached = np.load(cache_file)
        mu_real, sigma_real = cached["mu"], cached["sigma"]
    else:
        mu_real, sigma_real = extract_features_from_folder(
            real_path,
            extractor=extractor,
            device=device,
            batch_size=batch_size,
            num_workers=num_workers,
            max_images=max_images,
            desc="FID - Ảnh thật",
        )
        if cache_real_stats:
            np.savez(cache_file, mu=mu_real, sigma=sigma_real)

    mu_fake, sigma_fake = extract_features_from_folder(
        fake_path,
        extractor=extractor,
        device=device,
        batch_size=batch_size,
        num_workers=num_workers,
        max_images=max_images,
        desc="FID - Ảnh sinh",
    )

    return calculate_frechet_distance(mu_real, sigma_real, mu_fake, sigma_fake)


@torch.no_grad()
def evaluate_model_fid(
    model: GenerativeModel,
    real_dir: str | Path,
    n_samples: int = 10_000,
    batch_size: int = 128,
    device: str | torch.device | None = None,
    num_workers: int = 2,
    cache_real_stats: bool = True,
) -> float:
    """Sinh trực tiếp `n_samples` ảnh từ mô hình (không cần ghi ra ổ cứng) và tính FID."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)

    model.eval()
    model.to(device)

    extractor = InceptionV3FeatureExtractor().to(device)
    extractor.eval()

    real_path = Path(real_dir)
    cache_file = real_path.parent / f"{real_path.name}_stats_{n_samples}.npz"
    if cache_real_stats and cache_file.exists():
        cached = np.load(cache_file)
        mu_real, sigma_real = cached["mu"], cached["sigma"]
    else:
        mu_real, sigma_real = extract_features_from_folder(
            real_path,
            extractor=extractor,
            device=device,
            batch_size=batch_size,
            num_workers=num_workers,
            max_images=n_samples,
            desc="FID - Ảnh thật",
        )
        if cache_real_stats:
            np.savez(cache_file, mu=mu_real, sigma=sigma_real)

    # Sinh ảnh theo từng batch và trích xuất đặc trưng trực tiếp trên GPU
    feats_list: list[np.ndarray] = []
    remaining = n_samples
    pbar = tqdm(total=n_samples, desc=f"Sinh & đo FID ({model.__class__.__name__})", leave=False)
    while remaining > 0:
        curr_b = min(batch_size, remaining)
        fake_imgs = model.sample(n=curr_b, device=device)
        inp = preprocess_tensor_for_inception(fake_imgs)
        feats = extractor(inp)
        feats_list.append(feats.cpu().numpy())
        remaining -= curr_b
        pbar.update(curr_b)
    pbar.close()

    all_fake_feats = np.concatenate(feats_list, axis=0)
    mu_fake = np.mean(all_fake_feats, axis=0)
    sigma_fake = np.cov(all_fake_feats, rowvar=False)

    return calculate_frechet_distance(mu_real, sigma_real, mu_fake, sigma_fake)