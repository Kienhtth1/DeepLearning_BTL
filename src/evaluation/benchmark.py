"""[Phase 3 - TV3] Đánh giá tốc độ sinh, chi phí tính toán và độ đa dạng (LPIPS).

Dùng chung cho các mô hình kế thừa `src.models.base.GenerativeModel` (VAE, DCGAN, DDPM).
Có thể gọi trực tiếp như module hoặc import các hàm vào notebook/script Phase 4:

    python -m src.evaluation.benchmark --config configs/ddpm.yaml --ckpt checkpoints/ddpm_best.pth
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from src.models.base import GenerativeModel
from src.utils import ROOT_DIR, get_device, load_config, set_seed


def resolve_project_path(path: str | Path) -> Path:
    """Chuyển đường dẫn tương đối thành tuyệt đối theo ROOT_DIR."""
    p = Path(path)
    return p if p.is_absolute() else ROOT_DIR / p


def _format_rel_path(path: Path) -> str:
    """Định dạng đường dẫn dạng POSIX tương đối với ROOT_DIR nếu nằm trong dự án."""
    try:
        return path.resolve().relative_to(ROOT_DIR.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def _load_torch_checkpoint(ckpt_path: Path, map_location: str | torch.device) -> dict[str, Any]:
    """Đọc checkpoint PyTorch tương thích cả PyTorch cũ và PyTorch >= 2.6."""
    try:
        return torch.load(ckpt_path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(ckpt_path, map_location=map_location)


def measure_generation_speed(
    model: GenerativeModel,
    n_images: int = 64,
    batch_size: int = 16,
    warmup_batches: int = 1,
    device: str | torch.device | None = None,
    y: torch.Tensor | None = None,
) -> dict[str, Any]:
    """Đo tốc độ sinh ảnh và peak VRAM của một GenerativeModel.

    Thực hiện warm-up trước khi bấm giờ, đồng bộ CUDA (nếu dùng GPU) và chạy trong
    `torch.inference_mode()`.
    """
    if n_images <= 0:
        raise ValueError(f"n_images phải > 0, nhận được {n_images}")
    if batch_size <= 0:
        raise ValueError(f"batch_size phải > 0, nhận được {batch_size}")
    if warmup_batches < 0:
        raise ValueError(f"warmup_batches phải >= 0, nhận được {warmup_batches}")

    resolved_device = torch.device(device) if device is not None else get_device()
    model = model.to(resolved_device)
    model.eval()

    is_cuda = resolved_device.type == "cuda"
    if y is not None:
        y = y.to(resolved_device)
        if len(y) < n_images:
            raise ValueError(f"Số lượng nhãn y ({len(y)}) nhỏ hơn n_images ({n_images})")

    with torch.inference_mode():
        # Warm-up để ổn định kernel CUDA và allocator trước khi đo.
        warmup_n = min(batch_size, n_images)
        warmup_y = y[:warmup_n] if y is not None else None
        for _ in range(warmup_batches):
            _ = model.sample(n=warmup_n, y=warmup_y, device=resolved_device)

        if is_cuda:
            torch.cuda.synchronize(resolved_device)
            torch.cuda.reset_peak_memory_stats(resolved_device)

        start = time.perf_counter()
        generated = 0
        while generated < n_images:
            cur_bs = min(batch_size, n_images - generated)
            cur_y = y[generated : generated + cur_bs] if y is not None else None
            _ = model.sample(n=cur_bs, y=cur_y, device=resolved_device)
            generated += cur_bs

        if is_cuda:
            torch.cuda.synchronize(resolved_device)

        elapsed = max(time.perf_counter() - start, 1e-9)
        peak_vram_mb = (
            round(torch.cuda.max_memory_allocated(resolved_device) / (1024**2), 2)
            if is_cuda
            else 0.0
        )

    return {
        "device": str(resolved_device),
        "speed_n_images": int(n_images),
        "batch_size": int(batch_size),
        "total_gen_time_sec": round(elapsed, 4),
        "images_per_sec": round(n_images / elapsed, 4),
        "sec_per_image": round(elapsed / n_images, 6),
        "peak_vram_mb": peak_vram_mb,
    }


def _find_default_checkpoint(model_name: str) -> Path | None:
    """Tìm file checkpoint mặc định theo các quy ước đặt tên của nhóm."""
    candidates = [
        ROOT_DIR / "checkpoints" / f"{model_name}_best.pth",
        ROOT_DIR / "checkpoints" / model_name / f"{model_name}_best.pth",
        ROOT_DIR / "checkpoints" / model_name / "best.pth",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


def measure_model_cost(
    model: GenerativeModel,
    model_name: str | None = None,
    checkpoint_path: str | Path | None = None,
    train_log_path: str | Path | None = None,
) -> dict[str, Any]:
    """Đo chi phí mô hình: số tham số, kích thước checkpoint (MB) và thời gian huấn luyện."""
    num_params = int(model.num_params())

    resolved_ckpt: Path | None = None
    if checkpoint_path is not None:
        resolved_ckpt = resolve_project_path(checkpoint_path)
    elif model_name:
        resolved_ckpt = _find_default_checkpoint(model_name)

    checkpoint_size_mb: float | None = None
    ckpt_str: str | None = None
    if resolved_ckpt is not None and resolved_ckpt.is_file():
        checkpoint_size_mb = round(resolved_ckpt.stat().st_size / (1024**2), 3)
        ckpt_str = _format_rel_path(resolved_ckpt)

    resolved_log: Path | None = None
    if train_log_path is not None:
        resolved_log = resolve_project_path(train_log_path)
    elif model_name:
        resolved_log = ROOT_DIR / "outputs" / f"{model_name}_train_log.json"

    total_time_sec: float | None = None
    avg_epoch_sec: float | None = None
    completed_epochs: int | None = None
    log_str: str | None = None

    if resolved_log is not None and resolved_log.is_file():
        log_str = _format_rel_path(resolved_log)
        with resolved_log.open("r", encoding="utf-8") as fh:
            log_data = json.load(fh)

        history = log_data.get("history", [])
        if isinstance(history, list):
            durations = [
                float(row["duration_sec"])
                for row in history
                if isinstance(row, dict) and row.get("duration_sec") is not None
            ]
        else:
            durations = []

        if log_data.get("completed_epochs") is not None:
            completed_epochs = int(log_data["completed_epochs"])
        elif history:
            completed_epochs = len(history)

        if log_data.get("total_time_sec") is not None:
            total_time_sec = round(float(log_data["total_time_sec"]), 2)
        elif durations:
            total_time_sec = round(sum(durations), 2)

        if durations:
            avg_epoch_sec = round(sum(durations) / len(durations), 2)
        elif total_time_sec is not None and completed_epochs:
            avg_epoch_sec = round(total_time_sec / completed_epochs, 2)

    return {
        "num_params": num_params,
        "checkpoint_path": ckpt_str,
        "checkpoint_size_mb": checkpoint_size_mb,
        "train_log_path": log_str,
        "completed_epochs": completed_epochs,
        "total_time_sec": total_time_sec,
        "avg_epoch_sec": avg_epoch_sec,
    }


def _prepare_for_lpips(images: torch.Tensor, min_size: int = 64) -> torch.Tensor:
    """Chuẩn hoá tensor ảnh về dạng 3 kênh RGB trong [-1, 1] và kích thước >= min_size.

    AlexNet/VGG trong `lpips` yêu cầu ảnh 3 kênh và kích thước tối thiểu 64x64 để tránh
    feature map bị giảm xuống 0x0 qua các tầng pooling.
    """
    if images.ndim != 4:
        raise ValueError(f"Kỳ vọng tensor 4D (N, C, H, W), nhận được shape {tuple(images.shape)}")

    x = images.detach().to(dtype=torch.float32).clamp(-1.0, 1.0)
    if x.shape[1] == 1:
        x = x.repeat(1, 3, 1, 1)
    elif x.shape[1] != 3:
        raise ValueError(f"LPIPS chỉ hỗ trợ 1 hoặc 3 kênh, nhận được {x.shape[1]} kênh")

    h, w = x.shape[-2], x.shape[-1]
    if h < min_size or w < min_size:
        target_h = max(h, min_size)
        target_w = max(w, min_size)
        x = F.interpolate(x, size=(target_h, target_w), mode="bilinear", align_corners=False)
    return x


def _sample_images(
    model: GenerativeModel,
    n_images: int,
    batch_size: int,
    device: torch.device,
    y: torch.Tensor | None = None,
) -> torch.Tensor:
    """Sinh `n_images` ảnh từ `model.sample()` theo từng batch."""
    model = model.to(device)
    model.eval()
    if y is not None:
        y = y.to(device)
        if len(y) < n_images:
            raise ValueError(f"Số lượng nhãn y ({len(y)}) nhỏ hơn n_images ({n_images})")

    chunks: list[torch.Tensor] = []
    generated = 0
    with torch.inference_mode():
        while generated < n_images:
            cur_bs = min(batch_size, n_images - generated)
            cur_y = y[generated : generated + cur_bs] if y is not None else None
            batch = model.sample(n=cur_bs, y=cur_y, device=device)
            chunks.append(batch.detach().cpu())
            generated += cur_bs
    return torch.cat(chunks, dim=0)


def compute_lpips_diversity(
    images: torch.Tensor | None = None,
    model: GenerativeModel | None = None,
    n_images: int = 256,
    n_pairs: int = 500,
    batch_size: int = 32,
    net: str = "alex",
    min_size: int = 64,
    seed: int = 42,
    device: str | torch.device | None = None,
    y: torch.Tensor | None = None,
    lpips_fn: Any | None = None,
) -> dict[str, Any]:
    """Tính độ đa dạng ảnh sinh bằng LPIPS trung bình giữa các cặp ảnh ngẫu nhiên.

    Hỗ trợ truyền sẵn `images` (tái sử dụng ảnh sinh từ `fid_score.py` ở Phase 4) hoặc
    tự sinh mới từ `model`.
    """
    if n_pairs <= 0:
        raise ValueError(f"n_pairs phải > 0, nhận được {n_pairs}")
    if batch_size <= 0:
        raise ValueError(f"batch_size phải > 0, nhận được {batch_size}")

    resolved_device = torch.device(device) if device is not None else get_device()

    if images is None:
        if model is None:
            raise ValueError("Cần cung cấp `images` hoặc `model` để tính LPIPS diversity.")
        if n_images < 2:
            raise ValueError(f"n_images phải >= 2 để tạo cặp so sánh, nhận được {n_images}")
        images = _sample_images(model, n_images=n_images, batch_size=batch_size, device=resolved_device, y=y)
    else:
        if len(images) < 2:
            raise ValueError(f"Cần ít nhất 2 ảnh để tính LPIPS, nhận được {len(images)}")

    prepared = _prepare_for_lpips(images, min_size=min_size)
    num_available = prepared.shape[0]

    if lpips_fn is None:
        try:
            import lpips  # lazy import để module vẫn import được khi test mock hoặc chưa cài lpips
        except ImportError as exc:
            raise ImportError(
                "Chưa cài gói `lpips`. Hãy bỏ comment `lpips` trong requirements.txt và chạy `pip install lpips`."
            ) from exc
        lpips_fn = lpips.LPIPS(net=net)

    if hasattr(lpips_fn, "to"):
        lpips_fn = lpips_fn.to(resolved_device)
    if hasattr(lpips_fn, "eval"):
        lpips_fn.eval()

    rng = torch.Generator().manual_seed(seed)
    idx_a = torch.randint(0, num_available, (n_pairs,), generator=rng)
    idx_b = torch.randint(0, num_available - 1, (n_pairs,), generator=rng)
    # Dịch chỉ số khi idx_b >= idx_a để đảm bảo mọi cặp luôn gồm 2 ảnh khác nhau.
    idx_b = torch.where(idx_b >= idx_a, idx_b + 1, idx_b)

    distances: list[torch.Tensor] = []
    with torch.inference_mode():
        for start in range(0, n_pairs, batch_size):
            end = min(start + batch_size, n_pairs)
            a_batch = prepared[idx_a[start:end]].to(resolved_device)
            b_batch = prepared[idx_b[start:end]].to(resolved_device)
            dist = lpips_fn(a_batch, b_batch)
            distances.append(dist.reshape(-1).detach().cpu().to(torch.float32))

    all_dist = torch.cat(distances, dim=0)
    return {
        "lpips_diversity": round(float(all_dist.mean().item()), 6),
        "lpips_std": round(float(all_dist.std(unbiased=False).item()), 6),
        "lpips_n_images": int(num_available),
        "lpips_n_pairs": int(n_pairs),
        "lpips_net": net,
    }


def benchmark_model(
    model: GenerativeModel,
    model_name: str,
    checkpoint_path: str | Path | None = None,
    train_log_path: str | Path | None = None,
    generated_images: torch.Tensor | None = None,
    n_speed_images: int = 64,
    n_diversity_images: int = 256,
    n_lpips_pairs: int = 500,
    batch_size: int = 32,
    warmup_batches: int = 1,
    lpips_net: str = "alex",
    min_size: int = 64,
    seed: int = 42,
    device: str | torch.device | None = None,
    y_speed: torch.Tensor | None = None,
    y_diversity: torch.Tensor | None = None,
    lpips_fn: Any | None = None,
    output_path: str | Path | None = None,
    save_json: bool = True,
) -> dict[str, Any]:
    """Chạy toàn bộ benchmark cho một GenerativeModel và lưu kết quả JSON."""
    set_seed(seed)
    resolved_device = torch.device(device) if device is not None else get_device()

    cost_metrics = measure_model_cost(
        model=model,
        model_name=model_name,
        checkpoint_path=checkpoint_path,
        train_log_path=train_log_path,
    )
    speed_metrics = measure_generation_speed(
        model=model,
        n_images=n_speed_images,
        batch_size=batch_size,
        warmup_batches=warmup_batches,
        device=resolved_device,
        y=y_speed,
    )
    diversity_metrics = compute_lpips_diversity(
        images=generated_images,
        model=model if generated_images is None else None,
        n_images=n_diversity_images,
        n_pairs=n_lpips_pairs,
        batch_size=batch_size,
        net=lpips_net,
        min_size=min_size,
        seed=seed,
        device=resolved_device,
        y=y_diversity,
        lpips_fn=lpips_fn,
    )

    result: dict[str, Any] = {
        "model": model_name,
        "seed": int(seed),
        **cost_metrics,
        **speed_metrics,
        **diversity_metrics,
    }

    if save_json:
        out_file = (
            resolve_project_path(output_path)
            if output_path is not None
            else ROOT_DIR / "outputs" / f"{model_name}_benchmark.json"
        )
        out_file.parent.mkdir(parents=True, exist_ok=True)
        with out_file.open("w", encoding="utf-8") as fh:
            json.dump(result, fh, indent=2, ensure_ascii=False)
        result["benchmark_path"] = _format_rel_path(out_file)

    return result


def _load_ddpm_model(cfg: dict[str, Any], ckpt_path: Path, device: torch.device) -> GenerativeModel:
    """Nạp DDPM từ checkpoint (kèm EMA weights nếu có)."""
    from src.data import get_dataloader_from_config
    from src.models.ddpm import DDPM
    from src.training.train_ddpm import EMA

    ckpt = _load_torch_checkpoint(ckpt_path, map_location=device)
    if "model_config" in ckpt:
        m_cfg = ckpt["model_config"]
    else:
        _, info = get_dataloader_from_config(cfg, split="val")
        model_cfg = cfg.get("model", {})
        m_cfg = {
            "channels": info.channels,
            "image_size": info.image_size,
            "num_classes": info.num_classes,
            "timesteps": int(model_cfg.get("timesteps", 1000)),
            "beta_start": float(model_cfg.get("beta_start", 1e-4)),
            "beta_end": float(model_cfg.get("beta_end", 0.02)),
            "beta_schedule": str(model_cfg.get("beta_schedule", "linear")),
            "base_channels": int(model_cfg.get("base_channels", 64)),
            "channel_mults": tuple(model_cfg.get("channel_mults", (1, 2, 2))),
            "num_res_blocks": int(model_cfg.get("num_res_blocks", 2)),
        }

    model = DDPM(**m_cfg).to(device)
    model.load_state_dict(ckpt["model_state_dict"])

    if ckpt.get("ema_state_dict") is not None:
        ema = EMA(model, decay=ckpt["ema_state_dict"].get("decay", 0.999))
        ema.load_state_dict(ckpt["ema_state_dict"])
        ema.copy_to(model)

    model.eval()
    return model


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Benchmark tốc độ sinh, chi phí và độ đa dạng LPIPS.")
    parser.add_argument("--config", default="configs/ddpm.yaml", help="Đường dẫn file config YAML")
    parser.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE", help="Ghi đè config")
    parser.add_argument("--ckpt", default=None, help="Đường dẫn checkpoint .pth")
    parser.add_argument("--train-log", default=None, help="Đường dẫn file *_train_log.json")
    parser.add_argument("--n-speed", type=int, default=64, help="Số ảnh dùng đo tốc độ sinh")
    parser.add_argument("--n-diversity", type=int, default=256, help="Số ảnh sinh dùng đo LPIPS")
    parser.add_argument("--n-pairs", type=int, default=500, help="Số cặp ảnh ngẫu nhiên tính LPIPS")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size khi sample và tính LPIPS")
    parser.add_argument("--warmup", type=int, default=1, help="Số batch warm-up trước khi đo tốc độ")
    parser.add_argument("--lpips-net", default="alex", choices=["alex", "vgg", "squeeze"])
    parser.add_argument("--output", default=None, help="Đường dẫn file JSON kết quả")
    args = parser.parse_args(argv)

    cfg = load_config(resolve_project_path(args.config), args.set)
    model_name = cfg.get("model", {}).get("name", "ddpm").lower()
    device = get_device()

    ckpt_path = (
        resolve_project_path(args.ckpt)
        if args.ckpt is not None
        else resolve_project_path(cfg.get("output", {}).get("checkpoint_dir", "checkpoints"))
        / f"{model_name}_best.pth"
    )
    if not ckpt_path.is_file():
        sys.exit(f"Lỗi: Không tìm thấy checkpoint tại {ckpt_path}")

    if model_name == "ddpm":
        model = _load_ddpm_model(cfg, ckpt_path, device)
    else:
        sys.exit(f"CLI benchmark cho mô hình '{model_name}' chưa được nối (cần cài đặt mô hình trước).")

    result = benchmark_model(
        model=model,
        model_name=model_name,
        checkpoint_path=ckpt_path,
        train_log_path=args.train_log,
        n_speed_images=args.n_speed,
        n_diversity_images=args.n_diversity,
        n_lpips_pairs=args.n_pairs,
        batch_size=args.batch_size,
        warmup_batches=args.warmup,
        lpips_net=args.lpips_net,
        seed=int(cfg.get("seed", 42)),
        device=device,
        output_path=args.output,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
