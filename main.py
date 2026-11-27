"""Điểm chạy chung của dự án.

    python main.py data     [--config configs/base.yaml] [--set key=value ...]   # Phase 1 ✔
    python main.py train    --config configs/dcgan.yaml                          # Phase 2 ✔ (TV2)
    python main.py sample   --config configs/dcgan.yaml --ckpt checkpoints/dcgan_best.pth  # Phase 2 ✔ (TV2)
    python main.py train    --config configs/cdcgan.yaml                         # Phase 3 ✔ (TV2)
    python main.py sample   --config configs/cdcgan.yaml --ckpt checkpoints/cdcgan_best.pth --n-samples 100
    python main.py train    --config configs/ddpm.yaml                           # Phase 2 ✔ (TV3)
    python main.py sample   --config configs/ddpm.yaml --ckpt checkpoints/ddpm_best.pth  # Phase 2 ✔ (TV3)
    python main.py evaluate                                                      # Phase 4
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

import torch

from src.utils import get_device, load_config, save_image_grid, set_seed


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="BTL Deep Learning - Đề tài 4: VAE vs GAN vs Diffusion")
    p.add_argument("command", choices=["data", "train", "sample", "evaluate"])
    p.add_argument("--config", default="configs/base.yaml")
    p.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE",
                   help="Ghi đè config, vd: --set data.batch_size=64 data.subset_size=2000")
    p.add_argument("--ckpt", type=str, default=None,
                   help="Đường dẫn checkpoint (.pth) để nạp khi chạy lệnh sample")
    p.add_argument("--n-samples", type=int, default=16,
                   help="Số lượng ảnh cần sinh khi chạy lệnh sample (mặc định: 16)")
    p.add_argument("--class-id", type=int, default=None,
                   help="Chỉ định nhãn lớp cần sinh cho cDCGAN (vd: 0..9 trên Fashion-MNIST)")
    p.add_argument("--n-fid", type=int, default=10_000,
                   help="Số lượng ảnh dùng để xuất/tính điểm FID (mặc định: 10000)")
    p.add_argument("--fid-batch-size", type=int, default=128,
                   help="Batch size khi trích xuất đặc trưng Inception-v3 tính FID (mặc định: 128)")
    p.add_argument("--real-dir", type=str, default=None,
                   help="Đường dẫn thư mục ảnh thật fid_real (tự suy ra từ config nếu bỏ trống)")
    p.add_argument("--all", action="store_true",
                   help="Đánh giá FID cho tất cả mô hình tìm thấy checkpoint trong checkpoints/")
    p.add_argument("--skip-fid", action="store_true", help="Bỏ qua bước export ảnh thật")
    args = p.parse_args(argv)

    cfg = load_config(args.config, args.set)

    if args.command == "data":
        from src.data.prepare import run
        run(cfg, n_fid=args.n_fid, skip_fid=args.skip_fid)

    elif args.command == "train":
        model_name = cfg.get("model", {}).get("name", "ddpm").lower()
        if model_name == "ddpm":
            from src.training.train_ddpm import train_ddpm
            train_ddpm(cfg)
        elif model_name == "dcgan":
            from src.training.train_dcgan import train_dcgan
            train_dcgan(cfg)
        elif model_name == "cdcgan":
            from src.training.train_cdcgan import train_cdcgan
            train_cdcgan(cfg)
        else:
            sys.exit(f"Huấn luyện cho mô hình '{model_name}' chưa được triển khai (thuộc TV1/TV2).")

    elif args.command == "sample":
        model_name = cfg.get("model", {}).get("name", "ddpm").lower()
        if model_name == "ddpm":
            _run_ddpm_sample(cfg, ckpt_path=args.ckpt, n_samples=args.n_samples)
        elif model_name == "dcgan":
            _run_dcgan_sample(cfg, ckpt_path=args.ckpt, n_samples=args.n_samples)
        elif model_name == "cdcgan":
            _run_cdcgan_sample(cfg, ckpt_path=args.ckpt, n_samples=args.n_samples, class_id=args.class_id)    
        else:
            sys.exit(f"Sinh ảnh cho mô hình '{model_name}' chưa được triển khai (thuộc TV1/TV2).")

    elif args.command == "evaluate":
        _run_evaluate_fid(
            cfg,
            ckpt_path=args.ckpt,
            n_fid=args.n_fid,
            batch_size=args.fid_batch_size,
            real_dir=args.real_dir,
            evaluate_all=args.all,
        )

def _load_model_from_checkpoint(
    model_name: str,
    ckpt_file: Path,
    cfg: dict,
    device: torch.device,
) -> torch.nn.Module:
    """Nạp đúng kiến trúc và trọng số của mô hình từ checkpoint."""
    from src.data import get_dataloader_from_config

    ckpt = torch.load(ckpt_file, map_location=device)
    model_name = model_name.lower()

    if model_name == "dcgan":
        from src.models.dcgan import DCGAN
        if "model_config" in ckpt:
            m_cfg = ckpt["model_config"]
        else:
            _, info = get_dataloader_from_config(cfg, split="val")
            model_cfg = cfg.get("model", {})
            m_cfg = {
                "channels": info.channels,
                "image_size": info.image_size,
                "num_classes": info.num_classes,
                "latent_dim": int(model_cfg.get("latent_dim", 100)),
                "ngf": int(model_cfg.get("ngf", 64)),
                "ndf": int(model_cfg.get("ndf", 64)),
                "label_smoothing": float(model_cfg.get("label_smoothing", 0.0)),
            }
        model = DCGAN(**m_cfg).to(device)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()
        return model

    if model_name == "cdcgan":
        from src.models.cdcgan import ConditionalDCGAN
        if "model_config" in ckpt:
            m_cfg = ckpt["model_config"]
        else:
            _, info = get_dataloader_from_config(cfg, split="val")
            model_cfg = cfg.get("model", {})
            m_cfg = {
                "channels": info.channels,
                "image_size": info.image_size,
                "num_classes": info.num_classes or 10,
                "latent_dim": int(model_cfg.get("latent_dim", 100)),
                "embed_dim": int(model_cfg.get("embed_dim", 32)),
                "ngf": int(model_cfg.get("ngf", 64)),
                "ndf": int(model_cfg.get("ndf", 32)),
                "label_smoothing": float(model_cfg.get("label_smoothing", 0.1)),
            }
        model = ConditionalDCGAN(**m_cfg).to(device)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()
        return model

    if model_name == "ddpm":
        from src.models.ddpm import DDPM
        from src.training.train_ddpm import EMA
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

    if model_name == "vae":
        from src.models.vae import VAE
        if "model_config" in ckpt:
            m_cfg = ckpt["model_config"]
        else:
            _, info = get_dataloader_from_config(cfg, split="val")
            model_cfg = cfg.get("model", {})
            m_cfg = {
                "channels": info.channels,
                "image_size": info.image_size,
                "latent_dim": int(model_cfg.get("latent_dim", 64)),
            }
        model = VAE(**m_cfg).to(device)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()
        return model

    raise ValueError(f"Không hỗ trợ mô hình '{model_name}'.")

def _run_evaluate_fid(
    cfg: dict,
    ckpt_path: str | None = None,
    n_fid: int = 10_000,
    batch_size: int = 128,
    real_dir: str | None = None,
    evaluate_all: bool = False,
) -> None:
    """Tính điểm FID cho 1 mô hình chỉ định hoặc tất cả mô hình có checkpoint."""
    from src.evaluation.fid_score import evaluate_model_fid

    seed = cfg.get("seed", 42)
    set_seed(seed)
    device = get_device()
    print(f"[FID Evaluate] Thiết bị: {device} | Số mẫu đánh giá: {n_fid}")

    # Xác định thư mục ảnh thật fid_real đã export ở Phase 1
    if real_dir is None:
        data_cfg = cfg.get("data", {})
        dataset_name = data_cfg.get("name", "fashion_mnist")
        image_size = int(data_cfg.get("image_size", 32))
        processed_root = Path(data_cfg.get("processed_dir", "data/processed"))
        real_path = processed_root / f"{dataset_name}_{image_size}" / "fid_real"
    else:
        real_path = Path(real_dir)

    if not real_path.exists():
        sys.exit(
            f"Lỗi: Không tìm thấy thư mục ảnh thật tại '{real_path}'.\n"
            f"Hãy chạy lệnh sau trước: python main.py data --config configs/base.yaml --n-fid {n_fid}"
        )

    ckpt_dir = Path(cfg.get("output", {}).get("checkpoint_dir", "checkpoints"))
    targets: list[tuple[str, Path]] = []

    if evaluate_all:
        candidates = [
            ("vae", ckpt_dir / "vae_best.pth"),
            ("dcgan", ckpt_dir / "dcgan_best.pth"),
            ("cdcgan", ckpt_dir / "cdcgan_best.pth"),
            ("ddpm", ckpt_dir / "ddpm_best.pth"),
        ]
        for m_name, path in candidates:
            if path.exists():
                targets.append((m_name, path))
        if not targets:
            sys.exit(f"Lỗi: Không tìm thấy checkpoint nào trong thư mục '{ckpt_dir}'.")
    else:
        model_name = cfg.get("model", {}).get("name", "dcgan").lower()
        target_ckpt = Path(ckpt_path) if ckpt_path else (ckpt_dir / f"{model_name}_best.pth")
        if not target_ckpt.exists():
            sys.exit(f"Lỗi: Không tìm thấy checkpoint tại '{target_ckpt}'.")
        targets.append((model_name, target_ckpt))

    results: list[dict] = []
    num_workers = int(cfg.get("data", {}).get("num_workers", 2))

    for m_name, target_ckpt in targets:
        print(f"\n[FID Evaluate] Đang đánh giá '{m_name.upper()}' từ checkpoint: {target_ckpt}")
        model = _load_model_from_checkpoint(m_name, target_ckpt, cfg, device)
        fid = evaluate_model_fid(
            model=model,
            real_dir=real_path,
            n_samples=n_fid,
            batch_size=batch_size,
            device=device,
            num_workers=num_workers,
            cache_real_stats=True,
        )
        print(f"  => FID ({m_name.upper()} | {n_fid} samples): {fid:.4f}")
        results.append({
            "model": m_name,
            "checkpoint": str(target_ckpt),
            "n_samples": n_fid,
            "num_params": model.num_params(),
            "fid": round(fid, 4),
        })

    # In bảng tổng hợp và lưu JSON vào outputs/
    print("\n" + "=" * 56)
    print(f"{'Mô hình':<12} | {'Tham số':>12} | {'Số mẫu':>8} | {'FID ↓':>10}")
    print("-" * 56)
    for r in results:
        print(f"{r['model'].upper():<12} | {r['num_params']:>12,} | {r['n_samples']:>8} | {r['fid']:>10.4f}")
    print("=" * 56)

    outputs_dir = Path(cfg.get("output", {}).get("log_dir", "outputs"))
    outputs_dir.mkdir(parents=True, exist_ok=True)
    out_json = outputs_dir / "fid_results.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"[FID Evaluate] Đã lưu bảng điểm FID tại: {out_json}")

def _run_dcgan_sample(cfg: dict, ckpt_path: str | None = None, n_samples: int = 16) -> None:
    """Sinh ảnh từ mô hình DCGAN đã huấn luyện và lưu file dạng lưới."""
    from src.data import get_dataloader_from_config
    from src.models.dcgan import DCGAN

    seed = cfg.get("seed", 42)
    set_seed(seed)
    device = get_device()
    print(f"[DCGAN Sample] Thiết bị: {device}")

    if ckpt_path is None:
        ckpt_path = str(Path(cfg.get("output", {}).get("checkpoint_dir", "checkpoints")) / "dcgan_best.pth")

    ckpt_file = Path(ckpt_path)
    if not ckpt_file.exists():
        sys.exit(f"Lỗi: Không tìm thấy checkpoint tại {ckpt_file}. Vui lòng kiểm tra lại đường dẫn!")

    print(f"[DCGAN Sample] Nạp checkpoint: {ckpt_file}")
    ckpt = torch.load(ckpt_file, map_location=device)

    if "model_config" in ckpt:
        m_cfg = ckpt["model_config"]
    else:
        _, info = get_dataloader_from_config(cfg, split="val")
        model_cfg = cfg.get("model", {})
        m_cfg = {
            "channels": info.channels,
            "image_size": info.image_size,
            "num_classes": info.num_classes,
            "latent_dim": int(model_cfg.get("latent_dim", 100)),
            "ngf": int(model_cfg.get("ngf", 64)),
            "ndf": int(model_cfg.get("ndf", 64)),
            "label_smoothing": float(model_cfg.get("label_smoothing", 0.0)),
        }

    model = DCGAN(**m_cfg).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    print(f"[DCGAN Sample] Đang sinh {n_samples} ảnh từ vector nhiễu z (latent_dim={model.latent_dim})...")
    with torch.no_grad():
        samples = model.sample(n=n_samples, device=device)

    sample_out_dir = Path(cfg.get("output", {}).get("sample_dir", "outputs/samples"))
    sample_out_path = sample_out_dir / "dcgan_sampled.png"
    sample_out_dir.mkdir(parents=True, exist_ok=True)

    nrow = int(math.isqrt(n_samples)) or 4
    save_image_grid(samples, sample_out_path, nrow=nrow)
    print(f"[DCGAN Sample] Đã lưu lưới ảnh sinh thành công: {sample_out_path}")

def _run_cdcgan_sample(
    cfg: dict,
    ckpt_path: str | None = None,
    n_samples: int = 100,
    class_id: int | None = None,
) -> None:
    """Sinh ảnh có điều kiện từ mô hình Conditional DCGAN và lưu file dạng lưới."""
    from src.data import get_dataloader_from_config
    from src.models.cdcgan import ConditionalDCGAN

    seed = cfg.get("seed", 42)
    set_seed(seed)
    device = get_device()
    print(f"[cDCGAN Sample] Thiết bị: {device}")

    if ckpt_path is None:
        ckpt_path = str(Path(cfg.get("output", {}).get("checkpoint_dir", "checkpoints")) / "cdcgan_best.pth")

    ckpt_file = Path(ckpt_path)
    if not ckpt_file.exists():
        sys.exit(f"Lỗi: Không tìm thấy checkpoint tại {ckpt_file}. Vui lòng kiểm tra lại đường dẫn!")

    print(f"[cDCGAN Sample] Nạp checkpoint: {ckpt_file}")
    ckpt = torch.load(ckpt_file, map_location=device)

    if "model_config" in ckpt:
        m_cfg = ckpt["model_config"]
    else:
        _, info = get_dataloader_from_config(cfg, split="val")
        model_cfg = cfg.get("model", {})
        m_cfg = {
            "channels": info.channels,
            "image_size": info.image_size,
            "num_classes": info.num_classes or 10,
            "latent_dim": int(model_cfg.get("latent_dim", 100)),
            "embed_dim": int(model_cfg.get("embed_dim", 32)),
            "ngf": int(model_cfg.get("ngf", 64)),
            "ndf": int(model_cfg.get("ndf", 32)),
            "label_smoothing": float(model_cfg.get("label_smoothing", 0.1)),
        }

    model = ConditionalDCGAN(**m_cfg).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    sample_out_dir = Path(cfg.get("output", {}).get("sample_dir", "outputs/samples"))
    sample_out_dir.mkdir(parents=True, exist_ok=True)

    with torch.no_grad():
        if class_id is not None:
            if not (0 <= class_id < model.num_classes):
                sys.exit(f"Lỗi: class_id phải nằm trong [0, {model.num_classes - 1}], nhận được: {class_id}")
            print(f"[cDCGAN Sample] Đang sinh {n_samples} ảnh thuộc lớp {class_id}...")
            y = torch.full((n_samples,), class_id, dtype=torch.long, device=device)
            samples = model.sample(n=n_samples, y=y, device=device)
            sample_out_path = sample_out_dir / f"cdcgan_class_{class_id}.png"
            nrow = int(math.isqrt(n_samples)) or 4
        else:
            per_class = max(1, n_samples // model.num_classes)
            total = per_class * model.num_classes
            print(f"[cDCGAN Sample] Đang sinh lưới {model.num_classes} lớp x {per_class} ảnh ({total} ảnh)...")
            y = torch.arange(model.num_classes, device=device).repeat_interleave(per_class)
            samples = model.sample(n=total, y=y, device=device)
            sample_out_path = sample_out_dir / "cdcgan_sampled_grid.png"
            nrow = per_class

    save_image_grid(samples, sample_out_path, nrow=nrow)
    print(f"[cDCGAN Sample] Đã lưu lưới ảnh sinh thành công: {sample_out_path}")

def _run_ddpm_sample(cfg: dict, ckpt_path: str | None = None, n_samples: int = 16) -> None:
    """Sinh ảnh từ mô hình DDPM đã huấn luyện và lưu file dạng lưới."""
    from src.data import get_dataloader_from_config
    from src.models.ddpm import DDPM
    from src.training.train_ddpm import EMA

    seed = cfg.get("seed", 42)
    set_seed(seed)
    device = get_device()
    print(f"[DDPM Sample] Thiết bị: {device}")

    # Xác định checkpoint path
    if ckpt_path is None:
        ckpt_path = str(Path(cfg.get("output", {}).get("checkpoint_dir", "checkpoints")) / "ddpm_best.pth")

    ckpt_file = Path(ckpt_path)
    if not ckpt_file.exists():
        sys.exit(f"Lỗi: Không tìm thấy checkpoint tại {ckpt_file}. Vui lòng kiểm tra lại đường dẫn!")

    print(f"[DDPM Sample] Nạp checkpoint: {ckpt_file}")
    ckpt = torch.load(ckpt_file, map_location=device)

    # Đọc cấu hình mô hình từ checkpoint hoặc từ cfg
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

    # Khôi phục EMA weights nếu có
    if ckpt.get("ema_state_dict") is not None:
        ema = EMA(model, decay=ckpt["ema_state_dict"].get("decay", 0.999))
        ema.load_state_dict(ckpt["ema_state_dict"])
        ema.copy_to(model)
        print("[DDPM Sample] Đã khôi phục EMA weights.")

    model.eval()
    print(f"[DDPM Sample] Đang sinh {n_samples} ảnh qua reverse diffusion ({model.timesteps} timesteps)...")
    with torch.no_grad():
        samples = model.sample(n=n_samples, device=device)

    sample_out_dir = Path(cfg.get("output", {}).get("sample_dir", "outputs/samples"))
    sample_out_path = sample_out_dir / "ddpm_sampled.png"
    sample_out_dir.mkdir(parents=True, exist_ok=True)

    nrow = int(math.isqrt(n_samples)) or 4
    save_image_grid(samples, sample_out_path, nrow=nrow)
    print(f"[DDPM Sample] Đã lưu lưới ảnh sinh thành công: {sample_out_path}")

if __name__ == "__main__":
    main()
