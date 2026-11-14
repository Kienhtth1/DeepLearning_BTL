"""Điểm chạy chung của dự án.

    python main.py data     [--config configs/base.yaml] [--set key=value ...]   # Phase 1 ✔
    python main.py train    --config configs/ddpm.yaml                           # Phase 2 ✔ (TV3)
    python main.py sample   --config configs/ddpm.yaml --ckpt checkpoints/ddpm_best.pth  # Phase 2 ✔ (TV3)
    python main.py evaluate                                                      # Phase 4
"""
from __future__ import annotations

import argparse
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
    p.add_argument("--n-fid", type=int, default=10_000, help="Số ảnh thật export cho FID")
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
        else:
            sys.exit(f"Huấn luyện cho mô hình '{model_name}' chưa được triển khai (thuộc TV1/TV2).")

    elif args.command == "sample":
        model_name = cfg.get("model", {}).get("name", "ddpm").lower()
        if model_name == "ddpm":
            _run_ddpm_sample(cfg, ckpt_path=args.ckpt, n_samples=args.n_samples)
        else:
            sys.exit(f"Sinh ảnh cho mô hình '{model_name}' chưa được triển khai (thuộc TV1/TV2).")

    elif args.command == "evaluate":
        sys.exit(f"'{args.command}' sẽ được cài đặt ở Phase 4 (xem docs/PHASE_PLAN.md).")


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
