"""Điểm chạy chung của dự án.

    python main.py data     [--config configs/base.yaml] [--set key=value ...]   # Phase 1 ✔
    python main.py train    --config configs/vae.yaml                            # Phase 2
    python main.py sample   --config configs/vae.yaml --ckpt ...                 # Phase 2-3
    python main.py evaluate                                                      # Phase 4
"""
from __future__ import annotations

import argparse
import sys

from src.utils import load_config


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="BTL Deep Learning - Đề tài 4: VAE vs GAN vs Diffusion")
    p.add_argument("command", choices=["data", "train", "sample", "evaluate"])
    p.add_argument("--config", default="configs/base.yaml")
    p.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE",
                   help="Ghi đè config, vd: --set data.batch_size=64 data.subset_size=2000")
    p.add_argument("--n-fid", type=int, default=10_000, help="Số ảnh thật export cho FID")
    p.add_argument("--skip-fid", action="store_true", help="Bỏ qua bước export ảnh thật")
    args = p.parse_args(argv)

    cfg = load_config(args.config, args.set)

    if args.command == "data":
        from src.data.prepare import run
        run(cfg, n_fid=args.n_fid, skip_fid=args.skip_fid)
    else:
        sys.exit(f"'{args.command}' sẽ được cài đặt ở Phase sau (xem docs/PHASE_PLAN.md).")


if __name__ == "__main__":
    main()
