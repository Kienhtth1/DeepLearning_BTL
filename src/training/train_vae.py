"""[Phase 2 - TV1] Training script cho VAE.

    python main.py train  --config configs/vae.yaml
    python main.py sample --config configs/vae.yaml --ckpt checkpoints/vae_best.pth

Cùng quy ước đầu ra với train_ddpm.py để Phase 4 đánh giá chung:
- checkpoints/vae_best.pth, checkpoints/vae_last.pth (có `model_config` để nạp lại)
- outputs/samples/vae/sample_epoch_XXX.png  (ảnh sinh từ CÙNG một z cố định -> thấy tiến triển)
- outputs/samples/vae/recon_epoch_XXX.png   (hàng trên: ảnh thật val, hàng dưới: ảnh tái tạo)
- outputs/vae_loss_curve.png, outputs/vae_train_log.json

KL warm-up: beta tăng tuyến tính từ 0 -> beta trong `kl_warmup_epochs` epoch đầu để tránh
posterior collapse. Val loss luôn tính với beta đầy đủ nên dùng được để chọn checkpoint tốt nhất.
"""
from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from tqdm import tqdm

from src.data import get_dataloader_from_config
from src.models.vae import VAE
from src.utils import ROOT_DIR, get_device, save_image_grid, set_seed


def build_vae(cfg: dict, info) -> tuple[VAE, dict]:
    m = cfg.get("model", {})
    model_config = {
        "channels": info.channels,
        "image_size": info.image_size,
        "num_classes": info.num_classes,
        "latent_dim": int(m.get("latent_dim", 128)),
        "hidden_dims": list(m.get("hidden_dims", [32, 64, 128])),
        "beta": float(m.get("beta", 1.0)),
        "conditional": bool(m.get("conditional", False)),
    }
    return VAE(**model_config), model_config


def load_vae(ckpt_path: str | Path, device: str | torch.device = "cpu") -> tuple[VAE, dict]:
    """Nạp VAE từ checkpoint (dùng cho sample / Phase 4)."""
    ckpt = torch.load(ckpt_path, map_location=device)
    model = VAE(**ckpt["model_config"]).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, ckpt


@torch.no_grad()
def evaluate(model: VAE, loader, device) -> dict[str, float]:
    model.eval()
    tot = {"loss": 0.0, "recon": 0.0, "kl": 0.0}
    n = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        out = model.training_loss(x, y if model.conditional else None)
        b = x.size(0)
        for k in tot:
            tot[k] += out[k].item() * b
        n += b
    return {k: v / max(1, n) for k, v in tot.items()}


def plot_loss_curve(history: list[dict], save_path: Path) -> None:
    ep = [h["epoch"] for h in history]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, key, title in zip(axes, ["loss", "recon", "kl"], ["Tổng loss (recon + β·KL)", "Reconstruction (MSE)", "KL divergence"]):
        ax.plot(ep, [h[f"train_{key}"] for h in history], label="Train", color="#2563eb", linewidth=2)
        ax.plot(ep, [h[f"val_{key}"] for h in history], label="Val", color="#dc2626", linewidth=2, linestyle="--")
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.grid(True, linestyle=":", alpha=0.6)
        ax.legend()
    fig.suptitle("VAE Training & Validation Loss")
    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def train_vae(
    cfg: dict,
    device: torch.device | str | None = None,
    checkpoint_dir: str | Path | None = None,
    sample_dir: str | Path | None = None,
) -> dict[str, Any]:
    seed = cfg.get("seed", 42)
    set_seed(seed)
    device = get_device() if device is None else torch.device(device)
    print(f"[VAE Train] Thiết bị: {device}")

    train_loader, info = get_dataloader_from_config(cfg, split="train")
    val_loader, _ = get_dataloader_from_config(cfg, split="val")
    print(f"[VAE Train] Dataset: {info.name} | Shape: ({info.channels}, {info.image_size}, {info.image_size}) "
          f"| Train: {info.num_samples} | Val: {len(val_loader.dataset)}")

    model, model_config = build_vae(cfg, info)
    model.to(device)
    print(f"[VAE Train] Số tham số: {model.num_params():,} | latent_dim={model.latent_dim} "
          f"| beta={model.beta} | conditional={model.conditional}")

    t = cfg.get("train", {})
    epochs = int(t.get("epochs", 30))
    lr = float(t.get("lr", 1e-3))
    weight_decay = float(t.get("weight_decay", 0.0))
    kl_warmup_epochs = float(t.get("kl_warmup_epochs", 5))
    grad_clip = float(t.get("grad_clip", 0.0))
    sample_every = int(t.get("sample_every", 5))
    n_preview = int(t.get("n_preview", 64))

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr * 0.05)

    out = cfg.get("output", {})
    ckpt_dir = Path(checkpoint_dir or out.get("checkpoint_dir", ROOT_DIR / "checkpoints"))
    samples_dir = Path(sample_dir or out.get("sample_dir", ROOT_DIR / "outputs" / "samples")) / "vae"
    log_dir = Path(out.get("log_dir", ROOT_DIR / "outputs"))
    for d in (ckpt_dir, samples_dir, log_dir):
        d.mkdir(parents=True, exist_ok=True)

    # z và ảnh val cố định cho ảnh xem trước -> so sánh được giữa các epoch
    g = torch.Generator().manual_seed(seed)
    fixed_z = torch.randn(n_preview, model.latent_dim, generator=g).to(device)
    x_val, y_val = next(iter(val_loader))
    x_val, y_val = x_val[:8].to(device), y_val[:8].to(device)

    steps_per_epoch = max(1, len(train_loader))
    total_warmup_steps = kl_warmup_epochs * steps_per_epoch
    step = 0
    best_val = float("inf")
    history: list[dict[str, Any]] = []
    t_start = time.time()

    for epoch in range(1, epochs + 1):
        t_ep = time.time()
        model.train()
        run = {"loss": 0.0, "recon": 0.0, "kl": 0.0}
        n_seen = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch:02d}/{epochs:02d}", leave=False)
        for x, y in pbar:
            x, y = x.to(device), y.to(device)
            beta_t = model.beta * min(1.0, step / total_warmup_steps) if total_warmup_steps > 0 else model.beta
            loss_dict = model.training_loss(x, y if model.conditional else None, beta=beta_t)

            optimizer.zero_grad(set_to_none=True)
            loss_dict["loss"].backward()
            if grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            step += 1

            b = x.size(0)
            run["loss"] += (loss_dict["recon"] + model.beta * loss_dict["kl"]).item() * b  # ghi với beta đầy đủ
            run["recon"] += loss_dict["recon"].item() * b
            run["kl"] += loss_dict["kl"].item() * b
            n_seen += b
            pbar.set_postfix(recon=f"{loss_dict['recon'].item():.1f}", kl=f"{loss_dict['kl'].item():.1f}",
                             beta=f"{beta_t:.2f}")
        scheduler.step()

        train_m = {k: v / max(1, n_seen) for k, v in run.items()}
        val_m = evaluate(model, val_loader, device)
        ep_time = time.time() - t_ep
        rec = {"epoch": epoch, "beta": round(beta_t, 4), "lr": optimizer.param_groups[0]["lr"],
               "duration_sec": round(ep_time, 2)}
        rec.update({f"train_{k}": round(v, 4) for k, v in train_m.items()})
        rec.update({f"val_{k}": round(v, 4) for k, v in val_m.items()})
        history.append(rec)
        print(f"Epoch [{epoch:02d}/{epochs:02d}] train {train_m['loss']:.2f} (recon {train_m['recon']:.2f}, "
              f"kl {train_m['kl']:.2f}) | val {val_m['loss']:.2f} | beta {beta_t:.2f} | {ep_time:.1f}s")

        payload = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_loss": val_m["loss"],
            "best_val_loss": min(best_val, val_m["loss"]),
            "model_config": model_config,
            "train_config": t,
            "data_config": cfg.get("data", {}),
            "seed": seed,
        }
        torch.save(payload, ckpt_dir / "vae_last.pth")
        if val_m["loss"] < best_val:
            best_val = val_m["loss"]
            payload["best_val_loss"] = best_val
            torch.save(payload, ckpt_dir / "vae_best.pth")
            print(f"  -> Cập nhật vae_best.pth (val loss {best_val:.2f})")

        if n_preview > 0 and (epoch % sample_every == 0 or epoch == epochs or epoch == 1):
            nrow = int(math.isqrt(n_preview)) or 8
            save_image_grid(model.sample(n_preview, device=device, z=fixed_z),
                            samples_dir / f"sample_epoch_{epoch:03d}.png", nrow=nrow)
            yv = y_val if model.conditional else None
            save_image_grid(torch.cat([x_val, model.reconstruct(x_val, yv)]),
                            samples_dir / f"recon_epoch_{epoch:03d}.png", nrow=x_val.size(0))
            print(f"  -> Lưu ảnh mẫu & tái tạo: {samples_dir}")

    total = time.time() - t_start
    print(f"[VAE Train] Xong trong {total:.1f}s. Best val loss: {best_val:.2f}")

    plot_loss_curve(history, log_dir / "vae_loss_curve.png")
    with open(log_dir / "vae_train_log.json", "w", encoding="utf-8") as fh:
        json.dump({
            "model": "vae",
            "dataset": info.name,
            "num_params": model.num_params(),
            "model_config": model_config,
            "epochs": epochs,
            "best_val_loss": best_val,
            "total_time_sec": round(total, 1),
            "avg_epoch_sec": round(total / epochs, 2),
            "device": str(device),
            "history": history,
        }, fh, indent=2, ensure_ascii=False)
    print(f"[VAE Train] Đã lưu {log_dir / 'vae_loss_curve.png'} và {log_dir / 'vae_train_log.json'}")

    return {
        "best_val_loss": best_val,
        "history": history,
        "best_ckpt": str(ckpt_dir / "vae_best.pth"),
        "last_ckpt": str(ckpt_dir / "vae_last.pth"),
    }
