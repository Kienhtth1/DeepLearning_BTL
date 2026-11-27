"""[Phase 3 - TV2] Training script cho Conditional DCGAN (cDCGAN)."""
from __future__ import annotations

import json
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
from src.models.cdcgan import ConditionalDCGAN
from src.utils import ROOT_DIR, get_device, save_image_grid, set_seed


def evaluate_loss(model: ConditionalDCGAN, loader, device: torch.device) -> dict[str, float]:
    """Tính D loss, G loss và xác suất dự đoán trung bình trên tập validation có điều kiện."""
    model.eval()
    total_d_loss = 0.0
    total_g_loss = 0.0
    total_d_real = 0.0
    total_d_fake = 0.0
    count = 0

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            b = x.size(0)
            d_loss, d_metrics = model.discriminator_loss(x, y)
            g_loss, _ = model.generator_loss(b, y, device)

            total_d_loss += d_loss.item() * b
            total_g_loss += g_loss.item() * b
            total_d_real += d_metrics["d_real"] * b
            total_d_fake += d_metrics["d_fake"] * b
            count += b

    denom = max(1, count)
    avg_d = total_d_loss / denom
    avg_g = total_g_loss / denom
    return {
        "loss": avg_d + avg_g,
        "d_loss": avg_d,
        "g_loss": avg_g,
        "d_real": total_d_real / denom,
        "d_fake": total_d_fake / denom,
    }


def plot_loss_curve(history: list[dict], save_path: Path) -> None:
    """Vẽ và lưu biểu đồ loss curve của Conditional DCGAN."""
    epochs = [h["epoch"] for h in history]
    train_d = [h["train_d_loss"] for h in history]
    train_g = [h["train_g_loss"] for h in history]
    val_d = [h["val_d_loss"] for h in history]
    val_g = [h["val_g_loss"] for h in history]

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))

    axes[0].plot(epochs, train_d, label="Train D Loss", color="#2563eb", linewidth=2)
    axes[0].plot(epochs, train_g, label="Train G Loss", color="#16a34a", linewidth=2)
    axes[0].plot(epochs, val_d, label="Val D Loss", color="#2563eb", linewidth=1.5, linestyle="--")
    axes[0].plot(epochs, val_g, label="Val G Loss", color="#16a34a", linewidth=1.5, linestyle="--")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("BCE Loss")
    axes[0].set_title("cDCGAN Generator & Discriminator Loss")
    axes[0].grid(True, linestyle=":", alpha=0.6)
    axes[0].legend()

    d_real = [h["d_real"] for h in history]
    d_fake = [h["d_fake"] for h in history]
    axes[1].plot(epochs, d_real, label="D(x, y) - Ảnh thật", color="#dc2626", linewidth=2)
    axes[1].plot(epochs, d_fake, label="D(G(z, y), y) - Ảnh sinh", color="#9333ea", linewidth=2)
    axes[1].axhline(0.5, color="gray", linestyle=":", label="Cân bằng Nash (0.5)")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Xác suất")
    axes[1].set_ylim(0.0, 1.05)
    axes[1].set_title("Xác suất phân biệt có điều kiện của Discriminator")
    axes[1].grid(True, linestyle=":", alpha=0.6)
    axes[1].legend()

    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.close()


def train_cdcgan(
    cfg: dict,
    device: torch.device | str | None = None,
    checkpoint_dir: str | Path | None = None,
    sample_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Hàm huấn luyện Conditional DCGAN hoàn chỉnh."""
    seed = cfg.get("seed", 42)
    set_seed(seed)

    if device is None:
        device = get_device()
    else:
        device = torch.device(device)
    print(f"[cDCGAN Train] Sử dụng thiết bị: {device}")

    train_loader, info = get_dataloader_from_config(cfg, split="train")
    val_loader, _ = get_dataloader_from_config(cfg, split="val")
    if info.num_classes is None or info.num_classes <= 0:
        raise ValueError(f"Dataset '{info.name}' không có nhãn phân loại để huấn luyện cDCGAN.")

    print(
        f"[cDCGAN Train] Dataset: {info.name} | Classes: {info.num_classes} | "
        f"Shape: ({info.channels}, {info.image_size}, {info.image_size}) | Train samples: {info.num_samples}"
    )

    model_cfg = cfg.get("model", {})
    latent_dim = int(model_cfg.get("latent_dim", 100))
    embed_dim = int(model_cfg.get("embed_dim", 32))
    ngf = int(model_cfg.get("ngf", 64))
    ndf = int(model_cfg.get("ndf", 32))
    label_smoothing = float(model_cfg.get("label_smoothing", 0.1))

    model = ConditionalDCGAN(
        channels=info.channels,
        image_size=info.image_size,
        num_classes=info.num_classes,
        latent_dim=latent_dim,
        embed_dim=embed_dim,
        ngf=ngf,
        ndf=ndf,
        label_smoothing=label_smoothing,
    ).to(device)

    g_params = sum(p.numel() for p in model.generator.parameters())
    d_params = sum(p.numel() for p in model.discriminator.parameters())
    print(f"[cDCGAN Train] Tổng tham số: {model.num_params():,} (Generator: {g_params:,} | Discriminator: {d_params:,})")

    train_cfg = cfg.get("train", {})
    epochs = int(train_cfg.get("epochs", 30))
    warmup_epochs = int(train_cfg.get("warmup_epochs", epochs // 2))
    lr = float(train_cfg.get("lr", 2.0e-4))
    lr_g = float(train_cfg.get("lr_g", lr))
    lr_d = float(train_cfg.get("lr_d", lr))
    beta1 = float(train_cfg.get("beta1", 0.5))
    beta2 = float(train_cfg.get("beta2", 0.999))
    d_steps = int(train_cfg.get("d_steps", 1))
    g_steps = int(train_cfg.get("g_steps", 1))
    grad_clip = float(train_cfg.get("grad_clip", 0.0))
    sample_every = int(train_cfg.get("sample_every", 5))
    samples_per_class = int(train_cfg.get("samples_per_class", 10))

    optimizer_d = torch.optim.Adam(model.discriminator.parameters(), lr=lr_d, betas=(beta1, beta2))
    optimizer_g = torch.optim.Adam(model.generator.parameters(), lr=lr_g, betas=(beta1, beta2))

    # Lưới cố định: mỗi hàng là 1 lớp (0 -> num_classes-1), mỗi cột là 1 biến thể nhiễu z
    n_classes = info.num_classes
    n_preview = n_classes * samples_per_class
    fixed_noise = torch.randn(max(1, n_preview), latent_dim, device=device)
    fixed_labels = torch.arange(n_classes, device=device).repeat_interleave(samples_per_class)

    out_cfg = cfg.get("output", {})
    ckpt_dir = Path(checkpoint_dir or out_cfg.get("checkpoint_dir", ROOT_DIR / "checkpoints"))
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    samples_dir = Path(sample_dir or out_cfg.get("sample_dir", ROOT_DIR / "outputs" / "samples"))
    samples_dir = samples_dir / "cdcgan"
    samples_dir.mkdir(parents=True, exist_ok=True)

    outputs_dir = Path(out_cfg.get("log_dir", ROOT_DIR / "outputs"))
    outputs_dir.mkdir(parents=True, exist_ok=True)

    best_val_loss = float("inf")
    best_score = float("inf")
    history: list[dict[str, Any]] = []

    print(f"[cDCGAN Train] Bắt đầu huấn luyện {epochs} epochs (xét best từ epoch {warmup_epochs + 1})...")
    total_start_time = time.time()

    for epoch in range(1, epochs + 1):
        epoch_start_time = time.time()
        model.train()

        running_d_loss = 0.0
        running_g_loss = 0.0
        running_d_real = 0.0
        running_d_fake = 0.0
        train_samples = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch:02d}/{epochs:02d}", leave=False)
        for x, y in pbar:
            x = x.to(device)
            y = y.to(device)
            batch_size_curr = x.size(0)

            # 1. Cập nhật Discriminator có điều kiện
            d_loss_val = 0.0
            d_real_val = 0.0
            d_fake_val = 0.0
            for _ in range(d_steps):
                optimizer_d.zero_grad()
                d_loss, d_metrics = model.discriminator_loss(x, y)
                d_loss.backward()
                if grad_clip > 0.0:
                    nn.utils.clip_grad_norm_(model.discriminator.parameters(), grad_clip)
                optimizer_d.step()

                d_loss_val += d_metrics["d_loss"] / d_steps
                d_real_val += d_metrics["d_real"] / d_steps
                d_fake_val += d_metrics["d_fake"] / d_steps

            # 2. Cập nhật Generator có điều kiện
            g_loss_val = 0.0
            for _ in range(g_steps):
                optimizer_g.zero_grad()
                g_loss, g_metrics = model.generator_loss(batch_size_curr, y, device)
                g_loss.backward()
                if grad_clip > 0.0:
                    nn.utils.clip_grad_norm_(model.generator.parameters(), grad_clip)
                optimizer_g.step()

                g_loss_val += g_metrics["g_loss"] / g_steps

            running_d_loss += d_loss_val * batch_size_curr
            running_g_loss += g_loss_val * batch_size_curr
            running_d_real += d_real_val * batch_size_curr
            running_d_fake += d_fake_val * batch_size_curr
            train_samples += batch_size_curr

            pbar.set_postfix({
                "D_loss": f"{d_loss_val:.4f}",
                "G_loss": f"{g_loss_val:.4f}",
                "D(x,y)": f"{d_real_val:.2f}",
            })

        denom = max(1, train_samples)
        train_d_loss = running_d_loss / denom
        train_g_loss = running_g_loss / denom
        train_loss = train_d_loss + train_g_loss
        avg_d_real = running_d_real / denom
        avg_d_fake = running_d_fake / denom

        val_metrics = evaluate_loss(model, val_loader, device)
        val_loss = val_metrics["loss"]
        balance_penalty = abs(val_metrics["d_real"] - 0.5) + abs(val_metrics["d_fake"] - 0.5)
        selection_score = val_loss + balance_penalty
        epoch_time = time.time() - epoch_start_time

        history.append({
            "epoch": epoch,
            "train_loss": round(train_loss, 5),
            "train_d_loss": round(train_d_loss, 5),
            "train_g_loss": round(train_g_loss, 5),
            "val_loss": round(val_loss, 5),
            "val_d_loss": round(val_metrics["d_loss"], 5),
            "val_g_loss": round(val_metrics["g_loss"], 5),
            "d_real": round(avg_d_real, 4),
            "d_fake": round(avg_d_fake, 4),
            "duration_sec": round(epoch_time, 2),
        })

        print(
            f"Epoch [{epoch:02d}/{epochs:02d}] "
            f"D Loss: {train_d_loss:.4f} | G Loss: {train_g_loss:.4f} | "
            f"D(x,y)/D(G(z,y)): {avg_d_real:.2f}/{avg_d_fake:.2f} | Thời gian: {epoch_time:.1f}s"
        )

        full_model_config = {
            "channels": info.channels,
            "image_size": info.image_size,
            "num_classes": info.num_classes,
            "latent_dim": latent_dim,
            "embed_dim": embed_dim,
            "ngf": ngf,
            "ndf": ndf,
            "label_smoothing": label_smoothing,
        }

        ckpt_payload = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "generator_state_dict": model.generator.state_dict(),
            "discriminator_state_dict": model.discriminator.state_dict(),
            "optimizer_d_state_dict": optimizer_d.state_dict(),
            "optimizer_g_state_dict": optimizer_g.state_dict(),
            "train_loss": train_loss,
            "val_loss": val_loss,
            "best_val_loss": min(best_val_loss, val_loss),
            "model_config": full_model_config,
            "train_config": train_cfg,
            "seed": seed,
        }

        torch.save(ckpt_payload, ckpt_dir / "cdcgan_last.pth")

        if epoch > warmup_epochs and selection_score < best_score:
            best_score = selection_score
            best_val_loss = val_loss
            ckpt_payload["best_val_loss"] = best_val_loss
            torch.save(ckpt_payload, ckpt_dir / "cdcgan_best.pth")
            print(f"  -> Đã cập nhật checkpoint tốt nhất: {ckpt_dir / 'cdcgan_best.pth'} (Val Loss: {val_loss:.4f})")

        if (epoch % sample_every == 0 or epoch == epochs) and samples_per_class > 0:
            preview_path = samples_dir / f"sample_epoch_{epoch:03d}.png"
            model.eval()
            with torch.no_grad():
                preview_imgs = model.generator(fixed_noise, fixed_labels).clamp(-1.0, 1.0)
            save_image_grid(preview_imgs, preview_path, nrow=samples_per_class)
            print(f"  -> Lưu lưới ảnh mẫu theo từng lớp (10x{samples_per_class}): {preview_path}")

    total_time = time.time() - total_start_time
    print(f"[cDCGAN Train] Hoàn thành huấn luyện trong {total_time:.1f}s. Best Val Loss: {best_val_loss:.4f}")

    loss_curve_path = outputs_dir / "cdcgan_loss_curve.png"
    plot_loss_curve(history, loss_curve_path)
    print(f"[cDCGAN Train] Đã lưu biểu đồ loss: {loss_curve_path}")

    log_path = outputs_dir / "cdcgan_train_log.json"
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "model": "cdcgan",
                "epochs": epochs,
                "best_val_loss": best_val_loss,
                "total_time_sec": total_time,
                "history": history,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )
    print(f"[cDCGAN Train] Đã lưu log huấn luyện: {log_path}")

    return {
        "best_val_loss": best_val_loss,
        "history": history,
        "best_ckpt": str(ckpt_dir / "cdcgan_best.pth"),
        "last_ckpt": str(ckpt_dir / "cdcgan_last.pth"),
    }