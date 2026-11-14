"""[Phase 2 - TV3] Training script cho DDPM.

Tích hợp hoàn chỉnh:
- Đọc cấu hình từ YAML kế thừa (configs/ddpm.yaml -> configs/base.yaml).
- Lấy DataLoader qua get_dataloader_from_config.
- Cập nhật Exponential Moving Average (EMA).
- Validation định kỳ sau mỗi epoch.
- Lưu checkpoint checkpoints/ddpm_best.pth và ddpm_last.pth.
- Vẽ biểu đồ loss outputs/ddpm_loss_curve.png.
- Sinh ảnh mẫu xem trước outputs/samples/ddpm/.
"""
from __future__ import annotations

import json
import math
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")  # headless backend cho matplotlib
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from tqdm import tqdm

from src.data import get_dataloader_from_config
from src.models.ddpm import DDPM
from src.utils import ROOT_DIR, get_device, load_config, save_image_grid, set_seed


class EMA:
    """Exponential Moving Average cho trọng số mô hình."""

    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = decay
        self.shadow: dict[str, torch.Tensor] = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone().detach()

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.shadow[name].mul_(self.decay).add_(param.data, alpha=1.0 - self.decay)

    def state_dict(self) -> dict[str, Any]:
        return {
            "decay": self.decay,
            "shadow": {k: v.cpu() for k, v in self.shadow.items()},
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        self.decay = state_dict.get("decay", self.decay)
        self.shadow = {k: v.clone() for k, v in state_dict["shadow"].items()}

    def copy_to(self, model: nn.Module) -> None:
        """Nạp trọng số shadow vào model."""
        for name, param in model.named_parameters():
            if name in self.shadow:
                param.data.copy_(self.shadow[name].to(param.device))

    def store(self, model: nn.Module) -> dict[str, torch.Tensor]:
        """Lưu lại bản sao trọng số hiện tại của model trước khi nạp EMA."""
        return {
            name: param.data.clone().detach()
            for name, param in model.named_parameters()
            if name in self.shadow
        }

    def restore(self, model: nn.Module, backup: dict[str, torch.Tensor]) -> None:
        """Khôi phục lại trọng số gốc từ bản backup."""
        for name, param in model.named_parameters():
            if name in backup:
                param.data.copy_(backup[name].to(param.device))


@contextmanager
def ema_scope(model: nn.Module, ema: EMA | None):
    """Context manager để tạm thời áp dụng EMA weights (cho validation / sampling)."""
    if ema is None:
        yield
        return
    backup = ema.store(model)
    ema.copy_to(model)
    try:
        yield
    finally:
        ema.restore(model, backup)


def evaluate_loss(model: DDPM, loader, device: torch.device, ema: EMA | None = None) -> float:
    """Tính loss trung bình trên tập validation."""
    model.eval()
    total_loss = 0.0
    count = 0

    with torch.no_grad():
        with ema_scope(model, ema):
            for x, y in loader:
                x = x.to(device)
                y = y.to(device) if y is not None else None
                loss_dict = model.training_loss(x, y)
                total_loss += loss_dict["loss"].item() * x.size(0)
                count += x.size(0)

    return total_loss / max(1, count)


def plot_loss_curve(history: list[dict], save_path: Path) -> None:
    """Vẽ và lưu biểu đồ training/validation loss curve."""
    epochs = [h["epoch"] for h in history]
    train_losses = [h["train_loss"] for h in history]
    val_losses = [h["val_loss"] for h in history]

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_losses, label="Train Loss (MSE)", color="#2563eb", linewidth=2)
    plt.plot(epochs, val_losses, label="Val Loss (MSE)", color="#dc2626", linewidth=2, linestyle="--")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("DDPM Training & Validation Loss Curve")
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend()
    plt.tight_layout()

    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.close()


def train_ddpm(
    cfg: dict,
    device: torch.device | str | None = None,
    checkpoint_dir: str | Path | None = None,
    sample_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Hàm huấn luyện DDPM hoàn chỉnh."""
    seed = cfg.get("seed", 42)
    set_seed(seed)

    if device is None:
        device = get_device()
    else:
        device = torch.device(device)
    print(f"[DDPM Train] Sử dụng thiết bị: {device}")

    # Lấy DataLoader qua get_dataloader_from_config
    train_loader, info = get_dataloader_from_config(cfg, split="train")
    val_loader, _ = get_dataloader_from_config(cfg, split="val")
    print(
        f"[DDPM Train] Dataset: {info.name} | Shape: ({info.channels}, {info.image_size}, {info.image_size}) "
        f"| Train samples: {info.num_samples}"
    )

    # Đọc cấu hình mô hình
    model_cfg = cfg.get("model", {})
    timesteps = int(model_cfg.get("timesteps", 1000))
    beta_start = float(model_cfg.get("beta_start", 1e-4))
    beta_end = float(model_cfg.get("beta_end", 0.02))
    beta_schedule = str(model_cfg.get("beta_schedule", "linear"))
    base_channels = int(model_cfg.get("base_channels", 64))
    channel_mults = tuple(model_cfg.get("channel_mults", (1, 2, 2)))
    num_res_blocks = int(model_cfg.get("num_res_blocks", 2))
    dropout = float(model_cfg.get("dropout", 0.0))

    model = DDPM(
        channels=info.channels,
        image_size=info.image_size,
        num_classes=info.num_classes,
        timesteps=timesteps,
        beta_start=beta_start,
        beta_end=beta_end,
        beta_schedule=beta_schedule,
        base_channels=base_channels,
        channel_mults=channel_mults,
        num_res_blocks=num_res_blocks,
        dropout=dropout,
    ).to(device)

    print(f"[DDPM Train] Số tham số mô hình: {model.num_params():,}")

    # Cấu hình huấn luyện
    train_cfg = cfg.get("train", {})
    epochs = int(train_cfg.get("epochs", 30))
    lr = float(train_cfg.get("lr", 2.0e-4))
    weight_decay = float(train_cfg.get("weight_decay", 1e-4))
    ema_decay = float(train_cfg.get("ema_decay", 0.999))
    grad_clip = float(train_cfg.get("grad_clip", 1.0))
    sample_every = int(train_cfg.get("sample_every", 5))
    n_preview = int(train_cfg.get("n_preview", 16))

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    ema = EMA(model, decay=ema_decay) if ema_decay > 0.0 else None

    # Thư mục lưu checkpoint và outputs
    out_cfg = cfg.get("output", {})
    ckpt_dir = Path(checkpoint_dir or out_cfg.get("checkpoint_dir", ROOT_DIR / "checkpoints"))
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    samples_dir = Path(sample_dir or out_cfg.get("sample_dir", ROOT_DIR / "outputs" / "samples"))
    samples_dir = samples_dir / "ddpm"
    samples_dir.mkdir(parents=True, exist_ok=True)

    outputs_dir = Path(cfg.get("output", {}).get("log_dir", ROOT_DIR / "outputs"))
    outputs_dir.mkdir(parents=True, exist_ok=True)

    best_val_loss = float("inf")
    history: list[dict[str, Any]] = []

    print(f"[DDPM Train] Bắt đầu huấn luyện {epochs} epochs...")

    total_start_time = time.time()

    for epoch in range(1, epochs + 1):
        epoch_start_time = time.time()
        model.train()
        running_train_loss = 0.0
        train_samples = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch:02d}/{epochs:02d}", leave=False)
        for x, y in pbar:
            x = x.to(device)
            y = y.to(device) if y is not None else None

            optimizer.zero_grad()
            loss_dict = model.training_loss(x, y)
            loss = loss_dict["loss"]
            loss.backward()

            if grad_clip > 0.0:
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

            optimizer.step()

            if ema is not None:
                ema.update(model)

            batch_size_curr = x.size(0)
            running_train_loss += loss.item() * batch_size_curr
            train_samples += batch_size_curr
            pbar.set_postfix({"train_loss": f"{loss.item():.4f}"})

        train_loss = running_train_loss / max(1, train_samples)
        val_loss = evaluate_loss(model, val_loader, device, ema=ema)
        epoch_time = time.time() - epoch_start_time

        history.append({
            "epoch": epoch,
            "train_loss": round(train_loss, 5),
            "val_loss": round(val_loss, 5),
            "duration_sec": round(epoch_time, 2),
        })

        print(
            f"Epoch [{epoch:02d}/{epochs:02d}] "
            f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Thời gian: {epoch_time:.1f}s"
        )

        # Lưu checkpoint
        full_model_config = {
            "channels": info.channels,
            "image_size": info.image_size,
            "num_classes": info.num_classes,
            "timesteps": timesteps,
            "beta_start": beta_start,
            "beta_end": beta_end,
            "beta_schedule": beta_schedule,
            "base_channels": base_channels,
            "channel_mults": list(channel_mults),
            "num_res_blocks": num_res_blocks,
            "dropout": dropout,
        }

        ckpt_payload = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "ema_state_dict": ema.state_dict() if ema is not None else None,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "best_val_loss": min(best_val_loss, val_loss),
            "model_config": full_model_config,
            "train_config": train_cfg,
            "seed": seed,
        }

        # Lưu ddpm_last.pth
        torch.save(ckpt_payload, ckpt_dir / "ddpm_last.pth")

        # Lưu ddpm_best.pth nếu đạt val_loss tốt nhất
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            ckpt_payload["best_val_loss"] = best_val_loss
            torch.save(ckpt_payload, ckpt_dir / "ddpm_best.pth")
            print(f"  -> Đã cập nhật checkpoint tốt nhất: {ckpt_dir / 'ddpm_best.pth'} (Val Loss: {val_loss:.4f})")

        # Sinh ảnh xem trước định kỳ hoặc ở epoch cuối
        if (epoch % sample_every == 0 or epoch == epochs) and n_preview > 0:
            preview_path = samples_dir / f"sample_epoch_{epoch:03d}.png"
            with ema_scope(model, ema):
                with torch.no_grad():
                    preview_imgs = model.sample(n=n_preview, device=device)
            save_image_grid(preview_imgs, preview_path, nrow=int(math.isqrt(n_preview)) or 4)
            print(f"  -> Lưu ảnh mẫu xem trước: {preview_path}")

    total_time = time.time() - total_start_time
    print(f"[DDPM Train] Hoàn thành huấn luyện trong {total_time:.1f}s. Best Val Loss: {best_val_loss:.4f}")

    # Vẽ biểu đồ loss
    loss_curve_path = outputs_dir / "ddpm_loss_curve.png"
    plot_loss_curve(history, loss_curve_path)
    print(f"[DDPM Train] Đã lưu biểu đồ loss: {loss_curve_path}")

    # Lưu log lịch sử huấn luyện
    log_path = outputs_dir / "ddpm_train_log.json"
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "model": "ddpm",
                "epochs": epochs,
                "best_val_loss": best_val_loss,
                "total_time_sec": total_time,
                "history": history,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )
    print(f"[DDPM Train] Đã lưu log huấn luyện: {log_path}")

    return {
        "best_val_loss": best_val_loss,
        "history": history,
        "best_ckpt": str(ckpt_dir / "ddpm_best.pth"),
        "last_ckpt": str(ckpt_dir / "ddpm_last.pth"),
    }
