"""Kiểm thử huấn luyện Conditional DCGAN (cDCGAN) - Phase 3."""
from pathlib import Path
import json
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.models.cdcgan import ConditionalDCGAN
from src.training.train_cdcgan import evaluate_loss, plot_loss_curve, train_cdcgan


def test_evaluate_loss_and_plot_curve(tmp_path):
    """Kiểm tra hàm evaluate_loss và vẽ biểu đồ plot_loss_curve cho cDCGAN."""
    model = ConditionalDCGAN(
        channels=1,
        image_size=32,
        num_classes=10,
        latent_dim=32,
        embed_dim=16,
        ngf=16,
        ndf=16,
    )
    dummy_x = torch.randn(8, 1, 32, 32)
    dummy_y = torch.randint(0, 10, (8,), dtype=torch.long)
    loader = DataLoader(TensorDataset(dummy_x, dummy_y), batch_size=4)

    metrics = evaluate_loss(model, loader, device=torch.device("cpu"))
    assert set(metrics.keys()) == {"loss", "d_loss", "g_loss", "d_real", "d_fake"}
    assert metrics["loss"] > 0.0
    assert 0.0 <= metrics["d_real"] <= 1.0
    assert 0.0 <= metrics["d_fake"] <= 1.0

    # Kiểm tra vẽ biểu đồ loss
    history = [
        {"epoch": 1, "train_d_loss": 1.25, "train_g_loss": 0.92, "val_d_loss": 1.15, "val_g_loss": 0.98, "d_real": 0.64, "d_fake": 0.36},
        {"epoch": 2, "train_d_loss": 1.05, "train_g_loss": 1.08, "val_d_loss": 1.06, "val_g_loss": 1.04, "d_real": 0.54, "d_fake": 0.46},
    ]
    curve_path = tmp_path / "cdcgan_loss_curve.png"
    plot_loss_curve(history, curve_path)
    assert curve_path.exists()


def test_train_cdcgan_smoke(tmp_path):
    """Smoke test quá trình train_cdcgan trên subset nhỏ (1 epoch)."""
    cfg = {
        "seed": 42,
        "data": {
            "name": "fashion_mnist",
            "image_size": 32,
            "batch_size": 16,
            "num_workers": 0,
            "subset_size": 32,
        },
        "model": {
            "name": "cdcgan",
            "latent_dim": 32,
            "embed_dim": 16,
            "ngf": 16,
            "ndf": 16,
            "label_smoothing": 0.1,
        },
        "train": {
            "epochs": 1,
            "warmup_epochs": 0,
            "lr": 2e-4,
            "lr_g": 2e-4,
            "lr_d": 1e-4,
            "beta1": 0.5,
            "beta2": 0.999,
            "sample_every": 1,
            "samples_per_class": 2,
        },
        "output": {
            "checkpoint_dir": str(tmp_path / "checkpoints"),
            "sample_dir": str(tmp_path / "samples"),
            "log_dir": str(tmp_path / "outputs"),
        },
    }

    result = train_cdcgan(
        cfg,
        device="cpu",
        checkpoint_dir=tmp_path / "checkpoints",
        sample_dir=tmp_path / "samples",
    )

    best_ckpt = Path(result["best_ckpt"])
    last_ckpt = Path(result["last_ckpt"])
    assert best_ckpt.exists(), "Chưa tạo cdcgan_best.pth"
    assert last_ckpt.exists(), "Chưa tạo cdcgan_last.pth"
    assert (tmp_path / "samples" / "cdcgan" / "sample_epoch_001.png").exists()
    assert (tmp_path / "outputs" / "cdcgan_loss_curve.png").exists()

    log_file = tmp_path / "outputs" / "cdcgan_train_log.json"
    assert log_file.exists()
    log_data = json.loads(log_file.read_text(encoding="utf-8"))
    assert log_data["model"] == "cdcgan"
    assert len(log_data["history"]) == 1

    # Nạp và kiểm tra nội dung checkpoint
    ckpt = torch.load(best_ckpt, map_location="cpu")
    assert "model_state_dict" in ckpt
    assert "generator_state_dict" in ckpt
    assert "discriminator_state_dict" in ckpt
    assert "optimizer_d_state_dict" in ckpt
    assert "optimizer_g_state_dict" in ckpt
    assert "model_config" in ckpt
    assert ckpt["model_config"]["channels"] == 1
    assert ckpt["model_config"]["image_size"] == 32
    assert ckpt["model_config"]["num_classes"] == 10
    assert ckpt["model_config"]["embed_dim"] == 16