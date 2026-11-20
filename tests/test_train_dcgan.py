"""Kiểm thử huấn luyện DCGAN - Phase 2."""
from pathlib import Path
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.models.dcgan import DCGAN
from src.training.train_dcgan import evaluate_loss, plot_loss_curve, train_dcgan


def test_evaluate_loss_and_plot_curve(tmp_path):
    """Kiểm tra hàm evaluate_loss và vẽ biểu đồ plot_loss_curve."""
    model = DCGAN(
        channels=1,
        image_size=32,
        latent_dim=32,
        ngf=16,
        ndf=16,
    )
    dummy_x = torch.randn(8, 1, 32, 32)
    dummy_y = torch.zeros(8, dtype=torch.long)
    loader = DataLoader(TensorDataset(dummy_x, dummy_y), batch_size=4)

    metrics = evaluate_loss(model, loader, device=torch.device("cpu"))
    assert set(metrics.keys()) == {"loss", "d_loss", "g_loss", "d_real", "d_fake"}
    assert metrics["loss"] > 0.0
    assert 0.0 <= metrics["d_real"] <= 1.0
    assert 0.0 <= metrics["d_fake"] <= 1.0

    # Kiểm tra vẽ biểu đồ loss
    history = [
        {"epoch": 1, "train_d_loss": 1.2, "train_g_loss": 0.9, "val_d_loss": 1.1, "val_g_loss": 0.95, "d_real": 0.65, "d_fake": 0.35},
        {"epoch": 2, "train_d_loss": 1.0, "train_g_loss": 1.1, "val_d_loss": 1.05, "val_g_loss": 1.05, "d_real": 0.55, "d_fake": 0.45},
    ]
    curve_path = tmp_path / "dcgan_loss_curve.png"
    plot_loss_curve(history, curve_path)
    assert curve_path.exists()


def test_train_dcgan_smoke(tmp_path):
    """Smoke test quá trình train_dcgan trên subset nhỏ (1 epoch)."""
    cfg = {
        "seed": 42,
        "data": {
            "name": "fashion_mnist",
            "image_size": 32,
            "batch_size": 16,
            "num_workers": 0,
            "subset_size": 32,  # Chỉ lấy 32 ảnh để chạy trong vài giây
        },
        "model": {
            "name": "dcgan",
            "latent_dim": 32,
            "ngf": 16,
            "ndf": 16,
            "label_smoothing": 0.0,
        },
        "train": {
            "epochs": 1,
            "lr": 2e-4,
            "beta1": 0.5,
            "beta2": 0.999,
            "sample_every": 1,
            "n_preview": 4,
        },
        "output": {
            "checkpoint_dir": str(tmp_path / "checkpoints"),
            "sample_dir": str(tmp_path / "samples"),
            "log_dir": str(tmp_path / "outputs"),
        },
    }

    result = train_dcgan(
        cfg,
        device="cpu",
        checkpoint_dir=tmp_path / "checkpoints",
        sample_dir=tmp_path / "samples",
    )

    best_ckpt = Path(result["best_ckpt"])
    last_ckpt = Path(result["last_ckpt"])
    assert best_ckpt.exists(), "Chưa tạo dcgan_best.pth"
    assert last_ckpt.exists(), "Chưa tạo dcgan_last.pth"
    assert (tmp_path / "samples" / "dcgan" / "sample_epoch_001.png").exists()
    assert (tmp_path / "outputs" / "dcgan_loss_curve.png").exists()
    assert (tmp_path / "outputs" / "dcgan_train_log.json").exists()

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