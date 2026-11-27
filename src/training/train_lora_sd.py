"""[Phase 3 - TV3] Fine-tune Stable Diffusion 1.5 với LoRA theo phong cách Tranh dân gian Đông Hồ.

Chạy độc lập (tối ưu cho Kaggle GPU T4 16GB VRAM):
    python -m src.training.train_lora_sd --config configs/lora_sd.yaml
"""
from __future__ import annotations

import argparse
import gc
import json
import math
import textwrap
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, UnidentifiedImageError
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T

from src.data.custom_dataset import IMG_EXTS, FolderImageDataset
from src.utils import ROOT_DIR, get_device, load_config, set_seed


def resolve_project_path(path: str | Path) -> Path:
    """Chuyển đường dẫn tương đối thành tuyệt đối theo ROOT_DIR."""
    p = Path(path)
    return p if p.is_absolute() else ROOT_DIR / p


def _format_rel_path(path: Path) -> str:
    """Định dạng đường dẫn POSIX tương đối với ROOT_DIR nếu nằm trong workspace."""
    try:
        return path.resolve().relative_to(ROOT_DIR.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def _to_rgb(img: Image.Image) -> Image.Image:
    return img.convert("RGB") if img.mode != "RGB" else img


def build_lora_transform(
    resolution: int = 512,
    center_crop: bool = True,
    random_flip: bool = True,
) -> T.Compose:
    """Pipeline transform RGB chuẩn [-1, 1] cho Stable Diffusion 1.5."""
    ops: list[Any] = [
        T.Lambda(_to_rgb),
        T.Resize(resolution, interpolation=T.InterpolationMode.BICUBIC, antialias=True),
        T.CenterCrop(resolution) if center_crop else T.Resize((resolution, resolution), interpolation=T.InterpolationMode.BICUBIC, antialias=True),
    ]
    if random_flip:
        ops.append(T.RandomHorizontalFlip(p=0.5))
    ops.extend([
        T.ToTensor(),
        T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])
    return T.Compose(ops)


def _has_image_subdirs(root: Path) -> bool:
    """Kiểm tra thư mục có chứa thư mục con (không ẩn) có file ảnh hay không."""
    for entry in root.iterdir():
        if entry.is_dir() and not entry.name.startswith("."):
            if any(p.is_file() and p.suffix.lower() in IMG_EXTS for p in entry.rglob("*")):
                return True
    return False


class DongHoStyleDataset(Dataset):
    """Dataset nạp ảnh phong cách cho LoRA Stable Diffusion.

    Ưu tiên tái sử dụng `FolderImageDataset` từ Phase 1 khi dữ liệu tổ chức theo thư mục lớp
    (vd: `data/custom/tranh_dan_gian/dong_ho/`), đồng thời hỗ trợ fallback đọc ảnh đặt phẳng
    trực tiếp trong thư mục gốc.
    """

    def __init__(
        self,
        root: str | Path,
        prompt: str,
        resolution: int = 512,
        center_crop: bool = True,
        random_flip: bool = True,
        min_size: int = 32,
    ):
        self.root = resolve_project_path(root)
        if not self.root.exists():
            raise FileNotFoundError(
                f"Không tìm thấy thư mục dữ liệu LoRA tại {self.root}. "
                "Kiểm tra lại data/custom/tranh_dan_gian/dong_ho/ hoặc truyền --data-dir."
            )

        self.prompt = prompt
        self.transform = build_lora_transform(
            resolution=resolution,
            center_crop=center_crop,
            random_flip=random_flip,
        )
        self.samples: list[Path] = []
        self.skipped: list[tuple[Path, str]] = []
        self._folder_dataset: FolderImageDataset | None = None

        if _has_image_subdirs(self.root):
            # Tái sử dụng FolderImageDataset của Phase 1 cho cấu trúc có subfolder.
            try:
                ds = FolderImageDataset(self.root, transform=self.transform, min_size=min_size)
                self._folder_dataset = ds
                self.samples = [path for path, _ in ds.samples]
                self.skipped = list(ds.skipped)
            except RuntimeError:
                self._load_flat_directory(min_size=min_size)
        else:
            self._load_flat_directory(min_size=min_size)

        if not self.samples:
            raise RuntimeError(f"Không có ảnh hợp lệ nào trong {self.root}")

        print(
            f"[LoRA SD] Đã nạp {len(self.samples)} ảnh hợp lệ từ {self.root} "
            f"(bỏ qua {len(self.skipped)} file không hợp lệ/quá nhỏ).",
            flush=True,
        )

    def _load_flat_directory(self, min_size: int = 32) -> None:
        candidates = sorted(
            p for p in self.root.iterdir()
            if p.is_file() and not p.name.startswith(".") and p.suffix.lower() in IMG_EXTS
        )
        for path in candidates:
            try:
                with Image.open(path) as im:
                    if min(im.size) < min_size:
                        self.skipped.append((path, "too_small"))
                        continue
            except (UnidentifiedImageError, OSError):
                self.skipped.append((path, "corrupted"))
                continue
            self.samples.append(path)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        if self._folder_dataset is not None:
            img_tensor, _ = self._folder_dataset[idx]
        else:
            path = self.samples[idx]
            with Image.open(path) as im:
                im.load()
                img_tensor = self.transform(im)
        return {
            "pixel_values": img_tensor,
            "prompt": self.prompt,
        }


def plot_lora_loss_curve(
    history: list[dict[str, Any]],
    save_path: str | Path,
    ma_window: int = 20,
) -> Path:
    """Vẽ và lưu biểu đồ loss (raw loss + moving average) của quá trình fine-tune LoRA."""
    save_file = Path(save_path)
    save_file.parent.mkdir(parents=True, exist_ok=True)

    if not history:
        raise ValueError("history rỗng, không thể vẽ biểu đồ loss.")

    steps = np.array([int(row["step"]) for row in history], dtype=np.int64)
    losses = np.array([float(row["loss"]) for row in history], dtype=np.float64)

    window = max(1, min(ma_window, len(losses)))
    if window > 1:
        cumsum = np.cumsum(np.insert(losses, 0, 0.0))
        # Tính moving average với cửa sổ co dần ở đầu chuỗi để khớp độ dài `steps`.
        ma_losses = np.empty_like(losses)
        for i in range(len(losses)):
            start = max(0, i - window + 1)
            ma_losses[i] = (cumsum[i + 1] - cumsum[start]) / (i - start + 1)
    else:
        ma_losses = losses

    plt.figure(figsize=(9, 5))
    plt.plot(steps, losses, label="Raw Loss (MSE)", color="#93c5fd", alpha=0.75, linewidth=1.2)
    plt.plot(steps, ma_losses, label=f"Moving Avg (window={window})", color="#1d4ed8", linewidth=2.2)
    plt.xlabel("Training Step")
    plt.ylabel("Noise Prediction Loss (MSE)")
    plt.title("Stable Diffusion 1.5 LoRA Fine-tuning Loss (Dong Ho Style)")
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_file, dpi=150)
    plt.close()
    return save_file


def create_comparison_grid(
    base_images: Sequence[Image.Image | str | Path],
    lora_images: Sequence[Image.Image | str | Path],
    prompts: Sequence[str],
    save_path: str | Path,
) -> Path:
    """Tạo ảnh ghép so sánh 2 cột: Base SD 1.5 vs. LoRA SD 1.5 (Dong Ho)."""
    if not (len(base_images) == len(lora_images) == len(prompts)):
        raise ValueError("Số lượng base_images, lora_images và prompts phải bằng nhau.")
    if not prompts:
        raise ValueError("Danh sách prompts không được rỗng.")

    def _as_pil(item: Image.Image | str | Path) -> Image.Image:
        if isinstance(item, Image.Image):
            return item.convert("RGB")
        with Image.open(item) as im:
            return im.convert("RGB")

    n_rows = len(prompts)
    fig, axes = plt.subplots(n_rows, 2, figsize=(10.5, 4.6 * n_rows + 0.8), squeeze=False)
    fig.suptitle(
        "So sánh sinh ảnh: Base SD 1.5 vs. LoRA SD 1.5 (Tranh dân gian Đông Hồ)",
        fontsize=14,
        fontweight="bold",
        y=0.995,
    )

    for row_idx, (base_item, lora_item, prompt_text) in enumerate(zip(base_images, lora_images, prompts)):
        base_img = _as_pil(base_item)
        lora_img = _as_pil(lora_item)

        ax_base = axes[row_idx, 0]
        ax_lora = axes[row_idx, 1]

        ax_base.imshow(base_img)
        ax_lora.imshow(lora_img)
        ax_base.axis("off")
        ax_lora.axis("off")

        if row_idx == 0:
            ax_base.set_title("Base SD 1.5", fontsize=12, fontweight="bold", pad=10)
            ax_lora.set_title("LoRA SD 1.5 (Dong Ho)", fontsize=12, fontweight="bold", pad=10)

        short_prompt = textwrap.shorten(prompt_text, width=95, placeholder="...")
        wrapped = "\n".join(textwrap.wrap(f"Prompt {row_idx + 1}: {short_prompt}", width=85))
        ax_base.text(
            1.02,
            -0.05,
            wrapped,
            transform=ax_base.transAxes,
            ha="center",
            va="top",
            fontsize=9.5,
            color="#1f2937",
        )

    plt.tight_layout(rect=(0.0, 0.03, 1.0, 0.965))
    plt.subplots_adjust(hspace=0.28, wspace=0.08)

    out_path = Path(save_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def _save_train_artifacts(
    history: list[dict[str, Any]],
    outputs_dir: Path,
    cfg: dict[str, Any],
    num_images: int,
    trainable_params: int,
    total_unet_params: int,
    max_train_steps: int,
    total_time_sec: float,
    status: str,
) -> None:
    outputs_dir.mkdir(parents=True, exist_ok=True)
    if history:
        plot_lora_loss_curve(history, outputs_dir / "lora_sd_loss_curve.png")

    ratio_pct = round((trainable_params / max(1, total_unet_params)) * 100.0, 4)
    payload = {
        "model": "lora_sd",
        "status": status,
        "config": cfg,
        "num_images": int(num_images),
        "trainable_params": int(trainable_params),
        "total_unet_params": int(total_unet_params),
        "trainable_ratio_pct": ratio_pct,
        "max_train_steps": int(max_train_steps),
        "completed_steps": len(history),
        "total_time_sec": round(total_time_sec, 2),
        "history": history,
    }
    with (outputs_dir / "lora_sd_train_log.json").open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=False)


def _cosine_lr_lambda(current_step: int, warmup_steps: int, max_steps: int) -> float:
    if warmup_steps > 0 and current_step < warmup_steps:
        return float(current_step + 1) / float(max(1, warmup_steps))
    if max_steps <= warmup_steps:
        return 1.0
    progress = float(current_step - warmup_steps) / float(max_steps - warmup_steps)
    progress = min(max(progress, 0.0), 1.0)
    return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))


def _make_autocast(use_fp16: bool):
    if not use_fp16:
        return nullcontext()
    if hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
        return torch.amp.autocast("cuda", dtype=torch.float16)
    return torch.cuda.amp.autocast(dtype=torch.float16)


def _make_grad_scaler(use_fp16: bool):
    if not use_fp16:
        return None
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        return torch.amp.GradScaler("cuda")
    return torch.cuda.amp.GradScaler()


def _release_cuda_memory(*objs: Any) -> None:
    for obj in objs:
        del obj
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def generate_eval_comparison(
    cfg: dict[str, Any],
    checkpoint_dir: str | Path,
    sample_dir: str | Path | None = None,
    device: torch.device | str | None = None,
) -> dict[str, Any]:
    """Sinh ảnh đối chứng Base SD 1.5 vs. LoRA SD 1.5 với cùng prompt và cùng seed."""
    from diffusers import DPMSolverMultistepScheduler, StableDiffusionPipeline
    from peft import PeftModel

    resolved_device = torch.device(device) if device is not None else get_device()
    ckpt_dir = resolve_project_path(checkpoint_dir)
    if not ckpt_dir.exists():
        raise FileNotFoundError(f"Không tìm thấy LoRA checkpoint tại {ckpt_dir}")

    model_cfg = cfg.get("model", {})
    prompt_cfg = cfg.get("prompt", {})
    inf_cfg = cfg.get("inference", {})
    out_cfg = cfg.get("output", {})

    pretrained_model = str(model_cfg.get("pretrained_model", "stable-diffusion-v1-5/stable-diffusion-v1-5"))
    seed = int(cfg.get("train", {}).get("seed", cfg.get("seed", 42)))
    eval_prompts = list(prompt_cfg.get("eval_prompts", []))
    if not eval_prompts:
        base_prompt = str(prompt_cfg.get("base_prompt", "dongho style folk painting"))
        eval_prompts = [base_prompt]
    negative_prompt = str(prompt_cfg.get("negative_prompt", ""))
    num_steps = int(inf_cfg.get("num_inference_steps", 30))
    guidance_scale = float(inf_cfg.get("guidance_scale", 7.5))
    resolution = int(cfg.get("data", {}).get("resolution", 512))

    samples_root = resolve_project_path(sample_dir or out_cfg.get("sample_dir", "outputs/samples"))
    lora_sample_dir = samples_root / "lora_sd"
    lora_sample_dir.mkdir(parents=True, exist_ok=True)

    is_cuda = resolved_device.type == "cuda"
    pipe_dtype = torch.float16 if is_cuda else torch.float32

    print(f"[LoRA SD Eval] Nạp pipeline '{pretrained_model}' ({pipe_dtype}) trên {resolved_device}...", flush=True)
    pipe = StableDiffusionPipeline.from_pretrained(
        pretrained_model,
        torch_dtype=pipe_dtype,
        safety_checker=None,
        requires_safety_checker=False,
    )
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to(resolved_device)
    pipe.enable_attention_slicing()
    pipe.set_progress_bar_config(disable=True)

    base_images: list[Image.Image] = []
    base_paths: list[Path] = []
    print(f"[LoRA SD Eval] Bước A: Sinh {len(eval_prompts)} ảnh đối chứng từ Base SD 1.5...", flush=True)
    with torch.inference_mode():
        for idx, p_text in enumerate(eval_prompts):
            gen = torch.Generator(device=resolved_device).manual_seed(seed + idx)
            img = pipe(
                prompt=p_text,
                negative_prompt=negative_prompt or None,
                num_inference_steps=num_steps,
                guidance_scale=guidance_scale,
                height=resolution,
                width=resolution,
                generator=gen,
            ).images[0]
            out_img = lora_sample_dir / f"base_{idx:02d}.png"
            img.save(out_img)
            base_images.append(img)
            base_paths.append(out_img)

    print(f"[LoRA SD Eval] Bước B: Nạp LoRA adapter từ {ckpt_dir} và sinh lại cùng seed...", flush=True)
    pipe.unet = PeftModel.from_pretrained(pipe.unet, str(ckpt_dir))
    pipe.unet = pipe.unet.to(device=resolved_device, dtype=pipe_dtype)
    pipe.unet.eval()

    lora_images: list[Image.Image] = []
    lora_paths: list[Path] = []
    with torch.inference_mode():
        for idx, p_text in enumerate(eval_prompts):
            gen = torch.Generator(device=resolved_device).manual_seed(seed + idx)
            img = pipe(
                prompt=p_text,
                negative_prompt=negative_prompt or None,
                num_inference_steps=num_steps,
                guidance_scale=guidance_scale,
                height=resolution,
                width=resolution,
                generator=gen,
            ).images[0]
            out_img = lora_sample_dir / f"lora_{idx:02d}.png"
            img.save(out_img)
            lora_images.append(img)
            lora_paths.append(out_img)

    grid_path = create_comparison_grid(
        base_images=base_images,
        lora_images=lora_images,
        prompts=eval_prompts,
        save_path=lora_sample_dir / "base_vs_lora.png",
    )
    print(f"[LoRA SD Eval] Đã lưu lưới ảnh so sánh: {grid_path}", flush=True)

    _release_cuda_memory(pipe)
    return {
        "comparison_grid": _format_rel_path(grid_path),
        "base_images": [_format_rel_path(p) for p in base_paths],
        "lora_images": [_format_rel_path(p) for p in lora_paths],
    }


def train_lora_sd(
    cfg: dict[str, Any],
    data_dir: str | Path | None = None,
    checkpoint_dir: str | Path | None = None,
    output_dir: str | Path | None = None,
    sample_dir: str | Path | None = None,
    device: str | torch.device | None = None,
    skip_eval: bool = False,
) -> dict[str, Any]:
    """Fine-tune Stable Diffusion 1.5 với LoRA trên dữ liệu phong cách tùy chỉnh."""
    from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
    from peft import LoraConfig, get_peft_model
    from transformers import CLIPTextModel, CLIPTokenizer

    train_cfg = cfg.get("train", {})
    data_cfg = cfg.get("data", {})
    model_cfg = cfg.get("model", {})
    lora_cfg = cfg.get("lora", {})
    prompt_cfg = cfg.get("prompt", {})
    out_cfg = cfg.get("output", {})

    seed = int(train_cfg.get("seed", cfg.get("seed", 42)))
    set_seed(seed)

    resolved_device = torch.device(device) if device is not None else get_device()
    is_cuda = resolved_device.type == "cuda"
    mixed_precision = str(train_cfg.get("mixed_precision", "fp16")).lower()
    use_fp16 = is_cuda and mixed_precision == "fp16"
    frozen_dtype = torch.float16 if use_fp16 else torch.float32

    print(f"[LoRA SD Train] Thiết bị: {resolved_device} | mixed_precision={mixed_precision} (fp16={use_fp16})", flush=True)

    # 1. Dataset & DataLoader
    custom_subdir = str(data_cfg.get("custom_subdir", "tranh_dan_gian"))
    resolved_data_dir = (
        resolve_project_path(data_dir)
        if data_dir is not None
        else ROOT_DIR / "data" / "custom" / custom_subdir
    )
    resolution = int(data_cfg.get("resolution", 512))
    center_crop = bool(data_cfg.get("center_crop", True))
    random_flip = bool(data_cfg.get("random_flip", True))
    base_prompt = str(
        prompt_cfg.get(
            "base_prompt",
            "dongho style, traditional Vietnamese Dong Ho folk woodblock painting",
        )
    )

    dataset = DongHoStyleDataset(
        root=resolved_data_dir,
        prompt=base_prompt,
        resolution=resolution,
        center_crop=center_crop,
        random_flip=random_flip,
    )
    batch_size = int(train_cfg.get("batch_size", 1))
    loader_gen = torch.Generator().manual_seed(seed)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        drop_last=False,
        generator=loader_gen,
    )

    # 2. Tải pretrained components
    pretrained_model = str(model_cfg.get("pretrained_model", "stable-diffusion-v1-5/stable-diffusion-v1-5"))
    print(f"[LoRA SD Train] Tải pretrained model: {pretrained_model}...", flush=True)

    tokenizer = CLIPTokenizer.from_pretrained(pretrained_model, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(pretrained_model, subfolder="text_encoder", torch_dtype=frozen_dtype)
    vae = AutoencoderKL.from_pretrained(pretrained_model, subfolder="vae", torch_dtype=frozen_dtype)
    unet = UNet2DConditionModel.from_pretrained(pretrained_model, subfolder="unet", torch_dtype=frozen_dtype)
    noise_scheduler = DDPMScheduler.from_pretrained(pretrained_model, subfolder="scheduler")

    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    unet.requires_grad_(False)
    vae.eval()
    text_encoder.eval()

    vae.to(resolved_device)
    text_encoder.to(resolved_device)

    # Precompute text embedding cho base_prompt rồi giải phóng text_encoder để tiết kiệm VRAM.
    with torch.no_grad():
        tokens = tokenizer(
            [base_prompt],
            padding="max_length",
            max_length=tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        cached_prompt_emb = text_encoder(tokens.input_ids.to(resolved_device))[0].detach()
    _release_cuda_memory(text_encoder, tokenizer)

    # 3. Gắn LoRA vào UNet
    if bool(train_cfg.get("gradient_checkpointing", True)):
        unet.enable_gradient_checkpointing()
        if hasattr(unet, "enable_input_require_grads"):
            unet.enable_input_require_grads()

    peft_config = LoraConfig(
        r=int(lora_cfg.get("rank", 8)),
        lora_alpha=int(lora_cfg.get("alpha", 8)),
        lora_dropout=float(lora_cfg.get("dropout", 0.05)),
        init_lora_weights="gaussian",
        target_modules=list(lora_cfg.get("target_modules", ["to_q", "to_k", "to_v", "to_out.0"])),
    )
    unet = get_peft_model(unet, peft_config)
    unet.to(resolved_device)

    # Giữ tham số trainable LoRA ở float32 để GradScaler unscale an toàn trong chế độ FP16.
    trainable_params_list: list[nn.Parameter] = []
    for param in unet.parameters():
        if param.requires_grad:
            param.data = param.data.to(torch.float32)
            trainable_params_list.append(param)

    trainable_params = sum(p.numel() for p in trainable_params_list)
    total_unet_params = sum(p.numel() for p in unet.parameters())
    ratio_pct = (trainable_params / max(1, total_unet_params)) * 100.0
    print(
        f"[LoRA SD Train] Trainable LoRA params: {trainable_params:,} / {total_unet_params:,} ({ratio_pct:.3f}%)",
        flush=True,
    )

    # 4. Optimizer, Scheduler & GradScaler
    lr = float(train_cfg.get("lr", 1.0e-4))
    weight_decay = float(train_cfg.get("weight_decay", 1.0e-2))
    max_grad_norm = float(train_cfg.get("max_grad_norm", 1.0))
    grad_accum = max(1, int(train_cfg.get("gradient_accumulation_steps", 4)))
    max_train_steps = int(train_cfg.get("max_train_steps", 400))
    warmup_steps = int(train_cfg.get("warmup_steps", 40))
    log_every = max(1, int(train_cfg.get("log_every", 20)))
    save_every = int(train_cfg.get("save_every", 200))

    optimizer = torch.optim.AdamW(trainable_params_list, lr=lr, weight_decay=weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step: _cosine_lr_lambda(step, warmup_steps=warmup_steps, max_steps=max_train_steps),
    )
    scaler = _make_grad_scaler(use_fp16)

    ckpt_root = resolve_project_path(checkpoint_dir or Path(out_cfg.get("checkpoint_dir", "checkpoints")) / "lora_sd")
    ckpt_root.mkdir(parents=True, exist_ok=True)
    outputs_root = resolve_project_path(output_dir or out_cfg.get("log_dir", "outputs"))
    outputs_root.mkdir(parents=True, exist_ok=True)

    # 5. Vòng lặp huấn luyện
    history: list[dict[str, Any]] = []
    global_step = 0
    accum_loss = 0.0
    micro_step = 0
    start_time = time.perf_counter()

    unet.train()
    optimizer.zero_grad(set_to_none=True)
    print(f"[LoRA SD Train] Bắt đầu train {max_train_steps} optimizer steps (grad_accum={grad_accum})...", flush=True)

    while global_step < max_train_steps:
        for batch in dataloader:
            pixel_values = batch["pixel_values"].to(device=resolved_device, dtype=frozen_dtype)

            with torch.no_grad():
                latents = vae.encode(pixel_values).latent_dist.sample()
                latents = latents * vae.config.scaling_factor

            noise = torch.randn_like(latents)
            bsz = latents.shape[0]
            timesteps = torch.randint(
                0,
                noise_scheduler.config.num_train_timesteps,
                (bsz,),
                device=resolved_device,
                dtype=torch.long,
            )
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
            if bool(train_cfg.get("gradient_checkpointing", True)):
                noisy_latents.requires_grad_(True)

            encoder_hidden_states = cached_prompt_emb.expand(bsz, -1, -1)

            with _make_autocast(use_fp16):
                model_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample
                loss = F.mse_loss(model_pred.float(), noise.float(), reduction="mean")
                scaled_loss = loss / grad_accum

            if scaler is not None:
                scaler.scale(scaled_loss).backward()
            else:
                scaled_loss.backward()

            accum_loss += float(loss.detach().item())
            micro_step += 1

            if micro_step % grad_accum == 0:
                if scaler is not None:
                    scaler.unscale_(optimizer)
                if max_grad_norm > 0.0:
                    nn.utils.clip_grad_norm_(trainable_params_list, max_grad_norm)

                if scaler is not None:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()

                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=True)

                global_step += 1
                step_loss = accum_loss / grad_accum
                accum_loss = 0.0
                current_lr = float(lr_scheduler.get_last_lr()[0])
                elapsed = time.perf_counter() - start_time

                history.append({
                    "step": global_step,
                    "loss": round(step_loss, 6),
                    "lr": round(current_lr, 8),
                    "elapsed_sec": round(elapsed, 2),
                })

                if global_step % log_every == 0 or global_step == 1 or global_step == max_train_steps:
                    print(
                        f"Step [{global_step:04d}/{max_train_steps:04d}] "
                        f"loss={step_loss:.4f} | lr={current_lr:.2e} | elapsed={elapsed:.1f}s",
                        flush=True,
                    )
                    _save_train_artifacts(
                        history=history,
                        outputs_dir=outputs_root,
                        cfg=cfg,
                        num_images=len(dataset),
                        trainable_params=trainable_params,
                        total_unet_params=total_unet_params,
                        max_train_steps=max_train_steps,
                        total_time_sec=elapsed,
                        status="running",
                    )

                if save_every > 0 and global_step % save_every == 0 and global_step < max_train_steps:
                    step_ckpt_dir = ckpt_root / f"step_{global_step:04d}"
                    step_ckpt_dir.mkdir(parents=True, exist_ok=True)
                    unet.save_pretrained(str(step_ckpt_dir))
                    print(f"  -> Đã lưu checkpoint trung gian: {step_ckpt_dir}", flush=True)

                if global_step >= max_train_steps:
                    break

    total_time = time.perf_counter() - start_time
    unet.save_pretrained(str(ckpt_root))
    print(f"[LoRA SD Train] Hoàn thành trong {total_time:.1f}s. Đã lưu LoRA adapter tại: {ckpt_root}", flush=True)

    _save_train_artifacts(
        history=history,
        outputs_dir=outputs_root,
        cfg=cfg,
        num_images=len(dataset),
        trainable_params=trainable_params,
        total_unet_params=total_unet_params,
        max_train_steps=max_train_steps,
        total_time_sec=total_time,
        status="completed",
    )

    # Giải phóng toàn bộ bộ nhớ train trước khi nạp pipeline inference.
    _release_cuda_memory(unet, vae, cached_prompt_emb, optimizer, lr_scheduler, scaler)

    eval_info: dict[str, Any] = {}
    if not skip_eval:
        eval_info = generate_eval_comparison(
            cfg=cfg,
            checkpoint_dir=ckpt_root,
            sample_dir=sample_dir,
            device=resolved_device,
        )

    return {
        "checkpoint_dir": _format_rel_path(ckpt_root),
        "log_path": _format_rel_path(outputs_root / "lora_sd_train_log.json"),
        "loss_curve_path": _format_rel_path(outputs_root / "lora_sd_loss_curve.png"),
        "num_images": len(dataset),
        "trainable_params": trainable_params,
        "total_unet_params": total_unet_params,
        "total_time_sec": round(total_time, 2),
        **eval_info,
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Fine-tune Stable Diffusion 1.5 với LoRA (Tranh Đông Hồ).")
    parser.add_argument("--config", default="configs/lora_sd.yaml", help="Đường dẫn file config YAML")
    parser.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE", help="Ghi đè tham số YAML")
    parser.add_argument("--data-dir", default=None, help="Thư mục ảnh train (mặc định: data/custom/<custom_subdir>)")
    parser.add_argument("--output-dir", default=None, help="Thư mục lưu log và biểu đồ loss (mặc định: outputs)")
    parser.add_argument("--checkpoint-dir", default=None, help="Thư mục lưu adapter LoRA (mặc định: checkpoints/lora_sd)")
    parser.add_argument("--max-steps", type=int, default=None, help="Ghi đè số step huấn luyện")
    parser.add_argument("--seed", type=int, default=None, help="Ghi đè random seed")
    parser.add_argument("--skip-eval", action="store_true", help="Bỏ qua bước sinh ảnh đối chứng Base vs LoRA")
    parser.add_argument("--eval-only", action="store_true", help="Chỉ chạy sinh ảnh đối chứng từ checkpoint LoRA đã lưu")
    args = parser.parse_args(argv)

    cfg = load_config(resolve_project_path(args.config), args.set)
    if args.max_steps is not None:
        cfg.setdefault("train", {})["max_train_steps"] = args.max_steps
    if args.seed is not None:
        cfg["seed"] = args.seed
        cfg.setdefault("train", {})["seed"] = args.seed

    ckpt_dir = args.checkpoint_dir or Path(cfg.get("output", {}).get("checkpoint_dir", "checkpoints")) / "lora_sd"

    if args.eval_only:
        result = generate_eval_comparison(cfg=cfg, checkpoint_dir=ckpt_dir)
    else:
        result = train_lora_sd(
            cfg=cfg,
            data_dir=args.data_dir,
            checkpoint_dir=ckpt_dir,
            output_dir=args.output_dir,
            skip_eval=args.skip_eval,
        )
    print(json.dumps(result, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
