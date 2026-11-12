"""Tiện ích chung: seed, device, đọc config, lưu lưới ảnh."""
from __future__ import annotations

import copy
import os
import random
from pathlib import Path

import numpy as np
import torch
import yaml

ROOT_DIR = Path(__file__).resolve().parents[2]


def set_seed(seed: int = 42, deterministic: bool = False) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_config(path: str | Path, overrides: list[str] | None = None) -> dict:
    """Đọc YAML; hỗ trợ `_base_: base.yaml` để kế thừa và override dạng `data.batch_size=64`."""
    path = Path(path)
    with open(path, encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh) or {}
    base_name = cfg.pop("_base_", None)
    if base_name:
        cfg = _deep_merge(load_config(path.parent / base_name), cfg)
    for ov in overrides or []:
        key, val = ov.split("=", 1)
        _set_by_path(cfg, key, yaml.safe_load(val))
    return cfg


def _deep_merge(a: dict, b: dict) -> dict:
    out = copy.deepcopy(a)
    for k, v in b.items():
        out[k] = _deep_merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def _set_by_path(cfg: dict, dotted: str, value) -> None:
    keys = dotted.split(".")
    d = cfg
    for k in keys[:-1]:
        d = d.setdefault(k, {})
    d[keys[-1]] = value


def save_image_grid(x: torch.Tensor, path: str | Path, nrow: int = 8) -> Path:
    """Lưu batch ảnh trong [-1, 1] thành 1 file PNG dạng lưới."""
    from torchvision.utils import save_image

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    save_image((x.detach().cpu() + 1) / 2, path, nrow=nrow, padding=2)
    return path
