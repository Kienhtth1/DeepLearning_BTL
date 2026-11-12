# src/training - Phase 2

Mỗi thành viên viết `train_<model>.py` (vd `train_vae.py`), đọc config từ `configs/<model>.yaml`,
lấy dữ liệu bằng `get_dataloader_from_config(cfg)` và lưu checkpoint vào `checkpoints/<model>/`.
