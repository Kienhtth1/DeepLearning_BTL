# Kế hoạch theo Phase

## Phase 1 - Khởi tạo dự án & tiền xử lý dữ liệu (08/11 - 11/11) ✅

**Phụ trách:** TV1 + cả nhóm · **Bàn giao:** `data_loader.py`, dataset hoàn chỉnh

- [x] Cấu trúc repo, `requirements.txt`, `.gitignore`, README
- [x] `src/data/transforms.py` - pipeline chuẩn: grayscale/RGB → crop → resize → [-1, 1]
- [x] `src/data/data_loader.py` - `get_dataloader()` cho Fashion-MNIST, CelebA, ảnh tự thu thập
- [x] Chia train/val/test cố định theo seed (Fashion-MNIST: 54k/6k/10k)
- [x] `src/data/custom_dataset.py` - tự loại ảnh hỏng, ảnh quá nhỏ; thư mục con = lớp
- [x] `src/data/prepare.py` (`python main.py data`) - kiểm tra shape/khoảng giá trị, mean/std, phân bố lớp, lưới ảnh mẫu
- [x] Export 10.000 ảnh thật của tập test ra PNG để tính FID ở Phase 3
- [x] Config YAML kế thừa (`base.yaml` → `vae/dcgan/ddpm.yaml`), override từ dòng lệnh
- [x] Giao diện chung `GenerativeModel` cho 3 mô hình
- [x] `notebooks/01_data_exploration.ipynb`, `tests/test_data_loader.py`
- [ ] **Cả nhóm chốt dataset chính** (đề xuất: Fashion-MNIST làm chính vì DDPM train kịp trên Colab; CelebA 64 nếu còn GPU)
- [ ] Mỗi người chạy `python main.py data` + `pytest -q` trên máy/Colab của mình, báo OK trong nhóm
- [ ] (Nếu làm ảnh tự thu thập) gom ảnh vào Drive chung theo cấu trúc trong `data/README.md`

## Phase 2 - Train 3 mô hình cơ sở (12/11 - 20/11)

| Ai | File | Gợi ý |
|---|---|---|
| TV1 | `src/models/vae.py`, `src/training/train_vae.py`, `notebooks/02_vae.ipynb` | Conv VAE, latent 128, loss MSE + β·KL |
| TV2 | `src/models/dcgan.py`, `src/training/train_dcgan.py`, `notebooks/03_dcgan.ipynb` | Kiến trúc DCGAN chuẩn, Adam 2e-4, β1=0.5 |
| TV3 | `src/models/ddpm.py`, `src/training/train_ddpm.py`, `notebooks/04_ddpm.ipynb` | U-Net nhỏ, T=1000, EMA |

Chung: lưu checkpoint `checkpoints/<model>/epoch_XX.pth` (đẩy lên Drive), log loss, lưu lưới ảnh sinh mỗi vài epoch vào `outputs/samples/<model>/`, ghi lại thời gian train/epoch và GPU dùng.

## Phase 3 - Mở rộng + code đánh giá (21/11 - 27/11)

- `src/evaluation/fid_score.py` - FID so với `data/processed/<tag>/fid_real/` (torchmetrics hoặc torch-fidelity; ảnh xám nhân 3 kênh)
- `src/evaluation/benchmark.py` - ảnh/giây khi sinh, số tham số, độ đa dạng (LPIPS giữa các cặp ảnh sinh)
- Mở rộng: nội suy không gian ẩn (VAE/GAN), CGAN / conditional DDPM theo nhãn lớp, LoRA Stable Diffusion theo phong cách tranh dân gian

## Phase 4 - Đánh giá trực diện & code freeze (28/11 - 30/11)

`notebooks/05_evaluation_all.ipynb`: load 3 checkpoint tốt nhất → sinh 10.000 ảnh mỗi mô hình → bảng FID / tốc độ / đa dạng / số tham số + lưới ảnh so sánh. Hoàn thiện `main.py train|sample|evaluate`. Sau 30/11 không thêm tính năng.

## Phase 5 - Báo cáo & slide (01/12 - 04/12)

`report/report.pdf`, `slides/presentation.pptx`.

## Dự phòng 05/12 - 08/12 · Nộp 09/12
