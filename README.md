# BTL Deep Learning - Đề tài 4: Mô hình sinh ảnh VAE, GAN và Diffusion

Cài đặt **VAE**, **DCGAN** và **DDPM (bản nhỏ)** trên cùng một tập dữ liệu, rồi so sánh
**chất lượng ảnh (FID)**, **độ đa dạng** và **chi phí tính toán (tốc độ sinh, số tham số, thời gian train)**.

| | |
|---|---|
| **Đầu vào** | Lúc train: tập ảnh. Lúc sinh: vector nhiễu ngẫu nhiên (+ nhãn lớp nếu conditional) |
| **Đầu ra** | Ảnh mới cùng phân phối với dữ liệu train |
| **Dữ liệu** | Fashion-MNIST 32×32 (mặc định) · CelebA 64×64 · ảnh tranh dân gian / trang phục tự thu thập |
| **Kiến thức** | Autoencoder, VAE, GAN, DDPM, FID |
| **Mở rộng** | Conditional generation · nội suy không gian ẩn · fine-tune Stable Diffusion bằng LoRA |

## Tiến độ

| Phase | Thời gian | Nội dung | Trạng thái |
|---|---|---|---|
| 1 | 08/11 - 11/11 | Khởi tạo dự án & tiền xử lý dữ liệu | ✅ Xong |
| 2 | 12/11 - 20/11 | Train 3 mô hình cơ sở (TV1 VAE, TV2 DCGAN, TV3 DDPM) | ⏳ |
| 3 | 21/11 - 27/11 | Phần mở rộng + `fid_score.py`, `benchmark.py` | ⏳ |
| 4 | 28/11 - 30/11 | Đánh giá trực diện & code freeze | ⏳ |
| 5 | 01/12 - 04/12 | Báo cáo & slide | ⏳ |
| Dự phòng | 05/12 - 08/12 | Fix lỗi, sửa báo cáo | |
| **Nộp** | **09/12** | | |

Chi tiết từng phase và checklist: [`docs/PHASE_PLAN.md`](docs/PHASE_PLAN.md) · Quy trình làm việc nhóm: [`docs/CONTRIBUTING.md`](docs/CONTRIBUTING.md)

## Cài đặt nhanh

```bash
git clone https://github.com/Kienhtth1/DeepLearning_BTL.git
cd DeepLearning_BTL
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python main.py data      # tải Fashion-MNIST, kiểm tra, lưu ảnh mẫu, export ảnh thật cho FID
pytest -q                # chạy kiểm thử
```

Trên **Google Colab**: `!git clone ...`, `%cd DeepLearning_BTL`, `!pip install -r requirements.txt`, `!python main.py data`.

## Cấu trúc thư mục

```
DeepLearning_BTL/
├── main.py                    # điểm chạy chung: data | train | sample | evaluate
├── requirements.txt
├── configs/
│   ├── base.yaml              # cấu hình dữ liệu dùng chung (seed, dataset, batch size...)
│   ├── celeba.yaml            # ghi đè để chạy CelebA 64×64
│   ├── vae.yaml  dcgan.yaml  ddpm.yaml   # siêu tham số từng mô hình (Phase 2)
├── src/
│   ├── data/                  # ✅ PHASE 1
│   │   ├── data_loader.py     #    get_dataloader() - API dữ liệu duy nhất cho cả nhóm
│   │   ├── transforms.py      #    tiền xử lý chuẩn: resize, crop, chuẩn hoá [-1, 1]
│   │   ├── custom_dataset.py  #    đọc ảnh tự thu thập, tự loại ảnh hỏng
│   │   └── prepare.py         #    kiểm tra dữ liệu + export ảnh thật cho FID
│   ├── models/
│   │   ├── base.py            #    giao diện chung GenerativeModel (training_loss, sample)
│   │   ├── vae.py             #    Phase 2 - TV1
│   │   ├── dcgan.py           #    Phase 2 - TV2
│   │   └── ddpm.py            #    Phase 2 - TV3
│   ├── training/              # Phase 2: train_vae.py, train_dcgan.py, train_ddpm.py
│   ├── evaluation/            # Phase 3: fid_score.py, benchmark.py
│   └── utils/common.py        # seed, device, đọc config, lưu lưới ảnh
├── notebooks/
│   └── 01_data_exploration.ipynb   # ✅ Phase 1 (02-04 cho từng mô hình, 05_evaluation_all ở Phase 4)
├── tests/test_data_loader.py
├── data/                      # KHÔNG commit dữ liệu - xem data/README.md
├── checkpoints/               # file .pth (không commit, lưu Drive)
├── outputs/                   # ảnh sinh, biểu đồ
├── docs/                      # kế hoạch, quy trình nhóm
├── report/                    # report.pdf (Phase 5)
└── slides/                    # presentation.pptx (Phase 5)
```

## Quy ước chung (mọi người phải theo)

1. **Lấy dữ liệu** chỉ qua `get_dataloader` / `get_dataloader_from_config` - không tự viết loader riêng.
2. **Ảnh luôn trong [-1, 1]**, shape `(B, C, H, W)`. Decoder/Generator dùng `Tanh` ở cuối; VAE dùng MSE thay cho BCE.
3. Mô hình kế thừa `src.models.base.GenerativeModel`, có `sample(n)` trả ảnh trong [-1, 1] - để Phase 4 đánh giá chung một đoạn code.
4. **Cùng seed (42) và cùng cách chia train/val** (đã cố định trong code) để so sánh công bằng.
5. Không commit dữ liệu, checkpoint `.pth` - lưu trên Google Drive chung của nhóm.

## Dùng data loader trong code của bạn

```python
from src.data import get_dataloader, denormalize
from src.utils import load_config, set_seed

cfg = load_config("configs/vae.yaml")          # kế thừa base.yaml
set_seed(cfg["seed"])
loader, info = get_dataloader("fashion_mnist", split="train", batch_size=128)
print(info.channels, info.image_size, info.num_classes)   # 1 32 10

for x, y in loader:          # x ∈ [-1, 1], shape (128, 1, 32, 32)
    ...
```

Đổi dataset không cần sửa code:
```bash
python main.py data --config configs/celeba.yaml
python main.py data --set data.name=custom data.custom_subdir=tranh_dan_gian
python main.py data --set data.subset_size=2000 --skip-fid    # debug nhanh
```

## Thành viên

| | Họ tên | Phụ trách chính |
|---|---|---|
| TV1 | _(điền tên)_ | Dữ liệu (Phase 1) · VAE |
| TV2 | _(điền tên)_ | DCGAN |
| TV3 | _(điền tên)_ | DDPM |
