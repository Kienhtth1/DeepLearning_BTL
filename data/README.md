# Dữ liệu

Thư mục `raw/`, `processed/`, `custom/` **không được commit** lên GitHub (đã có trong `.gitignore`).
Mỗi thành viên tự tải về máy / Colab / Drive theo hướng dẫn dưới đây.

## 1. Fashion-MNIST (mặc định) - tự tải

```bash
python main.py data
```

Ảnh 28×28 xám được resize lên **32×32** (để U-Net của DDPM và DCGAN chia đôi 3 lần gọn: 32 → 16 → 8 → 4).
Chia: train 54.000 / val 6.000 (tách cố định từ 60k train gốc) / test 10.000 (test gốc).

## 2. CelebA 64×64 - tải thủ công

Link Google Drive của torchvision hay bị giới hạn lượt tải, nên dùng bản Kaggle:

```bash
pip install kaggle   # cần file ~/.kaggle/kaggle.json
kaggle datasets download -d jessicali9530/celeba-dataset -p data/raw/celeba --unzip
python main.py data --config configs/celeba.yaml
```

Cấu trúc cần có:
```
data/raw/celeba/
├── img_align_celeba/          # 202.599 ảnh .jpg (bản Kaggle lồng thêm 1 lớp cùng tên - code tự xử lý)
├── list_eval_partition.csv    # chia train/val/test chuẩn (162.770 / 19.867 / 19.962)
└── list_attr_celeba.csv       # tuỳ chọn - dùng làm nhãn khi làm conditional (vd Smiling)
```
Tiền xử lý: crop giữa 140×140 (bỏ nền) → resize 64×64, lật ngang ngẫu nhiên ở tập train.

> Train DDPM trên CelebA đầy đủ rất tốn GPU. Có thể dùng `--set data.subset_size=50000`.

## 3. Ảnh tự thu thập (tranh dân gian / trang phục truyền thống)

```
data/custom/tranh_dan_gian/
├── dong_ho/        # mỗi thư mục con = 1 lớp (nếu muốn conditional)
├── hang_trong/
└── kim_hoang/
```
```bash
python main.py data --set data.name=custom data.custom_subdir=tranh_dan_gian data.image_size=64
```
Ảnh hỏng hoặc cạnh < 32px sẽ tự bị loại. Chia train 90% / val 10% cố định theo seed.
Nên có **≥ 2.000 ảnh** để GAN/Diffusion học được; nếu ít hơn thì dùng cho phần mở rộng LoRA.

## Đầu ra của Phase 1

| Đường dẫn | Nội dung |
|---|---|
| `outputs/data_samples/<tag>.png` | Lưới 64 ảnh mẫu sau tiền xử lý |
| `data/processed/<tag>/dataset_info.json` | Số ảnh mỗi split, shape, mean/std, phân bố lớp |
| `data/processed/<tag>/fid_real/` | 10.000 ảnh thật (tập test) dạng PNG - Phase 3 dùng để tính FID |
