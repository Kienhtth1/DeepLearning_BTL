# Quy trình làm việc nhóm

## Nhánh Git

```
main                 # luôn chạy được - chỉ merge qua Pull Request
├── feat/data        # Phase 1
├── feat/vae         # TV1
├── feat/dcgan       # TV2
├── feat/ddpm        # TV3
└── feat/eval        # Phase 3-4
```

```bash
git checkout main && git pull
git checkout -b feat/vae
# ... code ...
git add src/models/vae.py src/training/train_vae.py
git commit -m "feat(vae): encoder/decoder + loss"
git push -u origin feat/vae      # rồi tạo Pull Request vào main, nhờ 1 người review
```

Commit message: `feat(...)`, `fix(...)`, `docs(...)`, `exp(...)` (kết quả thí nghiệm).

## Trước khi tạo PR

- `pytest -q` pass
- Không commit dữ liệu, `.pth`, ảnh lớn (đã chặn trong `.gitignore`)
- Notebook: *Restart & Run All* trước khi commit để output đúng thứ tự

## Lưu trữ chung (Google Drive)

```
DeepLearning_BTL_Drive/
├── data/           # CelebA, ảnh tự thu thập
├── checkpoints/    # vae/, dcgan/, ddpm/
└── results/        # ảnh sinh, bảng số liệu
```

Trên Colab: `from google.colab import drive; drive.mount('/content/drive')` rồi tạo symlink
`!ln -s /content/drive/MyDrive/DeepLearning_BTL_Drive/checkpoints checkpoints`.
