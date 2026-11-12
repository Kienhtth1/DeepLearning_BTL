"""[Phase 2 - TV3] DDPM nhỏ (Ho et al., 2020).

Kế thừa src.models.base.GenerativeModel. Gợi ý: U-Net nhỏ + time embedding sinusoidal,
T=1000, beta linear 1e-4 -> 0.02, loss MSE trên noise; Phase 3 có thể thêm sampler DDIM để tăng tốc.
"""
