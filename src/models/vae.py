"""[Phase 2 - TV1] Variational Autoencoder.

Kế thừa src.models.base.GenerativeModel. Gợi ý: encoder Conv -> (mu, logvar),
reparameterization, decoder ConvTranspose; loss = MSE (ảnh [-1,1]) + beta * KL.
"""
