import sys
sys.path.append('..')

import torch
from models.VAE.CNN_VAE import ConvVAE

encoder = ConvVAE(latent_dim=64)
encoder.load_ckpt('../ckpts/CNNVAE_1.ckpt')

sample = torch.randn(1, 3, 64, 64)
state = encoder.obs_to_z(sample)
reconstructed = encoder.sample(state)

print(reconstructed.shape)
print(state.shape)

print("Success!")