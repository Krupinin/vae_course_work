""" Вариационный автокодировщик """
import torch
import torch.nn as nn
import torch.nn.functional as F
from config import latent_dim

class ConvEncoder(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        # Input: 1x28x28
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, 4, 2, 1),  # 32x14x14
            nn.ReLU(True),
            nn.Conv2d(32, 64, 4, 2, 1), # 64x7x7
            nn.ReLU(True),
            nn.Conv2d(64, 128, 3, 1, 1),# 128x7x7
            nn.ReLU(True),
        )
        self.flat_dim = 128 * 7 * 7
        self.fc_mu = nn.Linear(self.flat_dim, latent_dim)
        self.fc_logvar = nn.Linear(self.flat_dim, latent_dim)

    def forward(self, x):
        h = self.conv(x)
        h = h.view(h.size(0), -1)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

class ConvDecoder(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.fc = nn.Linear(latent_dim, 128 * 7 * 7)
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 3, 1, 1), # 64x7x7
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),  # 32x14x14
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 1, 4, 2, 1),   # 1x28x28
            # Sigmoid will be applied in forward
        )

    def forward(self, z):
        h = self.fc(z)
        h = h.view(h.size(0), 128, 7, 7)
        x_recon = self.deconv(h)
        return torch.sigmoid(x_recon)

class ConvVAE(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.encoder = ConvEncoder(latent_dim)
        self.decoder = ConvDecoder(latent_dim)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decoder(z)
        return x_recon, mu, logvar, z
