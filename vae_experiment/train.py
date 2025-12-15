""" Обучение эпохи """

from tqdm import tqdm
import torch
from losses import recon_mse, kl_divergence
from config import device


def train_epoch(epoch, model, optimizer, train_loader, alpha):
    model.train()
    total_loss = 0.0
    pbar = tqdm(train_loader, desc=f"Train {epoch}")
    for x, y in pbar:
        x = x.to(device)  # device из config, но нужно импортировать
        optimizer.zero_grad()
        recon_x, mu, logvar, z = model(x)
        mse_loss = recon_mse(recon_x, x).mean()  # Используем MSE для reconstruction loss
        kl = kl_divergence(mu, logvar).mean()
        loss = mse_loss + alpha * kl  # loss = MSE + alpha * KLD
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * x.size(0)
        pbar.set_postfix({"loss": loss.item(), "mse": mse_loss.item(), "kl": kl.item()})
    return total_loss / len(train_loader.dataset)
