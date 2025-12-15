import matplotlib.pyplot as plt
from torchvision import utils
import torch
import os
from data import test_loader
from config import *

def visulize_model_recon_examples(model):
    # Распечатать несколько примеров реконструкции (нормальные и аномальные)
    model.eval()
    x_batch, y_batch = next(iter(test_loader))
    x_batch = x_batch.to(device)
    with torch.no_grad():
        recon_x, mu, logvar, z = model(x_batch)
    # Выберем первые 8 изображений и покажем оригинал/реконструкцию
    n_show = 8
    orig = x_batch[:n_show].cpu()
    recon = recon_x[:n_show].cpu()
    grid = torch.cat([orig, recon], dim=0)
    grid_img = utils.make_grid(grid, nrow=n_show, pad_value=1.0)
    plt.figure(figsize=(12,4))
    plt.title("Top row: original (first 8) | Bottom row: reconstructions")
    plt.axis('off')
    plt.imshow(grid_img.permute(1,2,0).squeeze(), cmap='gray')
    os.makedirs('diagrams', exist_ok=True)
    plt.savefig('diagrams/recon_examples.png')
    plt.close()
