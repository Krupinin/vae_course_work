import matplotlib.pyplot as plt
from torchvision import utils
import torch
import os
from data import test_loader
from config import *
from sklearn.metrics import roc_auc_score, roc_curve


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

def visualize_ROC_curves(test_metrics):
    plt.figure(figsize=(10, 8))

    # Negative ELBO ROC
    fpr_elbo, tpr_elbo, _ = roc_curve(test_metrics["labels"], test_metrics["neg_elbo"])
    auc_elbo = roc_auc_score(test_metrics["labels"], test_metrics["neg_elbo"])

    # Reconstruction Error ROC
    fpr_recon, tpr_recon, _ = roc_curve(test_metrics["labels"], test_metrics["recon_err"])
    auc_recon = roc_auc_score(test_metrics["labels"], test_metrics["recon_err"])

    # Latent Energy ROC
    fpr_latent, tpr_latent, _ = roc_curve(test_metrics["labels"], test_metrics["latent_energy"])
    auc_latent = roc_auc_score(test_metrics["labels"], test_metrics["latent_energy"])

    plt.plot(fpr_elbo, tpr_elbo, label=f'Negative ELBO (AUC = {auc_elbo:.3f})', linewidth=2)
    plt.plot(fpr_recon, tpr_recon, label=f'Reconstruction Error (AUC = {auc_recon:.3f})', linewidth=2)
    plt.plot(fpr_latent, tpr_latent, label=f'Latent Energy (AUC = {auc_latent:.3f})', linewidth=2)
    plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves for Different Anomaly Detection Methods')
    plt.legend()
    plt.grid(True)
    os.makedirs('diagrams', exist_ok=True)
    plt.savefig('diagrams/roc_curves.png')
    plt.close()
