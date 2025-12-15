import matplotlib.pyplot as plt
from torchvision import utils
import torch
import os
from data import test_loader, val_loader
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

def visualize_per_class_auc(per_class_results):
    import numpy as np
    classes = sorted(per_class_results.keys())
    auc_recon = [per_class_results[c]['auc_recon'] for c in classes]
    auc_elbo = [per_class_results[c]['auc_elbo'] for c in classes]
    auc_latent = [per_class_results[c]['auc_latent'] for c in classes]

    # FashionMNIST class names
    class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

    x = np.arange(len(classes))
    width = 0.25

    fig, ax = plt.subplots(figsize=(14,8))
    ax.bar(x - width, auc_recon, width, label='Recon Error AUC', alpha=0.8)
    ax.bar(x, auc_elbo, width, label='Neg ELBO AUC', alpha=0.8)
    ax.bar(x + width, auc_latent, width, label='Latent Energy AUC', alpha=0.8)

    ax.set_xlabel('Anomalous Class')
    ax.set_ylabel('AUC Score')
    ax.set_title('AUC for Distinguishing Sweater (Class 2) from Each Anomalous Class')
    ax.set_xticks(x)
    ax.set_xticklabels([f'{c}: {class_names[c]}' for c in classes], rotation=45, ha='right')
    ax.legend()
    ax.grid(True, axis='y')
    ax.set_ylim(0, 1)
    plt.tight_layout()
    os.makedirs('diagrams', exist_ok=True)
    plt.savefig('diagrams/per_class_auc.png')
    plt.close()

def visualize_latent_space(model):
    # 4. Визуализация латентного пространства (2D проекция)
    if latent_dim >= 2:
        model.eval()
        latents = []
        labels_list = []

        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(device)
                _, mu, _, _ = model(x)
                latents.append(mu.cpu().numpy())
                labels_list.append(y.numpy())

        latents = np.concatenate(latents)
        labels_list = np.concatenate(labels_list)

        # Используем PCA для визуализации многомерного пространства
        from sklearn.decomposition import PCA
        pca = PCA(n_components=2)
        latents_2d = pca.fit_transform(latents)

        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(latents_2d[:, 0], latents_2d[:, 1],
                            c=labels_list, cmap='coolwarm', alpha=0.6)
        plt.colorbar(scatter, label='Class (0=Normal, 1=Anomaly)')
        plt.xlabel('First Principal Component')
        plt.ylabel('Second Principal Component')
        plt.title('2D Projection of Latent Space (PCA)')
        # plt.show()
        os.makedirs('diagrams', exist_ok=True)
        plt.savefig('diagrams/latent_space.png')
        plt.close()