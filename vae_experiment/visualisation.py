import matplotlib.pyplot as plt
from torchvision import utils
import torch
import os
import numpy as np
from data import test_loader, val_loader
from config import *
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.decomposition import PCA

def visulize_model_recon_examples(model):
    """ Несколько примеров реконструкции (нормальные и аномальные) """
    model.eval()
    x_batch, y_batch = next(iter(test_loader))
    x_batch = x_batch.to(device)
    with torch.no_grad():
        recon_x, mu, logvar, z = model(x_batch)
    # Выберем первые 8 изображений и покажем оригинал/реконструкцию
    n_show = 20
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
    """ Визуализация ROC кривых """
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
    """ Визуализация латентного пространства (2D проекция) """
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
        pca = PCA(n_components=2)
        latents_2d = pca.fit_transform(latents)

        # FashionMNIST class names
        class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

        colors = plt.cm.tab10(np.linspace(0, 1, 10)) # Colors for each class

        plt.figure(figsize=(10, 8))
        for i in range(10):
            mask = labels_list == i
            plt.scatter(latents_2d[mask, 0], latents_2d[mask, 1], color=colors[i], label=class_names[i], alpha=0.6)
        plt.xlabel('First Principal Component')
        plt.ylabel('Second Principal Component')
        plt.title('2D Projection of Latent Space (PCA)')
        plt.legend(loc='upper right')
        os.makedirs('diagrams', exist_ok=True)
        plt.savefig('diagrams/latent_space.png')
        plt.close()

def visualize_distribution_of_scores(test_metrics):
    """ График распределения скоров для нормальных и аномальных примеров """
    plt.figure(figsize=(12, 4))

    # Negative ELBO
    plt.subplot(1, 3, 1)
    normal_scores = test_metrics["neg_elbo"][test_metrics["labels"] == 0]
    anomaly_scores = test_metrics["neg_elbo"][test_metrics["labels"] == 1]
    plt.hist(normal_scores, bins=50, alpha=0.7, label='Normal', color='blue')
    plt.hist(anomaly_scores, bins=50, alpha=0.7, label='Anomaly', color='red')
    # Calculate best threshold using Youden's J statistic
    fpr, tpr, thresholds = roc_curve(test_metrics["labels"], test_metrics["neg_elbo"])
    best_thresh_elbo = thresholds[np.argmax(tpr - fpr)]
    plt.axvline(x=best_thresh_elbo, color='black', linestyle='--', label=f'Threshold: {best_thresh_elbo:.2f}')
    plt.xlabel('Negative ELBO Score')
    plt.ylabel('Frequency')
    plt.title('Distribution of Negative ELBO Scores')
    plt.legend()

    # Reconstruction Error
    plt.subplot(1, 3, 2)
    normal_scores = test_metrics["recon_err"][test_metrics["labels"] == 0]
    anomaly_scores = test_metrics["recon_err"][test_metrics["labels"] == 1]
    plt.hist(normal_scores, bins=50, alpha=0.7, label='Normal', color='blue')
    plt.hist(anomaly_scores, bins=50, alpha=0.7, label='Anomaly', color='red')
    # Calculate best threshold
    fpr, tpr, thresholds = roc_curve(test_metrics["labels"], test_metrics["recon_err"])
    best_thresh_recon = thresholds[np.argmax(tpr - fpr)]
    plt.axvline(x=best_thresh_recon, color='black', linestyle='--', label=f'Threshold: {best_thresh_recon:.2f}')
    plt.xlabel('Reconstruction Error')
    plt.ylabel('Frequency')
    plt.title('Distribution of Reconstruction Errors')
    plt.legend()

    # Latent Energy
    plt.subplot(1, 3, 3)
    normal_scores = test_metrics["latent_energy"][test_metrics["labels"] == 0]
    anomaly_scores = test_metrics["latent_energy"][test_metrics["labels"] == 1]
    plt.hist(normal_scores, bins=50, alpha=0.7, label='Normal', color='blue')
    plt.hist(anomaly_scores, bins=50, alpha=0.7, label='Anomaly', color='red')
    # Calculate best threshold
    fpr, tpr, thresholds = roc_curve(test_metrics["labels"], test_metrics["latent_energy"])
    best_thresh_latent = thresholds[np.argmax(tpr - fpr)]
    plt.axvline(x=best_thresh_latent, color='black', linestyle='--', label=f'Threshold: {best_thresh_latent:.2f}')
    plt.xlabel('Latent Energy')
    plt.ylabel('Frequency')
    plt.title('Distribution of Latent Energy')
    plt.legend()

    plt.tight_layout()
    os.makedirs('diagrams', exist_ok=True)
    plt.savefig('diagrams/distribution_of_scores.png')
    plt.close()
