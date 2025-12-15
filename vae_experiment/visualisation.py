import matplotlib.pyplot as plt
from torchvision import utils
import torch
import os
import numpy as np
from data import test_loader, val_loader
from config import *
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve, average_precision_score, confusion_matrix
from sklearn.decomposition import PCA
import seaborn as sns

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
    """ Визуализация ROC кривых с указанием оптимальных порогов """
    plt.figure(figsize=(10, 8))

    # Negative ELBO ROC
    fpr_elbo, tpr_elbo, thresholds_elbo = roc_curve(test_metrics["labels"], test_metrics["neg_elbo"])
    auc_elbo = roc_auc_score(test_metrics["labels"], test_metrics["neg_elbo"])
    thresh_elbo = test_metrics["thresh_elbo"]
    idx_elbo = np.argmin(np.abs(thresholds_elbo - thresh_elbo))
    plt.plot(fpr_elbo, tpr_elbo, label=f'Negative ELBO (AUC = {auc_elbo:.3f})', linewidth=2)
    plt.plot(fpr_elbo[idx_elbo], tpr_elbo[idx_elbo], 'ro', markersize=8, label=f'ELBO Threshold ({thresh_elbo:.2f})')

    # Reconstruction Error ROC
    fpr_recon, tpr_recon, thresholds_recon = roc_curve(test_metrics["labels"], test_metrics["recon_err"])
    auc_recon = roc_auc_score(test_metrics["labels"], test_metrics["recon_err"])
    thresh_recon = test_metrics["thresh_recon"]
    idx_recon = np.argmin(np.abs(thresholds_recon - thresh_recon))
    plt.plot(fpr_recon, tpr_recon, label=f'Reconstruction Error (AUC = {auc_recon:.3f})', linewidth=2)
    plt.plot(fpr_recon[idx_recon], tpr_recon[idx_recon], 'go', markersize=8, label=f'Recon Threshold ({thresh_recon:.2f})')

    # Latent Energy ROC
    fpr_latent, tpr_latent, thresholds_latent = roc_curve(test_metrics["labels"], test_metrics["latent_energy"])
    auc_latent = roc_auc_score(test_metrics["labels"], test_metrics["latent_energy"])
    thresh_latent = test_metrics["thresh_latent"]
    idx_latent = np.argmin(np.abs(thresholds_latent - thresh_latent))
    plt.plot(fpr_latent, tpr_latent, label=f'Latent Energy (AUC = {auc_latent:.3f})', linewidth=2)
    plt.plot(fpr_latent[idx_latent], tpr_latent[idx_latent], 'mo', markersize=8, label=f'Latent Threshold ({thresh_latent:.2f})')

    # Latent PCA p-value ROC (higher anomaly score = more anomalous)
    if "latent_p_value" in test_metrics and np.any(test_metrics["latent_p_value"] != 0):
        anomaly_score_pca = 1 - test_metrics["latent_p_value"]
        fpr_pca, tpr_pca, thresholds_pca = roc_curve(test_metrics["labels"], anomaly_score_pca)
        auc_pca = roc_auc_score(test_metrics["labels"], anomaly_score_pca)
        thresh_pca = test_metrics["thresh_latent_pca"]
        idx_pca = np.argmin(np.abs(thresholds_pca - thresh_pca))
        plt.plot(fpr_pca, tpr_pca, label=f'Latent PCA p-value (AUC = {auc_pca:.3f})', linewidth=2)
        plt.plot(fpr_pca[idx_pca], tpr_pca[idx_pca], 'co', markersize=8, label=f'PCA Threshold ({thresh_pca:.2f})')

    plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves with Optimal Thresholds')
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
    auc_latent_pca = [per_class_results[c]['auc_latent_pca'] for c in classes]

    # FashionMNIST class names
    class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

    x = np.arange(len(classes))
    width = 0.2

    fig, ax = plt.subplots(figsize=(14,8))
    ax.bar(x - 1.5*width, auc_recon, width, label='Recon Error AUC', alpha=0.8)
    ax.bar(x - 0.5*width, auc_elbo, width, label='Neg ELBO AUC', alpha=0.8)
    ax.bar(x + 0.5*width, auc_latent, width, label='Latent Energy AUC', alpha=0.8)
    ax.bar(x + 1.5*width, auc_latent_pca, width, label='Latent PCA AUC', alpha=0.8)

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
    plt.figure(figsize=(16, 4))

    # Negative ELBO
    plt.subplot(1, 4, 1)
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
    plt.subplot(1, 4, 2)
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
    plt.subplot(1, 4, 3)
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

    # Latent PCA Anomaly Score (1 - p_value)
    if "latent_p_value" in test_metrics and np.any(test_metrics["latent_p_value"] != 0):
        plt.subplot(1, 4, 4)
        anomaly_score_pca = 1 - test_metrics["latent_p_value"]
        normal_scores = anomaly_score_pca[test_metrics["labels"] == 0]
        anomaly_scores = anomaly_score_pca[test_metrics["labels"] == 1]
        plt.hist(normal_scores, bins=50, alpha=0.7, label='Normal', color='blue')
        plt.hist(anomaly_scores, bins=50, alpha=0.7, label='Anomaly', color='red')
        # Threshold at 1 - 0.05 = 0.95 (since p_value < 0.05 -> anomaly_score > 0.95)
        plt.axvline(x=0.95, color='black', linestyle='--', label='Threshold: 0.95')
        plt.xlabel('Latent PCA Anomaly Score (1 - p-value)')
        plt.ylabel('Frequency')
        plt.title('Distribution of Latent PCA Anomaly Scores')
        plt.legend()

    plt.tight_layout()
    os.makedirs('diagrams', exist_ok=True)
    plt.savefig('diagrams/distribution_of_scores.png')
    plt.close()

def visualize_precision_recall_curves(test_metrics):
    """ Визуализация Precision-Recall кривых """
    plt.figure(figsize=(10, 8))

    # Negative ELBO PR
    precision_elbo, recall_elbo, _ = precision_recall_curve(test_metrics["labels"], test_metrics["neg_elbo"])
    ap_elbo = average_precision_score(test_metrics["labels"], test_metrics["neg_elbo"])
    plt.plot(recall_elbo, precision_elbo, label=f'Negative ELBO (AP = {ap_elbo:.3f})', linewidth=2)

    # Reconstruction Error PR
    precision_recon, recall_recon, _ = precision_recall_curve(test_metrics["labels"], test_metrics["recon_err"])
    ap_recon = average_precision_score(test_metrics["labels"], test_metrics["recon_err"])
    plt.plot(recall_recon, precision_recon, label=f'Reconstruction Error (AP = {ap_recon:.3f})', linewidth=2)

    # Latent Energy PR
    precision_latent, recall_latent, _ = precision_recall_curve(test_metrics["labels"], test_metrics["latent_energy"])
    ap_latent = average_precision_score(test_metrics["labels"], test_metrics["latent_energy"])
    plt.plot(recall_latent, precision_latent, label=f'Latent Energy (AP = {ap_latent:.3f})', linewidth=2)

    # Latent PCA p-value PR
    if "latent_p_value" in test_metrics and np.any(test_metrics["latent_p_value"] != 0):
        anomaly_score_pca = 1 - test_metrics["latent_p_value"]
        precision_pca, recall_pca, _ = precision_recall_curve(test_metrics["labels"], anomaly_score_pca)
        ap_pca = average_precision_score(test_metrics["labels"], anomaly_score_pca)
        plt.plot(recall_pca, precision_pca, label=f'Latent PCA p-value (AP = {ap_pca:.3f})', linewidth=2)

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curves')
    plt.legend()
    plt.grid(True)
    os.makedirs('diagrams', exist_ok=True)
    plt.savefig('diagrams/precision_recall_curves.png')
    plt.close()

def visualize_confusion_matrices(test_metrics):
    """ Визуализация confusion matrices для каждого детектора """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.ravel()

    detectors = [
        ('neg_elbo', 'Negative ELBO', test_metrics["thresh_elbo"]),
        ('recon_err', 'Reconstruction Error', test_metrics["thresh_recon"]),
        ('latent_energy', 'Latent Energy', test_metrics["thresh_latent"]),
        ('latent_pca', 'Latent PCA', test_metrics["thresh_latent_pca"])
    ]

    for i, (key, name, thresh) in enumerate(detectors):
        if key == 'latent_pca':
            if "latent_p_value" in test_metrics and np.any(test_metrics["latent_p_value"] != 0):
                scores = 1 - test_metrics["latent_p_value"]
            else:
                continue
        else:
            scores = test_metrics[key]

        predictions = (scores >= thresh).astype(int)
        cm = confusion_matrix(test_metrics["labels"], predictions)

        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[i],
                    xticklabels=['Normal', 'Anomaly'], yticklabels=['Normal', 'Anomaly'])
        axes[i].set_title(f'{name}\nThreshold: {thresh:.2f}')
        axes[i].set_ylabel('True Label')
        axes[i].set_xlabel('Predicted Label')

    plt.tight_layout()
    os.makedirs('diagrams', exist_ok=True)
    plt.savefig('diagrams/confusion_matrices.png')
    plt.close()
