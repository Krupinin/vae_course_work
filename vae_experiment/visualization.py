import matplotlib.pyplot as plt
from torchvision import utils
import torch
from config import device
from model import ConvVAE
from config import latent_dim
from data import test_loader

#############################
#############################
# Графики
#############################
#############################

def plot_reconstructions(model, test_loader, n_show=8):
    model.eval()
    x_batch, y_batch = next(iter(test_loader))
    x_batch = x_batch.to(device)
    with torch.no_grad():
        recon_x, mu, logvar, z = model(x_batch)
    # Выберем первые n_show изображений и покажем оригинал/реконструкцию
    orig = x_batch[:n_show].cpu()
    recon = recon_x[:n_show].cpu()
    grid = torch.cat([orig, recon], dim=0)
    grid_img = utils.make_grid(grid, nrow=n_show, pad_value=1.0)
    plt.figure(figsize=(12,4))
    plt.title("Top row: original (first 8) | Bottom row: reconstructions")
    plt.axis('off')
    plt.imshow(grid_img.permute(1,2,0).squeeze(), cmap='gray')
    plt.show()

def plot_experiment_results(experiment_results, optimal_alpha):
    alphas = [r['alpha'] for r in experiment_results]
    min_ratios = [r['min_ratio'] for r in experiment_results]
    avg_ratios = [r['avg_ratio'] for r in experiment_results]

    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.semilogx(alphas, min_ratios, 'o-', linewidth=2, label='Min ratio: MSE / (α*KLD)')
    plt.axhline(y=1, color='red', linestyle='--')
    plt.axvline(x=optimal_alpha, color='blue', linestyle='--', label=f'Optimal α={optimal_alpha}')
    plt.xlabel('Alpha')
    plt.ylabel('Min Ratio')
    plt.title('Min Ratio vs Alpha')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.semilogx(alphas, avg_ratios, 'o-', linewidth=2, color='green', label='Avg ratio: MSE / (α*KLD)')
    plt.axhline(y=1, color='red', linestyle='--')
    plt.xlabel('Alpha')
    plt.ylabel('Avg Ratio')
    plt.title('Avg Ratio vs Alpha')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()

# # ============ Дополнительная визуализация и анализ ============

# # 1. График распределения скоров для нормальных и аномальных примеров
# plt.figure(figsize=(12, 4))

# plt.subplot(1, 3, 1)
# normal_scores = test_metrics["neg_elbo"][test_metrics["labels"] == 0]
# anomaly_scores = test_metrics["neg_elbo"][test_metrics["labels"] == 1]
# plt.hist(normal_scores, bins=50, alpha=0.7, label='Normal', color='blue')
# plt.hist(anomaly_scores, bins=50, alpha=0.7, label='Anomaly', color='red')
# plt.xlabel('Negative ELBO Score')
# plt.ylabel('Frequency')
# plt.title('Distribution of Negative ELBO Scores')
# plt.legend()
# plt.axvline(x=best_thresh, color='black', linestyle='--', label=f'Threshold: {best_thresh:.1f}')
# plt.legend()

# plt.subplot(1, 3, 2)
# normal_scores = test_metrics["recon_err"][test_metrics["labels"] == 0]
# anomaly_scores = test_metrics["recon_err"][test_metrics["labels"] == 1]
# plt.hist(normal_scores, bins=50, alpha=0.7, label='Normal', color='blue')
# plt.hist(anomaly_scores, bins=50, alpha=0.7, label='Anomaly', color='red')
# plt.xlabel('Reconstruction Error')
# plt.ylabel('Frequency')
# plt.title('Distribution of Reconstruction Errors')
# plt.legend()

# plt.subplot(1, 3, 3)
# normal_scores = test_metrics["latent_energy"][test_metrics["labels"] == 0]
# anomaly_scores = test_metrics["latent_energy"][test_metrics["labels"] == 1]
# plt.hist(normal_scores, bins=50, alpha=0.7, label='Normal', color='blue')
# plt.hist(anomaly_scores, bins=50, alpha=0.7, label='Anomaly', color='red')
# plt.xlabel('Latent Energy')
# plt.ylabel('Frequency')
# plt.title('Distribution of Latent Energy')
# plt.legend()

# plt.tight_layout()
# plt.show()

# print("График 1: Распределение трех метрик аномальности")
# print("• Левый: Negative ELBO - комбинированная метрика (reconstruction + KL divergence)")
# print("• Средний: Reconstruction Error - ошибка восстановления пикселей")
# print("• Правый: Latent Energy - расстояние от центра латентного пространства")
# print("Синий цвет = нормальные примеры (свитеры), Красный цвет = аномалии (другие классы)")

# # 2. ROC кривые для всех трех методов
# plt.figure(figsize=(10, 8))

# # Negative ELBO ROC
# fpr_elbo, tpr_elbo, _ = roc_curve(test_metrics["labels"], test_metrics["neg_elbo"])
# auc_elbo = roc_auc_score(test_metrics["labels"], test_metrics["neg_elbo"])

# # Reconstruction Error ROC
# fpr_recon, tpr_recon, _ = roc_curve(test_metrics["labels"], test_metrics["recon_err"])
# auc_recon = roc_auc_score(test_metrics["labels"], test_metrics["recon_err"])

# # Latent Energy ROC
# fpr_latent, tpr_latent, _ = roc_curve(test_metrics["labels"], test_metrics["latent_energy"])
# auc_latent = roc_auc_score(test_metrics["labels"], test_metrics["latent_energy"])

# plt.plot(fpr_elbo, tpr_elbo, label=f'Negative ELBO (AUC = {auc_elbo:.3f})', linewidth=2)
# plt.plot(fpr_recon, tpr_recon, label=f'Reconstruction Error (AUC = {auc_recon:.3f})', linewidth=2)
# plt.plot(fpr_latent, tpr_latent, label=f'Latent Energy (AUC = {auc_latent:.3f})', linewidth=2)
# plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.title('ROC Curves for Different Anomaly Detection Methods')
# plt.legend()
# plt.grid(True)
# plt.show()

# print("График 2: ROC кривые трех методов обнаружения аномалий")
# print("• Negative ELBO: комбинированный score (Reconstruction + KL)")
# print("• Reconstruction Error: только ошибка восстановления")
# print("• Latent Energy: только расстояние в латентном пространстве")
# print("Чем выше кривая и больше AUC, тем лучше метод разделяет классы")

# # 4. Визуализация латентного пространства (2D проекция)
# if latent_dim >= 2:
#     model.eval()
#     latents = []
#     labels_list = []

#     with torch.no_grad():
#         for x, y in val_loader:
#             x = x.to(device)
#             _, mu, _, _ = model(x)
#             latents.append(mu.cpu().numpy())
#             labels_list.append(y.numpy())

#     latents = np.concatenate(latents)
#     labels_list = np.concatenate(labels_list)

#     # Используем PCA для визуализации многомерного пространства
#     from sklearn.decomposition import PCA
#     pca = PCA(n_components=2)
#     latents_2d = pca.fit_transform(latents)

#     plt.figure(figsize=(10, 8))
#     scatter = plt.scatter(latents_2d[:, 0], latents_2d[:, 1],
#                          c=labels_list, cmap='coolwarm', alpha=0.6)
#     plt.colorbar(scatter, label='Class (0=Normal, 1=Anomaly)')
#     plt.xlabel('First Principal Component')
#     plt.ylabel('Second Principal Component')
#     plt.title('2D Projection of Latent Space (PCA)')
#     plt.show()

#     print("График 4: 2D проекция латентного пространства")
#     print("• Синие точки: нормальные примеры (свитеры)")
#     print("• Красные точки: аномальные примеры (другие классы)")
#     print("• Показывает, насколько хорошо VAE разделяет классы в латентном пространстве")

# # Вывод итоговых результатов
# print("\\n" + "="*60)
# print("ИТОГОВЫЕ РЕЗУЛЬТАТЫ")
# print("="*60)
# print(f"Лучший метод: Latent Energy (AUC = {auc_latent:.4f})")
# print(f"Второй метод: Negative ELBO (AUC = {auc_elbo:.4f})")
# print(f"Третий метод: Reconstruction Error (AUC = {auc_recon:.4f})")
# print(f"\\nВыводы:")
# print(f"1. Метод латентного пространства работает лучше всего")
# print(f"2. Комбинированный метод (ELBO) показывает средние результаты")
# print(f"3. Reconstruction error alone плохо разделяет классы")
# print(f"4. Модель успешно отличает свитеры от других классов FashionMNIST")
