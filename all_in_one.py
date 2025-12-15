import os
import random
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset, Dataset
from torchvision import datasets, transforms, utils
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve, average_precision_score
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt

#############################
#############################
# настройки
#############################
#############################

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

batch_size = 128
lr = 1e-3
num_epochs = 30
latent_dim = 16
beta = 1.0  # коэффициент при KL (можно менять)
class_idx = 2  # класс FashionMNIST, на котором обучаемся (0..9). По умолчанию 2 = Свитер
save_interval = 5

#############################
#############################
# Датасеты
#############################
#############################

transform = transforms.Compose([
    transforms.ToTensor(),   # [0,1]
])

root = './data'
train_all = datasets.FashionMNIST(root=root, train=True, download=True, transform=transform)
test_all  = datasets.FashionMNIST(root=root, train=False, download=True, transform=transform)

# Фильтруем: train содержит только класс со Свитерами
train_idxs = [i for i, (_, y) in enumerate(train_all) if y == class_idx]
train_ds = Subset(train_all, train_idxs)

# Для валидации/теста подготовим смесь: нормальные (class_idx) + аномальные (все остальные)
def make_eval_subset(dataset, include_all_other_classes=True, max_per_class=None):
    idxs_normal = [i for i, (_, y) in enumerate(dataset) if y == class_idx]
    idxs_anom = [i for i, (_, y) in enumerate(dataset) if y != class_idx]
    if max_per_class is not None:
        # ограничение: взять равное количество из аномалий
        idxs_anom = idxs_anom[:max_per_class]
    idxs = idxs_normal + idxs_anom
    return Subset(dataset, idxs), len(idxs_normal), len(idxs_anom)

val_ds, n_norm_val, n_anom_val = make_eval_subset(test_all, max_per_class=1000)
test_ds, n_norm_test, n_anom_test = make_eval_subset(test_all, max_per_class=None)

train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=False)
val_loader   = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=False)
test_loader  = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=False)

print("Train size (normal only):", len(train_ds))
print("Val size (normal/anom):", len(val_ds), f"({n_norm_val} normal, {n_anom_val} anom)")
print("Test size (normal/anom):", len(test_ds), f"({n_norm_test} normal, {n_anom_test} anom)")


#############################
#############################
# Вариационный автокодировщик
#############################
#############################

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

model = ConvVAE(latent_dim).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

#############################
#############################
# Функции потерь
#############################
#############################

# Reconstruction loss: use BCE per-pixel (since inputs in [0,1]); we compute per-sample sums
def recon_bce(recon_x, x, eps=1e-8):
    # return per-sample reconstruction loss (sum over pixels)
    bce = F.binary_cross_entropy(recon_x, x, reduction='none')
    return bce.view(bce.size(0), -1).sum(dim=1)

def kl_divergence(mu, logvar):
    # returns per-sample KL divergence between q(z|x)=N(mu,var) and p(z)=N(0,I)
    return 0.5 * torch.sum(mu.pow(2) + logvar.exp() - 1.0 - logvar, dim=1)

def recon_mse(recon_x, x):
    mse = F.mse_loss(recon_x, x, reduction='none')
    return mse.view(mse.size(0), -1).sum(dim=1)  # Sum over all pixels for each sample


#############################
#############################
# Обучение
#############################
#############################

def train_epoch(epoch, model, optimizer, train_loader, alpha):
    model.train()
    total_loss = 0.0
    pbar = tqdm(train_loader, desc=f"Train {epoch}")
    for x, y in pbar:
        x = x.to(device)
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



#############################
#############################
# Оценка
#############################
#############################

@torch.no_grad()
def compute_mse_kl_stats(loader, model, alpha):
    model.eval()
    mse_losses = []
    kl_losses = []
    for x, y in loader:
        x = x.to(device)
        recon_x, mu, logvar, z = model(x)
        mse_per_sample = recon_mse(recon_x, x).cpu().numpy()  # per sample MSE sums
        kl_per_sample = kl_divergence(mu, logvar).cpu().numpy()  # per sample KL
        mse_losses.append(mse_per_sample)
        kl_losses.append(kl_per_sample)
    mse_all = np.concatenate(mse_losses)
    kl_all = np.concatenate(kl_losses)
    return {
        'min_mse': np.min(mse_all),
        'min_kl': np.min(kl_all),
        'alpha_min_kl': alpha * np.min(kl_all),
        'avg_mse': np.mean(mse_all),
        'avg_kl': np.mean(kl_all),
        'alpha_avg_kl': alpha * np.mean(kl_all)
    }

@torch.no_grad()
def compute_scores(loader, alpha):
    model.eval()
    all_labels = []
    recon_errors = []
    neg_elbos = []
    latent_neg_logp = []

    for x, y in loader:
        x = x.to(device)
        recon_x, mu, logvar, z = model(x)

        # --- use MSE reconstruction error ---
        rec_per_sample = recon_mse(recon_x, x).cpu().numpy()

        # --- KL divergence ---
        kl_per_sample = kl_divergence(mu, logvar).cpu().numpy()

        # --- negative ELBO proxy ---
        neg_elbo = rec_per_sample + alpha * kl_per_sample

        # --- latent energy = 0.5 * ||z||^2 ---
        z_np = z.cpu().numpy()
        energy = 0.5 * np.sum(z_np**2, axis=1)

        # define anomaly label
        label = (y.numpy() != class_idx).astype(int)

        all_labels.append(label)
        recon_errors.append(rec_per_sample)
        neg_elbos.append(neg_elbo)
        latent_neg_logp.append(energy)

    all_labels = np.concatenate(all_labels)
    recon_errors = np.concatenate(recon_errors)
    neg_elbos = np.concatenate(neg_elbos)
    latent_neg_logp = np.concatenate(latent_neg_logp)

    return all_labels, recon_errors, neg_elbos, latent_neg_logp


def evaluate(loader, alpha, split_name="val"):
    labels, recon_err, neg_elbo, latent_energy = compute_scores(loader, alpha)
    # For each score we compute AUC (higher score = more anomalous)
    auc_recon = roc_auc_score(labels, recon_err)
    auc_elbo  = roc_auc_score(labels, neg_elbo)
    auc_latent = roc_auc_score(labels, latent_energy)
    print(f"[{split_name}] AUC (recon error): {auc_recon:.4f} | AUC (neg ELBO): {auc_elbo:.4f} | AUC (latent energy): {auc_latent:.4f}")
    # also compute average precision
    ap_recon = average_precision_score(labels, recon_err)
    ap_elbo = average_precision_score(labels, neg_elbo)
    ap_latent = average_precision_score(labels, latent_energy)
    print(f"[{split_name}] AP (recon): {ap_recon:.4f} | AP (ELBO): {ap_elbo:.4f} | AP (latent): {ap_latent:.4f}")
    return {
        "labels": labels,
        "recon_err": recon_err,
        "neg_elbo": neg_elbo,
        "latent_energy": latent_energy,
        "auc_recon": auc_recon,
        "auc_elbo": auc_elbo,
        "auc_latent": auc_latent
    }

#############################
#############################
# Запуск
#############################
#############################

# ============ Experiment: finding optimal alpha ============
alpha_values = [0.18, 0.2, 0.25]  # Test different alphas
epoch_fraction = 10  # Reduced training for faster experiments

experiment_results = []

for alpha in alpha_values:
    print(f"\n--- Experiment: alpha = {alpha} ---")
    
    # Create fresh model and optimizer for each alpha experiment
    model = ConvVAE(latent_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    # Train for reduced epochs
    for epoch in range(1, epoch_fraction + 1):
        train_loss = train_epoch(epoch, model, optimizer, train_loader, alpha)
        print(f"Epoch {epoch}: Train loss = {train_loss:.4f}")
    
    # Evaluate MSE and KL stats on validation set
    mse_kl_stats = compute_mse_kl_stats(val_loader, model, alpha)
    
    # Compute ratios for both min and avg values
    min_ratio = mse_kl_stats['min_mse'] / mse_kl_stats['alpha_min_kl'] if mse_kl_stats['min_kl'] > 0 else float('inf')
    avg_ratio = mse_kl_stats['avg_mse'] / mse_kl_stats['alpha_avg_kl']
    
    print(f"Val: min MSE={mse_kl_stats['min_mse']:.2f}, alpha*min KLD={mse_kl_stats['alpha_min_kl']:.2f}, min_ratio={min_ratio:.2f}")
    print(f"Val: avg MSE={mse_kl_stats['avg_mse']:.2f}, alpha*avg KLD={mse_kl_stats['alpha_avg_kl']:.2f}, avg_ratio={avg_ratio:.2f}")
    
    experiment_results.append({
        'alpha': alpha,
        'min_mse': mse_kl_stats['min_mse'],
        'alpha_min_kl': mse_kl_stats['alpha_min_kl'],
        'min_ratio': min_ratio,
        'avg_mse': mse_kl_stats['avg_mse'],
        'alpha_avg_kl': mse_kl_stats['alpha_avg_kl'],
        'avg_ratio': avg_ratio
    })

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
    plt.show()

# Find optimal alpha where the ratio is closest to 1 (for minimum values as per task)
best_experiment = min(experiment_results, key=lambda x: abs(x['min_ratio'] - 1.0) if x['min_ratio'] != float('inf') else 999)
optimal_alpha = best_experiment['alpha']

print(f"\nOptimal alpha (by min ratio): {optimal_alpha}, min_ratio={best_experiment['min_ratio']:.2f}")

# Visualization
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



#############################
#############################
# Графики
#############################
#############################

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

#############################



#############################
#############################
#############################