import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve, average_precision_score
import torch
from losses import recon_mse, kl_divergence
from config import class_idx, device

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
        x = x.to(device)  # device из config
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
