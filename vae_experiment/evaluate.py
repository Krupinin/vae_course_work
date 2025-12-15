import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve, average_precision_score
from sklearn.decomposition import PCA
from scipy.stats import multivariate_normal
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
def prepare_latent_density_model(model, val_loader):
    """
    Prepare PCA and multivariate gaussian model for latent space density.
    Returns PCA, multivariate_normal, and log_likelihoods for normal validation samples.
    """
    model.eval()
    latents = []
    with torch.no_grad():
        for x, y in val_loader:
            x = x.to(device)
            _, _, _, z = model(x)
            # Only normal samples (y == class_idx)
            mask = y == class_idx
            latents.append(z[mask].cpu().numpy())
    latents = np.concatenate(latents)

    # PCA to 2D
    pca = PCA(n_components=2)
    latents_2d = pca.fit_transform(latents)

    # Fit multivariate gaussian
    mean = np.mean(latents_2d, axis=0)
    cov = np.cov(latents_2d.T)
    rv = multivariate_normal(mean=mean, cov=cov)

    # Log likelihoods for normal samples
    log_lik_normal = rv.logpdf(latents_2d)

    return pca, rv, log_lik_normal

@torch.no_grad()
def compute_scores(model, loader, alpha, pca=None, rv=None, log_lik_normal=None):
    model.eval()
    all_labels = []
    recon_errors = []
    neg_elbos = []
    latent_neg_logp = []
    latent_p_value = []

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

        # --- latent 2D PCA log likelihood p-value ---
        if pca is not None and rv is not None and log_lik_normal is not None:
            latent_2d = pca.transform(z_np)
            latent_log_lik = rv.logpdf(latent_2d)
            # p_value: fraction of normal samples with log_lik <= this (lower = more anomalous)
            p_values = np.array([np.sum(log_lik_normal <= lik) / len(log_lik_normal) for lik in latent_log_lik])
            latent_p_value.append(p_values)
        else:
            latent_p_value.append(np.zeros_like(energy))  # placeholder

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
    latent_p_value = np.concatenate(latent_p_value)

    return all_labels, recon_errors, neg_elbos, latent_neg_logp, latent_p_value


def evaluate_per_class(model, test_loader, alpha, pca=None, rv=None, log_lik_normal=None):
    """
    Evaluate model's ability to distinguish training class (class_idx) from each other class.
    Returns dict with AUC for each anomalous class.
    """
    model.eval()

    # Collect all test data
    all_x, all_y = [], []
    for x, y in test_loader:
        all_x.append(x)
        all_y.append(y)
    all_x = torch.cat(all_x)
    all_y = torch.cat(all_y)

    results = {}
    for c in range(10):
        if c == class_idx:
            continue

        # Filter for class_idx and c
        mask = (all_y == class_idx) | (all_y == c)
        x_subset = all_x[mask]
        y_subset = all_y[mask]

        if len(x_subset) == 0:
            continue

        # Compute scores
        x_subset = x_subset.to(device)
        with torch.no_grad():
            recon_x, mu, logvar, z = model(x_subset)

        rec_err = recon_mse(recon_x, x_subset).cpu().numpy()
        kl = kl_divergence(mu, logvar).cpu().numpy()
        neg_elbo = rec_err + alpha * kl
        z_np = z.cpu().numpy()
        energy = 0.5 * np.sum(z_np**2, axis=1)

        # Compute p_values if available
        if pca is not None and rv is not None and log_lik_normal is not None:
            latent_2d = pca.transform(z_np)
            latent_log_lik = rv.logpdf(latent_2d)
            p_values = np.array([np.sum(log_lik_normal <= lik) / len(log_lik_normal) for lik in latent_log_lik])
            auc_latent_pca = roc_auc_score((y_subset.numpy() == c).astype(int), 1 - p_values)
        else:
            auc_latent_pca = 0.5

        # Labels: 1 for anomalous (c), 0 for normal (class_idx)
        labels = (y_subset.numpy() == c).astype(int)

        # Compute AUC
        auc_recon = roc_auc_score(labels, rec_err)
        auc_elbo = roc_auc_score(labels, neg_elbo)
        auc_latent = roc_auc_score(labels, energy)

        results[c] = {
            'auc_recon': auc_recon,
            'auc_elbo': auc_elbo,
            'auc_latent': auc_latent,
            'auc_latent_pca': auc_latent_pca,
            'n_samples': len(labels)
        }

    return results

def evaluate(model, loader, alpha, split_name="val", val_loader=None):
    # Prepare latent density model if val_loader provided
    if val_loader is not None:
        pca, rv, log_lik_normal = prepare_latent_density_model(model, val_loader)
        labels, recon_err, neg_elbo, latent_energy, latent_p_value = compute_scores(model, loader, alpha, pca, rv, log_lik_normal)
    else:
        labels, recon_err, neg_elbo, latent_energy, latent_p_value = compute_scores(model, loader, alpha)

    # For each score we compute AUC (higher score = more anomalous)
    auc_recon = roc_auc_score(labels, recon_err)
    auc_elbo  = roc_auc_score(labels, neg_elbo)
    auc_latent = roc_auc_score(labels, latent_energy)
    # For p_value: lower p_value = more anomalous, so use 1 - p_value for AUC
    auc_latent_pca = roc_auc_score(labels, 1 - latent_p_value) if np.any(latent_p_value != 0) else 0.5
    print(f"[{split_name}] AUC (recon error): {auc_recon:.4f} | AUC (neg ELBO): {auc_elbo:.4f} | AUC (latent energy): {auc_latent:.4f} | AUC (latent PCA p-value): {auc_latent_pca:.4f}")
    # also compute average precision
    ap_recon = average_precision_score(labels, recon_err)
    ap_elbo = average_precision_score(labels, neg_elbo)
    ap_latent = average_precision_score(labels, latent_energy)
    ap_latent_pca = average_precision_score(labels, 1 - latent_p_value) if np.any(latent_p_value != 0) else 0.5
    print(f"[{split_name}] AP (recon): {ap_recon:.4f} | AP (ELBO): {ap_elbo:.4f} | AP (latent): {ap_latent:.4f} | AP (latent PCA): {ap_latent_pca:.4f}")
    return {
        "labels": labels,
        "recon_err": recon_err,
        "neg_elbo": neg_elbo,
        "latent_energy": latent_energy,
        "latent_p_value": latent_p_value,
        "auc_recon": auc_recon,
        "auc_elbo": auc_elbo,
        "auc_latent": auc_latent,
        "auc_latent_pca": auc_latent_pca
    }
