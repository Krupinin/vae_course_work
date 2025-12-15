from config import *
from data import *
from model import ConvVAE
from train import train_epoch
from evaluate import compute_mse_kl_stats, evaluate
from visualization import plot_reconstructions, plot_experiment_results
import torch

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
    plot_reconstructions(model, test_loader)

# Find optimal alpha where the ratio is closest to 1 (for minimum values as per task)
best_experiment = min(experiment_results, key=lambda x: abs(x['min_ratio'] - 1.0) if x['min_ratio'] != float('inf') else 999)
optimal_alpha = best_experiment['alpha']

print(f"\nOptimal alpha (by min ratio): {optimal_alpha}, min_ratio={best_experiment['min_ratio']:.2f}")

# Visualization
plot_experiment_results(experiment_results, optimal_alpha)
