""" Основной скрипт запуска проекта. """

from config import *
from data import train_loader, val_loader, test_loader
from model import ConvVAE
from train import train_epoch
from evaluate import compute_mse_kl_stats, evaluate, evaluate_per_class, prepare_latent_density_model
from visualisation import visulize_model_recon_examples, visualize_ROC_curves, visualize_per_class_auc, visualize_per_class_roc_curves, visualize_latent_space, visualize_distribution_of_scores, visualize_precision_recall_curves, visualize_confusion_matrices
import torch
import os

def find_optimal_alpha(alpha_values=None, epoch_fraction=10):
    """ Находит оптимальную alpha тестируя разные значения """
    if alpha_values is None:
        alpha_values = [0.18, 0.2, 0.25]  # Default test values

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

    # Find optimal alpha where the ratio is closest to 1 (for minimum values as per task)
    best_experiment = min(experiment_results, key=lambda x: abs(x['min_ratio'] - 1.0) if x['min_ratio'] != float('inf') else 999)
    optimal_alpha = best_experiment['alpha']

    print(f"\nOptimal alpha (by min ratio): {optimal_alpha}, min_ratio={best_experiment['min_ratio']:.2f}")

    return optimal_alpha

def train_full_model(optimal_alpha):
    """
    Обучает полную модель VAE с оптимальным значением альфа.
    Возвращает обученную модель.
    """
    print(f"\n--- Full Training with optimal alpha = {optimal_alpha} ---")

    # Create model and optimizer
    model = ConvVAE(latent_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Full training
    for epoch in range(1, num_epochs + 1):
        train_loss = train_epoch(epoch, model, optimizer, train_loader, optimal_alpha)
        print(f"Epoch {epoch}/{num_epochs}: Train loss = {train_loss:.4f}")

        # Save model checkpoint at intervals
        if epoch % save_interval == 0:
            save_path = f"vae_experiment/checkpoints/model_epoch_{epoch}_alpha_{optimal_alpha}.pth"
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            torch.save(model.state_dict(), save_path)
            print(f"Model saved at epoch {epoch}")

    

    return model

def print_in_frame(text, frame_char="#", padding=1):
    """ Форматирование строк вывода. Вывод в рамке. """
    print("\n")
    lines = text.split('\n')    
    max_length = max(len(line) for line in lines)
    border = frame_char * (max_length + 2 * padding + 2)
    print(border)
    if padding > 0:
        empty_line = frame_char + ' ' * (max_length + 2 * padding) + frame_char
        for _ in range(padding):
            print(empty_line)
    for line in lines:
        spaces_needed = max_length - len(line)
        line_with_padding = ' ' * padding + line + ' ' * (spaces_needed + padding)
        print(f"{frame_char}{line_with_padding}{frame_char}")
    if padding > 0:
        for _ in range(padding):
            print(empty_line)
    print(border)
    print("\n")

def main():
    print_in_frame("Step 1: Find optimal alpha")
    # optimal_alpha = find_optimal_alpha()
    optimal_alpha = 0.25
    print(f"optimal_alpha: {optimal_alpha}")

    print_in_frame("Step 2: Train full model with optimal alpha")
    final_save_path = f"vae_experiment/model_optimal_alpha_{optimal_alpha}.pth"

    # Check if model already exists
    if os.path.exists(final_save_path):
        print(f"Model already exists at {final_save_path}, loading...")
        model = ConvVAE(latent_dim).to(device)
        model.load_state_dict(torch.load(final_save_path))
        trained_model = model
    else:
        # Train full model with optimal alpha
        trained_model = train_full_model(optimal_alpha)

        # Save final model
        os.makedirs(os.path.dirname(final_save_path), exist_ok=True)
        torch.save(trained_model.state_dict(), final_save_path)
        print(f"Final model saved to {final_save_path}")

    visulize_model_recon_examples(trained_model)
    visualize_latent_space(trained_model)

    print_in_frame("Step 3: Evaluate on test set")
    test_results = evaluate(trained_model, test_loader, optimal_alpha, split_name="test", val_loader=val_loader)
    visualize_ROC_curves(test_results)
    visualize_distribution_of_scores(test_results)
    visualize_precision_recall_curves(test_results)
    visualize_confusion_matrices(test_results)

    print_in_frame("Step 4: Per-class evaluation")
    # Prepare PCA model once for per-class evaluation
    pca, rv, log_lik_normal = prepare_latent_density_model(trained_model, val_loader)
    per_class_results = evaluate_per_class(trained_model, test_loader, optimal_alpha, pca, rv, log_lik_normal)
    for c, res in sorted(per_class_results.items()):
        print(f"Class {c}: AUC recon {res['auc_recon']:.4f}, ELBO {res['auc_elbo']:.4f}, Latent {res['auc_latent']:.4f}, Latent PCA {res['auc_latent_pca']:.4f} (n={res['n_samples']})")
    visualize_per_class_auc(per_class_results)
    visualize_per_class_roc_curves(per_class_results)


if __name__ == '__main__':
    main()
