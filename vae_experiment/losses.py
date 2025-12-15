import torch
import torch.nn.functional as F

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
