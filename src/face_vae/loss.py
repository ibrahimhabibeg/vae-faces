import torch
import torch.nn.functional as F


def bce_reconstruction_loss(recon_x, x):
    """
    Binary Cross Entropy reconstruction loss for VAE.
    Args:
        recon_x: Reconstructed image (output of decoder).
        x: Original image (input to encoder).
    Returns:
        BCE loss value.
    """
    assert recon_x.shape == x.shape, (
        "Reconstructed and original images must have the same shape"
    )
    bce_loss = F.binary_cross_entropy(recon_x, x, reduction="sum")
    return bce_loss


def mse_reconstruction_loss(recon_x, x):
    """
    Mean Squared Error reconstruction loss for VAE.
    Args:
        recon_x: Reconstructed image (output of decoder).
        x: Original image (input to encoder).
    Returns:
        MSE loss value.
    """
    assert recon_x.shape == x.shape, (
        "Reconstructed and original images must have the same shape"
    )
    mse_loss = F.mse_loss(recon_x, x, reduction="sum")
    return mse_loss


def kl_divergence_loss(mu, logvar):
    """
    KL Divergence loss for VAE.
    Assumes the prior p(z) is a standard normal distribution N(0, I) and the approximate posterior q(z|x) is a diagonal Gaussian with mean mu and log variance logvar.
    This is based on the closed-form solution for KL divergence between two Gaussians: KL(q(z|x) || p(z)) = -0.5 * sum(1 + logvar - mu^2 - variance)
    Args:
        mu: Mean of the latent distribution (output of encoder).
        logvar: Log variance of the latent distribution (output of encoder).
    Returns:
        KL divergence loss value.
    """
    var = torch.exp(logvar)
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - var)
    return kl_loss


def vae_loss(recon_x, x, mu, logvar, recon_loss_fn=bce_reconstruction_loss, beta=1.0):
    """
    Total VAE loss combining reconstruction loss and regularization (KL divergence) loss.
    The KL divergence is weighted by a factor.
    The used regularization assumes p(z) is a standard normal distribution N(0, I) and q(z|x) is a diagonal Gaussian with mean mu and log variance logvar.
    Increase the beta factor will make images generated from random latent vectors sampled from the prior look more like real images, but may reduce the quality of reconstructions.
    Decrease it will improve reconstructions but may make generated images look worse as th latent space might have gaps and q(z|x) might not match p(z) well.
        recon_x: Reconstructed image (output of decoder).
        x: Original image (input to encoder).
        mu: Mean of the latent distribution (output of encoder).
        logvar: Log variance of the latent distribution (output of encoder).
        recon_loss_fn: Reconstruction loss function to use (default is binary cross-entropy).
        beta: Weighting factor for the KL divergence loss (default is 1.0).
    Returns:
        Total VAE loss value.
    """
    recon_loss = recon_loss_fn(recon_x, x)
    kl_loss = kl_divergence_loss(mu, logvar)
    total_loss = recon_loss + beta * kl_loss
    return total_loss
