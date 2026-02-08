import torch
from .model import VAE


def morph_images(
    vae: VAE, img1: torch.Tensor, img2: torch.Tensor, steps: int
) -> list[torch.Tensor]:
    """
    Morph between two images in the latent space of the VAE.
    Args:
        vae: The trained VAE model used for encoding and decoding.
        img1: The first image tensor (C x H x W).
        img2: The second image tensor (C x H x W).
        steps: The number of intermediate images to generate between img1 and img2.
    Returns:
        A list of image tensors representing the morphing sequence from img1 to img2.
    """
    vae.eval()

    with torch.no_grad():
        # Get the mean latent vectors (mu) for both images
        mu1, _ = vae.encode(img1)
        mu2, _ = vae.encode(img2)

        # Generate intermediate latent vectors by linear interpolation
        # Use only the mean instead of sampling to get a smoother morphing sequence
        morph_images = []
        for alpha in torch.linspace(0, 1, steps):
            z = (1 - alpha) * mu1 + alpha * mu2  # Linear interpolation in latent space
            recon_img = vae.decode(z)
            morph_images.append(recon_img.squeeze(0))

    return morph_images
