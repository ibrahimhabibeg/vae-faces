from simple_parsing import ArgumentParser
from dataclasses import dataclass
import torch
import numpy as np
from face_vae.model import VAE, Encoder, Decoder, Prior, ConvEncoderNet, DecoderNet
from face_vae.data import CelebAMaskHQ
import random
import matplotlib.pyplot as plt


@dataclass
class ReconstructionGridConfig:
    """
    Configuration for creating a reconstruction grid visualization.
    """

    data_root: str = "../data/celebamask-hq-processed"  # Root directory of preprocessed CelebAMaskHQ dataset
    vae_checkpoint: str = "../checkpoints/vae_epoch_20.pt"  # Path to the trained VAE checkpoint
    num_cols: int = 16  # Number of cols in the grid
    num_rows: int = 16  # Number of rows in the grid
    output_path: str = "reconstruction_grid.png"  # Path to save the output grid image
    seed: int = 42  # Random seed for reproducible image selection
    grid_padding: int = 5  # Padding between images in the grid
    grid_color: tuple = (0, 0, 0)  # Background color


def load_model(config: ReconstructionGridConfig) -> VAE:
    with torch.serialization.safe_globals([]):
        checkpoint = torch.load(config.vae_checkpoint, weights_only=False)
    print(f"Loaded VAE checkpoint from {config.vae_checkpoint}")
    latent_dim = checkpoint["model_state_dict"]["encoder.head_mu.weight"].shape[0]
    encoder_net = ConvEncoderNet()
    encoder = Encoder(
        encoder_net=encoder_net,
        latent_dim=latent_dim,
    )

    decoder_net = DecoderNet(latent_dim=latent_dim)
    decoder = Decoder(decoder_net=decoder_net)

    prior = Prior()

    vae = VAE(encoder=encoder, decoder=decoder, prior=prior)
    vae.load_state_dict(checkpoint["model_state_dict"])
    vae.eval()
    return vae


def select_random_images(dataset: CelebAMaskHQ, num_images: int, seed: int) -> torch.Tensor:
    random.seed(seed)
    indices = random.sample(range(len(dataset)), num_images)
    images = [dataset[i] for i in indices]
    return torch.stack(images)


def reconstruct_images(vae: VAE, images: torch.Tensor) -> torch.Tensor:
    vae.eval()
    with torch.no_grad():
        recon_images, _, _ = vae(images)
    return recon_images


def create_reconstruction_grid(
    original_images: torch.Tensor,
    reconstructed_images: torch.Tensor,
    num_rows: int,
    num_cols: int,
    padding: int = 5,
    color=(255, 255, 255),
):
    B, C, H, W = original_images.shape
    # Create a blank canvas for the grid
    grid_width = num_cols * W + (num_cols - 1) * padding
    grid_height = num_rows * H + (num_rows - 1) * padding
    grid_array = np.full((grid_height, grid_width, 3), color, dtype=np.uint8)
    
    for r, c in [(r, c) for r in range(num_rows) for c in range(num_cols)]:
        idx = (r // 2) * num_cols + c
        
        if r % 2 == 0:
            img = original_images[idx].permute(1, 2, 0).cpu().numpy()
        else:
            img = reconstructed_images[idx].permute(1, 2, 0).cpu().numpy()

        starting_x = c * (W + padding)
        starting_y = r * (H + padding)
        img_uint8 = (img * 255).astype(np.uint8)
        grid_array[starting_y:starting_y + H, starting_x:starting_x + W] = img_uint8
    
    return grid_array

def create_image(grid_array: np.ndarray, output_path: str, num_rows: int):
    height, width, _ = grid_array.shape
    dpi = 100
    figsize = ((width + 150) / dpi, height / dpi)
    
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi, facecolor='black')
    ax.set_facecolor('black')
    ax.imshow(grid_array)
    ax.set_xticks([])
    
    row_height = height / num_rows
    y_positions = [(i + 0.5) * row_height for i in range(num_rows)]
    y_labels = []
    for i in range(num_rows):
        if i % 2 == 0:
            y_labels.append("Original")
        else:
            y_labels.append("Reconstruction")
    
    ax.set_yticks(y_positions)
    ax.set_yticklabels(y_labels, fontsize=12, color='white')
    
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight', dpi=dpi, facecolor='black')
    plt.close(fig)
    print(f"Grid image saved to {output_path}")


def create_reconstruction_visualization(config: ReconstructionGridConfig):
    assert config.num_rows > 0, "num_rows must be greater than 0"
    assert config.num_cols > 0, "num_cols must be greater than 0"
    assert config.num_rows % 2 == 0, "num_rows must be even"
    
    print("=" * 80)
    print("Creating Reconstruction Grid Visualization")
    print("=" * 80)
    print(f"Data root: {config.data_root}")
    print(f"VAE checkpoint: {config.vae_checkpoint}")
    print(f"Number of columns: {config.num_cols}")
    print(f"Number of rows: {config.num_rows}")
    print(f"Output path: {config.output_path}")
    print(f"Random seed: {config.seed}")
    print("=" * 80)
    
    print("\nLoading CelebAMaskHQ dataset...")
    dataset = CelebAMaskHQ(root=config.data_root)
    
    num_images = config.num_cols * config.num_rows // 2  # Total images needed is half the grid size since each image has a reconstruction
    print(f"\nSelecting {num_images} random images...")
    original_images = select_random_images(dataset, num_images, config.seed)
    
    print("\nLoading VAE model...")
    vae = load_model(config)
    
    print("\nReconstructing images...")
    reconstructed_images = reconstruct_images(vae, original_images)
    
    # Create and save grid
    print("\nCreating visualization grid...")
    grid_array = create_reconstruction_grid(
        original_images=original_images,
        reconstructed_images=reconstructed_images,
        num_rows=config.num_rows,
        num_cols=config.num_cols,
        padding=config.grid_padding,
        color=config.grid_color,
    )
    print(f"\nSaving grid image to {config.output_path}...")
    create_image(
        grid_array=grid_array,
        output_path=config.output_path,
        num_rows=config.num_rows
    )
    
    print("\n" + "=" * 80)
    print("Done!")
    print("=" * 80)


if __name__ == "__main__":
    parser = ArgumentParser(description="Create a reconstruction grid visualization from CelebAMaskHQ dataset.")
    parser.add_arguments(ReconstructionGridConfig, dest="config")
    args = parser.parse_args()
    create_reconstruction_visualization(args.config)
