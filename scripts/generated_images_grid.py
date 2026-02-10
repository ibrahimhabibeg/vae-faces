from simple_parsing import ArgumentParser
from dataclasses import dataclass
import torch
import numpy as np
from face_vae.model import VAE, Encoder, Decoder, Prior, ConvEncoderNet, DecoderNet
from PIL import Image



@dataclass
class GeneratedImagesGridConfig:
    """
    Configuration for creating a grid of generated images.
    """

    vae_checkpoint: str = "../checkpoints/vae_epoch_20.pt"  # Path to the trained VAE checkpoint
    num_cols: int = 16  # Number of columns in the grid
    num_rows: int = 16  # Number of rows in the grid
    output_path: str = "generated_images_grid.png"  # Path to save the output grid image
    seed: int = 42  # Random seed for reproducible image generation
    grid_padding: int = 5  # Padding between images in the grid
    grid_color: tuple = (0, 0, 0)  # Background color


def load_model(config: GeneratedImagesGridConfig) -> tuple[VAE, int]:
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
    
    return vae, latent_dim


def generate_images(vae: VAE, latent_dim: int, num_images: int, seed: int) -> torch.Tensor:
    torch.manual_seed(seed)
    vae.eval()
    
    with torch.no_grad():
        generated_images = vae.sample(
            batch_size=num_images,
            latent_dim=latent_dim,
            device='cpu',
        )
    
    return generated_images


def create_generated_grid(
    generated_images: torch.Tensor,
    num_rows: int,
    num_cols: int,
    padding: int = 5,
    color=(0, 0, 0),
) -> np.ndarray:
    B, C, H, W = generated_images.shape
    
    grid_width = num_cols * W + (num_cols - 1) * padding
    grid_height = num_rows * H + (num_rows - 1) * padding
    grid_array = np.full((grid_height, grid_width, 3), color, dtype=np.uint8)
    
    for r in range(num_rows):
        for c in range(num_cols):
            idx = r * num_cols + c
            img = generated_images[idx].permute(1, 2, 0).cpu().numpy()
            
            starting_x = c * (W + padding)
            starting_y = r * (H + padding)
            img_uint8 = (img * 255).astype(np.uint8)
            grid_array[starting_y:starting_y + H, starting_x:starting_x + W] = img_uint8
    
    return grid_array


def save_grid_image(grid_array: np.ndarray, output_path: str):    
    grid_image = Image.fromarray(grid_array)
    grid_image.save(output_path)
    print(f"Generated images grid saved to {output_path}")


def create_generated_visualization(config: GeneratedImagesGridConfig):
    assert config.num_rows > 0, "num_rows must be greater than 0"
    assert config.num_cols > 0, "num_cols must be greater than 0"
    
    print("=" * 80)
    print("Creating Generated Images Grid Visualization")
    print("=" * 80)
    print(f"VAE checkpoint: {config.vae_checkpoint}")
    print(f"Number of columns: {config.num_cols}")
    print(f"Number of rows: {config.num_rows}")
    print(f"Output path: {config.output_path}")
    print(f"Random seed: {config.seed}")
    print("=" * 80)
    
    print("\nLoading VAE model...")
    vae, latent_dim = load_model(config)
    print(f"Latent dimension: {latent_dim}")
    
    num_images = config.num_rows * config.num_cols
    print(f"\nGenerating {num_images} images from VAE prior...")
    generated_images = generate_images(vae, latent_dim, num_images, config.seed)
    
    print("\nCreating visualization grid...")
    grid_array = create_generated_grid(
        generated_images=generated_images,
        num_rows=config.num_rows,
        num_cols=config.num_cols,
        padding=config.grid_padding,
        color=config.grid_color,
    )
    
    print(f"\nSaving grid image to {config.output_path}...")
    save_grid_image(grid_array, config.output_path)
    
    print("\n" + "=" * 80)
    print("Done!")
    print("=" * 80)


if __name__ == "__main__":
    parser = ArgumentParser(description="Create a grid of generated images from VAE prior.")
    parser.add_arguments(GeneratedImagesGridConfig, dest="config")
    args = parser.parse_args()
    create_generated_visualization(args.config)
