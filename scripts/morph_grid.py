from simple_parsing import ArgumentParser
from dataclasses import dataclass
import torch
import numpy as np
from face_vae.model import VAE, Encoder, Decoder, Prior, ConvEncoderNet, DecoderNet
from face_vae.data import CelebAMaskHQ
from face_vae.morph import morph_images
from PIL import Image
import random


@dataclass
class MorphGridConfig:
    """
    Configuration for creating a morphing grid animation.
    """

    data_root: str = "../data/celebamask-hq-processed"  # Root directory of preprocessed CelebAMaskHQ dataset
    vae_checkpoint: str = (
        "../checkpoints/vae_epoch_20.pt"  # Path to the trained VAE checkpoint
    )
    num_cols: int = 8  # Number of columns in the grid
    num_rows: int = 8  # Number of rows in the grid
    fps: int = 30  # Frames per second for the output morphing sequence
    duration: int = 5  # Duration of the morphing sequence in seconds
    output_path: str = "morph_grid.gif"  # Path to save the output morphing GIF
    seed: int = 42  # Random seed for reproducible image selection
    grid_padding: int = 5  # Padding between images in the grid
    grid_color: tuple = (0, 0, 0)  # Background color


def load_model(config: MorphGridConfig) -> VAE:
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


def select_random_image_pairs(
    dataset: CelebAMaskHQ, num_pairs: int, seed: int
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    random.seed(seed)
    indices = random.sample(range(len(dataset)), num_pairs * 2)

    first_images = [dataset[indices[i * 2]] for i in range(num_pairs)]
    second_images = [dataset[indices[i * 2 + 1]] for i in range(num_pairs)]

    return first_images, second_images


def create_morph_grid_frame(
    morph_images_list: list[torch.Tensor],
    frame_idx: int,
    num_rows: int,
    num_cols: int,
    padding: int = 5,
    color=(0, 0, 0),
) -> np.ndarray:
    sample_img = morph_images_list[0][frame_idx]
    C, H, W = sample_img.shape

    grid_width = num_cols * W + (num_cols - 1) * padding
    grid_height = num_rows * H + (num_rows - 1) * padding
    grid_array = np.full((grid_height, grid_width, 3), color, dtype=np.uint8)

    for r in range(num_rows):
        for c in range(num_cols):
            idx = r * num_cols + c
            img = morph_images_list[idx][frame_idx].permute(1, 2, 0).cpu().numpy()

            starting_x = c * (W + padding)
            starting_y = r * (H + padding)
            img_uint8 = (img * 255).astype(np.uint8)
            grid_array[starting_y : starting_y + H, starting_x : starting_x + W] = (
                img_uint8
            )

    return grid_array


def save_morph_grid_gif(grid_frames: list[np.ndarray], output_path: str, fps: int):
    images = []
    for frame_array in grid_frames:
        images.append(Image.fromarray(frame_array))

    images[0].save(
        output_path,
        save_all=True,
        append_images=images[1:],
        duration=1000 / fps,
        loop=0,
    )
    print(f"Morphing grid GIF saved to {output_path}")


def create_morph_grid_animation(config: MorphGridConfig):
    assert config.num_rows > 0, "num_rows must be greater than 0"
    assert config.num_cols > 0, "num_cols must be greater than 0"

    print("=" * 80)
    print("Creating Morphing Grid Animation")
    print("=" * 80)
    print(f"Data root: {config.data_root}")
    print(f"VAE checkpoint: {config.vae_checkpoint}")
    print(f"Grid size: {config.num_rows}x{config.num_cols}")
    print(f"FPS: {config.fps}")
    print(f"Duration: {config.duration}s")
    print(f"Output path: {config.output_path}")
    print(f"Random seed: {config.seed}")
    print("=" * 80)

    print("\nLoading CelebAMaskHQ dataset...")
    dataset = CelebAMaskHQ(root=config.data_root)

    num_pairs = config.num_rows * config.num_cols
    print(f"\nSelecting {num_pairs} random image pairs...")
    first_images, second_images = select_random_image_pairs(
        dataset, num_pairs, config.seed
    )

    print("\nLoading VAE model...")
    vae = load_model(config)

    steps = config.fps * config.duration
    print(f"\nCreating morphing sequences ({steps} frames per pair)...")
    morph_sequences = []
    for i, (img1, img2) in enumerate(zip(first_images, second_images)):
        if (i + 1) % 4 == 0 or i == 0:
            print(f"  Processing pair {i + 1}/{num_pairs}...")
        morph_seq = morph_images(vae, img1, img2, steps)
        morph_sequences.append(morph_seq)

    print("\nCreating grid frames...")
    grid_frames = []
    for frame_idx in range(steps):
        if (frame_idx + 1) % 30 == 0 or frame_idx == 0:
            print(f"  Frame {frame_idx + 1}/{steps}...")
        grid_frame = create_morph_grid_frame(
            morph_images_list=morph_sequences,
            frame_idx=frame_idx,
            num_rows=config.num_rows,
            num_cols=config.num_cols,
            padding=config.grid_padding,
            color=config.grid_color,
        )
        grid_frames.append(grid_frame)

    print(f"\nSaving morphing grid GIF to {config.output_path}...")
    save_morph_grid_gif(grid_frames, config.output_path, config.fps)

    print("\n" + "=" * 80)
    print("Done!")
    print("=" * 80)


if __name__ == "__main__":
    parser = ArgumentParser(
        description="Create a morphing grid animation from CelebAMaskHQ dataset."
    )
    parser.add_arguments(MorphGridConfig, dest="config")
    args = parser.parse_args()
    create_morph_grid_animation(args.config)
