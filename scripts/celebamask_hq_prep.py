import numpy as np
import PIL.Image
from pathlib import Path
from simple_parsing import ArgumentParser
from dataclasses import dataclass
from tqdm import tqdm
from torchvision import transforms


@dataclass
class PrepConfig:
    """
    Configuration for preprocessing the CelebAMask-HQ dataset.
    This script applies masks to images, resizes them, and saves the processed images.
    """

    root: str = (
        "../data/celebamask-hq"  # Root directory containing CelebAMask-HQ dataset
    )
    output_dir: str = (
        "../data/celebamask-hq-processed"  # Output directory for processed images
    )
    img_size: int = 64  # Target image size (0 means no resize)
    background_color: tuple = (0, 0, 0)  # RGB color for background pixels


def build_mask_lookup(mask_dir: Path) -> dict[int, list[Path]]:
    """
    Pre-compute a lookup table mapping each image ID to its mask file paths.

    Args:
        mask_dir: Path to mask directory

    Returns:
        Dictionary mapping img_id to list of mask file paths
    """
    mask_lookup = {}

    for i in range(15):
        mask_folder = mask_dir / str(i)
        if not mask_folder.is_dir():
            continue

        for mask_file in mask_folder.glob("*.png"):
            # Extract image ID from filename (e.g., "00001_shair.png" -> 1)
            img_id = int(mask_file.stem.split("_")[0])
            if img_id not in mask_lookup:
                mask_lookup[img_id] = []
            mask_lookup[img_id].append(mask_file)

    return mask_lookup


def merge_masks(mask_files: list, img_size: tuple) -> np.ndarray:
    """
    Merge all mask files into a single binary mask.

    Args:
        mask_files: List of mask file paths
        img_size: Size of the image (height, width)

    Returns:
        Binary mask as numpy array (height, width) with values 0 or 1
    """
    merged_mask = np.zeros((img_size[0], img_size[1]), dtype=np.uint8)

    for mask_file in mask_files:
        mask = PIL.Image.open(mask_file).convert("L")
        mask_array = np.array(mask)
        merged_mask = np.maximum(merged_mask, (mask_array > 0).astype(np.uint8))

    return merged_mask


def apply_mask_to_image(
    img: PIL.Image.Image, mask: np.ndarray, background_color: tuple
) -> PIL.Image.Image:
    """
    Apply mask to image, replacing background with specified color.

    Args:
        img: PIL Image (RGB)
        mask: Binary mask (height, width) with values 0 or 1
        background_color: RGB color tuple for background

    Returns:
        PIL Image with background replaced
    """
    img_array = np.array(img)
    background = np.full_like(img_array, background_color)
    mask_3ch = np.stack([mask] * 3, axis=-1)
    result = img_array * mask_3ch + background * (1 - mask_3ch)
    return PIL.Image.fromarray(result.astype(np.uint8))


def process_single_image(
    img_file: Path,
    mask_files: list[Path],
    cfg: PrepConfig,
):
    img = PIL.Image.open(img_file).convert("RGB")

    # Resize to 512x512 to match mask size
    img = transforms.Resize(512)(img)
    img = transforms.CenterCrop(512)(img)

    merged_mask = merge_masks(mask_files, img.size)
    img = apply_mask_to_image(img, merged_mask, cfg.background_color)

    img = transforms.Resize(cfg.img_size)(img)
    img = transforms.CenterCrop(cfg.img_size)(img)
    return img


def preprocess_dataset(config: PrepConfig):
    """
    Preprocess the CelebAMask-HQ dataset: apply masks, resize, and save images sequentially in batches.
    """
    # Setup paths
    root = Path(config.root)
    img_dir = root / "CelebAMask-HQ" / "CelebA-HQ-img"
    mask_dir = root / "CelebAMask-HQ" / "CelebAMask-HQ-mask-anno"
    output_dir = Path(config.output_dir)

    if not img_dir.is_dir():
        raise RuntimeError(f"Image directory not found at {img_dir}")
    if not mask_dir.is_dir():
        raise RuntimeError(f"Mask directory not found at {mask_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")

    # Get list of image files
    image_files = sorted([f for f in img_dir.glob("*.jpg")], key=lambda x: int(x.stem))

    if len(image_files) == 0:
        raise RuntimeError(f"No image files found in {img_dir}")

    print(f"Found {len(image_files)} images to process")
    print("Configuration:")
    print(f"  Output height: {config.img_size}")
    print(f"  Output width: {config.img_size}")
    print(f"  Background color: {config.background_color}")

    # Pre-compute mask file path lookup
    print("\nBuilding mask lookup table...")
    mask_lookup = build_mask_lookup(mask_dir)
    print(f"  Found masks for {len(mask_lookup)} images")

    # Process images
    print("\nProcessing images...")
    for img_file in tqdm(image_files, desc="Processing images"):
        img_id = int(img_file.stem)
        mask_files = mask_lookup.get(img_id, [])
        img = process_single_image(img_file, mask_files, config)
        img.save(output_dir / f"{img_id:05d}.png")
    print("\nProcessing completed!")
    print(f"  Saved to: {output_dir}")


if __name__ == "__main__":
    parser = ArgumentParser(
        description="Preprocess CelebAMask-HQ dataset: apply masks and resize images."
    )
    parser.add_arguments(PrepConfig, dest="config")
    args = parser.parse_args()

    preprocess_dataset(args.config)
