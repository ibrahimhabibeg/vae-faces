import os
import pandas as pd
import torch
import PIL.Image
from torchvision.datasets import VisionDataset
from typing import Optional, Callable, Tuple, Any, Union
from torchvision import transforms
from pathlib import Path
import numpy as np


class CelebA(VisionDataset):
    """
    Custom CelebA dataset loader inspired by torchvision.datasets.CelebA but adapted for a different directory structure.
    Doesn't require data download from Google Drive.
    Expects the data structure found in https://www.kaggle.com/datasets/jessicali9530/celeba-dataset
    Data should be downloaded into the specified root directory before using this class.
    Check the data_download.py script for the data downloading process.
    """

    def __init__(
        self,
        root: str,
        split: str = "train",
        transform: Optional[Callable] = transforms.Compose(
            [
                transforms.Resize(64),
                transforms.CenterCrop(64),
                transforms.ToTensor(),
            ]
        ),
        target_transform: Optional[Callable] = None,
        return_attributes: bool = False,
    ) -> None:
        """
        Args:
            root (string): Root directory containing the CSV files and the nested
                           'img_align_celeba/img_align_celeba' folder.
            split (string): One of {'train', 'valid', 'test', 'all'}.
                            Selects the split based on list_eval_partition.csv.
            transform (callable, optional): A function/transform that takes in a PIL image
                and returns a transformed version. E.g, ``transforms.ToTensor``
            target_transform (callable, optional): A function/transform that takes in the
                attributes tensor and transforms it.
            return_attributes (bool): Whether to return face attributes. Default is False.
                If False, __getitem__ returns only the image.
                If True, __getitem__ returns a tuple (image, attributes) where attributes is a tensor of face attributes (0 or 1).
        """
        super().__init__(root, transform=transform, target_transform=target_transform)

        self.split = split.lower()
        self.return_attributes = return_attributes
        valid_splits = ("train", "valid", "test", "all")
        if self.split not in valid_splits:
            raise ValueError(f"Split must be one of {valid_splits}, got {self.split}")

        # 1. Define paths based on the expected directory structure
        self.img_dir = os.path.join(self.root, "img_align_celeba", "img_align_celeba")
        partition_csv_path = os.path.join(self.root, "list_eval_partition.csv")
        attr_csv_path = os.path.join(self.root, "list_attr_celeba.csv")

        # Basic checks
        if not os.path.isdir(self.img_dir):
            raise RuntimeError(
                f"Image directory not found at {self.img_dir}. Check structure."
            )
        if not os.path.isfile(partition_csv_path) or not os.path.isfile(attr_csv_path):
            raise RuntimeError("Necessary CSV files not found in root directory.")

        # 2. Define split mapping (CelebA standard: 0=train, 1=valid, 2=test)
        split_map = {
            "train": 0,
            "valid": 1,
            "test": 2,
            "all": None,
        }
        split_int = split_map[self.split]

        # 3. Load CSVs using pandas
        # Assume the image filename (e.g., '000001.jpg') is the index (column 0)
        df_partition = pd.read_csv(partition_csv_path, index_col=0)
        df_attr = pd.read_csv(attr_csv_path, index_col=0)

        # 4. Filter based on requested split
        if split_int is not None:
            # Assuming the partition column is named 'partition'
            mask = df_partition[df_partition.columns[0]] == split_int
            selected_filenames = df_partition[mask].index
        else:
            selected_filenames = df_partition.index

        # 5. Align attributes data with selected filenames
        selected_attrs_df = df_attr.loc[selected_filenames]

        # Store filenames for __getitem__
        self.filenames = selected_filenames.values

        # 6. Convert attributes to tensor and normalize
        self.attr_names = list(selected_attrs_df.columns)
        # Convert dataframe values to torch long tensor
        self.attr_data = torch.as_tensor(selected_attrs_df.values, dtype=torch.long)

        # CelebA dataset uses -1 for negative and 1 for positive.
        # Map this to standard 0 and 1.
        self.attr_data = (self.attr_data + 1) // 2

        print(
            f"Loaded KaggleCelebA Dataset. Split: {self.split}. Samples: {len(self.filenames)}"
        )

    def __getitem__(self, index: int) -> Union[Any, Tuple[Any, Any]]:
        """
        Args:
            index (int): Index
        Returns:
            If return_attributes is False: returns just the image.
            If return_attributes is True: returns tuple (image, attributes) where attributes is a tensor of face attributes (0 or 1).
        """
        # Load Image
        img_filename = self.filenames[index]
        img_path = os.path.join(self.img_dir, img_filename)

        # Open and ensure RGB
        img = PIL.Image.open(img_path).convert("RGB")

        # Apply Transforms
        if self.transform is not None:
            img = self.transform(img)

        # Return only image if attributes not requested
        if not self.return_attributes:
            return img

        # Get Attributes
        attributes = self.attr_data[index]

        if self.target_transform is not None:
            attributes = self.target_transform(attributes)

        return img, attributes

    def __len__(self) -> int:
        return len(self.filenames)

    def extra_repr(self) -> str:
        # Extra info to the print(dataset) output
        return f"Split: {self.split}"


class CelebAMaskHQ(VisionDataset):
    """
    CelebAMask-HQ dataset loader with background removal.
    This dataset loads face images from CelebA-HQ-img and applies masks from CelebAMask-HQ-mask-anno
    to remove backgrounds, keeping only facial features.
    Based on data structure found in https://www.kaggle.com/datasets/ipythonx/celebamaskhq
    """

    def __init__(
        self,
        root: str,
        transform: Optional[Callable] = transforms.Compose(
            [
                transforms.Resize(64),
                transforms.CenterCrop(64),
                transforms.ToTensor(),
            ]
        ),
        background_color: Tuple[int, int, int] = (0, 0, 0),
    ) -> None:
        """
        Args:
            root (string): Root directory containing 'CelebA-HQ-img' and 'CelebAMask-HQ-mask-anno' folders.
            transform (callable, optional): A function/transform that takes in a PIL image
                and returns a transformed version. E.g, ``transforms.ToTensor``
            background_color (tuple): RGB color tuple (0-255) for background pixels. Default is black (0, 0, 0).
        """
        super().__init__(root, transform=transform)

        self.background_color = background_color

        # Transform images to 512x512 to match mask size
        self.resize_transform = transforms.Compose(
            [
                transforms.Resize(512),
                transforms.CenterCrop(512),
            ]
        )

        self.img_dir = Path(root) / "CelebAMask-HQ" / "CelebA-HQ-img"
        self.mask_dir = Path(root) / "CelebAMask-HQ" / "CelebAMask-HQ-mask-anno"

        if not self.img_dir.is_dir():
            raise RuntimeError(
                f"Image directory not found at {self.img_dir}. Check structure."
            )
        if not self.mask_dir.is_dir():
            raise RuntimeError(
                f"Mask directory not found at {self.mask_dir}. Check structure."
            )

        # Get list of all image files
        self.image_files = sorted(
            [f for f in self.img_dir.glob("*.jpg")], key=lambda x: int(x.stem)
        )

        if len(self.image_files) == 0:
            raise RuntimeError(f"No image files found in {self.img_dir}")

        # Build mask folder mapping (0-14)
        self.mask_folders = [self.mask_dir / str(i) for i in range(15)]

        for folder in self.mask_folders:
            if not folder.is_dir():
                raise RuntimeError(
                    f"Mask folder not found at {folder}. Check structure."
                )

        print(f"Loaded CelebAMaskHQ Dataset. Samples: {len(self.image_files)}")

    def _get_mask_files_for_image(self, img_id: int) -> list:
        """
        Get all mask files for a given image ID.

        Args:
            img_id: Image ID (e.g., 0, 1, 2, ...)

        Returns:
            List of Path objects for all mask files belonging to this image
        """
        # Format image ID with 5 digits (e.g., 0 -> 00000)
        img_id_str = f"{img_id:05d}"

        mask_files = []

        # Search through all mask folders (0-14)
        for mask_folder in self.mask_folders:
            # Find all masks for this image (pattern: {img_id}_*.png)
            matching_masks = list(mask_folder.glob(f"{img_id_str}_*.png"))
            mask_files.extend(matching_masks)

        return mask_files

    def _merge_masks(self, mask_files: list, img_size: Tuple[int, int]) -> np.ndarray:
        """
        Merge all mask files into a single binary mask.

        Args:
            mask_files: List of mask file paths
            img_size: Size of the image (width, height)

        Returns:
            Binary mask as numpy array (height, width) with values 0 or 1
        """
        # Create empty mask
        merged_mask = np.zeros((img_size[1], img_size[0]), dtype=np.uint8)

        # Load and merge all masks
        for mask_file in mask_files:
            mask = PIL.Image.open(mask_file).convert("L")
            mask_array = np.array(mask)
            # Any non-zero pixel in the mask is considered part of the face
            merged_mask = np.maximum(merged_mask, (mask_array > 0).astype(np.uint8))

        return merged_mask

    def _apply_mask_to_image(
        self, img: PIL.Image.Image, mask: np.ndarray
    ) -> PIL.Image.Image:
        """
        Apply mask to image, replacing background with specified color.

        Args:
            img: PIL Image (RGB)
            mask: Binary mask (height, width) with values 0 or 1

        Returns:
            PIL Image with background replaced
        """
        img_array = np.array(img)
        background = np.full_like(img_array, self.background_color)
        mask_3ch = np.stack([mask] * 3, axis=-1)
        result = img_array * mask_3ch + background * (1 - mask_3ch)
        return PIL.Image.fromarray(result.astype(np.uint8))

    def __getitem__(self, index: int) -> Any:
        """
        Args:
            index (int): Index

        Returns:
            Image with background removed (masked)
        """
        # Get image
        img_file = self.image_files[index]
        img_id = int(img_file.stem)
        img = PIL.Image.open(img_file).convert("RGB")

        # Images are 1024x1024, masks are 512x512
        img = self.resize_transform(img)

        # Mask image
        mask_files = self._get_mask_files_for_image(img_id)
        assert len(mask_files) > 0, (
            f"No mask files found for image ID {img_id} at index {index}"
        )
        merged_mask = self._merge_masks(mask_files, img.size)
        img = self._apply_mask_to_image(img, merged_mask)

        # Apply transforms
        if self.transform is not None:
            img = self.transform(img)

        return img

    def __len__(self) -> int:
        return len(self.image_files)

    def extra_repr(self) -> str:
        return f"Samples: {len(self.image_files)}, Background color: {self.background_color}"
