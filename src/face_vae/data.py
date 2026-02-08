import os
import pandas as pd
import torch
import PIL.Image
from torchvision.datasets import VisionDataset
from typing import Optional, Callable, Tuple, Any, Union

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
        transform: Optional[Callable] = None,
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
