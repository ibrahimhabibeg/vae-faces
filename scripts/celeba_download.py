import kagglehub
from simple_parsing import ArgumentParser
from dataclasses import dataclass


@dataclass
class DataDownloadConfig:
    """
    Configuration for downloading the CelebA dataset.
    This script downloads the dataset from Kaggle, which is more reliable than Google Drive.
    The dataset will be downloaded into the specified root directory and extracted automatically.
    """

    root: str = (
        "../data/celeba"  # Directory where the dataset will be downloaded and extracted
    )
    dataset: str = "jessicali9530/celeba-dataset"  # Kaggle dataset id


def download_celeba(config: DataDownloadConfig):
    """
    Downloads the CelebA dataset from Kaggle using the kagglehub library.
    Args:
        config (DataDownloadConfig): Configuration for the download process.
    """
    print(f"Starting download of CelebA dataset to {config.root}...")

    # Use kagglehub to download and extract the dataset
    try:
        kagglehub.dataset_download(config.dataset, output_dir=config.root)
        print("Download and extraction completed successfully.")
    except Exception as e:
        print(f"An error occurred during download: {e}")


if __name__ == "__main__":
    parser = ArgumentParser(description="Download the CelebA dataset from Kaggle.")
    parser.add_arguments(DataDownloadConfig, dest="config")
    args = parser.parse_args()

    download_celeba(args.config)
