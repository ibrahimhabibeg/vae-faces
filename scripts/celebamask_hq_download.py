import kagglehub
from simple_parsing import ArgumentParser
from dataclasses import dataclass


@dataclass
class DataDownloadConfig:
    """
    Configuration for downloading the CelebAMask-HQ dataset.
    This script downloads the dataset from Kaggle, which is more reliable than Google Drive.
    The dataset will be downloaded into the specified root directory and extracted automatically.
    """

    root: str = "../data/celebamask-hq"  # Directory where the dataset will be downloaded and extracted
    dataset: str = "ipythonx/celebamaskhq"  # Kaggle dataset id


def download_celebamask_hq(config: DataDownloadConfig):
    """
    Downloads the CelebAMask-HQ dataset from Kaggle using the kagglehub library.
    Args:
        config (DataDownloadConfig): Configuration for the download process.
    """
    print(f"Starting download of CelebAMask-HQ dataset to {config.root}...")

    # Use kagglehub to download and extract the dataset
    try:
        kagglehub.dataset_download(config.dataset, output_dir=config.root)
        print("Download and extraction completed successfully.")
    except Exception as e:
        print(f"An error occurred during download: {e}")


if __name__ == "__main__":
    parser = ArgumentParser(
        description="Download the CelebAMask-HQ dataset from Kaggle."
    )
    parser.add_arguments(DataDownloadConfig, dest="config")
    args = parser.parse_args()

    download_celebamask_hq(args.config)
