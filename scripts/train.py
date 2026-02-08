import torch
from torch.utils.data import DataLoader
import wandb
from simple_parsing import ArgumentParser, field
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union
from face_vae.data import CelebA, CelebAMaskHQ
from face_vae.model import VAE, Encoder, Decoder, Prior, ConvEncoderNet, DecoderNet
from face_vae.train import train_vae
from face_vae.loss import vae_loss, bce_reconstruction_loss, mse_reconstruction_loss


################################################
# Configuration dataclasses
################################################


@dataclass
class DataConfig:
    """Data loading configuration."""

    dataset_name: str = (
        "CelebAMaskHQ"  # Name of the dataset to use ('CelebA' or 'CelebAMaskHQ')
    )
    data_root: str = "../data/celebamask-hq-processed"  # Root directory of the CelebA dataset
    split: str = "train"  # Dataset split to use: 'train', 'valid', 'test', or 'all'
    batch_size: int = 32  # Number of samples per batch
    shuffle: bool = True  # Whether to shuffle the data at every epoch
    num_workers: int = 1  # Number of workers for data loading


@dataclass
class ModelConfig:
    """Model architecture configuration."""

    latent_dim: int = 128  # Dimension of the latent space


@dataclass
class TrainingConfig:
    """Training hyperparameters configuration."""

    num_epochs: int = 20  # Number of training epochs
    learning_rate: float = 1e-3  # Learning rate for the optimizer
    beta: float = 1.0  # Beta parameter for KL divergence loss weighting
    reconstruction_loss: str = "bce"  # Reconstruction loss: 'bce' or 'mse'
    early_stopping_patience: Optional[int] = (
        None  # Early stopping patience (epochs with no improvement)
    )
    early_stopping_min_delta: float = (
        0.0  # Minimum change in loss to qualify as an improvement
    )
    device: str = "auto"  # Device to use: 'auto', 'cuda', 'mps', or 'cpu'


@dataclass
class WandbConfig:
    """Weights & Biases logging configuration."""

    use_wandb: bool = True  # Whether to use Weights & Biases for logging
    wandb_project: str = "face_vae"  # W&B project name
    wandb_run_name: Optional[str] = None  # W&B run name (optional)
    wandb_entity: Optional[str] = None  # W&B entity name (optional)


@dataclass
class CheckpointConfig:
    """Checkpointing configuration."""

    save_checkpoint: bool = True  # Whether to save model checkpoints
    checkpoint_dir: str = "../checkpoints"  # Directory to save checkpoints


@dataclass
class TrainConfig:
    """
    Main configuration for training the VAE model on CelebA dataset.
    """

    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    wandb: WandbConfig = field(default_factory=WandbConfig)
    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)


def cfg_integrity_check(config: TrainConfig):
    """
    Perform integrity checks on the training configuration.
    Raises ValueError if any configuration parameter is invalid.
    """
    # Check dataset name
    valid_datasets = ["CelebA", "CelebAMaskHQ"]
    assert config.data.dataset_name in valid_datasets, (
        f"Invalid dataset_name: {config.data.dataset_name}. Must be one of: {valid_datasets}"
    )

    # Check reconstruction loss type
    valid_loss_types = ["bce", "mse"]
    assert config.training.reconstruction_loss in valid_loss_types, (
        f"Invalid reconstruction_loss: {config.training.reconstruction_loss}. Must be one of: {valid_loss_types}"
    )

    # Check device configuration
    assert config.training.device in ["auto", "cuda", "mps", "cpu"], (
        f"Invalid device: {config.training.device}. Must be one of: ['auto', 'cuda', 'mps', 'cpu']"
    )


##################################################
# Helper functions for training
##################################################


def get_device(device_config: str) -> torch.device:
    """
    Get the appropriate device based on the configuration.
    """
    if device_config == "auto":
        return torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )
    if device_config == "cuda":
        assert torch.cuda.is_available(), (
            "CUDA is not available but 'cuda' was specified as the device."
        )
    elif device_config == "mps":
        assert torch.backends.mps.is_available(), (
            "MPS is not available but 'mps' was specified as the device."
        )
    return torch.device(device_config)


def get_reconstruction_loss_fn(loss_type: str):
    """
    Get the reconstruction loss function based on the configuration.
    """
    if loss_type == "bce":
        return bce_reconstruction_loss
    return mse_reconstruction_loss


def create_vae_model(config: TrainConfig) -> VAE:
    """
    Create and return a VAE model based on the configuration.
    """
    encoder_net = ConvEncoderNet()
    encoder = Encoder(
        encoder_net=encoder_net,
        latent_dim=config.model.latent_dim,
    )

    decoder_net = DecoderNet(latent_dim=config.model.latent_dim)
    decoder = Decoder(decoder_net=decoder_net)

    prior = Prior()

    vae = VAE(encoder=encoder, decoder=decoder, prior=prior)
    return vae


def create_dataset(config: TrainConfig) -> Union[CelebA, CelebAMaskHQ]:
    """
    Create and return the dataset based on the configuration.
    """
    if config.data.dataset_name == "CelebA":
        return CelebA(
            root=config.data.data_root,
            split=config.data.split,
            return_attributes=False,
        )
    else:
        return CelebAMaskHQ(config.data.data_root)


def create_dataloader(
    dataset: torch.utils.data.Dataset, config: TrainConfig
) -> DataLoader:
    """
    Create and return a DataLoader for the given dataset based on the configuration.
    """
    return DataLoader(
        dataset,
        batch_size=config.data.batch_size,
        shuffle=config.data.shuffle,
        num_workers=config.data.num_workers,
    )


def get_loss_fn(config: TrainConfig):
    """
    Get the loss function based on the configuration.
    """
    recon_loss_fn = get_reconstruction_loss_fn(config.training.reconstruction_loss)

    def loss_fn(recon_x, x, mu, logvar):
        return vae_loss(
            recon_x,
            x,
            mu,
            logvar,
            recon_loss_fn=recon_loss_fn,
            beta=config.training.beta,
        )

    return loss_fn


################################################
# Main training function
################################################


def train(config: TrainConfig):
    """
    Main training function that orchestrates the entire training process.
    """
    print("=" * 80)
    print("Training VAE on CelebA Dataset")
    print("=" * 80)
    print("Configuration:")
    print(f"  Data root: {config.data.data_root}")
    print(f"  Split: {config.data.split}")
    print(f"  Batch size: {config.data.batch_size}")
    print(f"  Latent dimension: {config.model.latent_dim}")
    print(f"  Number of epochs: {config.training.num_epochs}")
    print(f"  Learning rate: {config.training.learning_rate}")
    print(f"  Beta: {config.training.beta}")
    print(f"  Reconstruction loss: {config.training.reconstruction_loss}")
    print(f"  Device: {config.training.device}")
    print(f"  Use W&B: {config.wandb.use_wandb}")
    print("=" * 80)

    # Get device
    device = get_device(config.training.device)
    print(f"\nUsing device: {device}")

    # Initialize wandb if requested
    if config.wandb.use_wandb:
        wandb_config = {
            "dataset": config.data.dataset_name,
            "latent_dim": config.model.latent_dim,
            "batch_size": config.data.batch_size,
            "learning_rate": config.training.learning_rate,
            "num_epochs": config.training.num_epochs,
            "beta": config.training.beta,
            "reconstruction_loss": config.training.reconstruction_loss,
            "split": config.data.split,
        }
        wandb.init(
            project=config.wandb.wandb_project,
            name=config.wandb.wandb_run_name,
            entity=config.wandb.wandb_entity,
            config=wandb_config,
        )
        print(f"Initialized W&B project: {config.wandb.wandb_project}")

    # Load dataset
    print("\nLoading dataset...")
    dataset = create_dataset(config)
    dataloader = create_dataloader(dataset, config)
    print(f"Dataset loaded with {len(dataset)} samples")

    # Create model
    print("\nCreating VAE model...")
    vae = create_vae_model(config)
    total_params = sum(p.numel() for p in vae.parameters())
    trainable_params = sum(p.numel() for p in vae.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # Get loss function
    loss_fn = get_loss_fn(config)

    # Train the model
    print("\nStarting training...")
    print("=" * 80)
    train_vae(
        vae=vae,
        dataloader=dataloader,
        num_epochs=config.training.num_epochs,
        learning_rate=config.training.learning_rate,
        loss_fn=loss_fn,
        device=device,
        use_wandb=config.wandb.use_wandb,
        latent_dim=config.model.latent_dim,
        checkpoint_dir=config.checkpoint.checkpoint_dir
        if config.checkpoint.save_checkpoint
        else None,
        early_stopping_patience=config.training.early_stopping_patience,
        early_stopping_min_delta=config.training.early_stopping_min_delta,
    )
    print("=" * 80)
    print("Training completed!")

    # Save checkpoint if requested
    if config.checkpoint.save_checkpoint:
        checkpoint_dir = Path(config.checkpoint.checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_path = checkpoint_dir / "vae_final.pt"
        torch.save(
            {
                "model_state_dict": vae.state_dict(),
                "config": config,
            },
            checkpoint_path,
        )
        print(f"\nModel checkpoint saved to: {checkpoint_path}")

        # Upload final checkpoint to W&B artifacts
        if config.wandb.use_wandb:
            artifact = wandb.Artifact(
                name="vae-final",
                type="model",
                metadata={
                    "epochs": config.training.num_epochs,
                    "latent_dim": config.model.latent_dim,
                    "learning_rate": config.training.learning_rate,
                    "batch_size": config.data.batch_size,
                },
            )
            artifact.add_file(str(checkpoint_path))
            wandb.log_artifact(artifact)
            print("Final checkpoint uploaded to W&B artifacts")

    # Finish W&B run
    if config.wandb.use_wandb:
        wandb.finish()
        print("W&B run finished")


if __name__ == "__main__":
    parser = ArgumentParser(description="Train a VAE model on the CelebA dataset.")
    parser.add_arguments(TrainConfig, dest="config")
    args = parser.parse_args()
    cfg_integrity_check(args.config)
    train(args.config)
