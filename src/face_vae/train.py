import wandb
import torch
from typing import Optional
from torch import optim
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from pathlib import Path
from face_vae.model import VAE
from face_vae.loss import vae_loss
from dataclasses import dataclass


@dataclass
class EarlyStoppingState:
    best_loss: float = float("inf")
    patience_counter: int = 0


def check_early_stopping(
    avg_loss: float,
    state: EarlyStoppingState,
    epoch: int,
    early_stopping_patience: Optional[int],
    early_stopping_min_delta: float,
    use_wandb: bool = False,
) -> tuple[bool, EarlyStoppingState]:
    """
    Check if early stopping criteria are met and update the state accordingly.
    Args:
        avg_loss: The average loss for the current epoch.
        state: The current state of early stopping (best loss and patience counter).
        epoch: The current epoch number (used for logging).
        early_stopping_patience: Number of epochs with no improvement after which training will be stopped. If None, early stopping is disabled.
        early_stopping_min_delta: Minimum change in the monitored loss to qualify as an improvement. Default is 0.0.
        use_wandb: Whether to log early stopping status to Weights & Biases.
    Returns:
        A tuple (should_stop, new_state) where should_stop is a boolean indicating whether training should be stopped,
        and new_state is the updated EarlyStoppingState.
    """
    if early_stopping_patience is None:
        return False, state  # Early stopping disabled

    new_state = EarlyStoppingState(
        best_loss=state.best_loss, patience_counter=state.patience_counter
    )

    if avg_loss < state.best_loss - early_stopping_min_delta:
        if state.patience_counter > 0:
            tqdm.write(f"Loss improved to {avg_loss:.4f}. Resetting patience counter.")
        new_state.best_loss = avg_loss
        new_state.patience_counter = 0
    else:
        new_state.patience_counter += 1
        tqdm.write(
            f"Loss did not improve. Patience: {new_state.patience_counter}/{early_stopping_patience}"
        )
        if new_state.patience_counter >= early_stopping_patience:
            tqdm.write(
                f"Early stopping triggered after {epoch + 1} epochs (best loss: {new_state.best_loss:.4f})"
            )
            if use_wandb:
                wandb.log({"early_stopped": True, "early_stopped_epoch": epoch + 1})
            return True, new_state

    return False, new_state


def save_checkpoint(
    vae: VAE,
    optimizer: optim.Optimizer,
    epoch: int,
    loss: float,
    checkpoint_dir: Optional[str],
    use_wandb: bool,
):
    """
    Save a checkpoint of the VAE model and optimizer state.
    Args:
        vae: The VAE model to save.
        optimizer: The optimizer whose state to save.
        epoch: The current epoch number (used for naming the checkpoint file).
        loss: The loss value at the time of saving (stored in checkpoint metadata).
        checkpoint_dir: Directory to save the checkpoint file.
    """
    if checkpoint_dir is None:
        return  # Do not save if no checkpoint directory is specified

    checkpoint_path = Path(checkpoint_dir) / f"vae_epoch_{epoch}.pt"
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": vae.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": loss,
        },
        checkpoint_path,
    )
    print(f"Checkpoint saved to: {checkpoint_path}")

    # Upload checkpoint to W&B artifacts
    if use_wandb:
        artifact = wandb.Artifact(
            name=f"vae-checkpoint-epoch-{epoch}",
            type="model",
            metadata={
                "epoch": epoch,
                "loss": loss,
            },
        )
        artifact.add_file(str(checkpoint_path))
        wandb.log_artifact(artifact)


def end_of_epoch_logging(
    vae: VAE,
    dataloader: DataLoader,
    epoch: int,
    avg_loss: float,
    device: torch.device,
    use_wandb: bool,
    latent_dim: int,
):
    """
    Log original and reconstructed images to W&B at the end of each epoch.
    Args:
        vae: The trained VAE model.
        dataloader: DataLoader for the validation data (used for logging reconstructions).
        epoch: The current epoch number (used for logging).
        avg_loss: The average loss for the epoch (used for logging).
        device: The device to perform inference on.
        use_wandb: Whether to log to Weights & Biases.
        latent_dim: Dimension of the latent space (used for sampling).
    """
    if not use_wandb:
        return

    # Get a batch of images for visualization
    val_images = next(iter(dataloader)).to(device)
    with torch.no_grad():
        recon_images, _, _ = vae(val_images)

    wandb.log(
        {
            "originals": [wandb.Image(img) for img in val_images[:8]],
            "reconstructions": [wandb.Image(img) for img in recon_images[:8]],
            "epoch": epoch + 1,
            "avg_loss": avg_loss,
        }
    )

    # Log VAE generated samples
    with torch.no_grad():
        sampled_images = vae.sample(batch_size=8, latent_dim=latent_dim, device=device)
    wandb.log({"samples": [wandb.Image(img) for img in sampled_images]})


def train_vae(
    vae: VAE,
    dataloader: DataLoader,
    num_epochs: int = 20,
    learning_rate: float = 1e-3,
    loss_fn=vae_loss,
    device: torch.device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    ),
    use_wandb: bool = False,
    latent_dim: int = 128,
    checkpoint_dir: Optional[str] = None,
    early_stopping_patience: Optional[int] = None,
    early_stopping_min_delta: float = 0.0,
):
    """
    Train a VAE model.
    Args:
        vae: The VAE model to train.
        dataloader: DataLoader for training data.
        num_epochs: Number of training epochs.
        learning_rate: Learning rate for the optimizer.
        loss_fn: Loss function to use.
        device: Device to train on.
        use_wandb: Whether to log metrics to Weights & Biases.
                   Note: wandb.init() should be called before this function if use_wandb=True.
        latent_dim: Dimension of the latent space (used for sampling).
        checkpoint_dir: Directory to save epoch checkpoints. If None, checkpoints are not saved.
        early_stopping_patience: Number of epochs with no improvement after which training will be stopped. If None, early stopping is disabled.
        early_stopping_min_delta: Minimum change in the monitored loss to qualify as an improvement. Default is 0.0.
    """
    vae.to(device)
    optimizer = optim.Adam(vae.parameters(), lr=learning_rate)

    if use_wandb:
        wandb.watch(vae, log="all", log_freq=100)

    if checkpoint_dir:
        checkpoint_path = Path(checkpoint_dir)
        checkpoint_path.mkdir(parents=True, exist_ok=True)

    early_stopping_state = EarlyStoppingState(
        best_loss=float("inf"), patience_counter=0
    )

    epochs_pbar = tqdm(range(num_epochs), desc="Training VAE")

    for epoch in epochs_pbar:
        vae.train()
        total_loss = 0

        batch_pbar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{num_epochs}")
        for batch_idx, images in enumerate(batch_pbar):
            images = images.to(device)
            optimizer.zero_grad()
            recon_images, mu, logvar = vae(images)
            loss = loss_fn(recon_images, images, mu, logvar)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            batch_pbar.set_postfix({"loss": loss.item()})

            if use_wandb:
                wandb.log(
                    {
                        "batch_loss": loss.item(),
                        "batch": epoch * len(dataloader) + batch_idx,
                    }
                )

        avg_loss = total_loss / len(dataloader.dataset)  # type: ignore
        tqdm.write(f"Epoch [{epoch + 1}/{num_epochs}], Average Loss: {avg_loss:.4f}")

        end_of_epoch_logging(
            vae, dataloader, epoch, avg_loss, device, use_wandb, latent_dim
        )
        save_checkpoint(vae, optimizer, epoch + 1, avg_loss, checkpoint_dir, use_wandb)
        should_stop, early_stopping_state = check_early_stopping(
            avg_loss,
            early_stopping_state,
            epoch,
            early_stopping_patience,
            early_stopping_min_delta,
            use_wandb,
        )
        if should_stop:
            break
