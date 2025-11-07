"""
Training script for DVAE on volume pairs

Implements:
- VAE loss (reconstruction + KL divergence)
- Training loop with validation
- Checkpointing
- TensorBoard logging
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from pathlib import Path
from tqdm import tqdm
import argparse
import json
from typing import Tuple, Dict, Optional

from models import DVAE
from data import create_dataloader


class DVAELoss(nn.Module):
    """
    Combined loss for Decoupled VAE:
    1. Reconstruction loss (MSE between output and ground truth)
    2. KL divergence for content features (regularization)
    3. KL divergence for artifact features (regularization)

    Args:
        recon_weight: Weight for reconstruction loss
        kl_weight: Weight for KL divergence
        content_weight: Relative weight for content KL
        artifact_weight: Relative weight for artifact KL
    """

    def __init__(
        self,
        recon_weight: float = 1.0,
        kl_weight: float = 0.001,
        content_weight: float = 1.0,
        artifact_weight: float = 0.5
    ):
        super().__init__()
        self.recon_weight = recon_weight
        self.kl_weight = kl_weight
        self.content_weight = content_weight
        self.artifact_weight = artifact_weight

        self.mse = nn.MSELoss()

    def kl_divergence(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """
        Compute KL divergence between latent distribution and standard normal
        KL(N(mu, sigma) || N(0, 1))
        """
        return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    def forward(
        self,
        recon: torch.Tensor,
        target: torch.Tensor,
        content_mu: torch.Tensor,
        content_logvar: torch.Tensor,
        artifact_mu: torch.Tensor,
        artifact_logvar: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute total loss and individual components

        Returns:
            total_loss: Combined weighted loss
            loss_dict: Dictionary of individual loss components
        """
        # Reconstruction loss
        recon_loss = self.mse(recon, target)

        # KL divergences
        kl_content = self.kl_divergence(content_mu, content_logvar)
        kl_artifact = self.kl_divergence(artifact_mu, artifact_logvar)

        # Normalize KL by batch size and latent dimensions
        batch_size = recon.size(0)
        latent_dim = content_mu.size(1)
        kl_content = kl_content / (batch_size * latent_dim)
        kl_artifact = kl_artifact / (batch_size * latent_dim)

        # Total KL with relative weights
        kl_total = (
            self.content_weight * kl_content +
            self.artifact_weight * kl_artifact
        )

        # Combined loss
        total_loss = (
            self.recon_weight * recon_loss +
            self.kl_weight * kl_total
        )

        # Loss components for logging
        loss_dict = {
            'total': total_loss.item(),
            'reconstruction': recon_loss.item(),
            'kl_content': kl_content.item(),
            'kl_artifact': kl_artifact.item(),
            'kl_total': kl_total.item()
        }

        return total_loss, loss_dict


class Trainer:
    """
    Trainer for DVAE model

    Args:
        model: DVAE model
        train_loader: Training data loader
        val_loader: Validation data loader (optional)
        lr: Learning rate
        device: Device to train on
        checkpoint_dir: Directory to save checkpoints
        log_dir: Directory for TensorBoard logs
    """

    def __init__(
        self,
        model: DVAE,
        train_loader,
        val_loader=None,
        lr: float = 1e-4,
        device: str = 'cuda',
        checkpoint_dir: str = 'checkpoints',
        log_dir: str = 'logs'
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device

        # Loss and optimizer
        self.criterion = DVAELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

        # Checkpointing
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Logging
        self.writer = SummaryWriter(log_dir)
        self.global_step = 0
        self.epoch = 0

    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch"""
        self.model.train()
        epoch_losses = {
            'total': 0.0,
            'reconstruction': 0.0,
            'kl_content': 0.0,
            'kl_artifact': 0.0,
            'kl_total': 0.0
        }

        pbar = tqdm(self.train_loader, desc=f"Epoch {self.epoch}")
        for batch_idx, (lamino, tomo) in enumerate(pbar):
            lamino = lamino.to(self.device)
            tomo = tomo.to(self.device)

            # Forward pass
            recon, content_mu, content_logvar, artifact_mu, artifact_logvar = self.model(lamino)

            # Compute loss
            loss, loss_dict = self.criterion(
                recon, tomo,
                content_mu, content_logvar,
                artifact_mu, artifact_logvar
            )

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # Accumulate losses
            for key, value in loss_dict.items():
                epoch_losses[key] += value

            # Update progress bar
            pbar.set_postfix({
                'loss': f"{loss_dict['total']:.4f}",
                'recon': f"{loss_dict['reconstruction']:.4f}"
            })

            # Log to TensorBoard
            if batch_idx % 10 == 0:
                for key, value in loss_dict.items():
                    self.writer.add_scalar(f'train/{key}', value, self.global_step)

            self.global_step += 1

        # Average losses
        num_batches = len(self.train_loader)
        for key in epoch_losses:
            epoch_losses[key] /= num_batches

        return epoch_losses

    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """Validate the model"""
        if self.val_loader is None:
            return {}

        self.model.eval()
        val_losses = {
            'total': 0.0,
            'reconstruction': 0.0,
            'kl_content': 0.0,
            'kl_artifact': 0.0,
            'kl_total': 0.0
        }

        for lamino, tomo in tqdm(self.val_loader, desc="Validation"):
            lamino = lamino.to(self.device)
            tomo = tomo.to(self.device)

            # Forward pass
            recon, content_mu, content_logvar, artifact_mu, artifact_logvar = self.model(lamino)

            # Compute loss
            loss, loss_dict = self.criterion(
                recon, tomo,
                content_mu, content_logvar,
                artifact_mu, artifact_logvar
            )

            # Accumulate losses
            for key, value in loss_dict.items():
                val_losses[key] += value

        # Average losses
        num_batches = len(self.val_loader)
        for key in val_losses:
            val_losses[key] /= num_batches

        # Log to TensorBoard
        for key, value in val_losses.items():
            self.writer.add_scalar(f'val/{key}', value, self.epoch)

        return val_losses

    def save_checkpoint(self, filename: str = 'checkpoint.pth'):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': self.epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }
        path = self.checkpoint_dir / filename
        torch.save(checkpoint, path)
        print(f"Checkpoint saved to {path}")

    def load_checkpoint(self, filename: str = 'checkpoint.pth'):
        """Load model checkpoint"""
        path = self.checkpoint_dir / filename
        if not path.exists():
            print(f"Checkpoint {path} not found")
            return

        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        print(f"Checkpoint loaded from {path}")

    def train(self, num_epochs: int, save_every: int = 10):
        """
        Train the model

        Args:
            num_epochs: Number of epochs to train
            save_every: Save checkpoint every N epochs
        """
        print(f"Training for {num_epochs} epochs")
        print(f"Device: {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")

        best_val_loss = float('inf')

        for epoch in range(num_epochs):
            self.epoch = epoch

            # Train
            train_losses = self.train_epoch()
            print(f"Epoch {epoch} - Train Loss: {train_losses['total']:.4f}, "
                  f"Recon: {train_losses['reconstruction']:.4f}")

            # Validate
            if self.val_loader is not None:
                val_losses = self.validate()
                print(f"Epoch {epoch} - Val Loss: {val_losses['total']:.4f}, "
                      f"Recon: {val_losses['reconstruction']:.4f}")

                # Save best model
                if val_losses['total'] < best_val_loss:
                    best_val_loss = val_losses['total']
                    self.save_checkpoint('best_model.pth')

            # Save periodic checkpoint
            if (epoch + 1) % save_every == 0:
                self.save_checkpoint(f'checkpoint_epoch_{epoch+1}.pth')

        # Save final model
        self.save_checkpoint('final_model.pth')
        self.writer.close()
        print("Training complete!")


def main():
    parser = argparse.ArgumentParser(description='Train DVAE for artifact correction')
    parser.add_argument('--lamino-path', type=str, required=True, help='Path to laminography volume (.npy)')
    parser.add_argument('--tomo-path', type=str, required=True, help='Path to tomography volume (.npy)')
    parser.add_argument('--patch-size', type=int, default=64, help='Patch size')
    parser.add_argument('--num-patches', type=int, default=1000, help='Number of patches per epoch')
    parser.add_argument('--batch-size', type=int, default=2, help='Batch size')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--latent-dim', type=int, default=128, help='Latent dimension')
    parser.add_argument('--base-channels', type=int, default=32, help='Base channels')
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints', help='Checkpoint directory')
    parser.add_argument('--log-dir', type=str, default='logs', help='Log directory')
    args = parser.parse_args()

    # Load volumes
    print("Loading volumes...")
    lamino_volume = np.load(args.lamino_path)
    tomo_volume = np.load(args.tomo_path)
    print(f"Laminography shape: {lamino_volume.shape}")
    print(f"Tomography shape: {tomo_volume.shape}")

    # Create dataloader
    train_loader = create_dataloader(
        lamino_volume,
        tomo_volume,
        patch_size=args.patch_size,
        num_patches=args.num_patches,
        batch_size=args.batch_size,
        normalize=True,
        augment=True
    )

    # Create model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = DVAE(
        in_channels=1,
        base_channels=args.base_channels,
        latent_dim=args.latent_dim,
        patch_size=args.patch_size
    )

    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        lr=args.lr,
        device=device,
        checkpoint_dir=args.checkpoint_dir,
        log_dir=args.log_dir
    )

    # Train
    trainer.train(num_epochs=args.epochs, save_every=10)


if __name__ == "__main__":
    main()
