"""
Improved training script with better loss functions for high-quality reconstruction

This version includes:
- Multi-scale reconstruction loss
- Gradient-based loss for smoothness
- SSIM loss for structural similarity
- Optional perceptual loss
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from pathlib import Path
from tqdm import tqdm
import argparse
from typing import Tuple, Dict, Optional

from models import DVAE
from data import create_multi_pair_dataloader, load_volume_pairs_from_directory


class GradientLoss(nn.Module):
    """
    Gradient loss to encourage smooth reconstructions and reduce graininess
    """
    def __init__(self):
        super().__init__()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute gradient difference loss

        Args:
            pred: Predicted volume (B, C, D, H, W)
            target: Target volume (B, C, D, H, W)
        """
        # Compute gradients in all 3 dimensions
        pred_grad_d = pred[:, :, 1:, :, :] - pred[:, :, :-1, :, :]
        pred_grad_h = pred[:, :, :, 1:, :] - pred[:, :, :, :-1, :]
        pred_grad_w = pred[:, :, :, :, 1:] - pred[:, :, :, :, :-1]

        target_grad_d = target[:, :, 1:, :, :] - target[:, :, :-1, :, :]
        target_grad_h = target[:, :, :, 1:, :] - target[:, :, :, :-1, :]
        target_grad_w = target[:, :, :, :, 1:] - target[:, :, :, :, :-1]

        # L1 loss on gradients
        loss_d = F.l1_loss(pred_grad_d, target_grad_d)
        loss_h = F.l1_loss(pred_grad_h, target_grad_h)
        loss_w = F.l1_loss(pred_grad_w, target_grad_w)

        return (loss_d + loss_h + loss_w) / 3.0


class SSIMLoss(nn.Module):
    """
    SSIM loss for structural similarity (adapted for 3D)
    """
    def __init__(self, window_size: int = 11):
        super().__init__()
        self.window_size = window_size

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute SSIM loss (1 - SSIM)

        For simplicity, we compute 2D SSIM on random slices
        """
        # Take random z-slices
        B, C, D, H, W = pred.shape
        z_idx = torch.randint(0, D, (1,)).item()

        pred_slice = pred[:, :, z_idx, :, :]  # (B, C, H, W)
        target_slice = target[:, :, z_idx, :, :]

        # Compute SSIM
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2

        mu_pred = F.avg_pool2d(pred_slice, self.window_size, stride=1, padding=self.window_size//2)
        mu_target = F.avg_pool2d(target_slice, self.window_size, stride=1, padding=self.window_size//2)

        mu_pred_sq = mu_pred ** 2
        mu_target_sq = mu_target ** 2
        mu_pred_target = mu_pred * mu_target

        sigma_pred_sq = F.avg_pool2d(pred_slice ** 2, self.window_size, stride=1, padding=self.window_size//2) - mu_pred_sq
        sigma_target_sq = F.avg_pool2d(target_slice ** 2, self.window_size, stride=1, padding=self.window_size//2) - mu_target_sq
        sigma_pred_target = F.avg_pool2d(pred_slice * target_slice, self.window_size, stride=1, padding=self.window_size//2) - mu_pred_target

        ssim_map = ((2 * mu_pred_target + C1) * (2 * sigma_pred_target + C2)) / \
                   ((mu_pred_sq + mu_target_sq + C1) * (sigma_pred_sq + sigma_target_sq + C2))

        return 1 - ssim_map.mean()


class ImprovedDVAELoss(nn.Module):
    """
    Improved loss for DVAE with better reconstruction quality

    Args:
        recon_weight: Weight for reconstruction loss
        kl_weight: Weight for KL divergence
        gradient_weight: Weight for gradient loss (smoothness)
        ssim_weight: Weight for SSIM loss (structural similarity)
        l1_weight: Weight for L1 loss (vs L2/MSE)
        content_weight: Relative weight for content KL
        artifact_weight: Relative weight for artifact KL
    """

    def __init__(
        self,
        recon_weight: float = 1.0,
        kl_weight: float = 0.0001,  # Reduced from 0.001
        gradient_weight: float = 0.5,
        ssim_weight: float = 0.2,
        l1_weight: float = 0.5,
        content_weight: float = 1.0,
        artifact_weight: float = 0.3  # Reduced from 0.5
    ):
        super().__init__()
        self.recon_weight = recon_weight
        self.kl_weight = kl_weight
        self.gradient_weight = gradient_weight
        self.ssim_weight = ssim_weight
        self.l1_weight = l1_weight
        self.content_weight = content_weight
        self.artifact_weight = artifact_weight

        self.mse = nn.MSELoss()
        self.l1 = nn.L1Loss()
        self.gradient_loss = GradientLoss()
        self.ssim_loss = SSIMLoss()

    def kl_divergence(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Compute KL divergence"""
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
        """
        # Reconstruction losses
        mse_loss = self.mse(recon, target)
        l1_loss = self.l1(recon, target)
        recon_loss = (1 - self.l1_weight) * mse_loss + self.l1_weight * l1_loss

        # Gradient loss for smoothness
        grad_loss = self.gradient_loss(recon, target)

        # SSIM loss for structure
        ssim_loss = self.ssim_loss(recon, target)

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
            self.gradient_weight * grad_loss +
            self.ssim_weight * ssim_loss +
            self.kl_weight * kl_total
        )

        # Loss components for logging
        loss_dict = {
            'total': total_loss.item(),
            'reconstruction': recon_loss.item(),
            'mse': mse_loss.item(),
            'l1': l1_loss.item(),
            'gradient': grad_loss.item(),
            'ssim': ssim_loss.item(),
            'kl_content': kl_content.item(),
            'kl_artifact': kl_artifact.item(),
            'kl_total': kl_total.item()
        }

        return total_loss, loss_dict


class ImprovedTrainer:
    """
    Improved trainer with better optimization strategies
    """

    def __init__(
        self,
        model: DVAE,
        train_loader,
        val_loader=None,
        lr: float = 5e-5,  # Lower learning rate for stability
        device: str = 'cuda',
        checkpoint_dir: str = 'checkpoints',
        log_dir: str = 'logs',
        use_improved_loss: bool = True
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device

        # Loss function
        if use_improved_loss:
            self.criterion = ImprovedDVAELoss()
            print("Using improved loss with gradient and SSIM components")
        else:
            from train import DVAELoss
            self.criterion = DVAELoss()
            print("Using standard loss")

        # Optimizer with weight decay for regularization
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=lr,
            weight_decay=1e-5
        )

        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=10,
            verbose=True
        )

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
            'gradient': 0.0,
            'ssim': 0.0,
            'kl_content': 0.0,
            'kl_artifact': 0.0,
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

            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            self.optimizer.step()

            # Accumulate losses
            for key, value in loss_dict.items():
                if key in epoch_losses:
                    epoch_losses[key] += value

            # Update progress bar
            pbar.set_postfix({
                'loss': f"{loss_dict['total']:.4f}",
                'recon': f"{loss_dict['reconstruction']:.4f}",
                'grad': f"{loss_dict.get('gradient', 0):.4f}"
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
            'gradient': 0.0,
            'ssim': 0.0,
            'kl_content': 0.0,
            'kl_artifact': 0.0,
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
                if key in val_losses:
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
            'scheduler_state_dict': self.scheduler.state_dict(),
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
        if 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        print(f"Checkpoint loaded from {path}")

    def train(self, num_epochs: int, save_every: int = 10):
        """Train the model"""
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

                # Update learning rate
                self.scheduler.step(val_losses['total'])

                # Save best model
                if val_losses['total'] < best_val_loss:
                    best_val_loss = val_losses['total']
                    self.save_checkpoint('best_model.pth')
            else:
                # No validation, use train loss for scheduler
                self.scheduler.step(train_losses['total'])

            # Save periodic checkpoint
            if (epoch + 1) % save_every == 0:
                self.save_checkpoint(f'checkpoint_epoch_{epoch+1}.pth')

        # Save final model
        self.save_checkpoint('final_model.pth')
        self.writer.close()
        print("Training complete!")


def main():
    parser = argparse.ArgumentParser(description='Improved DVAE training for multiple volume pairs')
    parser.add_argument('--data-dir', type=str, help='Directory containing volume pairs')
    parser.add_argument('--pair-list', type=str, nargs='+', help='List of lamino,tomo pairs')
    parser.add_argument('--patch-size', type=int, default=64, help='Patch size')
    parser.add_argument('--patches-per-pair', type=int, default=200, help='Patches per pair per epoch')
    parser.add_argument('--batch-size', type=int, default=2, help='Batch size')
    parser.add_argument('--epochs', type=int, default=150, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=5e-5, help='Learning rate')
    parser.add_argument('--latent-dim', type=int, default=128, help='Latent dimension')
    parser.add_argument('--base-channels', type=int, default=32, help='Base channels')
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints_improved', help='Checkpoint directory')
    parser.add_argument('--log-dir', type=str, default='logs_improved', help='Log directory')
    parser.add_argument('--standard-loss', action='store_true', help='Use standard loss instead of improved')
    args = parser.parse_args()

    # Load volume pairs
    if args.data_dir:
        print(f"Loading volume pairs from {args.data_dir}")
        pairs = load_volume_pairs_from_directory(args.data_dir)
    elif args.pair_list:
        pairs = []
        for pair_str in args.pair_list:
            lamino, tomo = pair_str.split(',')
            pairs.append((lamino, tomo))
    else:
        raise ValueError("Must provide either --data-dir or --pair-list")

    print(f"Found {len(pairs)} volume pairs")

    # Create dataloader
    train_loader = create_multi_pair_dataloader(
        pairs,
        patch_size=args.patch_size,
        patches_per_pair=args.patches_per_pair,
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
    trainer = ImprovedTrainer(
        model=model,
        train_loader=train_loader,
        lr=args.lr,
        device=device,
        checkpoint_dir=args.checkpoint_dir,
        log_dir=args.log_dir,
        use_improved_loss=not args.standard_loss
    )

    # Train
    trainer.train(num_epochs=args.epochs, save_every=max(args.epochs // 10, 1))


if __name__ == "__main__":
    main()
