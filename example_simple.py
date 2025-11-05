"""
Minimal example of using DVAE for artifact correction

This script shows the simplest possible usage with synthetic data.
"""

import numpy as np
import torch
from models import DVAE
from data import create_dataloader
from train import Trainer
from inference import VolumeInference

# Set random seed for reproducibility
np.random.seed(42)
torch.manual_seed(42)

print("DVAE Simple Example")
print("=" * 60)

# 1. Create synthetic data (replace with your real data)
print("Creating synthetic data...")
size = 96  # Smaller for faster demo
lamino = np.random.randn(size, size, size).astype(np.float32)
tomo = np.random.randn(size, size, size).astype(np.float32)
print(f"Volume shape: {lamino.shape}")

# 2. Create dataloader
print("\nCreating dataloader...")
train_loader = create_dataloader(
    lamino, tomo,
    patch_size=64,
    num_patches=50,  # Small number for quick demo
    batch_size=2
)
print(f"Batches per epoch: {len(train_loader)}")

# 3. Create model
print("\nCreating model...")
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

model = DVAE(
    in_channels=1,
    base_channels=16,  # Smaller for faster training
    latent_dim=64,
    patch_size=64
)
print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

# 4. Train
print("\nTraining...")
trainer = Trainer(
    model=model,
    train_loader=train_loader,
    lr=1e-3,
    device=device,
    checkpoint_dir='checkpoints_simple',
    log_dir='logs_simple'
)

trainer.train(num_epochs=5, save_every=5)  # Just 5 epochs for demo

# 5. Inference
print("\nRunning inference...")
inference = VolumeInference(
    model=model,
    device=device,
    patch_size=64,
    overlap=0.5
)

corrected = inference.process_volume(lamino)
print(f"Output shape: {corrected.shape}")

print("\n" + "=" * 60)
print("Done! Check 'checkpoints_simple' for saved models.")
print("Run 'tensorboard --logdir logs_simple' to view training.")
