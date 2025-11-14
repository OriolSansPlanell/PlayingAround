"""
Example: Training on multiple volume pairs for better quality

This script demonstrates how to train on multiple laminography-tomography pairs
using the improved training pipeline for high-quality reconstruction.
"""

import numpy as np
import torch
from pathlib import Path

from models import DVAE
from data import create_multi_pair_dataloader, load_volume_pairs_from_directory
from train_improved import ImprovedTrainer

# Set random seed for reproducibility
np.random.seed(42)
torch.manual_seed(42)

print("="*70)
print("Multi-Pair DVAE Training Example")
print("="*70)

# ============================================================================
# METHOD 1: Load pairs from a directory
# ============================================================================
print("\nMETHOD 1: Auto-discover pairs in directory")
print("-"*70)

# Example: If you have files like:
#   data/sample1_lamino.npy, data/sample1_tomo.npy
#   data/sample2_lamino.npy, data/sample2_tomo.npy
#   etc.

data_dir = Path("./data")
if data_dir.exists():
    pairs = load_volume_pairs_from_directory(
        str(data_dir),
        lamino_pattern="*lamino*.npy",
        tomo_pattern="*tomo*.npy"
    )
    print(f"Found {len(pairs)} pairs in {data_dir}")
else:
    print(f"Directory {data_dir} not found, skipping auto-discovery")
    pairs = []

# ============================================================================
# METHOD 2: Manually specify pairs
# ============================================================================
print("\nMETHOD 2: Manually specify pairs")
print("-"*70)

# Option A: Specify file paths
manual_pairs = [
    # ('path/to/lamino1.npy', 'path/to/tomo1.npy'),
    # ('path/to/lamino2.npy', 'path/to/tomo2.npy'),
    # ('path/to/lamino3.npy', 'path/to/tomo3.npy'),
]

# Option B: Create synthetic pairs for demonstration
print("Creating synthetic pairs for demonstration...")
synthetic_pairs = []
for i in range(3):
    # Create synthetic volumes (replace with your real data)
    size = 96
    lamino = np.random.randn(size, size, size).astype(np.float32) * 0.3 + 0.5
    tomo = np.random.randn(size, size, size).astype(np.float32) * 0.3 + 0.5

    # Clip to [0, 1] range
    lamino = np.clip(lamino, 0, 1)
    tomo = np.clip(tomo, 0, 1)

    synthetic_pairs.append((lamino, tomo))
    print(f"  Created synthetic pair {i+1}: shape {lamino.shape}")

# Use synthetic pairs if no real pairs found
if not pairs and not manual_pairs:
    pairs = synthetic_pairs
    print(f"\nUsing {len(pairs)} synthetic pairs for demonstration")
elif manual_pairs:
    pairs = manual_pairs
    print(f"\nUsing {len(pairs)} manually specified pairs")

# ============================================================================
# Create dataloader
# ============================================================================
print("\n" + "="*70)
print("Creating Multi-Pair Dataloader")
print("="*70)

train_loader = create_multi_pair_dataloader(
    volume_pairs=pairs,
    patch_size=64,
    patches_per_pair=100,  # 100 patches from each pair per epoch
    batch_size=2,
    normalize=True,
    augment=True
)

total_patches = len(pairs) * 100
print(f"Total patches per epoch: {total_patches}")
print(f"Batches per epoch: {len(train_loader)}")

# ============================================================================
# Create model
# ============================================================================
print("\n" + "="*70)
print("Creating DVAE Model")
print("="*70)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Device: {device}")

model = DVAE(
    in_channels=1,
    base_channels=32,      # Increase to 48 or 64 for better quality
    latent_dim=128,
    patch_size=64
)

num_params = sum(p.numel() for p in model.parameters())
print(f"Model parameters: {num_params:,}")

# ============================================================================
# Create improved trainer
# ============================================================================
print("\n" + "="*70)
print("Creating Improved Trainer")
print("="*70)

trainer = ImprovedTrainer(
    model=model,
    train_loader=train_loader,
    val_loader=None,       # Add validation loader if you have one
    lr=5e-5,               # Lower LR for stability
    device=device,
    checkpoint_dir='checkpoints_multi_example',
    log_dir='logs_multi_example',
    use_improved_loss=True  # Use improved loss with gradient & SSIM
)

print("Using improved loss function with:")
print("  - MSE + L1 reconstruction loss")
print("  - Gradient loss for smoothness")
print("  - SSIM loss for structural similarity")
print("  - Reduced KL divergence weight")

# ============================================================================
# Train
# ============================================================================
print("\n" + "="*70)
print("Training")
print("="*70)

num_epochs = 10  # Increase to 100-150 for real training
print(f"Training for {num_epochs} epochs...")
print("(This is a short demo. For real training, use 100-150 epochs)")

trainer.train(num_epochs=num_epochs, save_every=5)

# ============================================================================
# Results
# ============================================================================
print("\n" + "="*70)
print("Training Complete!")
print("="*70)
print(f"Checkpoints saved to: checkpoints_multi_example/")
print(f"Logs saved to: logs_multi_example/")
print("\nTo view training progress:")
print("  tensorboard --logdir logs_multi_example")
print("\nTo use the trained model for inference:")
print("  See inference.py or USAGE_GUIDE.md")
print("="*70)

# ============================================================================
# Optional: Quick inference test
# ============================================================================
print("\n" + "="*70)
print("Quick Inference Test")
print("="*70)

from inference import VolumeInference

# Load a test volume (use first pair's laminography)
if isinstance(pairs[0][0], str):
    test_volume = np.load(pairs[0][0])
else:
    test_volume = pairs[0][0]

print(f"Test volume shape: {test_volume.shape}")

# Create inference object
inference = VolumeInference(
    model=model,
    device=device,
    patch_size=64,
    overlap=0.5,
    blend_mode='gaussian'
)

# Process (this may take a while for large volumes)
print("Processing test volume...")
corrected = inference.process_volume(test_volume[:, :80, :80])  # Small crop for demo

print(f"Corrected volume shape: {corrected.shape}")
print("\nInference test complete!")
