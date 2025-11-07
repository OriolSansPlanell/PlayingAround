"""
Demo script for DVAE artifact correction on a single volume pair

This script demonstrates the complete pipeline:
1. Generate synthetic volume pair (or load real data)
2. Train DVAE model
3. Apply model to correct artifacts
4. Visualize and evaluate results
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path
import argparse

from models import DVAE
from data import create_dataloader
from train import Trainer
from inference import VolumeInference
from utils.visualization import plot_slices, compute_metrics


def generate_synthetic_volumes(size: int = 128, artifact_strength: float = 0.3) -> tuple:
    """
    Generate synthetic volume pair for demonstration

    Creates a ground truth volume with structures, then adds defocusing artifacts
    to simulate laminography.

    Args:
        size: Volume size (cubic)
        artifact_strength: Strength of artifacts (0-1)

    Returns:
        laminography_volume, tomography_volume
    """
    print("Generating synthetic volumes...")

    # Create ground truth with various structures
    tomography = np.zeros((size, size, size), dtype=np.float32)

    # Add spheres
    center = size // 2
    for i in range(5):
        r = np.random.randint(5, 15)
        cx = np.random.randint(r, size - r)
        cy = np.random.randint(r, size - r)
        cz = np.random.randint(r, size - r)

        z, y, x = np.ogrid[:size, :size, :size]
        mask = (x - cx)**2 + (y - cy)**2 + (z - cz)**2 <= r**2
        tomography[mask] = np.random.uniform(0.5, 1.0)

    # Add some noise
    tomography += np.random.randn(size, size, size) * 0.05
    tomography = np.clip(tomography, 0, 1)

    # Simulate laminography artifacts (defocusing)
    # Apply depth-dependent blur
    from scipy.ndimage import gaussian_filter

    laminography = tomography.copy()

    # Apply varying blur along depth
    for z in range(size):
        # Blur increases away from focal plane
        blur_sigma = artifact_strength * abs(z - center) / center
        laminography[z] = gaussian_filter(laminography[z], sigma=blur_sigma * 2)

    # Add some additional artifacts
    # Reduced contrast
    laminography = laminography * 0.8 + 0.1

    # Add streak artifacts
    for _ in range(3):
        angle = np.random.uniform(0, np.pi)
        for z in range(size):
            streak = np.random.randn(size, size) * 0.1
            streak = gaussian_filter(streak, sigma=10)
            laminography[z] += streak * artifact_strength * 0.5

    laminography = np.clip(laminography, 0, 1)

    print(f"Generated volumes of shape: {tomography.shape}")
    print(f"Tomography range: [{tomography.min():.3f}, {tomography.max():.3f}]")
    print(f"Laminography range: [{laminography.min():.3f}, {laminography.max():.3f}]")

    return laminography, tomography


def run_demo(
    volume_size: int = 128,
    patch_size: int = 64,
    num_patches: int = 200,
    batch_size: int = 2,
    epochs: int = 50,
    latent_dim: int = 128,
    base_channels: int = 32,
    lr: float = 1e-4,
    device: str = 'cuda',
    output_dir: str = 'demo_output',
    use_synthetic: bool = True,
    lamino_path: str = None,
    tomo_path: str = None
):
    """
    Run complete demonstration pipeline

    Args:
        volume_size: Size of synthetic volumes
        patch_size: Patch size for training
        num_patches: Number of patches per epoch
        batch_size: Training batch size
        epochs: Number of training epochs
        latent_dim: Latent space dimension
        base_channels: Base number of channels
        lr: Learning rate
        device: Device to use ('cuda' or 'cpu')
        output_dir: Directory for outputs
        use_synthetic: Whether to use synthetic data
        lamino_path: Path to real laminography data (.npy)
        tomo_path: Path to real tomography data (.npy)
    """
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print("="*60)
    print("DVAE Laminography Artifact Correction - Demo")
    print("="*60)

    # Step 1: Load or generate data
    if use_synthetic:
        laminography, tomography = generate_synthetic_volumes(
            size=volume_size,
            artifact_strength=0.3
        )
        # Save synthetic volumes
        np.save(output_path / 'synthetic_laminography.npy', laminography)
        np.save(output_path / 'synthetic_tomography.npy', tomography)
    else:
        print(f"Loading volumes from {lamino_path} and {tomo_path}")
        laminography = np.load(lamino_path)
        tomography = np.load(tomo_path)
        print(f"Loaded volumes of shape: {laminography.shape}")

    # Initial comparison
    print("\n" + "="*60)
    print("Initial Comparison (before correction)")
    print("="*60)
    initial_metrics = compute_metrics(laminography, tomography)
    for key, value in initial_metrics.items():
        print(f"{key}: {value:.4f}")

    # Plot initial comparison
    fig = plot_slices(laminography, tomography, slice_idx=volume_size//2)
    plt.savefig(output_path / 'initial_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()

    # Step 2: Create dataloader
    print("\n" + "="*60)
    print("Creating training data")
    print("="*60)
    train_loader = create_dataloader(
        laminography,
        tomography,
        patch_size=patch_size,
        num_patches=num_patches,
        batch_size=batch_size,
        normalize=True,
        augment=True
    )
    print(f"Created dataloader with {len(train_loader)} batches per epoch")

    # Step 3: Create and train model
    print("\n" + "="*60)
    print("Training DVAE model")
    print("="*60)

    if device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        device = 'cpu'

    model = DVAE(
        in_channels=1,
        base_channels=base_channels,
        latent_dim=latent_dim,
        patch_size=patch_size
    )

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        lr=lr,
        device=device,
        checkpoint_dir=str(output_path / 'checkpoints'),
        log_dir=str(output_path / 'logs')
    )

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Training for {epochs} epochs...")

    trainer.train(num_epochs=epochs, save_every=max(epochs // 5, 1))

    # Step 4: Apply model to full volume
    print("\n" + "="*60)
    print("Applying trained model to full volume")
    print("="*60)

    inference = VolumeInference(
        model=model,
        device=device,
        patch_size=patch_size,
        overlap=0.5,
        blend_mode='gaussian'
    )

    corrected = inference.process_volume(laminography, normalize=True)

    # Save corrected volume
    np.save(output_path / 'corrected_volume.npy', corrected)
    print(f"Corrected volume saved to {output_path / 'corrected_volume.npy'}")

    # Step 5: Evaluate results
    print("\n" + "="*60)
    print("Final Results")
    print("="*60)

    final_metrics = compute_metrics(corrected, tomography)

    print("\nBefore correction:")
    for key, value in initial_metrics.items():
        print(f"  {key}: {value:.4f}")

    print("\nAfter correction:")
    for key, value in final_metrics.items():
        print(f"  {key}: {value:.4f}")

    print("\nImprovement:")
    for key in initial_metrics:
        if key in ['PSNR', 'SSIM']:
            improvement = final_metrics[key] - initial_metrics[key]
            print(f"  {key}: {improvement:+.4f}")
        else:  # MSE, MAE (lower is better)
            improvement = initial_metrics[key] - final_metrics[key]
            print(f"  {key}: {improvement:+.4f} (reduction)")

    # Step 6: Visualize results
    print("\n" + "="*60)
    print("Generating visualizations")
    print("="*60)

    # Plot comparison with correction
    fig = plot_slices(
        laminography,
        tomography,
        corrected,
        slice_idx=volume_size//2
    )
    plt.savefig(output_path / 'final_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved comparison to {output_path / 'final_comparison.png'}")

    # Plot multiple slices
    for axis in range(3):
        fig = plot_slices(
            laminography,
            tomography,
            corrected,
            slice_idx=volume_size//2,
            axis=axis
        )
        plt.savefig(output_path / f'comparison_axis{axis}.png', dpi=150, bbox_inches='tight')
        plt.close()

    print("\n" + "="*60)
    print("Demo complete!")
    print(f"All outputs saved to: {output_path}")
    print("="*60)


def main():
    parser = argparse.ArgumentParser(description='DVAE Demo for Laminography Artifact Correction')

    # Data options
    parser.add_argument('--use-real-data', action='store_true',
                        help='Use real data instead of synthetic')
    parser.add_argument('--lamino-path', type=str,
                        help='Path to laminography volume (.npy)')
    parser.add_argument('--tomo-path', type=str,
                        help='Path to tomography volume (.npy)')
    parser.add_argument('--volume-size', type=int, default=128,
                        help='Size of synthetic volumes (if not using real data)')

    # Training options
    parser.add_argument('--patch-size', type=int, default=64,
                        help='Patch size for training')
    parser.add_argument('--num-patches', type=int, default=200,
                        help='Number of patches per epoch')
    parser.add_argument('--batch-size', type=int, default=2,
                        help='Batch size')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate')

    # Model options
    parser.add_argument('--latent-dim', type=int, default=128,
                        help='Latent dimension')
    parser.add_argument('--base-channels', type=int, default=32,
                        help='Base number of channels')

    # Other options
    parser.add_argument('--device', type=str, default='cuda',
                        choices=['cuda', 'cpu'],
                        help='Device to use')
    parser.add_argument('--output-dir', type=str, default='demo_output',
                        help='Output directory')

    args = parser.parse_args()

    # Validate arguments
    if args.use_real_data:
        if not args.lamino_path or not args.tomo_path:
            parser.error("--lamino-path and --tomo-path required when using real data")

    # Run demo
    run_demo(
        volume_size=args.volume_size,
        patch_size=args.patch_size,
        num_patches=args.num_patches,
        batch_size=args.batch_size,
        epochs=args.epochs,
        latent_dim=args.latent_dim,
        base_channels=args.base_channels,
        lr=args.lr,
        device=args.device,
        output_dir=args.output_dir,
        use_synthetic=not args.use_real_data,
        lamino_path=args.lamino_path,
        tomo_path=args.tomo_path
    )


if __name__ == "__main__":
    main()
