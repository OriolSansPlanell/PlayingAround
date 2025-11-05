"""
Visualization and metrics utilities for comparing volumes
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional, Tuple


def compute_metrics(predicted: np.ndarray, ground_truth: np.ndarray) -> dict:
    """
    Compute comparison metrics between predicted and ground truth volumes

    Args:
        predicted: Predicted volume
        ground_truth: Ground truth volume

    Returns:
        Dictionary of metrics
    """
    # Mean Squared Error
    mse = np.mean((predicted - ground_truth) ** 2)

    # Peak Signal-to-Noise Ratio
    if mse > 0:
        max_val = ground_truth.max()
        psnr = 20 * np.log10(max_val / np.sqrt(mse))
    else:
        psnr = float('inf')

    # Mean Absolute Error
    mae = np.mean(np.abs(predicted - ground_truth))

    # Structural Similarity (simplified version)
    mu_pred = np.mean(predicted)
    mu_gt = np.mean(ground_truth)
    sigma_pred = np.std(predicted)
    sigma_gt = np.std(ground_truth)
    sigma_pred_gt = np.mean((predicted - mu_pred) * (ground_truth - mu_gt))

    c1 = 0.01 ** 2
    c2 = 0.03 ** 2

    ssim = ((2 * mu_pred * mu_gt + c1) * (2 * sigma_pred_gt + c2)) / \
           ((mu_pred**2 + mu_gt**2 + c1) * (sigma_pred**2 + sigma_gt**2 + c2))

    return {
        'MSE': mse,
        'MAE': mae,
        'PSNR': psnr,
        'SSIM': ssim
    }


def plot_slices(
    laminography: np.ndarray,
    tomography: np.ndarray,
    corrected: Optional[np.ndarray] = None,
    slice_idx: Optional[int] = None,
    axis: int = 0,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (15, 5)
):
    """
    Plot comparison slices from volumes

    Args:
        laminography: Input volume with artifacts
        tomography: Ground truth volume
        corrected: Corrected volume (optional)
        slice_idx: Slice index (if None, uses middle slice)
        axis: Axis along which to slice (0, 1, or 2)
        save_path: Path to save figure
        figsize: Figure size
    """
    # Get middle slice if not specified
    if slice_idx is None:
        slice_idx = laminography.shape[axis] // 2

    # Extract slices
    if axis == 0:
        lam_slice = laminography[slice_idx, :, :]
        tomo_slice = tomography[slice_idx, :, :]
        corr_slice = corrected[slice_idx, :, :] if corrected is not None else None
    elif axis == 1:
        lam_slice = laminography[:, slice_idx, :]
        tomo_slice = tomography[:, slice_idx, :]
        corr_slice = corrected[:, slice_idx, :] if corrected is not None else None
    else:
        lam_slice = laminography[:, :, slice_idx]
        tomo_slice = tomography[:, :, slice_idx]
        corr_slice = corrected[:, :, slice_idx] if corrected is not None else None

    # Determine number of plots
    num_plots = 3 if corrected is not None else 2
    fig, axes = plt.subplots(1, num_plots, figsize=figsize)

    # Plot laminography
    im0 = axes[0].imshow(lam_slice, cmap='gray')
    axes[0].set_title('Laminography (Input)')
    axes[0].axis('off')
    plt.colorbar(im0, ax=axes[0], fraction=0.046)

    # Plot tomography
    im1 = axes[1].imshow(tomo_slice, cmap='gray')
    axes[1].set_title('Tomography (Ground Truth)')
    axes[1].axis('off')
    plt.colorbar(im1, ax=axes[1], fraction=0.046)

    # Plot corrected if available
    if corrected is not None:
        im2 = axes[2].imshow(corr_slice, cmap='gray')
        axes[2].set_title('DVAE Corrected')
        axes[2].axis('off')
        plt.colorbar(im2, ax=axes[2], fraction=0.046)

        # Compute and display metrics for this slice
        metrics = compute_metrics(corr_slice, tomo_slice)
        fig.suptitle(
            f"Slice {slice_idx} (axis {axis}) - "
            f"MSE: {metrics['MSE']:.4f}, PSNR: {metrics['PSNR']:.2f} dB, "
            f"SSIM: {metrics['SSIM']:.4f}",
            fontsize=12
        )

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Figure saved to {save_path}")

    return fig


def plot_3d_comparison(
    laminography: np.ndarray,
    tomography: np.ndarray,
    corrected: Optional[np.ndarray] = None,
    num_slices: int = 3,
    save_path: Optional[str] = None
):
    """
    Plot multiple slices along all three axes

    Args:
        laminography: Input volume with artifacts
        tomography: Ground truth volume
        corrected: Corrected volume (optional)
        num_slices: Number of slices to show per axis
        save_path: Path to save figure
    """
    num_rows = 3  # One row per axis
    num_cols = num_slices * (3 if corrected is not None else 2)

    fig = plt.figure(figsize=(num_cols * 3, num_rows * 3))

    for axis in range(3):
        depth = laminography.shape[axis]
        indices = np.linspace(depth // 4, 3 * depth // 4, num_slices, dtype=int)

        for i, idx in enumerate(indices):
            # Extract slices
            if axis == 0:
                lam_slice = laminography[idx, :, :]
                tomo_slice = tomography[idx, :, :]
                corr_slice = corrected[idx, :, :] if corrected is not None else None
            elif axis == 1:
                lam_slice = laminography[:, idx, :]
                tomo_slice = tomography[:, idx, :]
                corr_slice = corrected[:, idx, :] if corrected is not None else None
            else:
                lam_slice = laminography[:, :, idx]
                tomo_slice = tomography[:, :, idx]
                corr_slice = corrected[:, :, idx] if corrected is not None else None

            base_col = i * (3 if corrected is not None else 2)

            # Laminography
            ax = plt.subplot(num_rows, num_cols, axis * num_cols + base_col + 1)
            ax.imshow(lam_slice, cmap='gray')
            if i == 0:
                ax.set_ylabel(f'Axis {axis}', fontsize=12)
            ax.set_title(f'Lamino {idx}', fontsize=10)
            ax.axis('off')

            # Tomography
            ax = plt.subplot(num_rows, num_cols, axis * num_cols + base_col + 2)
            ax.imshow(tomo_slice, cmap='gray')
            ax.set_title(f'Tomo {idx}', fontsize=10)
            ax.axis('off')

            # Corrected
            if corrected is not None:
                ax = plt.subplot(num_rows, num_cols, axis * num_cols + base_col + 3)
                ax.imshow(corr_slice, cmap='gray')
                ax.set_title(f'Corrected {idx}', fontsize=10)
                ax.axis('off')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Figure saved to {save_path}")

    return fig


def plot_training_progress(log_dir: str, save_path: Optional[str] = None):
    """
    Plot training metrics from TensorBoard logs

    Args:
        log_dir: Directory containing TensorBoard logs
        save_path: Path to save figure
    """
    try:
        from tensorboard.backend.event_processing import event_accumulator

        ea = event_accumulator.EventAccumulator(log_dir)
        ea.Reload()

        # Get available scalar tags
        tags = ea.Tags()['scalars']

        # Create subplots for different metrics
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        axes = axes.flatten()

        metrics_to_plot = [
            'train/total',
            'train/reconstruction',
            'train/kl_content',
            'train/kl_artifact'
        ]

        for idx, tag in enumerate(metrics_to_plot):
            if tag in tags:
                events = ea.Scalars(tag)
                steps = [e.step for e in events]
                values = [e.value for e in events]

                axes[idx].plot(steps, values)
                axes[idx].set_title(tag.replace('train/', '').replace('_', ' ').title())
                axes[idx].set_xlabel('Step')
                axes[idx].set_ylabel('Value')
                axes[idx].grid(True)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Training progress plot saved to {save_path}")

        return fig

    except ImportError:
        print("tensorboard package required for plotting training progress")
        return None
