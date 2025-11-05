"""
Inference script for applying trained DVAE to full volumes

Handles:
- Loading trained model
- Extracting overlapping patches
- Processing patches through model
- Reconstructing full volume with blending
"""

import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
import argparse
from typing import Optional

from models import DVAE
from data import OverlappingPatchExtractor


class VolumeInference:
    """
    Perform inference on full 3D volumes using trained DVAE

    Args:
        model: Trained DVAE model
        device: Device to run inference on
        patch_size: Size of patches (must match training)
        overlap: Overlap ratio for patches
        blend_mode: Blending method ('gaussian', 'linear', or 'average')
    """

    def __init__(
        self,
        model: DVAE,
        device: str = 'cuda',
        patch_size: int = 64,
        overlap: float = 0.5,
        blend_mode: str = 'gaussian'
    ):
        self.model = model.to(device)
        self.model.eval()
        self.device = device
        self.patch_size = patch_size
        self.overlap = overlap
        self.blend_mode = blend_mode

    def _normalize(self, volume: np.ndarray) -> tuple[np.ndarray, float, float]:
        """Normalize volume and return parameters for denormalization"""
        vmin, vmax = volume.min(), volume.max()
        if vmax > vmin:
            normalized = (volume - vmin) / (vmax - vmin)
        else:
            normalized = volume
        return normalized, vmin, vmax

    def _denormalize(self, volume: np.ndarray, vmin: float, vmax: float) -> np.ndarray:
        """Denormalize volume to original range"""
        return volume * (vmax - vmin) + vmin

    @torch.no_grad()
    def process_volume(self, volume: np.ndarray, normalize: bool = True) -> np.ndarray:
        """
        Process a full volume through the model

        Args:
            volume: Input volume with artifacts (D, H, W)
            normalize: Whether to normalize input

        Returns:
            Corrected volume (D, H, W)
        """
        # Normalize if requested
        if normalize:
            volume_norm, vmin, vmax = self._normalize(volume)
        else:
            volume_norm = volume
            vmin, vmax = 0.0, 1.0

        # Create patch extractor
        extractor = OverlappingPatchExtractor(
            volume_norm,
            patch_size=self.patch_size,
            overlap=self.overlap
        )

        print(f"Processing volume of shape {volume.shape}")
        print(f"Extracting {len(extractor)} overlapping patches...")

        # Process patches
        processed_patches = []
        for idx in tqdm(range(len(extractor)), desc="Processing patches"):
            patch, coord = extractor.get_patch(idx)

            # Prepare input tensor
            patch_tensor = torch.from_numpy(patch).unsqueeze(0).unsqueeze(0)  # (1, 1, D, H, W)
            patch_tensor = patch_tensor.to(self.device)

            # Forward pass
            recon, _, _, _, _ = self.model(patch_tensor)

            # Convert back to numpy
            recon_patch = recon.squeeze(0).squeeze(0).cpu().numpy()
            processed_patches.append(recon_patch)

        # Reconstruct full volume
        print("Reconstructing volume with blending...")
        output_volume = extractor.reconstruct_volume(
            processed_patches,
            blend=self.blend_mode
        )

        # Denormalize if needed
        if normalize:
            output_volume = self._denormalize(output_volume, vmin, vmax)

        return output_volume

    def process_and_save(
        self,
        input_path: str,
        output_path: str,
        normalize: bool = True
    ):
        """
        Load volume, process it, and save result

        Args:
            input_path: Path to input volume (.npy)
            output_path: Path to save output volume (.npy)
            normalize: Whether to normalize input
        """
        # Load volume
        print(f"Loading volume from {input_path}")
        volume = np.load(input_path)

        # Process
        output = self.process_volume(volume, normalize=normalize)

        # Save
        print(f"Saving output to {output_path}")
        np.save(output_path, output)

        print("Done!")


def load_model(
    checkpoint_path: str,
    patch_size: int = 64,
    base_channels: int = 32,
    latent_dim: int = 128,
    device: str = 'cuda'
) -> DVAE:
    """
    Load trained model from checkpoint

    Args:
        checkpoint_path: Path to checkpoint file
        patch_size: Patch size used in training
        base_channels: Base channels used in training
        latent_dim: Latent dimension used in training
        device: Device to load model on

    Returns:
        Loaded DVAE model
    """
    # Create model
    model = DVAE(
        in_channels=1,
        base_channels=base_channels,
        latent_dim=latent_dim,
        patch_size=patch_size
    )

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Handle different checkpoint formats
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")
    else:
        model.load_state_dict(checkpoint)

    model = model.to(device)
    model.eval()

    return model


def main():
    parser = argparse.ArgumentParser(description='Apply trained DVAE to volume')
    parser.add_argument('--input', type=str, required=True, help='Input volume path (.npy)')
    parser.add_argument('--output', type=str, required=True, help='Output volume path (.npy)')
    parser.add_argument('--checkpoint', type=str, required=True, help='Model checkpoint path')
    parser.add_argument('--patch-size', type=int, default=64, help='Patch size')
    parser.add_argument('--overlap', type=float, default=0.5, help='Overlap ratio (0-1)')
    parser.add_argument('--blend-mode', type=str, default='gaussian',
                        choices=['gaussian', 'linear', 'average'],
                        help='Blending mode for overlapping patches')
    parser.add_argument('--latent-dim', type=int, default=128, help='Latent dimension')
    parser.add_argument('--base-channels', type=int, default=32, help='Base channels')
    parser.add_argument('--no-normalize', action='store_true', help='Skip normalization')
    parser.add_argument('--device', type=str, default='cuda', help='Device (cuda/cpu)')

    args = parser.parse_args()

    # Check device
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        args.device = 'cpu'

    # Load model
    print("Loading model...")
    model = load_model(
        checkpoint_path=args.checkpoint,
        patch_size=args.patch_size,
        base_channels=args.base_channels,
        latent_dim=args.latent_dim,
        device=args.device
    )

    # Create inference object
    inference = VolumeInference(
        model=model,
        device=args.device,
        patch_size=args.patch_size,
        overlap=args.overlap,
        blend_mode=args.blend_mode
    )

    # Process volume
    inference.process_and_save(
        input_path=args.input,
        output_path=args.output,
        normalize=not args.no_normalize
    )


if __name__ == "__main__":
    main()
