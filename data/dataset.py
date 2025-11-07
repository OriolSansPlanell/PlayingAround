"""
Dataset classes for loading and processing 3D volume pairs (laminography + tomography)

Supports:
- Loading volume pairs from numpy arrays or files
- Random/systematic sub-cropping for patch extraction
- Overlapping patches for training augmentation
- Data normalization and augmentation
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, Optional, List
import random


class VolumePairDataset(Dataset):
    """
    Dataset for paired 3D volumes (laminography input, tomography ground truth)

    Args:
        laminography_volume: Input volume with artifacts (numpy array)
        tomography_volume: Ground truth volume (numpy array)
        patch_size: Size of patches to extract
        num_patches: Number of random patches to extract per volume pair
        normalize: Whether to normalize volumes to [0, 1]
        augment: Whether to apply data augmentation
    """

    def __init__(
        self,
        laminography_volume: np.ndarray,
        tomography_volume: np.ndarray,
        patch_size: int = 64,
        num_patches: int = 100,
        normalize: bool = True,
        augment: bool = True
    ):
        super().__init__()

        assert laminography_volume.shape == tomography_volume.shape, \
            "Laminography and tomography volumes must have the same shape"

        self.laminography = laminography_volume.astype(np.float32)
        self.tomography = tomography_volume.astype(np.float32)

        self.patch_size = patch_size
        self.num_patches = num_patches
        self.augment = augment

        # Normalize volumes
        if normalize:
            self.laminography = self._normalize(self.laminography)
            self.tomography = self._normalize(self.tomography)

        # Generate patch coordinates
        self.patch_coords = self._generate_patch_coordinates()

    def _normalize(self, volume: np.ndarray) -> np.ndarray:
        """Normalize volume to [0, 1] range"""
        vmin, vmax = volume.min(), volume.max()
        if vmax > vmin:
            return (volume - vmin) / (vmax - vmin)
        return volume

    def _generate_patch_coordinates(self) -> List[Tuple[int, int, int]]:
        """Generate random valid patch starting coordinates"""
        d, h, w = self.laminography.shape
        max_d = d - self.patch_size
        max_h = h - self.patch_size
        max_w = w - self.patch_size

        if max_d <= 0 or max_h <= 0 or max_w <= 0:
            raise ValueError(
                f"Volume shape {self.laminography.shape} is too small for patch size {self.patch_size}"
            )

        coords = []
        for _ in range(self.num_patches):
            z = random.randint(0, max_d)
            y = random.randint(0, max_h)
            x = random.randint(0, max_w)
            coords.append((z, y, x))

        return coords

    def _extract_patch(
        self, volume: np.ndarray, coord: Tuple[int, int, int]
    ) -> np.ndarray:
        """Extract a patch from volume at given coordinates"""
        z, y, x = coord
        ps = self.patch_size
        return volume[z:z+ps, y:y+ps, x:x+ps].copy()

    def _augment_patch(self, lamino_patch: np.ndarray, tomo_patch: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Apply random augmentations to patch pairs"""
        if not self.augment:
            return lamino_patch, tomo_patch

        # Random flips along each axis
        for axis in range(3):
            if random.random() > 0.5:
                lamino_patch = np.flip(lamino_patch, axis=axis).copy()
                tomo_patch = np.flip(tomo_patch, axis=axis).copy()

        # Random 90-degree rotations in xy plane
        k = random.randint(0, 3)
        if k > 0:
            lamino_patch = np.rot90(lamino_patch, k=k, axes=(1, 2)).copy()
            tomo_patch = np.rot90(tomo_patch, k=k, axes=(1, 2)).copy()

        return lamino_patch, tomo_patch

    def __len__(self) -> int:
        return self.num_patches

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            lamino_patch: Input patch with artifacts (1, D, H, W)
            tomo_patch: Ground truth patch (1, D, H, W)
        """
        coord = self.patch_coords[idx]

        # Extract patches
        lamino_patch = self._extract_patch(self.laminography, coord)
        tomo_patch = self._extract_patch(self.tomography, coord)

        # Augment
        lamino_patch, tomo_patch = self._augment_patch(lamino_patch, tomo_patch)

        # Convert to torch tensors and add channel dimension
        lamino_patch = torch.from_numpy(lamino_patch).unsqueeze(0)  # (1, D, H, W)
        tomo_patch = torch.from_numpy(tomo_patch).unsqueeze(0)      # (1, D, H, W)

        return lamino_patch, tomo_patch


class OverlappingPatchExtractor:
    """
    Extract overlapping patches from a full volume for inference

    Args:
        volume: Input volume (D, H, W)
        patch_size: Size of patches to extract
        overlap: Overlap ratio (0.0 to 1.0)
    """

    def __init__(self, volume: np.ndarray, patch_size: int = 64, overlap: float = 0.5):
        self.volume = volume
        self.patch_size = patch_size
        self.overlap = overlap

        # Calculate stride
        self.stride = int(patch_size * (1 - overlap))

        # Calculate patch grid
        self.patch_coords = self._calculate_patch_grid()

    def _calculate_patch_grid(self) -> List[Tuple[int, int, int]]:
        """Calculate systematic grid of overlapping patch coordinates"""
        d, h, w = self.volume.shape
        coords = []

        # Generate grid coordinates
        z_coords = list(range(0, d - self.patch_size + 1, self.stride))
        y_coords = list(range(0, h - self.patch_size + 1, self.stride))
        x_coords = list(range(0, w - self.patch_size + 1, self.stride))

        # Ensure we cover the entire volume
        if z_coords[-1] + self.patch_size < d:
            z_coords.append(d - self.patch_size)
        if y_coords[-1] + self.patch_size < h:
            y_coords.append(h - self.patch_size)
        if x_coords[-1] + self.patch_size < w:
            x_coords.append(w - self.patch_size)

        for z in z_coords:
            for y in y_coords:
                for x in x_coords:
                    coords.append((z, y, x))

        return coords

    def __len__(self) -> int:
        return len(self.patch_coords)

    def get_patch(self, idx: int) -> Tuple[np.ndarray, Tuple[int, int, int]]:
        """
        Get patch and its coordinates

        Returns:
            patch: Extracted patch (patch_size, patch_size, patch_size)
            coord: Starting coordinate (z, y, x)
        """
        coord = self.patch_coords[idx]
        z, y, x = coord
        ps = self.patch_size
        patch = self.volume[z:z+ps, y:y+ps, x:x+ps].copy()
        return patch, coord

    def reconstruct_volume(self, patches: List[np.ndarray], blend: str = 'gaussian') -> np.ndarray:
        """
        Reconstruct full volume from overlapping patches

        Args:
            patches: List of predicted patches
            blend: Blending method ('gaussian', 'linear', or 'average')

        Returns:
            Reconstructed volume
        """
        d, h, w = self.volume.shape
        output = np.zeros((d, h, w), dtype=np.float32)
        weights = np.zeros((d, h, w), dtype=np.float32)

        # Create blending weights
        if blend == 'gaussian':
            weight_patch = self._gaussian_weights_3d()
        elif blend == 'linear':
            weight_patch = self._linear_weights_3d()
        else:  # average
            weight_patch = np.ones((self.patch_size, self.patch_size, self.patch_size))

        # Accumulate patches
        for patch, coord in zip(patches, self.patch_coords):
            z, y, x = coord
            ps = self.patch_size

            output[z:z+ps, y:y+ps, x:x+ps] += patch * weight_patch
            weights[z:z+ps, y:y+ps, x:x+ps] += weight_patch

        # Normalize by weights
        output = np.divide(output, weights, where=weights > 0)

        return output

    def _gaussian_weights_3d(self) -> np.ndarray:
        """Create 3D Gaussian weighting for smooth blending"""
        size = self.patch_size
        center = size / 2
        sigma = size / 4

        z, y, x = np.ogrid[:size, :size, :size]
        dist_sq = (z - center)**2 + (y - center)**2 + (x - center)**2
        weights = np.exp(-dist_sq / (2 * sigma**2))

        return weights.astype(np.float32)

    def _linear_weights_3d(self) -> np.ndarray:
        """Create 3D linear weighting for blending"""
        size = self.patch_size
        center = size / 2

        z, y, x = np.ogrid[:size, :size, :size]
        dist = np.sqrt((z - center)**2 + (y - center)**2 + (x - center)**2)
        max_dist = np.sqrt(3 * (center**2))

        weights = 1 - (dist / max_dist)
        weights = np.clip(weights, 0, 1)

        return weights.astype(np.float32)


def create_dataloader(
    laminography_volume: np.ndarray,
    tomography_volume: np.ndarray,
    patch_size: int = 64,
    num_patches: int = 100,
    batch_size: int = 2,
    num_workers: int = 0,
    normalize: bool = True,
    augment: bool = True
) -> DataLoader:
    """
    Create DataLoader for training

    Args:
        laminography_volume: Input volume with artifacts
        tomography_volume: Ground truth volume
        patch_size: Size of patches
        num_patches: Number of patches to extract
        batch_size: Batch size for training
        num_workers: Number of workers for data loading
        normalize: Whether to normalize volumes
        augment: Whether to apply augmentation

    Returns:
        DataLoader instance
    """
    dataset = VolumePairDataset(
        laminography_volume,
        tomography_volume,
        patch_size=patch_size,
        num_patches=num_patches,
        normalize=normalize,
        augment=augment
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )

    return dataloader


if __name__ == "__main__":
    # Test dataset with dummy data
    print("Testing VolumePairDataset...")

    # Create dummy volumes
    lamino = np.random.randn(128, 128, 128).astype(np.float32)
    tomo = np.random.randn(128, 128, 128).astype(np.float32)

    # Create dataset
    dataset = VolumePairDataset(lamino, tomo, patch_size=64, num_patches=10)
    print(f"Dataset length: {len(dataset)}")

    # Test getting item
    lamino_patch, tomo_patch = dataset[0]
    print(f"Laminography patch shape: {lamino_patch.shape}")
    print(f"Tomography patch shape: {tomo_patch.shape}")

    # Test dataloader
    dataloader = create_dataloader(lamino, tomo, batch_size=2, num_patches=10)
    for batch_lamino, batch_tomo in dataloader:
        print(f"Batch laminography shape: {batch_lamino.shape}")
        print(f"Batch tomography shape: {batch_tomo.shape}")
        break

    # Test overlapping patch extractor
    print("\nTesting OverlappingPatchExtractor...")
    extractor = OverlappingPatchExtractor(lamino, patch_size=64, overlap=0.5)
    print(f"Number of overlapping patches: {len(extractor)}")

    patch, coord = extractor.get_patch(0)
    print(f"Patch shape: {patch.shape}, Coordinate: {coord}")
