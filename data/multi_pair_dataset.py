"""
Multi-pair dataset loader for training on multiple volume pairs

This module handles loading and combining multiple laminography-tomography pairs
for improved training with more diverse data.
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple, Optional
import random
from pathlib import Path


class MultiPairVolumePatchDataset(Dataset):
    """
    Dataset for multiple paired 3D volumes

    Args:
        volume_pairs: List of (laminography_path, tomography_path) tuples or (lamino_array, tomo_array)
        patch_size: Size of patches to extract
        patches_per_pair: Number of patches to extract from each pair per epoch
        normalize: Whether to normalize volumes to [0, 1]
        augment: Whether to apply data augmentation
    """

    def __init__(
        self,
        volume_pairs: List[Tuple],
        patch_size: int = 64,
        patches_per_pair: int = 100,
        normalize: bool = True,
        augment: bool = True
    ):
        super().__init__()

        self.patch_size = patch_size
        self.patches_per_pair = patches_per_pair
        self.augment = augment
        self.normalize = normalize

        # Load all volume pairs
        print(f"Loading {len(volume_pairs)} volume pairs...")
        self.volume_pairs = []

        for idx, pair in enumerate(volume_pairs):
            if isinstance(pair[0], str):
                # Load from file paths
                lamino = np.load(pair[0]).astype(np.float32)
                tomo = np.load(pair[1]).astype(np.float32)
                print(f"  Pair {idx+1}: Loaded from files, shape {lamino.shape}")
            else:
                # Use arrays directly
                lamino = pair[0].astype(np.float32)
                tomo = pair[1].astype(np.float32)
                print(f"  Pair {idx+1}: Using provided arrays, shape {lamino.shape}")

            assert lamino.shape == tomo.shape, \
                f"Pair {idx}: Shape mismatch {lamino.shape} vs {tomo.shape}"

            # Normalize if requested
            if normalize:
                lamino = self._normalize(lamino)
                tomo = self._normalize(tomo)

            self.volume_pairs.append((lamino, tomo))

        # Generate patch coordinates for all pairs
        self.patch_list = self._generate_all_patches()

        print(f"Total patches available: {len(self.patch_list)}")

    def _normalize(self, volume: np.ndarray) -> np.ndarray:
        """Normalize volume to [0, 1] range"""
        vmin, vmax = volume.min(), volume.max()
        if vmax > vmin:
            normalized = (volume - vmin) / (vmax - vmin)
        else:
            normalized = volume
        return normalized.astype(np.float32)

    def _generate_all_patches(self) -> List[Tuple[int, Tuple[int, int, int]]]:
        """
        Generate patch coordinates for all volume pairs

        Returns:
            List of (pair_index, (z, y, x)) tuples
        """
        patch_list = []

        for pair_idx, (lamino, tomo) in enumerate(self.volume_pairs):
            d, h, w = lamino.shape
            max_d = d - self.patch_size
            max_h = h - self.patch_size
            max_w = w - self.patch_size

            if max_d <= 0 or max_h <= 0 or max_w <= 0:
                print(f"Warning: Pair {pair_idx} shape {lamino.shape} too small for patch size {self.patch_size}, skipping")
                continue

            # Generate random patches for this pair
            for _ in range(self.patches_per_pair):
                z = random.randint(0, max_d)
                y = random.randint(0, max_h)
                x = random.randint(0, max_w)
                patch_list.append((pair_idx, (z, y, x)))

        return patch_list

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
        return len(self.patch_list)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            lamino_patch: Input patch with artifacts (1, D, H, W)
            tomo_patch: Ground truth patch (1, D, H, W)
        """
        pair_idx, coord = self.patch_list[idx]
        lamino_volume, tomo_volume = self.volume_pairs[pair_idx]

        # Extract patches
        lamino_patch = self._extract_patch(lamino_volume, coord)
        tomo_patch = self._extract_patch(tomo_volume, coord)

        # Augment
        lamino_patch, tomo_patch = self._augment_patch(lamino_patch, tomo_patch)

        # Convert to torch tensors and add channel dimension
        lamino_patch = torch.from_numpy(lamino_patch).unsqueeze(0)  # (1, D, H, W)
        tomo_patch = torch.from_numpy(tomo_patch).unsqueeze(0)      # (1, D, H, W)

        return lamino_patch, tomo_patch

    def reshuffle_patches(self):
        """Regenerate random patch coordinates for new epoch diversity"""
        self.patch_list = self._generate_all_patches()


def create_multi_pair_dataloader(
    volume_pairs: List[Tuple],
    patch_size: int = 64,
    patches_per_pair: int = 100,
    batch_size: int = 2,
    num_workers: int = 0,
    normalize: bool = True,
    augment: bool = True
) -> DataLoader:
    """
    Create DataLoader for training on multiple volume pairs

    Args:
        volume_pairs: List of (lamino_path, tomo_path) or (lamino_array, tomo_array)
        patch_size: Size of patches
        patches_per_pair: Number of patches per pair per epoch
        batch_size: Batch size for training
        num_workers: Number of workers for data loading
        normalize: Whether to normalize volumes
        augment: Whether to apply augmentation

    Returns:
        DataLoader instance

    Example:
        >>> pairs = [
        ...     ('lamino1.npy', 'tomo1.npy'),
        ...     ('lamino2.npy', 'tomo2.npy'),
        ...     ('lamino3.npy', 'tomo3.npy'),
        ... ]
        >>> dataloader = create_multi_pair_dataloader(
        ...     pairs, patch_size=64, patches_per_pair=200, batch_size=4
        ... )
    """
    dataset = MultiPairVolumePatchDataset(
        volume_pairs,
        patch_size=patch_size,
        patches_per_pair=patches_per_pair,
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


def load_volume_pairs_from_directory(
    directory: str,
    lamino_pattern: str = "*lamino*.npy",
    tomo_pattern: str = "*tomo*.npy"
) -> List[Tuple[str, str]]:
    """
    Auto-discover volume pairs in a directory based on naming patterns

    Args:
        directory: Directory containing volume pairs
        lamino_pattern: Glob pattern for laminography files
        tomo_pattern: Glob pattern for tomography files

    Returns:
        List of (lamino_path, tomo_path) tuples

    Example directory structure:
        data/
        ├── pair1_lamino.npy
        ├── pair1_tomo.npy
        ├── pair2_lamino.npy
        ├── pair2_tomo.npy
        └── ...
    """
    from pathlib import Path

    dir_path = Path(directory)
    lamino_files = sorted(dir_path.glob(lamino_pattern))
    tomo_files = sorted(dir_path.glob(tomo_pattern))

    if len(lamino_files) != len(tomo_files):
        print(f"Warning: Found {len(lamino_files)} lamino files but {len(tomo_files)} tomo files")

    pairs = []
    for lamino_file in lamino_files:
        # Try to find matching tomo file by replacing pattern
        tomo_file = None

        # Simple matching: replace 'lamino' with 'tomo' in filename
        potential_tomo = lamino_file.parent / lamino_file.name.replace('lamino', 'tomo')
        if potential_tomo.exists():
            tomo_file = potential_tomo
        else:
            # Try to find by common prefix
            prefix = lamino_file.stem.replace('_lamino', '').replace('lamino', '')
            for tf in tomo_files:
                if prefix in tf.stem:
                    tomo_file = tf
                    break

        if tomo_file:
            pairs.append((str(lamino_file), str(tomo_file)))
            print(f"Paired: {lamino_file.name} <-> {tomo_file.name}")
        else:
            print(f"Warning: No matching tomo file found for {lamino_file.name}")

    return pairs


if __name__ == "__main__":
    # Test with synthetic data
    print("Testing MultiPairVolumePatchDataset...")

    # Create dummy volume pairs
    pairs = []
    for i in range(3):
        lamino = np.random.randn(100, 100, 100).astype(np.float32)
        tomo = np.random.randn(100, 100, 100).astype(np.float32)
        pairs.append((lamino, tomo))

    # Create dataset
    dataset = MultiPairVolumePatchDataset(
        pairs,
        patch_size=64,
        patches_per_pair=50
    )
    print(f"Dataset length: {len(dataset)}")

    # Test getting item
    lamino_patch, tomo_patch = dataset[0]
    print(f"Laminography patch shape: {lamino_patch.shape}")
    print(f"Tomography patch shape: {tomo_patch.shape}")

    # Test dataloader
    dataloader = create_multi_pair_dataloader(
        pairs,
        patch_size=64,
        patches_per_pair=50,
        batch_size=4
    )

    for batch_lamino, batch_tomo in dataloader:
        print(f"Batch laminography shape: {batch_lamino.shape}")
        print(f"Batch tomography shape: {batch_tomo.shape}")
        break

    print("\nTest complete!")
