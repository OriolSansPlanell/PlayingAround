# DVAE for Laminography Artifact Correction

A PyTorch implementation of a Decoupled Variational Autoencoder (DVAE) for correcting defocusing artifacts in laminography volumes using tomography as ground truth.

## Overview

This project trains a deep learning model to remove artifacts from laminography scans by learning from paired laminography-tomography volumes. The DVAE architecture separates content features (structure) from artifact features in the latent space, allowing reconstruction of artifact-free volumes.

## Features

- **3D Convolutional DVAE**: Fully 3D architecture that preserves spatial relationships
- **Decoupled latent space**: Separates content and artifact features
- **Patch-based training**: Memory-efficient training with overlapping patch extraction
- **Data augmentation**: Random flips and rotations for improved generalization
- **Overlapping inference**: Gaussian/linear blending for smooth reconstruction
- **Comprehensive metrics**: MSE, MAE, PSNR, SSIM for evaluation
- **Visualization tools**: Slice comparison and multi-axis visualization

## Project Structure

```
.
├── models/
│   ├── __init__.py
│   └── dvae.py                 # DVAE architecture
├── data/
│   ├── __init__.py
│   └── dataset.py              # Dataset and patch extraction
├── utils/
│   ├── __init__.py
│   └── visualization.py        # Visualization and metrics
├── train.py                    # Training script
├── inference.py                # Inference on full volumes
├── demo.py                     # Complete demonstration
├── requirements.txt            # Dependencies
└── README.md                   # This file
```

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd PlayingAround
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Quick Start

### Demo with Synthetic Data

Run a complete demonstration with synthetic volumes:

```bash
python demo.py --epochs 50 --patch-size 64 --num-patches 200
```

This will:
1. Generate synthetic laminography and tomography volumes
2. Train the DVAE model
3. Apply the model to correct artifacts
4. Save results and visualizations to `demo_output/`

### Using Real Data

If you have real volume pairs (as .npy files):

```bash
python demo.py \
    --use-real-data \
    --lamino-path /path/to/laminography.npy \
    --tomo-path /path/to/tomography.npy \
    --epochs 100 \
    --patch-size 64 \
    --num-patches 1000
```

## Training

Train on your own data:

```bash
python train.py \
    --lamino-path /path/to/laminography.npy \
    --tomo-path /path/to/tomography.npy \
    --patch-size 64 \
    --num-patches 1000 \
    --batch-size 2 \
    --epochs 100 \
    --lr 1e-4 \
    --checkpoint-dir checkpoints \
    --log-dir logs
```

### Training Parameters

- `--patch-size`: Size of 3D patches (default: 64)
  - Larger patches capture more context but require more memory
  - Must be divisible by 8 (due to downsampling)

- `--num-patches`: Number of random patches per epoch (default: 1000)
  - More patches = more training examples and better coverage

- `--batch-size`: Batch size (default: 2)
  - Adjust based on available GPU memory
  - 3D volumes are memory-intensive

- `--epochs`: Number of training epochs (default: 100)

- `--lr`: Learning rate (default: 1e-4)

- `--latent-dim`: Dimension of latent space (default: 128)

- `--base-channels`: Base number of feature channels (default: 32)
  - Affects model capacity and memory usage

### Monitor Training

View training progress with TensorBoard:

```bash
tensorboard --logdir logs
```

## Inference

Apply a trained model to a full volume:

```bash
python inference.py \
    --input /path/to/laminography.npy \
    --output /path/to/corrected.npy \
    --checkpoint checkpoints/best_model.pth \
    --patch-size 64 \
    --overlap 0.5 \
    --blend-mode gaussian
```

### Inference Parameters

- `--overlap`: Overlap ratio for patches (default: 0.5)
  - Higher overlap = smoother results but slower inference
  - Recommended: 0.25-0.5

- `--blend-mode`: Blending method for overlapping patches
  - `gaussian`: Smooth Gaussian weighting (recommended)
  - `linear`: Linear distance-based weighting
  - `average`: Simple averaging

## Model Architecture

### DVAE Components

1. **Encoder**:
   - 3D convolutional layers with progressive downsampling
   - Splits into two paths: content features and artifact features
   - Each path produces mean (μ) and log-variance (log σ²) for VAE

2. **Latent Space**:
   - Content latent: Captures volume structure
   - Artifact latent: Captures artifact patterns
   - Reparameterization trick for training

3. **Decoder**:
   - Uses ONLY content features (ignores artifacts)
   - 3D transposed convolutions for upsampling
   - Reconstructs artifact-free volume

### Loss Function

```python
Total Loss = α * Reconstruction + β * (KL_content + γ * KL_artifact)
```

- **Reconstruction Loss**: MSE between output and ground truth tomography
- **KL Divergence**: Regularizes latent space to follow standard normal distribution
- **Weights**:
  - α = 1.0 (reconstruction weight)
  - β = 0.001 (KL weight)
  - γ = 0.5 (artifact KL relative to content KL)

## Data Format

Volumes should be:
- **Format**: NumPy arrays saved as `.npy` files
- **Shape**: 3D arrays (D, H, W)
- **Data type**: float32 or float64
- **Requirements**:
  - Laminography and tomography must have the same shape
  - Volumes should be registered (aligned)
  - Tomography should be upscaled to match laminography resolution

### Preparing Your Data

```python
import numpy as np

# Load your volumes (adjust based on your file format)
laminography = load_your_laminography()  # Shape: (D, H, W)
tomography = load_your_tomography()      # Shape: (D, H, W)

# Ensure they're the same shape and registered
assert laminography.shape == tomography.shape

# Save as .npy
np.save('laminography.npy', laminography)
np.save('tomography.npy', tomography)
```

## Advanced Usage

### Custom Training Loop

```python
from models import DVAE
from data import create_dataloader
from train import Trainer
import numpy as np

# Load data
lamino = np.load('laminography.npy')
tomo = np.load('tomography.npy')

# Create dataloader
train_loader = create_dataloader(
    lamino, tomo,
    patch_size=64,
    num_patches=1000,
    batch_size=2
)

# Create model
model = DVAE(
    in_channels=1,
    base_channels=32,
    latent_dim=128,
    patch_size=64
)

# Train
trainer = Trainer(
    model=model,
    train_loader=train_loader,
    lr=1e-4,
    device='cuda'
)
trainer.train(num_epochs=100)
```

### Custom Inference

```python
from models import DVAE
from inference import VolumeInference, load_model
import numpy as np

# Load model
model = load_model(
    'checkpoints/best_model.pth',
    patch_size=64
)

# Create inference object
inference = VolumeInference(
    model=model,
    patch_size=64,
    overlap=0.5,
    blend_mode='gaussian'
)

# Process volume
lamino = np.load('laminography.npy')
corrected = inference.process_volume(lamino)
np.save('corrected.npy', corrected)
```

## Performance Tips

1. **GPU Memory**:
   - Reduce `patch_size` if running out of memory
   - Reduce `batch_size`
   - Use gradient accumulation for larger effective batch sizes

2. **Training Speed**:
   - Use larger `batch_size` if memory allows
   - Increase `num_workers` in dataloader (for multi-core CPUs)
   - Use mixed precision training (requires modifications)

3. **Model Quality**:
   - Train for more epochs
   - Increase `num_patches` for better data coverage
   - Use larger `patch_size` for more context
   - Increase `base_channels` or `latent_dim` for more capacity

4. **Inference Quality**:
   - Use higher `overlap` (0.5-0.75) for smoother results
   - Use `gaussian` blending mode
   - Ensure input is normalized consistently with training

## Evaluation Metrics

The project computes several metrics:

- **MSE** (Mean Squared Error): Lower is better
- **MAE** (Mean Absolute Error): Lower is better
- **PSNR** (Peak Signal-to-Noise Ratio): Higher is better (dB)
- **SSIM** (Structural Similarity): Higher is better (0-1)

## Troubleshooting

### Common Issues

1. **Out of memory errors**:
   - Reduce `patch_size` (e.g., 64 → 48 → 32)
   - Reduce `batch_size` (e.g., 2 → 1)
   - Use CPU instead of GPU (slower but more memory)

2. **Poor results**:
   - Train for more epochs
   - Increase model capacity (`base_channels`, `latent_dim`)
   - Check that volumes are properly registered
   - Verify data normalization

3. **Training instability**:
   - Reduce learning rate
   - Adjust KL weight in loss function
   - Check for NaN values in input data

## Citation

If you use this code in your research, please cite:

```bibtex
@software{dvae_laminography,
  title={DVAE for Laminography Artifact Correction},
  author={Your Name},
  year={2024},
  url={https://github.com/yourusername/yourrepo}
}
```

## License

See LICENSE file for details.

## Acknowledgments

This implementation is based on variational autoencoder and image-to-image translation techniques adapted for 3D medical imaging.
