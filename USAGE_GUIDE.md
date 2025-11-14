# Multi-Pair Training & Quality Improvement Guide

This guide shows you how to train on multiple volume pairs and improve reconstruction quality to reduce graininess.

## Table of Contents
1. [Training on Multiple Volume Pairs](#training-on-multiple-volume-pairs)
2. [Improving Reconstruction Quality](#improving-reconstruction-quality)
3. [Troubleshooting Grainy Results](#troubleshooting-grainy-results)
4. [Advanced Tips](#advanced-tips)

---

## Training on Multiple Volume Pairs

### Option 1: Organized Directory Structure

Organize your files like this:
```
data/
├── pair1_lamino.npy
├── pair1_tomo.npy
├── pair2_lamino.npy
├── pair2_tomo.npy
├── pair3_lamino.npy
└── pair3_tomo.npy
```

Then train with:
```bash
python train_improved.py \
    --data-dir data/ \
    --patch-size 64 \
    --patches-per-pair 200 \
    --batch-size 4 \
    --epochs 150 \
    --lr 5e-5
```

The script will automatically discover and pair files with "lamino" and "tomo" in their names.

### Option 2: Explicit Pair List

Specify pairs explicitly:
```bash
python train_improved.py \
    --pair-list \
        "/path/to/lamino1.npy,/path/to/tomo1.npy" \
        "/path/to/lamino2.npy,/path/to/tomo2.npy" \
        "/path/to/lamino3.npy,/path/to/tomo3.npy" \
    --patch-size 64 \
    --patches-per-pair 200 \
    --batch-size 4 \
    --epochs 150
```

### Option 3: Python Script

```python
from data import create_multi_pair_dataloader
from models import DVAE
from train_improved import ImprovedTrainer
import torch

# Define your volume pairs
pairs = [
    ('lamino1.npy', 'tomo1.npy'),
    ('lamino2.npy', 'tomo2.npy'),
    ('lamino3.npy', 'tomo3.npy'),
    # Add more pairs...
]

# Create dataloader
train_loader = create_multi_pair_dataloader(
    volume_pairs=pairs,
    patch_size=64,
    patches_per_pair=200,  # per pair per epoch
    batch_size=4,
    normalize=True,
    augment=True
)

# Total patches per epoch = num_pairs * patches_per_pair
# Example: 3 pairs * 200 patches = 600 patches/epoch

# Create model
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = DVAE(
    in_channels=1,
    base_channels=32,
    latent_dim=128,
    patch_size=64
)

# Create improved trainer
trainer = ImprovedTrainer(
    model=model,
    train_loader=train_loader,
    lr=5e-5,
    device=device,
    checkpoint_dir='checkpoints_multi',
    log_dir='logs_multi',
    use_improved_loss=True  # Important for quality!
)

# Train
trainer.train(num_epochs=150, save_every=15)
```

---

## Improving Reconstruction Quality

The improved training script includes several enhancements to reduce graininess:

### 1. **Improved Loss Function** (Default)

The `ImprovedDVAELoss` includes:

- **MSE + L1 Loss**: Combination reduces extreme errors
  - MSE: Penalizes large errors
  - L1: More robust to outliers

- **Gradient Loss** (NEW): Enforces smoothness by matching gradients
  - Reduces high-frequency noise and graininess
  - Weight: 0.5 by default

- **SSIM Loss** (NEW): Preserves structural similarity
  - Maintains texture and structure
  - Weight: 0.2 by default

- **Reduced KL Weight**: 0.0001 instead of 0.001
  - Less regularization = better reconstruction
  - Still maintains reasonable latent space

### 2. **Better Optimization**

- **AdamW optimizer**: Weight decay prevents overfitting
- **Lower learning rate**: 5e-5 instead of 1e-4 for stability
- **Gradient clipping**: Prevents training instability
- **Learning rate scheduler**: Automatically reduces LR when loss plateaus

### 3. **More Training Data**

With multiple pairs, you get:
- More diverse examples
- Better generalization
- Reduced overfitting
- Smoother reconstructions

---

## Troubleshooting Grainy Results

### Problem: Results are still grainy

#### Solution 1: Increase Gradient Loss Weight
```python
# In train_improved.py, modify ImprovedDVAELoss:
criterion = ImprovedDVAELoss(
    gradient_weight=1.0,  # Increase from 0.5
    ssim_weight=0.3,      # Increase from 0.2
)
```

#### Solution 2: Train Longer
```bash
python train_improved.py \
    --data-dir data/ \
    --epochs 200  # or more
```

#### Solution 3: Increase Model Capacity
```bash
python train_improved.py \
    --data-dir data/ \
    --base-channels 48  # or 64 instead of 32
    --latent-dim 256    # instead of 128
```

#### Solution 4: Use Larger Patches
```bash
python train_improved.py \
    --data-dir data/ \
    --patch-size 96  # or 128 instead of 64
```
Note: Requires more GPU memory

#### Solution 5: Adjust Inference Overlap
```python
# Higher overlap = smoother results
inference = VolumeInference(
    model=model,
    patch_size=64,
    overlap=0.75,  # Increase from 0.5
    blend_mode='gaussian'
)
```

### Problem: Training is unstable (loss jumps around)

#### Solution: Lower learning rate
```bash
python train_improved.py \
    --lr 1e-5  # instead of 5e-5
```

### Problem: Output is too blurry

#### Solution: Reduce gradient/SSIM weights
```python
criterion = ImprovedDVAELoss(
    gradient_weight=0.2,  # Reduce from 0.5
    ssim_weight=0.1,      # Reduce from 0.2
)
```

### Problem: Not enough GPU memory

#### Solution: Reduce batch size or patch size
```bash
python train_improved.py \
    --batch-size 1  # instead of 2-4
    --patch-size 48  # instead of 64
```

---

## Advanced Tips

### 1. **Progressive Training**

Start with standard loss, then fine-tune with improved loss:

```bash
# Stage 1: Train with standard loss
python train_improved.py \
    --data-dir data/ \
    --epochs 50 \
    --standard-loss

# Stage 2: Fine-tune with improved loss
python train_improved.py \
    --data-dir data/ \
    --epochs 100 \
    --checkpoint-dir checkpoints_improved
```

Then load the checkpoint from stage 1 and continue with improved loss.

### 2. **Custom Loss Weights**

Edit `train_improved.py` to customize:

```python
# For very noisy data, increase smoothness:
criterion = ImprovedDVAELoss(
    recon_weight=1.0,
    gradient_weight=1.5,    # Higher smoothness
    ssim_weight=0.5,        # Higher structure preservation
    kl_weight=0.00005,      # Lower regularization
)

# For preserving fine details:
criterion = ImprovedDVAELoss(
    recon_weight=1.0,
    gradient_weight=0.1,    # Lower smoothness
    ssim_weight=0.1,        # Lower structure constraint
    l1_weight=0.8,          # Higher L1 vs L2
)
```

### 3. **Monitor Training with TensorBoard**

```bash
tensorboard --logdir logs_improved
```

Watch these metrics:
- **reconstruction**: Should decrease steadily
- **gradient**: Should decrease (smoother outputs)
- **ssim**: Should decrease (better structure)
- **kl_total**: Should stay reasonable (not too high)

### 4. **Validation Split**

For better hyperparameter tuning, hold out one pair for validation:

```python
# Training pairs
train_pairs = [
    ('lamino1.npy', 'tomo1.npy'),
    ('lamino2.npy', 'tomo2.npy'),
]

# Validation pair
val_pairs = [
    ('lamino3.npy', 'tomo3.npy'),
]

train_loader = create_multi_pair_dataloader(train_pairs, ...)
val_loader = create_multi_pair_dataloader(val_pairs, patches_per_pair=50, ...)

trainer = ImprovedTrainer(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,  # Add validation
    ...
)
```

### 5. **Data Quality Matters**

Ensure your data is:
- **Well-registered**: Misalignment causes artifacts
- **Same resolution**: Tomography properly upscaled
- **Clean**: Pre-process to remove extreme outliers
- **Normalized consistently**: All pairs in similar intensity ranges

### 6. **Recommended Settings for Best Quality**

Based on experimentation, these settings work well:

```bash
python train_improved.py \
    --data-dir data/ \
    --patch-size 64 \
    --patches-per-pair 300 \
    --batch-size 4 \
    --epochs 150 \
    --lr 5e-5 \
    --base-channels 48 \
    --latent-dim 128
```

Then for inference:
```python
inference = VolumeInference(
    model=model,
    patch_size=64,
    overlap=0.6,
    blend_mode='gaussian'
)
```

---

## Comparison: Standard vs Improved Training

| Feature | Standard (`train.py`) | Improved (`train_improved.py`) |
|---------|----------------------|-------------------------------|
| Loss function | MSE + KL | MSE + L1 + Gradient + SSIM + KL |
| Supports multi-pair | ❌ | ✅ |
| Gradient loss | ❌ | ✅ |
| SSIM loss | ❌ | ✅ |
| Optimizer | Adam | AdamW (with weight decay) |
| Learning rate | 1e-4 | 5e-5 (lower, more stable) |
| LR scheduler | ❌ | ✅ (ReduceLROnPlateau) |
| Gradient clipping | ❌ | ✅ |
| KL weight | 0.001 | 0.0001 (lower) |
| Best for | Quick testing | Production quality |

---

## Quick Reference

### For Best Quality:
```bash
python train_improved.py --data-dir /path/to/data --epochs 150 --base-channels 48
```

### For Faster Iteration:
```bash
python train_improved.py --data-dir /path/to/data --epochs 50 --patch-size 48 --batch-size 2
```

### For Maximum Smoothness:
Modify loss in `train_improved.py`:
```python
criterion = ImprovedDVAELoss(gradient_weight=1.5, ssim_weight=0.5)
```

### For Maximum Detail:
Modify loss in `train_improved.py`:
```python
criterion = ImprovedDVAELoss(gradient_weight=0.1, l1_weight=0.9)
```

---

## Questions?

- Check `README.md` for basic usage
- See `demo.py` for single-pair examples
- Use `example_simple.py` for minimal examples
- Monitor training with TensorBoard: `tensorboard --logdir logs_improved`
