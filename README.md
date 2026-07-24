# TideGAN

Conditional GAN for generating tide-aware coastal satellite images.

## Overview

TideGAN takes a **reference image** (typically at low tide) and a **condition vector** encoding the target tide level and site, and generates a synthetic satellite image of that coastal scene at the requested tide level.

### Architecture

| Component | Architecture | Parameters |
|-----------|-------------|------------|
| **Generator** | U-Net with AdaIN condition injection | ~22M |
| **Discriminator** | PatchGAN (FCN) | ~2.8M |
| **Condition** | Tide level (normalized) + site one-hot | 4 values |

- **Generator**: 4-level U-Net encoder-decoder with skip connections. Condition is injected via AdaIN (Adaptive Instance Normalization) at every layer, allowing the network to modulate its features based on tide level and site.
- **Discriminator**: PatchGAN that outputs per-patch realism scores (70×70 receptive field), encouraging local texture coherence.
- **Loss**: Hinge GAN loss + gradient penalty (WGAN-GP style) + L1 reconstruction loss (weight=100) for content consistency.

## Data

Three coastal sites with RGB satellite imagery and tide measurements:

| Site | Images | Tide Range (m) |
|------|--------|----------------|
| **Foz** | 68 | [-1.90, 1.19] |
| **Santander** | 38 | [-2.18, 0.91] |
| **Villaviciosa** | 68 | [-2.17, 1.01] |

Image size: ~489×578 pixels. The dataset extracts 256×256 random patches with augmentation (horizontal flip, color jitter).

## Installation

```bash
pip install -r requirements.txt
```

Or individually:
```bash
pip install torch torchvision numpy Pillow tqdm tensorboard
```

## Usage

### Training

```bash
# Train on all sites
python train.py --epochs 200 --batch_size 8 --patch_size 256

# Train on a single site
python train.py --epochs 200 --batch_size 8 --site Foz

# Resume from checkpoint
python train.py --resume checkpoints/tidegan/checkpoint_epoch_50.pth
```

### CLI Options

| Option | Default | Description |
|--------|---------|-------------|
| `--epochs` | 100 | Number of training epochs |
| `--batch_size` | 8 | Training batch size |
| `--lr_g` | 2e-4 | Generator learning rate |
| `--lr_d` | 2e-4 | Discriminator learning rate |
| `--patch_size` | 256 | Patch size for training |
| `--site` | all | Sites to train on (e.g., `Foz Santander`) |
| `--save_dir` | `checkpoints/tidegan` | Checkpoint output directory |
| `--resume` | None | Path to checkpoint to resume from |
| `--num_workers` | 4 | DataLoader workers |

### Generation

```bash
python train.py --mode generate --checkpoint checkpoints/tidegan/checkpoint_epoch_200.pth --site Foz --output_dir outputs
```

This generates 10 images spanning the full tide range for the given site, using the lowest-tide image as reference.

## Project Structure

```
IntertidelGEN/
├── data/
│   ├── Foz/
│   │   ├── dataset_foz.csv          # Metadata + tide levels
│   │   ├── rgb_png/                  # RGB images
│   │   └── scl_png/                  # Classification masks (unused for now)
│   ├── Santander/
│   └── Villaviciosa/
├── tidegan_dataset.py               # Dataset + dataloader
├── tidegan_model.py                 # Generator + Discriminator + loss
├── train.py                         # Training loop + CLI
├── requirements.txt
└── README.md
```

## Design Decisions

1. **AdaIN for condition injection**: The tide level is embedded into a vector and used to scale/shift feature maps at every layer. This allows fine-grained control over the generated output.

2. **PatchGAN discriminator**: Instead of a single real/fake score for the whole image, PatchGAN evaluates overlapping patches, producing higher-quality textures.

3. **Gradient penalty (WGAN-GP)**: Ensures Lipschitz continuity of the discriminator, preventing mode collapse and training instability.

4. **L1 loss**: Encourages pixel-level consistency between generated and target images, important for preserving geographic features.

5. **Per-site normalization**: Tide values are normalized per-site using min-max scaling to [-1, 1], accounting for different tidal regimes.

6. **Patch-based training**: 256×256 patches provide implicit data augmentation through random cropping and horizontal flipping.

## Next Steps

- Integrate SCL classification masks as additional condition/input
- Add multi-scale discriminator for better global coherence
- Implement cycle-consistency loss for unsupervised domain adaptation
- Add attention mechanisms for long-range dependency modeling
- Train on all sites simultaneously with site-specific normalization
