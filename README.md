# TideGAN

Conditional image-to-image translation for tide-aware coastal satellite imagery.

## Overview

TideGAN takes a **reference image** (typically at low tide) and a **condition vector** encoding the target tide level and site, and generates a synthetic satellite image of that coastal scene at the requested tide level.

### Architecture

| Component | Architecture | Parameters |
|-----------|-------------|------------|
| **Generator** | U-Net with AdaIN condition injection + ResNet bottleneck | ~12M |
| **Condition** | Tide level (normalized to [-1, 1]) + site one-hot (3 sites) | 4 values |

- **Generator**: 4-level U-Net encoder-decoder (64→128→256→512 channels) with skip connections. A 3-block ResNet bottleneck sits between encoder and decoder. Condition is injected via AdaIN (Adaptive Instance Normalization) at every normalization layer, allowing fine-grained modulation based on tide level and site.
- **Loss**: L1 reconstruction loss (weight=5.0) for pixel-level content consistency.

## Data

Three coastal sites with RGB satellite imagery and tide measurements:

| Site | Images | Tide Range (m) |
|------|--------|----------------|
| **Foz** | 68 | [-1.90, 1.19] |
| **Santander** | 38 | [-2.18, 0.91] |
| **Villaviciosa** | 68 | [-2.17, 1.01] |

Image size: ~489×578 pixels. The dataset extracts 256×256 random patches with augmentation (horizontal flip, color jitter).

### Smart Patch Extraction

The dataset loader uses SCL (Scene Classification Layer) masks to avoid:
- Clouds and cloud shadows
- Nodata / saturated pixels
- Patches with no coastal elements (water + land)

This ensures training data contains meaningful coastal transitions.

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
python train.py --epochs 150 --batch_size 8 --patch_size 256

# Train on a single site
python train.py --epochs 150 --batch_size 8 --site Foz

# Train on specific GPUs
python train.py --epochs 150 --batch_size 8 --devices 0 1

# Resume from checkpoint
python train.py --resume checkpoints/tidegan/checkpoint_epoch_50.pth
```

### Generation

```bash
python train.py --mode generate --checkpoint checkpoints/tidegan/checkpoint_epoch_150.pth --site Foz --output_dir outputs
```

This generates 10 images spanning the full tide range for the given site, using the lowest-tide image as reference.

### CLI Options

| Option | Default | Description |
|--------|---------|-------------|
| `--mode` | `train` | Mode: `train` or `generate` |
| `--epochs` | 150 | Number of training epochs |
| `--batch_size` | 8 | Training batch size |
| `--lr_g` | 1e-4 | Generator learning rate |
| `--patch_size` | 256 | Patch size for training (must be divisible by 16) |
| `--site` | all | Sites to train on (e.g., `Foz Santander Villaviciosa`) |
| `--save_dir` | `checkpoints/tidegan` | Checkpoint output directory |
| `--resume` | None | Path to checkpoint to resume from |
| `--devices` | auto | CUDA device ids to use (e.g., `--devices 0 1`) |
| `--seed` | 42 | Random seed for reproducibility |
| `--num_workers` | 4 | DataLoader workers |
| `--no_progress_bar` | False | Hide tqdm progress bar |

**Generation mode options:**

| Option | Default | Description |
|--------|---------|-------------|
| `--checkpoint` | — | Path to generator checkpoint |
| `--site` | — | Single site to generate images for |
| `--output_dir` | `outputs` | Output directory for generated images |

## Project Structure

```
IntertidalGAN/
├── data/
│   ├── Foz/
│   │   ├── dataset_foz.csv          # Metadata + tide levels
│   │   ├── rgb_png/                  # RGB satellite images
│   │   └── scl_png/                  # SCL classification masks
│   ├── Santander/
│   └── Villaviciosa/
├── tidegan_dataset.py               # Dataset + dataloader + SCL filtering
├── tidegan_model.py                 # Generator (U-Net with AdaIN)
├── train.py                         # Training loop + CLI + generation
├── requirements.txt
└── README.md
```

## Design Decisions

1. **AdaIN for condition injection**: The tide level and site embedding are projected into a vector and used to scale/shift feature maps at every AdaIN layer. This allows fine-grained control over the generated output.

2. **U-Net with ResNet bottleneck**: The 4-level U-Net preserves spatial detail via skip connections, while the ResNet bottleneck blocks enable deeper feature learning at the lowest resolution.

3. **Supervised L1 training**: Instead of adversarial training, the model uses L1 reconstruction loss for stable, reproducible training. This prioritizes content fidelity over photorealism.

4. **Smart patch extraction with SCL masks**: Patches are filtered to exclude clouds, nodata, and non-coastal regions, ensuring high-quality training pairs.

5. **Per-site tide normalization**: Tide values are normalized to [-1, 1] using each site's min-max range, accounting for different tidal regimes.

6. **In-memory loading**: All images and SCL masks are loaded into RAM at initialization for fast training.

7. **Image-level train/val split**: To avoid leakage, images are split deterministically across the tide range rather than randomly, ensuring both splits cover the full tide spectrum.

8. **Deterministic training**: Seeds are set for Python, NumPy, and PyTorch (with `cudnn.deterministic=True`), making runs reproducible. Checkpoints store all RNG states for exact resume.

9. **Multi-GPU support**: DataParallel enables training across multiple GPUs via the `--devices` flag.

## Next Steps

- Add attention mechanisms for long-range dependency modeling
- Implement cycle-consistency loss for unsupervised domain adaptation
- Train on all sites simultaneously with site-specific normalization
