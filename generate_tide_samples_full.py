"""
Generate tide-conditioned full-image samples from a trained TideGAN checkpoint.

Unlike generate_tide_samples.py, this version processes the entire image at once
(no patch sliding). The U-Net with stride-2 conv + padding=1 handles arbitrary
sizes (min ~16x16 due to 4 down-sampling stages).

Usage:
    python generate_tide_samples_full.py --site Foz -c checkpoints/tidegan/checkpoint_epoch_1000.pth
    python generate_tide_samples_full.py --site Santander -c checkpoint.pth -o output_dir/
"""

import argparse
import os
import numpy as np
import torch
from PIL import Image
from matplotlib.gridspec import GridSpec
import matplotlib.pyplot as plt

from tidegan_dataset import TideGANDataset, SITES, SITE_TIDE_RANGES, _normalize_tide
from tidegan_model import Generator


STEP_SIZE = 0.05  # 5cm tide steps


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate full-image tide-conditioned samples (no patches).",
    )
    parser.add_argument(
        "--site", "-s",
        type=str,
        required=True,
        choices=SITES,
        help=f"Site to generate images for: {SITES}",
    )
    parser.add_argument(
        "--checkpoint", "-c",
        type=str,
        required=True,
        help="Path to the trained checkpoint (.pth).",
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Directory to save output. Default: {site}_tide_images. Use --display-only to show without saving.",
    )
    parser.add_argument(
        "--display-only",
        action="store_true",
        help="Only display the plot without saving.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use (e.g., cpu, cuda, cuda:0). Auto-detected if not provided.",
    )
    return parser.parse_args()


def load_model(checkpoint_path, device, default_n_sites=3, default_cond_dim=32):
    """Load the generator model from a checkpoint, auto-detecting config."""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    config = checkpoint.get("config", {})
    n_sites = config.get("n_sites", default_n_sites)
    cond_dim = config.get("cond_dim", default_cond_dim)

    G = Generator(n_sites=n_sites, cond_dim=cond_dim).to(device)
    G_state = checkpoint["generator_state"]

    # Handle DataParallel wrapper
    if list(G_state.keys())[0].startswith("module."):
        from collections import OrderedDict
        new_state = OrderedDict()
        for k, v in G_state.items():
            new_state[k[7:]] = v
        G.load_state_dict(new_state)
    else:
        G.load_state_dict(G_state)

    G.eval()
    n_params = sum(p.numel() for p in G.parameters() if p.requires_grad)
    print(f"  ✓ Generator loaded ({n_params:,} params, n_sites={n_sites}, cond_dim={cond_dim})")
    return G, n_sites, cond_dim


def build_condition(tide_norm, site_idx, n_sites):
    """Build condition tensor: [normalized_tide, site_one_hot]."""
    cond = np.zeros(1 + n_sites, dtype=np.float32)
    cond[0] = tide_norm
    cond[1 + site_idx] = 1.0
    return torch.from_numpy(cond).unsqueeze(0)  # (1, 1 + n_sites)


def generate_full_image(G, ref_tensor, cond_tensor, device):
    """
    Generate for the entire image at once, with padding to ensure
    dimensions are divisible by 16 (required by the 4-stage U-Net).
    ref_tensor: (3, H, W) in [-1, 1]
    cond_tensor: (1, 1 + n_sites)
    Returns: numpy array (H, W, 3) in [-1, 1]
    """
    H, W = ref_tensor.shape[1], ref_tensor.shape[2]

    # Pad to nearest multiple of 16 so encoder/decoder dims match for skip connections
    factor = 16  # 4 down-sampling stages
    pad_h = (factor - (H % factor)) % factor
    pad_w = (factor - (W % factor)) % factor

    if pad_h > 0 or pad_w > 0:
        ref_padded = torch.nn.functional.pad(
            ref_tensor, (0, pad_w, 0, pad_h), mode='reflect'
        )
    else:
        ref_padded = ref_tensor

    with torch.no_grad():
        gen_padded = G(ref_padded.unsqueeze(0), cond_tensor)  # (1, 3, H_p, W_p)

    # Crop back to original size
    gen = gen_padded.squeeze(0)[:, :H, :W]
    return gen.permute(1, 2, 0).cpu().numpy()  # (H, W, 3)


def main():
    args = parse_args()

    # ── Device ────────────────────────────────────────────────────────────
    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")
    print(f"Device: {device}\n")

    # ── Load model ────────────────────────────────────────────────────────
    print(f"Checkpoint: {args.checkpoint}")
    G, n_sites, cond_dim = load_model(args.checkpoint, device)

    # ── Site info ─────────────────────────────────────────────────────────
    site = args.site
    out_dir = args.output if args.output else f"{site}_tide_images"
    site_idx = SITES.index(site)
    t_min, t_max = SITE_TIDE_RANGES[site]
    print(f"Site: {site} (tide range: [{t_min:.3f}, {t_max:.3f}] m)")

    # ── Load full reference from dataset ──────────────────────────────────
    print("\nLoading reference image from dataset...")
    dataset = TideGANDataset(
        sites=[site],
        patch_size=256,  # just for loading, not used for generation
        augment=False,
        min_tide_diff=0.0,
        split=None,
    )

    ref_entry = dataset.ref_image_by_site[site]
    ref_img = ref_entry.rgb_data.copy()  # (H, W, 3) in [0, 255]
    H, W = ref_img.shape[:2]
    print(f"  ✓ Full reference: {H}x{W}px  ({ref_entry.date}, tide: {ref_entry.tide:.3f}m)")

    # Normalize full reference to [-1, 1] for model input
    ref_tensor = (ref_img.astype(np.float32) / 127.5 - 1.0)  # (H, W, 3)
    ref_tensor = torch.from_numpy(ref_tensor).permute(2, 0, 1).to(device)  # (3, H, W)
    ref_pil = Image.fromarray(ref_img.astype(np.uint8))

    # ── Figure size matching reference aspect ratio ─────────────────────────
    # Keep height fixed at 4in, scale width to match reference aspect ratio
    ref_fig_h = 4.0
    ref_fig_w = ref_fig_h * W / H
    # Cap width at 32in for usability
    ref_fig_w = min(ref_fig_w, 32.0)

    # ── Tide levels (every 5cm) ───────────────────────────────────────────
    tides = np.arange(t_min, t_max + STEP_SIZE / 2, STEP_SIZE)
    tides = tides[tides <= t_max]
    print(f"\nGenerating full images for {len(tides)} tide levels (every {STEP_SIZE*100:.0f}cm)...\n")

    # ── Generate one tide level at a time ─────────────────────────────────
    generated_full = []

    for i, tide_raw in enumerate(tides, 1):
        tide_norm = _normalize_tide(tide_raw, site)
        cond_tensor = build_condition(tide_norm, site_idx, n_sites).to(device)

        print(f"[{i}/{len(tides)}] Tide: {tide_raw:+.3f}m ... ", end="", flush=True)
        full_image = generate_full_image(G, ref_tensor, cond_tensor, device)
        generated_full.append((full_image, tide_raw))
        print("✓")

    # ── Plot ──────────────────────────────────────────────────────────────
    print("\nPlotting...")
    cols = max(8, len(generated_full))
    rows = (len(generated_full) + cols - 1) // cols

    fig = plt.figure(figsize=(ref_fig_w * cols / max(cols, 8) * 2, ref_fig_h * (rows + 1)))
    gs = GridSpec(rows + 1, cols, figure=fig, hspace=0.25, wspace=0.08)

    # Reference image
    ax_ref = fig.add_subplot(gs[0, 0])
    ax_ref.imshow(ref_pil)
    ax_ref.set_title("Reference", fontsize=12, fontweight="bold")
    ax_ref.axis("off")
    for i in range(1, cols):
        fig.add_subplot(gs[0, i]).axis("off")

    # Generated full images for each tide level
    for i, (img, tide) in enumerate(generated_full):
        row = (i // cols) + 1
        col = i % cols
        ax = fig.add_subplot(gs[row, col])
        # Denormalize to [0, 255] for display
        img_display = np.clip(img * 127.5 + 127.5, 0, 255).astype(np.uint8)
        ax.imshow(img_display)
        ax.set_title(f"{tide:+.2f}m", fontsize=10, fontweight="bold")
        ax.axis("off")

    fig.suptitle(
        f"TideGAN — {site} Full-Image Samples ({H}x{W}px, no patches, every {STEP_SIZE*100:.0f}cm)",
        fontsize=15, fontweight="bold", y=1.01
    )
    plt.tight_layout()

    # ── Save individual full-res images ───────────────────────────────────
    if args.display_only:
        plt.show()
    else:
        out_dir = args.output if args.output else out_dir
        os.makedirs(out_dir, exist_ok=True)

        # Save each image at full resolution
        for img, tide in generated_full:
            img_display = np.clip(img * 127.5 + 127.5, 0, 255).astype(np.uint8)
            tide_str = f"{tide:.3f}"  # e.g. "-1.896" or "+0.004"
            filename = f"tide_{tide_str}m.png"
            filepath = os.path.join(out_dir, filename)
            Image.fromarray(img_display).save(filepath, quality=95)

        # Also save a low-res montage
        os.makedirs(out_dir, exist_ok=True)
        montage_path = os.path.join(out_dir, "montage.png")
        plt.savefig(montage_path, dpi=150, bbox_inches="tight")
        print(f"  ✓ Saved {len(generated_full)} images at full resolution ({H}x{W}px)")
        print(f"  ✓ Saved montage to {montage_path}")

    print("Done!")


if __name__ == "__main__":
    main()
