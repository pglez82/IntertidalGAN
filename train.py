"""
Tide-conditioned image translation training loop.

This project learns to generate the target coastal image from a reference image
and a tide condition, using a supervised L1 objective instead of adversarial
training.

Usage:
    python train.py [--epochs 100] [--batch_size 8] [--lr_g 1e-4] [--patch_size 256]
                   [--site foz] [--save_dir checkpoints/tidegan] [--resume checkpoint.pth]

Features:
    - Conditional image-to-image translation from reference + tide
    - L1 reconstruction loss for content consistency
    - Per-site evaluation and visualization
    - TensorBoard logging
    - Checkpoint save/load
"""

import os
import sys
import argparse
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from tidegan_dataset import TideGANDataset, SITES, SITE_TIDE_RANGES, SITE_DIR
from tidegan_model import Generator, count_parameters
from PIL import Image, ImageDraw
import random


# ── Training configuration ────────────────────────────────────────────────
def get_config(args):
    """Merge CLI args with defaults."""
    config = {
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr_g": args.lr_g,
        "patch_size": args.patch_size,
        "site": args.site,
        # Conditions always use the stable global site vocabulary.  A subset
        # of sites must not change the input shape or one-hot indices.
        "n_sites": len(SITES),
        "cond_dim": 16,
        "beta1": 0.5,
        "beta2": 0.9,
        "log_interval": 25,
        "save_interval": 10,
        "eval_interval": 2,
        "num_workers": args.num_workers,
        "seed": args.seed,
    }
    return config


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Deterministic algorithms make resumed/ repeated runs comparable.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _worker_init_fn(worker_id):
    worker_seed = torch.initial_seed() % (2 ** 32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def resolve_devices(args):
    """Return the primary CUDA device and list of visible GPU ids."""
    if not torch.cuda.is_available():
        return torch.device("cpu"), []

    available = list(range(torch.cuda.device_count()))
    if args.devices is None:
        selected = available[: min(2, len(available))] if len(available) > 1 else available
    else:
        selected = [int(d) for d in args.devices]
        invalid = [d for d in selected if d not in available]
        if invalid:
            raise ValueError(f"GPU ids invalidos: {invalid}. GPUs disponibles: {available}")

    if len(selected) == 0:
        return torch.device("cuda:0"), [0]

    primary = torch.device(f"cuda:{selected[0]}")
    return primary, selected


@torch.no_grad()
def save_generated_samples(G, val_loader, save_dir, epoch, device):
    G.eval()
    os.makedirs(save_dir, exist_ok=True)

    for batch in val_loader:
        ref, target, condition = batch
        ref = ref.to(device)
        target = target.to(device)
        condition = condition.to(device)

        fake = G(ref, condition)
        sample_idx = 0

        norm_tide = float(condition[sample_idx, 0].item())
        site_idx = int(condition[sample_idx, 1:].argmax().item())
        site_name = SITES[site_idx]
        tmin, tmax = SITE_TIDE_RANGES[site_name]
        tide_m = ((norm_tide + 1.0) / 2.0) * (tmax - tmin) + tmin

        ref_img = ((ref[sample_idx].cpu().permute(1, 2, 0).clamp(-1, 1) + 1) / 2).numpy()
        fake_img = ((fake[sample_idx].cpu().permute(1, 2, 0).clamp(-1, 1) + 1) / 2).numpy()
        tgt_img = ((target[sample_idx].cpu().permute(1, 2, 0).clamp(-1, 1) + 1) / 2).numpy()

        panel_ref = Image.fromarray((ref_img * 255).astype(np.uint8))
        panel_fake = Image.fromarray((fake_img * 255).astype(np.uint8))
        panel_target = Image.fromarray((tgt_img * 255).astype(np.uint8))

        for panel, label in [(panel_ref, "Reference"), (panel_fake, "Generated"), (panel_target, "Target")]:
            draw = ImageDraw.Draw(panel)
            draw.rectangle((0, 0, panel.width - 1, 24), fill=(0, 0, 0, 180))
            draw.text((10, 6), label, fill=(255, 255, 255))

        montage_h = panel_ref.height + 42
        montage = Image.new(
            "RGB",
            (panel_ref.width * 3, montage_h),
            color=(30, 30, 30),
        )
        draw = ImageDraw.Draw(montage)
        draw.text(
            (20, 10),
            f"Target tide: {tide_m:.3f} m  |  site: {site_name}",
            fill=(255, 255, 255),
        )

        montage.paste(panel_ref, (0, 42))
        montage.paste(panel_fake, (panel_ref.width, 42))
        montage.paste(panel_target, (panel_ref.width * 2, 42))
        montage.save(os.path.join(save_dir, f"epoch_{epoch}_comparison.png"))
        break

    G.train()

# ── Training functions ────────────────────────────────────────────────────
@torch.no_grad()
def evaluate(G, data_loader, device, writer, global_step, prefix="eval"):
    """Run a quick evaluation epoch using supervised L1 loss only."""
    G.eval()

    l1_losses = []

    for batch in data_loader:
        ref, target, condition = batch
        ref, target, condition = ref.to(device), target.to(device), condition.to(device)

        fake = G(ref, condition)
        l1 = nn.L1Loss()(fake, target)
        l1_losses.append(l1.item())

    G.train()

    mean_l1 = np.mean(l1_losses) if l1_losses else 0

    if writer is not None:
        writer.add_scalar(f"{prefix}/l1", mean_l1, global_step)
        writer.add_image(
            f"{prefix}/comparison",
            (np.concatenate([
                ((ref[0].cpu().permute(1, 2, 0).clamp(-1, 1).numpy() + 1) / 2),
                ((fake[0].cpu().permute(1, 2, 0).clamp(-1, 1).numpy() + 1) / 2),
                ((target[0].cpu().permute(1, 2, 0).clamp(-1, 1).numpy() + 1) / 2),
            ], axis=1) * 255).astype(np.uint8),
            global_step,
            dataformats="HWC",
        )

    return mean_l1


# ── Save / Load ───────────────────────────────────────────────────────────
def save_checkpoint(config, G, optimizer_g, scheduler_g, epoch, global_step, path, data_rng=None):
    """Save generator-only training state."""
    G_state = G.module.state_dict() if hasattr(G, "module") else G.state_dict()

    torch.save({
        "config": config,
        "epoch": epoch,
        "global_step": global_step,
        "generator_state": G_state,
        "optimizer_g": optimizer_g.state_dict(),
        "scheduler_g": scheduler_g.state_dict(),
        "python_rng_state": random.getstate(),
        "numpy_rng_state": np.random.get_state(),
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_state": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        "data_rng_state": data_rng.get_state() if data_rng is not None else None,
    }, path)
    print(f"  ✓ Checkpoint saved: {path}")


def load_checkpoint(path, G, optimizer_g=None, scheduler_g=None, device="cpu", data_rng=None):
    """Load generator-only training state."""
    checkpoint = torch.load(path, map_location=device, weights_only=False)

    G_state = checkpoint["generator_state"]
    if hasattr(G, "module"):
        G.module.load_state_dict(G_state)
    else:
        G.load_state_dict(G_state)

    if optimizer_g is not None and "optimizer_g" in checkpoint:
        optimizer_g.load_state_dict(checkpoint["optimizer_g"])
    if scheduler_g is not None and "scheduler_g" in checkpoint:
        scheduler_g.load_state_dict(checkpoint["scheduler_g"])
    if "python_rng_state" in checkpoint:
        random.setstate(checkpoint["python_rng_state"])
        np.random.set_state(checkpoint["numpy_rng_state"])
        torch.set_rng_state(checkpoint["torch_rng_state"])
        if torch.cuda.is_available() and checkpoint.get("cuda_rng_state") is not None:
            torch.cuda.set_rng_state_all(checkpoint["cuda_rng_state"])
    if data_rng is not None and checkpoint.get("data_rng_state") is not None:
        data_rng.set_state(checkpoint["data_rng_state"])
    print(f"  ✓ Loaded checkpoint from epoch {checkpoint['epoch']}, step {checkpoint['global_step']}")
    return checkpoint.get("epoch", 0), checkpoint.get("global_step", 0)


# ── Main training loop ────────────────────────────────────────────────────
def train(args):
    config = get_config(args)
    if config["patch_size"] % 16:
        raise ValueError("--patch_size must be divisible by 16 for the U-Net skip shapes")
    seed_everything(config["seed"])
    device, device_ids = resolve_devices(args)
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"Using CUDA devices: {device_ids}")
        for i in device_ids:
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")

    if len(device_ids) > 1:
        print(f"Multi-GPU enabled with DataParallel across devices: {device_ids}")

    # ── Data ────────────────────────────────────────────────────────────
    print(f"\nLoading data (sites: {config['n_sites']})...")
    train_sites = config["site"] if config["site"] else SITES
    dataset = TideGANDataset(
        sites=train_sites,
        patch_size=config["patch_size"],
        augment=True,
        split="train",
        split_mode="image",
        val_ratio=0.2,
        seed=config["seed"],
    )
    data_rng = torch.Generator().manual_seed(config["seed"])
    data_loader = DataLoader(
        dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
        worker_init_fn=_worker_init_fn,
        generator=data_rng,
    )

    # Validation set: same site pool, but disjoint by image/tide split to avoid leakage.
    val_dataset = TideGANDataset(
        sites=train_sites,
        patch_size=config["patch_size"],
        augment=False,
        split="val",
        split_mode="image",
        val_ratio=0.2,
        seed=config["seed"],
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=min(4, config["batch_size"]),
        shuffle=False,
        num_workers=0,
    )

    # ── Models ──────────────────────────────────────────────────────────
    G = Generator(n_sites=config["n_sites"], cond_dim=config["cond_dim"]).to(device)

    if len(device_ids) > 1:
        G = nn.DataParallel(G, device_ids=device_ids)

    print(f"\nGenerator parameters: {count_parameters(G):,}")

    # ── Optimizer & scheduler ───────────────────────────────────────────
    optimizer_g = optim.Adam(G.parameters(), lr=config["lr_g"],
                             betas=(config["beta1"], config["beta2"]))
    scheduler_g = optim.lr_scheduler.CosineAnnealingLR(
        optimizer_g,
        T_max=max(1, config["epochs"]),
    )

    # ── Logging ─────────────────────────────────────────────────────────
    save_dir = args.save_dir
    os.makedirs(save_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=save_dir)

    start_epoch = 0
    global_step = 0

    if args.resume:
        start_epoch, global_step = load_checkpoint(
            args.resume, G, optimizer_g, scheduler_g, device, data_rng)

    # ── Training ────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"Starting training: {config['epochs']} epochs, "
          f"batch_size={config['batch_size']}, patch_size={config['patch_size']}")
    print(f"{'='*60}\n")

    for epoch in range(start_epoch, config["epochs"]):
        epoch_start = time.time()
        running_g_loss = 0.0
        n_batches = 0

        pbar = tqdm(data_loader, desc=f"Epoch {epoch+1}/{config['epochs']}",
                    disable=args.no_progress_bar)

        for batch in pbar:
            ref, target, condition = batch
            ref, target, condition = ref.to(device), target.to(device), condition.to(device)

            optimizer_g.zero_grad()
            fake = G(ref, condition)
            g_loss = nn.L1Loss()(fake, target)
            g_loss.backward()
            torch.nn.utils.clip_grad_norm_(G.parameters(), max_norm=5.0)
            optimizer_g.step()

            running_g_loss += float(g_loss.item())
            n_batches += 1

            if global_step % config["log_interval"] == 0:
                writer.add_scalar("loss/l1", g_loss.item(), global_step)
                writer.add_scalar("lr/g", optimizer_g.param_groups[0]["lr"], global_step)

            global_step += 1

        avg_g = running_g_loss / n_batches
        epoch_time = time.time() - epoch_start
        current_lr = optimizer_g.param_groups[0]["lr"]

        writer.add_scalar("metrics/g_loss", avg_g, epoch)
        writer.add_scalar("lr/g", current_lr, epoch)

        print(f"\nEpoch {epoch+1}/{config['epochs']}  "
              f"G={avg_g:.4f}  "
              f"L1={g_loss.item():.4f}  "
              f"lr={current_lr:.2e}  "
              f"Time: {epoch_time:.1f}s")

        if (epoch + 1) % config["eval_interval"] == 0 or epoch == 0:
            print(f"  Running evaluation...")
            eval_l1 = evaluate(G, val_loader, device, writer, epoch, "eval")
            print(f"  Eval L1: {eval_l1:.4f}")

        if (epoch + 1) % config["save_interval"] == 0 or epoch == 0:
            ckpt_path = os.path.join(save_dir, f"checkpoint_epoch_{epoch+1}.pth")
            save_checkpoint(config, G, optimizer_g, scheduler_g, epoch + 1, global_step, ckpt_path, data_rng)

        save_generated_samples(G, val_loader, os.path.join(save_dir, "samples"), epoch, device)

        scheduler_g.step()

    writer.close()
    print(f"\nTraining complete. Checkpoints in: {save_dir}")


# ── Inference script ──────────────────────────────────────────────────────
def generate_images(args):
    """Generate tide-conditioned images from a reference image."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    G = Generator(n_sites=3, cond_dim=16).to(device)
    load_checkpoint(args.checkpoint, G, device=device)
    G.eval()

    from tidegan_dataset import SITE_TIDE_RANGES, SITES

    # Load reference image
    if not args.site or len(args.site) != 1:
        raise ValueError("En modo generate debes indicar exactamente un sitio con --site.")
    site = args.site[0]  # must be capitalized e.g. "Foz"
    site_lower = site.lower()
    if site not in SITES:
        raise ValueError(f"Sitio desconocido: {site}. Opciones: {', '.join(SITES)}")
    csv_path = os.path.join(SITE_DIR, site, f"dataset_{site_lower}.csv")
    import csv as csv_mod
    with open(csv_path) as f:
        rows = list(csv_mod.DictReader(f))
    rows_sorted = sorted(rows, key=lambda r: float(r["marea_m"]))

    ref_path = os.path.join(SITE_DIR, site, rows_sorted[0]["imagen_rgb"])
    ref_img = Image.open(ref_path).convert("RGB")
    original_h, original_w = ref_img.height, ref_img.width
    pad_h = (-original_h) % 16
    pad_w = (-original_w) % 16
    ref_np = np.array(ref_img, dtype=np.float32) / 127.5 - 1.0  # [-1, 1]
    ref_tensor = torch.from_numpy(ref_np).permute(2, 0, 1).unsqueeze(0).to(device)
    if pad_h or pad_w:
        ref_tensor = F.pad(ref_tensor, (0, pad_w, 0, pad_h), mode="reflect")

    # Generate for different tide levels
    tmin, tmax = SITE_TIDE_RANGES[site]
    tide_levels = np.linspace(tmin, tmax, 10)

    output_dir = os.path.join(args.output_dir, site)
    os.makedirs(output_dir, exist_ok=True)

    for tide in tide_levels:
        norm_tide = 2.0 * (tide - tmin) / (tmax - tmin) - 1.0
        one_hot = [1.0 if SITES.index(site) == i else 0.0 for i in range(3)]
        condition = torch.tensor([[norm_tide] + one_hot], dtype=torch.float32).to(device)

        with torch.no_grad():
            gen = G(ref_tensor, condition)

        # Convert back to [0, 255]
        gen = gen[:, :, :original_h, :original_w]
        gen_np = (gen[0].cpu().permute(1, 2, 0).clamp(-1, 1).numpy() + 1) / 2 * 255
        gen_pil = Image.fromarray(gen_np.astype(np.uint8))
        gen_pil.save(os.path.join(output_dir, f"generated_tide_{tide:.3f}.png"))
        print(f"  Generated: tide={tide:.3f} m → {output_dir}/generated_tide_{tide:.3f}.png")


# ── CLI ────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="TideGAN: Conditional GAN for tide-aware coastal image generation")
    parser.add_argument("--mode", choices=["train", "generate"], default="train")
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr_g", type=float, default=1e-4)
    parser.add_argument("--patch_size", type=int, default=256)
    parser.add_argument("--site", nargs="+", default=None, help="Sites to use (e.g. Foz Santander Villaviciosa, default: all)")
    parser.add_argument("--save_dir", type=str, default="checkpoints/tidegan")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--devices", type=int, nargs="+", default=None,
                        help="CUDA device ids to use for training, e.g. --devices 0 1")
    parser.add_argument("--no_progress_bar", action="store_true")

    # Generate mode args
    parser.add_argument("--checkpoint", type=str, default=None, help="Checkpoint for generation")
    parser.add_argument("--output_dir", type=str, default="outputs")

    args = parser.parse_args()

    if args.mode == "train":
        train(args)
    elif args.mode == "generate":
        if args.checkpoint is None:
            print("Error: --checkpoint required for generate mode")
            sys.exit(1)
        generate_images(args)


if __name__ == "__main__":
    main()
