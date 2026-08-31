"""
TideGAN Training Loop

End-to-end training script for the conditional GAN.

Usage:
    python train.py [--epochs 100] [--batch_size 8] [--lr 2e-4] [--patch_size 256]
                   [--site foz] [--save_dir checkpoints] [--resume checkpoint.pth]

Features:
    - Adaptive training: alternate G and D updates with ratio 1:1
    - Gradient penalty for improved stability (WGAN-GP style)
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
from tidegan_model import Generator, Discriminator, GANLoss, count_parameters
from PIL import Image
import random


# ── Training configuration ────────────────────────────────────────────────
def get_config(args):
    """Merge CLI args with defaults."""
    config = {
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr_g": args.lr_g,
        "lr_d": args.lr_d,
        "patch_size": args.patch_size,
        "site": args.site,
        # Conditions always use the stable global site vocabulary.  A subset
        # of sites must not change the input shape or one-hot indices.
        "n_sites": len(SITES),
        "cond_dim": 16,
        "beta1": 0.5,
        "beta2": 0.999,
        "gradient_penalty": 10.0,
        "l1_weight": 100.0,
        "log_interval": 50,
        "save_interval": 5,
        "eval_interval": 20,
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


# ── Training functions ────────────────────────────────────────────────────
def train_step_g(G, D, optimizer_g, criterion, config, batch, device):
    """Single generator update step."""
    ref, target, condition = batch
    ref = ref.to(device)
    target = target.to(device)
    condition = condition.to(device)

    # Generate fake image
    fake = G(ref, condition)

    # Discriminator output on fake
    disc_fake = D(ref, fake, condition)

    # Generator loss: GAN + L1
    _, g_gan_loss = criterion(None, disc_fake)
    g_l1_loss = nn.L1Loss()(fake, target)
    g_loss = g_gan_loss + config["l1_weight"] * g_l1_loss

    optimizer_g.zero_grad()
    g_loss.backward()
    optimizer_g.step()

    return g_loss.item(), g_gan_loss.item(), g_l1_loss.item(), fake


def train_step_d(G, D, optimizer_d, criterion, config, ref, target, condition, fake, device):
    """Single discriminator update step with gradient penalty."""
    ref = ref.to(device)
    target = target.to(device)
    condition = condition.to(device)

    # Real
    disc_real = D(ref, target, condition)
    # Fake (detach generator to not backprop through it)
    with torch.no_grad():
        fake_detached = fake.detach()
    disc_fake = D(ref, fake_detached, condition)

    # Hinge D loss
    d_gan_loss, _ = criterion(disc_real, disc_fake)

    # Gradient penalty (WGAN-GP style)
    alpha = torch.rand(ref.size(0), 1, 1, 1, device=device)
    interpolates = alpha * target + (1 - alpha) * fake_detached
    interpolates.requires_grad_(True)
    disc_interp = D(ref, interpolates, condition)
    gradients = torch.autograd.grad(
        outputs=disc_interp.sum(),
        inputs=interpolates,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    grad_norm = gradients.norm(2, dim=1)
    gp = config["gradient_penalty"] * ((grad_norm - 1.0) ** 2).mean()

    d_loss = d_gan_loss + gp

    optimizer_d.zero_grad()
    d_loss.backward()
    optimizer_d.step()

    return d_loss.item(), d_gan_loss.item(), gp.item()


@torch.no_grad()
def evaluate(G, D, config, data_loader, device, writer, global_step, prefix="eval"):
    """Run a quick evaluation epoch."""
    G.eval()
    D.eval()

    l1_losses = []
    gan_losses = []

    for batch in data_loader:
        ref, target, condition = batch
        ref, target, condition = ref.to(device), target.to(device), condition.to(device)

        fake = G(ref, condition)
        disc_fake = D(ref, fake, condition)
        disc_real = D(ref, target, condition)

        _, g_gan = GANLoss("hinge")(disc_real, disc_fake)
        l1 = nn.L1Loss()(fake, target)

        l1_losses.append(l1.item())
        gan_losses.append(g_gan.item())

    G.train()
    D.train()

    mean_l1 = np.mean(l1_losses) if l1_losses else 0
    mean_gan = np.mean(gan_losses) if gan_losses else 0

    if writer is not None:
        writer.add_scalar(f"{prefix}/l1", mean_l1, global_step)
        writer.add_scalar(f"{prefix}/gan", mean_gan, global_step)
        writer.add_image(f"{prefix}/reference", (ref[0].cpu().permute(1, 2, 0).clamp(-1, 1).numpy() + 1) / 2,
                         global_step, dataformats="HWC")
        writer.add_image(f"{prefix}/generated", (fake[0].cpu().permute(1, 2, 0).clamp(-1, 1).numpy() + 1) / 2,
                         global_step, dataformats="HWC")
        writer.add_image(f"{prefix}/target", (target[0].cpu().permute(1, 2, 0).clamp(-1, 1).numpy() + 1) / 2,
                         global_step, dataformats="HWC")

    return mean_l1, mean_gan


# ── Save / Load ───────────────────────────────────────────────────────────
def save_checkpoint(config, G, D, optimizer_g, optimizer_d, epoch, global_step, path,
                    data_rng=None):
    """Save training state."""
    torch.save({
        "config": config,
        "epoch": epoch,
        "global_step": global_step,
        "generator_state": G.state_dict(),
        "discriminator_state": D.state_dict(),
        "optimizer_g": optimizer_g.state_dict(),
        "optimizer_d": optimizer_d.state_dict(),
        "python_rng_state": random.getstate(),
        "numpy_rng_state": np.random.get_state(),
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_state": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        "data_rng_state": data_rng.get_state() if data_rng is not None else None,
    }, path)
    print(f"  ✓ Checkpoint saved: {path}")


def load_checkpoint(path, G, D=None, optimizer_g=None, optimizer_d=None, device="cpu",
                    data_rng=None):
    """Load training state."""
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    G.load_state_dict(checkpoint["generator_state"])
    if D is not None:
        D.load_state_dict(checkpoint["discriminator_state"])
    if optimizer_g is not None and "optimizer_g" in checkpoint:
        optimizer_g.load_state_dict(checkpoint["optimizer_g"])
    if optimizer_d is not None and "optimizer_d" in checkpoint:
        optimizer_d.load_state_dict(checkpoint["optimizer_d"])
    if "python_rng_state" in checkpoint:
        random.setstate(checkpoint["python_rng_state"])
        np.random.set_state(checkpoint["numpy_rng_state"])
        torch.set_rng_state(checkpoint["torch_rng_state"])
        if torch.cuda.is_available() and checkpoint.get("cuda_rng_state") is not None:
            torch.cuda.set_rng_state_all(checkpoint["cuda_rng_state"])
    if data_rng is not None and checkpoint.get("data_rng_state") is not None:
        data_rng.set_state(checkpoint["data_rng_state"])
    print(f"  ✓ Loaded checkpoint from epoch {checkpoint['epoch']}, step {checkpoint['global_step']}")
    # Checkpoints store the one-based epoch that has just completed.
    return checkpoint.get("epoch", 0), checkpoint.get("global_step", 0)


# ── Main training loop ────────────────────────────────────────────────────
def train(args):
    config = get_config(args)
    if config["patch_size"] % 16:
        raise ValueError("--patch_size must be divisible by 16 for the U-Net skip shapes")
    seed_everything(config["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # ── Data ────────────────────────────────────────────────────────────
    print(f"\nLoading data (sites: {config['n_sites']})...")
    dataset = TideGANDataset(
        sites=config["site"] if config["site"] else SITES,
        patch_size=config["patch_size"],
        augment=True,
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

    # Validation set (small, no augmentation)
    val_dataset = TideGANDataset(
        sites=config["site"] if config["site"] else SITES,
        patch_size=config["patch_size"],
        augment=False,
        seed=123,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=min(4, config["batch_size"]),
        shuffle=False,
        num_workers=0,
    )

    # ── Models ──────────────────────────────────────────────────────────
    G = Generator(n_sites=config["n_sites"], cond_dim=config["cond_dim"]).to(device)
    D = Discriminator(n_sites=config["n_sites"], cond_dim=config["cond_dim"]).to(device)

    print(f"\nGenerator parameters: {count_parameters(G):,}")
    print(f"Discriminator parameters: {count_parameters(D):,}")

    # ── Optimizers & Loss ───────────────────────────────────────────────
    optimizer_g = optim.Adam(G.parameters(), lr=config["lr_g"],
                             betas=(config["beta1"], config["beta2"]))
    optimizer_d = optim.Adam(D.parameters(), lr=config["lr_d"],
                             betas=(config["beta1"], config["beta2"]))

    criterion = GANLoss("hinge")

    # ── Logging ─────────────────────────────────────────────────────────
    save_dir = args.save_dir
    os.makedirs(save_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=save_dir)

    start_epoch = 0
    global_step = 0

    if args.resume:
        start_epoch, global_step = load_checkpoint(
            args.resume, G, D, optimizer_g, optimizer_d, device, data_rng)

    # ── Training ────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"Starting training: {config['epochs']} epochs, "
          f"batch_size={config['batch_size']}, patch_size={config['patch_size']}")
    print(f"{'='*60}\n")

    D.train()

    for epoch in range(start_epoch, config["epochs"]):
        epoch_start = time.time()
        running_d_loss = 0
        running_g_loss = 0
        n_batches = 0

        # Progress bar
        pbar = tqdm(data_loader, desc=f"Epoch {epoch+1}/{config['epochs']}",
                    disable=args.no_progress_bar)

        for batch in pbar:
            ref, target, condition = batch
            ref, target, condition = ref.to(device), target.to(device), condition.to(device)

            # ── Update Generator ──────────────────────────────────────
            fake = G(ref, condition)
            disc_fake = D(ref, fake, condition)
            # GAN loss for G: maximize D(fake) (hinge: -mean(D(fake)))
            g_gan_loss = (-disc_fake).mean()
            g_l1_loss = nn.L1Loss()(fake, target)
            g_loss = g_gan_loss + config["l1_weight"] * g_l1_loss

            optimizer_g.zero_grad()
            g_loss.backward()
            optimizer_g.step()

            # ── Update Discriminator ──────────────────────────────────
            disc_real = D(ref, target, condition)
            with torch.no_grad():
                disc_fake = D(ref, fake.detach(), condition)
            d_gan_loss, _ = criterion(disc_real, disc_fake)

            # Gradient penalty
            alpha = torch.rand(ref.size(0), 1, 1, 1, device=device)
            interpolates = alpha * target + (1 - alpha) * fake.detach()
            interpolates.requires_grad_(True)
            disc_interp = D(ref, interpolates, condition)
            gradients = torch.autograd.grad(
                outputs=disc_interp.sum(),
                inputs=interpolates,
                create_graph=True,
                retain_graph=True,
                only_inputs=True,
            )[0]
            grad_norm = gradients.norm(2, dim=1)
            gp = config["gradient_penalty"] * ((grad_norm - 1.0) ** 2).mean()

            d_loss = d_gan_loss + gp

            optimizer_d.zero_grad()
            d_loss.backward()
            optimizer_d.step()

            # Logging
            running_d_loss += d_loss.item()
            running_g_loss += g_loss.item()
            n_batches += 1

            if global_step % config["log_interval"] == 0:
                writer.add_scalar("loss/g_gan", g_gan_loss.item(), global_step)
                writer.add_scalar("loss/g_l1", g_l1_loss.item(), global_step)
                writer.add_scalar("loss/d_gan", d_gan_loss.item(), global_step)
                writer.add_scalar("loss/gp", gp.item(), global_step)
                writer.add_scalar("lr/g", optimizer_g.param_groups[0]["lr"], global_step)
                writer.add_scalar("lr/d", optimizer_d.param_groups[0]["lr"], global_step)

            global_step += 1

        # Epoch metrics
        avg_d = running_d_loss / n_batches
        avg_g = running_g_loss / n_batches
        epoch_time = time.time() - epoch_start

        writer.add_scalar("metrics/d_loss", avg_d, epoch)
        writer.add_scalar("metrics/g_loss", avg_g, epoch)

        print(f"\nEpoch {epoch+1}/{config['epochs']}  "
              f"D={avg_d:.4f}  G={avg_g:.4f}  "
              f"L1={g_l1_loss.item():.4f}  "
              f"Time: {epoch_time:.1f}s")

        # ── Evaluation ──────────────────────────────────────────────
        if (epoch + 1) % config["eval_interval"] == 0 or epoch == 0:
            print(f"  Running evaluation...")
            eval_l1, eval_gan = evaluate(
                G, D, config, val_loader, device, writer, epoch, "eval")
            print(f"  Eval L1: {eval_l1:.4f}  Eval GAN: {eval_gan:.4f}")

        # ── Save checkpoint ─────────────────────────────────────────
        if (epoch + 1) % config["save_interval"] == 0 or epoch == 0:
            ckpt_path = os.path.join(save_dir, f"checkpoint_epoch_{epoch+1}.pth")
            save_checkpoint(config, G, D, optimizer_g, optimizer_d,
                            epoch + 1, global_step, ckpt_path, data_rng)

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
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr_g", type=float, default=2e-4)
    parser.add_argument("--lr_d", type=float, default=2e-4)
    parser.add_argument("--patch_size", type=int, default=256)
    parser.add_argument("--site", nargs="+", default=None, help="Sites to use (e.g. Foz Santander Villaviciosa, default: all)")
    parser.add_argument("--save_dir", type=str, default="checkpoints/tidegan")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
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
