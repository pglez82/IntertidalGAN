"""
TideGAN Dataset: Loads coastal RGB images paired with tide levels for conditional GAN training.

Each site has images spanning a range of tide levels. The dataset:
- Loads RGB images from disk
- Reads tide levels from CSV
- Extracts 256×256 patches with augmentation
- Pairs a reference (low-tide) image with a target image at a given tide level
- Normalizes tide values per-site for stable training
"""

import os
import csv
import random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image


# ── Site metadata ──────────────────────────────────────────────────────────
SITES = ["Foz", "Santander", "Villaviciosa"]
SITE_DIR = "data"

# lowercase versions for CSV naming
SITES_LOWERCASE = [s.lower() for s in SITES]

# Tide ranges per site (computed from full datasets)
SITE_TIDE_RANGES = {
    "Foz": (-1.896, 1.186),
    "Santander": (-2.178, 0.908),
    "Villaviciosa": (-2.169, 1.013),
}


# ── Helpers ────────────────────────────────────────────────────────────────
def _read_csv(path):
    """Return list of dicts from a CSV file."""
    with open(path, "r") as f:
        return list(csv.DictReader(f))


def _load_image(path):
    """Load a PNG image as numpy array (H, W, 3) in [0, 255]."""
    img = Image.open(path).convert("RGB")
    return np.array(img, dtype=np.float32)


def _normalize_tide(tide, site):
    """Min-max normalize tide to [-1, 1] range for the given site."""
    tmin, tmax = SITE_TIDE_RANGES[site]
    return 2.0 * (tide - tmin) / (tmax - tmin) - 1.0


def _denormalize_tide(norm_tide, site):
    """Inverse of _normalize_tide."""
    tmin, tmax = SITE_TIDE_RANGES[site]
    return (norm_tide + 1.0) / 2.0 * (tmax - tmin) + tmin


# ── Single-image entry ────────────────────────────────────────────────────
class TideImageEntry:
    """One image record from the CSV."""
    def __init__(self, row, site):
        self.site = site
        self.date = row["fecha"]
        self.tide = float(row["marea_m"])
        self.norm_tide = _normalize_tide(self.tide, site)
        self.rgb_path = os.path.join(SITE_DIR, site, row["imagen_rgb"])
        self.scl_path = os.path.join(SITE_DIR, site, row["imagen_scl"])
        self.coverage = float(row["cobertura_tile_pct"])
        self.nubes = float(row["nubes_pct"])

    def __repr__(self):
        return f"TideImage({self.site}, {self.date}, tide={self.tide:.3f})"


# ── Dataset ────────────────────────────────────────────────────────────────
class TideGANDataset(Dataset):
    """
    Conditional GAN dataset for tide-aware image generation.

    For each training sample we produce:
      - reference_rgb:  (3, H, W)  – low-tide image, normalized to [-1, 1]
      - target_rgb:     (3, H, W)  – image at the requested tide level
      - condition:      (C,)       – [norm_target_tide, site_one_hot]

    Data is drawn in *pairs*: we randomly pick a low-tide reference and
    pair it with a target image at a different tide level from the same site.
    """

    def __init__(self, sites=None, patch_size=256, augment=True,
                 min_tide_diff=0.2, seed=42):
        """
        Args:
            sites:          list of site names to include. Default = all.
            patch_size:     size of extracted patches (patch_size × patch_size).
            augment:        apply random horizontal flip + color jitter.
            min_tide_diff:  minimum tide difference (normalized) between
                            reference and target to avoid trivial pairs.
            seed:           random seed.
        """
        self.patch_size = patch_size
        self.augment = augment
        self.min_tide_diff = min_tide_diff
        random.seed(seed)
        np.random.seed(seed)

        if sites is None:
            sites = SITES

        # Load all entries, sorted by tide within each site
        self.entries_by_site = {}
        for site in sites:
            site_lower = site.lower()
            csv_path = os.path.join(SITE_DIR, site, f"dataset_{site_lower}.csv")
            rows = _read_csv(csv_path)
            entries = sorted([TideImageEntry(r, site) for r in rows],
                             key=lambda e: e.tide)
            self.entries_by_site[site] = entries

        # Pre-select reference candidates: the N lowest-tide images per site
        # (we pick randomly among the lowest 10)
        self.ref_candidates = {}
        for site, entries in self.entries_by_site.items():
            n_ref = min(10, len(entries))
            self.ref_candidates[site] = entries[:n_ref]

        # Build a flat list of all valid (ref, target) pairs for sampling
        # We'll sample dynamically instead for infinite variety
        self.sites_list = list(self.entries_by_site.keys())

    def __len__(self):
        # Arbitrary large number; the dataset is infinite in practice
        return 100000

    def _augment(self, img: np.ndarray) -> np.ndarray:
        """Apply random augmentations to a (H, W, 3) image in [0, 255]."""
        if not self.augment:
            return img

        # Horizontal flip
        if random.random() > 0.5:
            img = np.ascontiguousarray(img[:, ::-1, :])

        # Color jitter (small perturbations)
        if random.random() > 0.5:
            factor = random.uniform(0.85, 1.15)
            img = img * factor

        if random.random() > 0.5:
            offset = random.uniform(-10, 10)
            img = np.clip(img + offset, 0, 255)

        return img

    def _extract_patch(self, img: np.ndarray, patch_size: int):
        """
        Extract a random patch of given size from an (H, W, 3) image.
        Handles images smaller than patch_size by cropping center.
        """
        H, W, _ = img.shape
        if H <= patch_size and W <= patch_size:
            # Crop center
            y1 = (H - patch_size) // 2
            x1 = (W - patch_size) // 2
            return img[y1:y1+patch_size, x1:x1+patch_size]

        y1 = random.randint(0, max(0, H - patch_size))
        x1 = random.randint(0, max(0, W - patch_size))
        return img[y1:y1+patch_size, x1:x1+patch_size]

    def _to_tensor(self, img: np.ndarray) -> torch.Tensor:
        """Convert (H, W, 3) in [0, 255] to (3, H, W) in [-1, 1]."""
        img = img.astype(np.float32) / 127.5 - 1.0  # [-1, 1]
        return torch.from_numpy(img).permute(2, 0, 1)

    def _site_one_hot(self, site: str, n_sites: int = 3) -> torch.Tensor:
        idx = SITES.index(site)
        one_hot = torch.zeros(n_sites)
        one_hot[idx] = 1.0
        return one_hot

    def __getitem__(self, idx):
        """
        Returns:
            reference_rgb:  (3, patch_size, patch_size)  in [-1, 1]
            target_rgb:     (3, patch_size, patch_size)  in [-1, 1]
            condition:      (1 + n_sites,)                [norm_tide, one_hot]
        """
        site = random.choice(self.sites_list)
        entries = self.entries_by_site[site]
        n = len(entries)

        # Pick reference from low-tide candidates
        ref_entry = random.choice(self.ref_candidates[site])

        # Pick a target with sufficiently different tide
        for _ in range(20):  # retry up to 20 times
            target_entry = random.choice(entries)
            if abs(target_entry.norm_tide - ref_entry.norm_tide) > self.min_tide_diff:
                break

        # Load and augment
        ref_img = _load_image(ref_entry.rgb_path)
        target_img = _load_image(target_entry.rgb_path)

        if self.augment:
            ref_img = self._augment(ref_img)
            target_img = self._augment(target_img)

        # Extract patches
        ref_patch = self._extract_patch(ref_img, self.patch_size)
        target_patch = self._extract_patch(target_img, self.patch_size)

        # Convert to tensors
        ref_tensor = self._to_tensor(ref_patch)
        target_tensor = self._to_tensor(target_patch)

        # Condition: normalized target tide + site one-hot
        n_sites = len(self.sites_list)
        condition = torch.cat([
            torch.tensor([target_entry.norm_tide]),
            self._site_one_hot(site, n_sites)
        ])

        return ref_tensor, target_tensor, condition


# ── Utility: get min-tide reference path per site ──────────────────────────
def get_reference_images():
    """Return dict mapping site → path to the absolute lowest-tide RGB image."""
    refs = {}
    for site, entries in _read_csv_dummy().items():
        entries_sorted = sorted(entries, key=lambda e: e.tide)
        refs[site] = entries_sorted[0].rgb_path if entries_sorted else None
    return refs


def _read_csv_dummy():
    """Helper for reference image lookup."""
    result = {}
    for site in SITES:
        site_lower = site.lower()
        csv_path = os.path.join(SITE_DIR, site, f"dataset_{site_lower}.csv")
        rows = _read_csv(csv_path)
        entries = sorted([TideImageEntry(r, site) for r in rows],
                         key=lambda e: e.tide)
        result[site] = entries
    return result


# ── Test ───────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    ds = TideGANDataset(patch_size=256, augment=True)
    ref, target, cond = ds[0]
    print(f"Reference shape:   {ref.shape}   (range: {ref.min():.3f}, {ref.max():.3f})")
    print(f"Target shape:      {target.shape}  (range: {target.min():.3f}, {target.max():.3f})")
    print(f"Condition shape:   {cond.shape}   values: {cond.tolist()}")

    # Test DataLoader
    dl = DataLoader(ds, batch_size=8, shuffle=True, num_workers=2,
                    pin_memory=True, drop_last=False)
    batch_ref, batch_target, batch_cond = next(iter(dl))
    print(f"\nBatch shapes: {batch_ref.shape}, {batch_target.shape}, {batch_cond.shape}")

    # Print tide stats
    for site in SITES:
        site_lower = site.lower()
        csv_path = os.path.join(SITE_DIR, site, f"dataset_{site_lower}.csv")
        rows = _read_csv(csv_path)
        tides = [float(r["marea_m"]) for r in rows]
        print(f"\n{site}: {len(tides)} images, "
              f"min tide = {min(tides):.3f} m ({os.path.basename(min(rows, key=lambda r: float(r['marea_m']))['imagen_rgb'])}), "
              f"max tide = {max(tides):.3f} m ({os.path.basename(max(rows, key=lambda r: float(r['marea_m']))['imagen_rgb'])})")
