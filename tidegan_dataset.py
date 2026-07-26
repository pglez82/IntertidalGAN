"""
TideGAN Dataset: Loads coastal RGB images paired with tide levels for conditional GAN training.
Includes SCL-based smart patch extraction to avoid clouds and "dead" patches.
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

SITE_TIDE_RANGES = {
    "Foz": (-1.896, 1.186),
    "Santander": (-2.178, 0.908),
    "Villaviciosa": (-2.169, 1.013),
}

# ── Helpers ────────────────────────────────────────────────────────────────
def _read_csv(path):
    with open(path, "r") as f:
        return list(csv.DictReader(f))

def _load_image(path):
    """Load a PNG image as numpy array (H, W, 3) in [0, 255]."""
    img = Image.open(path).convert("RGB")
    return np.array(img, dtype=np.float32)

def _load_scl(path):
    """
    Loads RGB SCL image and converts it to a 2D class index array (H, W)
    based on the exact hex colors from the Sentinel-2 legend.
    """
    # Cargar como RGB
    img_rgb = np.array(Image.open(path).convert("RGB"))
    
    # Mapeo de colores RGB a IDs de clase (según tu tabla)
    COLOR_TO_CLASS = {
        (0, 0, 0): 0,          # #000000 - Sin datos
        (255, 0, 0): 1,        # #FF0000 - Saturado/Defectuoso
        (47, 47, 47): 2,       # #2F2F2F - Área oscura
        (100, 50, 0): 3,       # #643200 - Sombra de nube
        (0, 160, 0): 4,        # #00A000 - Vegetación
        (255, 230, 90): 5,     # #FFE65A - No vegetación
        (0, 0, 255): 6,        # #0000FF - Agua
        (128, 128, 128): 7,    # #808080 - No clasificado
        (192, 192, 192): 8,    # #C0C0C0 - Nube (prob. media)
        (255, 255, 255): 9,    # #FFFFFF - Nube (prob. alta)
        (100, 200, 255): 10,   # #64C8FF - Cirrus fino
        (255, 150, 255): 11    # #FF96FF - Nieve/Hielo
    }
    
    # Crear matriz 2D vacía (H, W)
    scl_2d = np.zeros(img_rgb.shape[:2], dtype=np.uint8)
    
    # Mapeo vectorizado súper rápido: busca dónde coincide el color RGB y asigna la clase
    for color, cls_id in COLOR_TO_CLASS.items():
        # np.all con axis=-1 comprueba que los 3 canales (R, G, B) coincidan exactamente
        mask = np.all(img_rgb == color, axis=-1)
        scl_2d[mask] = cls_id
        
    return scl_2d

def _normalize_tide(tide, site):
    tmin, tmax = SITE_TIDE_RANGES[site]
    return 2.0 * (tide - tmin) / (tmax - tmin) - 1.0

# ── Single-image entry ────────────────────────────────────────────────────
class TideImageEntry:
    def __init__(self, row, site):
        self.site = site
        self.date = row["fecha"]
        self.tide = float(row["marea_m"])
        self.norm_tide = _normalize_tide(self.tide, site)
        self.rgb_path = os.path.join(SITE_DIR, site, row["imagen_rgb"])
        self.scl_path = os.path.join(SITE_DIR, site, row["imagen_scl"])

# ── Dataset ────────────────────────────────────────────────────────────────
class TideGANDataset(Dataset):
    def __init__(self, sites=None, patch_size=256, augment=True,
                 min_tide_diff=0.2, max_cloud_pct=0.01, max_nodata_pct=0.10, seed=42):
        self.patch_size = patch_size
        self.augment = augment
        self.min_tide_diff = min_tide_diff
        
        # New independent thresholds
        self.max_cloud_pct = max_cloud_pct
        self.max_nodata_pct = max_nodata_pct
        
        # Class 0 is "No Data", the rest are actual clouds/sensor errors
        self.nodata_class = 0
        self.cloud_classes = [1, 3, 8, 9, 10]
        
        random.seed(seed)
        np.random.seed(seed)

        if sites is None: sites = SITES

        self.entries_by_site = {}
        for site in sites:
            csv_path = os.path.join(SITE_DIR, site, f"dataset_{site.lower()}.csv")
            rows = _read_csv(csv_path)
            entries = sorted([TideImageEntry(r, site) for r in rows], key=lambda e: e.tide)
            self.entries_by_site[site] = entries

        self.ref_candidates = {}
        for site, entries in self.entries_by_site.items():
            n_ref = min(10, len(entries))
            self.ref_candidates[site] = entries[:n_ref]

        self.sites_list = list(self.entries_by_site.keys())

    def __len__(self):
        return 100000

    def _color_jitter(self, img: np.ndarray) -> np.ndarray:
        """Aumentos de color independientes para cada imagen."""
        if random.random() > 0.5:
            factor = random.uniform(0.85, 1.15)
            img = img * factor
        if random.random() > 0.5:
            offset = random.uniform(-10, 10)
            img = np.clip(img + offset, 0, 255)
        return img

    def _get_valid_crop_coords(self, scl: np.ndarray, max_retries=50):
        """
        Searches for random (y, x) coordinates where the SCL patch:
        1. Has less than 'max_nodata_pct' black pixels (No Data).
        2. Has less than 'max_cloud_pct' cloud/error pixels.
        3. Contains both water and land (coastal dynamics).
        """
        H, W = scl.shape
        
        if H <= self.patch_size or W <= self.patch_size:
            return 0, 0

        for _ in range(max_retries):
            y1 = random.randint(0, H - self.patch_size)
            x1 = random.randint(0, W - self.patch_size)
            
            patch_scl = scl[y1:y1+self.patch_size, x1:x1+self.patch_size]
            
            # 1. Check No Data (Black pixels) threshold
            nodata_ratio = np.mean(patch_scl == self.nodata_class)
            if nodata_ratio > self.max_nodata_pct:
                continue
                
            # 2. Check Clouds/Errors threshold
            cloud_ratio = np.mean(np.isin(patch_scl, self.cloud_classes))
            if cloud_ratio > self.max_cloud_pct:
                continue
                
            # 3. Check for Water (6) and Land/Vegetation (2, 4, 5, 7)
            has_water = np.any(patch_scl == 6)
            has_land = np.any(np.isin(patch_scl, [2, 4, 5, 7]))
            
            if has_water and has_land:
                return y1, x1
                
        # Fallback if no perfect patch is found after max_retries
        return random.randint(0, H - self.patch_size), random.randint(0, W - self.patch_size)

    def _extract_and_pad(self, img: np.ndarray, y: int, x: int) -> np.ndarray:
        """Recorta la imagen. Si es más pequeña que el parche, aplica padding."""
        H, W = img.shape[:2]
        
        if H < self.patch_size or W < self.patch_size:
            # Padding con ceros (o reflejo) si la imagen es demasiado pequeña
            pad_h = max(0, self.patch_size - H)
            pad_w = max(0, self.patch_size - W)
            img = np.pad(img, ((0, pad_h), (0, pad_w), (0, 0)), mode='reflect')
            return img[0:self.patch_size, 0:self.patch_size]
            
        return img[y:y+self.patch_size, x:x+self.patch_size]

    def _to_tensor(self, img: np.ndarray) -> torch.Tensor:
        img = img.astype(np.float32) / 127.5 - 1.0  # [-1, 1]
        return torch.from_numpy(img).permute(2, 0, 1)

    def _site_one_hot(self, site: str, n_sites: int = 3) -> torch.Tensor:
        idx = SITES.index(site)
        one_hot = torch.zeros(n_sites)
        one_hot[idx] = 1.0
        return one_hot

    def __getitem__(self, idx):
        site = random.choice(self.sites_list)
        entries = self.entries_by_site[site]

        # Seleccionar Referencia y Target
        ref_entry = random.choice(self.ref_candidates[site])
        for _ in range(20):
            target_entry = random.choice(entries)
            if abs(target_entry.norm_tide - ref_entry.norm_tide) > self.min_tide_diff:
                break

        # Cargar RGB y SCL
        ref_img = _load_image(ref_entry.rgb_path)
        target_img = _load_image(target_entry.rgb_path)
        target_scl = _load_scl(target_entry.scl_path) # Usamos la SCL del target para validar el parche

        # 1. Buscar coordenadas válidas usando la SCL (Mismo recorte para ambos)
        y, x = self._get_valid_crop_coords(target_scl)

        # 2. Extraer parches con las MISMAS coordenadas
        ref_patch = self._extract_and_pad(ref_img, y, x)
        target_patch = self._extract_and_pad(target_img, y, x)

        # 3. Aumentos Geométricos Sincronizados (Mismo flip para ambos)
        if self.augment and random.random() > 0.5:
            ref_patch = np.ascontiguousarray(ref_patch[:, ::-1, :])
            target_patch = np.ascontiguousarray(target_patch[:, ::-1, :])

        # 4. Aumentos de Color Independientes
        if self.augment:
            ref_patch = self._color_jitter(ref_patch)
            target_patch = self._color_jitter(target_patch)

        # Convertir a tensores
        ref_tensor = self._to_tensor(ref_patch)
        target_tensor = self._to_tensor(target_patch)

        condition = torch.cat([
            torch.tensor([target_entry.norm_tide]),
            self._site_one_hot(site, len(self.sites_list))
        ])

        return ref_tensor, target_tensor, condition