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
SITE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

SITE_TIDE_RANGES = {
    "Foz": (-1.896, 1.186),
    "Santander": (-2.178, 0.908),
    "Villaviciosa": (-2.169, 1.013),
}

# ── Helpers ────────────────────────────────────────────────────────────────
def _read_csv(path):
    with open(path, "r", newline="", encoding="utf-8") as f:
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
        self.norm_tide = 0.0  # Se asigna luego globalmente
        
        self.rgb_path = os.path.join(SITE_DIR, site, row["imagen_rgb"])
        self.scl_path = os.path.join(SITE_DIR, site, row["imagen_scl"])
        
        # Contenedores para la caché en RAM
        self.rgb_data = None
        self.scl_data = None

    def load_into_ram(self):
        """Carga las imágenes en memoria y procesa la máscara SCL."""
        self.rgb_data = _load_image(self.rgb_path)
        self.scl_data = _load_scl(self.scl_path)
        if self.rgb_data.shape[:2] != self.scl_data.shape:
            raise ValueError(
                f"RGB y SCL tienen tamaños distintos para {self.rgb_path}: "
                f"{self.rgb_data.shape[:2]} != {self.scl_data.shape}"
            )

# ── Dataset ────────────────────────────────────────────────────────────────
class TideGANDataset(Dataset):
    def __init__(self, sites=None, patch_size=256, augment=True,
                 min_tide_diff=0.2, max_cloud_pct=0.01, max_nodata_pct=0.10,
                 split=None, split_mode="image", val_ratio=0.2, seed=42):
        self.patch_size = patch_size
        self.augment = augment
        self.min_tide_diff = min_tide_diff
        self.max_cloud_pct = max_cloud_pct
        self.max_nodata_pct = max_nodata_pct
        self.split = split
        self.split_mode = split_mode.lower() if split_mode is not None else None
        self.val_ratio = float(val_ratio)
        self.seed = seed

        self.nodata_class = 0
        self.cloud_classes = [1, 3, 8, 9, 10]

        random.seed(seed)
        np.random.seed(seed)

        if sites is None:
            sites = SITES

        self.entries_by_site = {}
        all_tides = []

        # 1. Parsear CSVs
        for site in sites:
            csv_path = os.path.join(SITE_DIR, site, f"dataset_{site.lower()}.csv")
            rows = _read_csv(csv_path)
            entries = sorted([TideImageEntry(r, site) for r in rows], key=lambda e: e.tide)
            entries = self._split_entries_for_site(entries, site)
            self.entries_by_site[site] = entries
            all_tides.extend([e.tide for e in entries])

        if not all_tides:
            raise ValueError(
                f"No se encontraron imágenes con datos de marea para el split solicitado "
                f"(split={self.split}, mode={self.split_mode}, val_ratio={self.val_ratio})."
            )

        # 2. Normalización global de mareas
        self.global_min_tide = min(all_tides)
        self.global_max_tide = max(all_tides)
        print(f"Dataset cargado. Rango global: [{self.global_min_tide}m, {self.global_max_tide}m]")

        # 3. CARGA MASIVA EN RAM
        print("Cargando todas las imágenes y máscaras en memoria RAM. Esto puede tardar un minuto...")
        for site, entries in self.entries_by_site.items():
            for entry in entries:
                # The model and inference normalize within each site's tide
                # range; training must use the same convention.
                entry.norm_tide = _normalize_tide(entry.tide, site)
                entry.load_into_ram()
        print("Carga en memoria completada. ¡Listo para entrenar a máxima velocidad!")

        # 4. Buscar UNA referencia perfecta por sitio (usando los datos ya en RAM)
        self.ref_image_by_site = {}
        for site, entries in self.entries_by_site.items():
            best_ref = None
            for entry in entries:
                scl = entry.scl_data
                cloud_ratio = np.mean(np.isin(scl, self.cloud_classes))

                if cloud_ratio <= self.max_cloud_pct:
                    best_ref = entry
                    print(f"[{site}] Referencia fija: {entry.date} (Marea: {entry.tide}m, Nubes: {cloud_ratio:.1%})")
                    break

            if best_ref is None:
                print(f"[{site}] Aviso: No se encontró referencia sin nubes. Usando marea más baja.")
                best_ref = entries[0]

            self.ref_image_by_site[site] = best_ref

        self.sites_list = list(self.entries_by_site.keys())

    def _split_entries_for_site(self, entries, site):
        """Split image-level data deterministically by tide values to avoid train/val leakage."""
        if self.split is None:
            return list(entries)

        if self.split_mode not in (None, "image"):
            raise ValueError(f"split_mode={self.split_mode!r} no está soportado. Usa 'image'.")

        if len(entries) <= 1:
            return list(entries) if self.split == "train" else []

        sorted_entries = sorted(entries, key=lambda e: e.tide)
        n_val = max(1, int(round(len(sorted_entries) * self.val_ratio)))
        n_val = min(n_val, len(sorted_entries) - 1)

        if n_val == 0:
            n_val = 1

        # Choose a few tide levels spread across the full range so both train and val
        # cover the tide spectrum instead of all the low/high values being in one split.
        val_positions = np.unique(np.linspace(0, len(sorted_entries) - 1, num=n_val, dtype=int))
        val_set = set(int(p) for p in val_positions)

        selected = [
            sorted_entries[i]
            for i in range(len(sorted_entries))
            if (i in val_set) == (self.split == "val")
        ]

        if not selected:
            raise ValueError(
                f"[{site}] El split {self.split} produjo un conjunto vacío con {len(entries)} imágenes. "
                f"Considera aumentar el número de imágenes o ajustar val_ratio."
            )

        return selected

    def __len__(self):
        return 1000

    def _color_jitter_pair(self, ref: np.ndarray, target: np.ndarray):
        """Aplica el mismo aumento fotométrico a las dos imágenes del par."""
        if random.random() > 0.5:
            factor = random.uniform(0.85, 1.15)
            ref = ref * factor
            target = target * factor
        if random.random() > 0.5:
            offset = random.uniform(-10, 10)
            ref = np.clip(ref + offset, 0, 255)
            target = np.clip(target + offset, 0, 255)
        return ref, target

    def _get_valid_crop_coords(self, ref_scl: np.ndarray, target_scl: np.ndarray, max_retries=50):
        """
        Returns: (y, x, success_boolean, min_bad_ratio)
        """
        H, W = target_scl.shape
        if H < self.patch_size or W < self.patch_size:
            return 0, 0, False, float('inf')

        best_y, best_x = 0, 0
        min_bad_ratio = float('inf')

        for _ in range(max_retries):
            y1 = random.randint(0, H - self.patch_size)
            x1 = random.randint(0, W - self.patch_size)
            
            patch_target = target_scl[y1:y1+self.patch_size, x1:x1+self.patch_size]
            patch_ref = ref_scl[y1:y1+self.patch_size, x1:x1+self.patch_size]
            
            # Calculate ratio of invalid pixels in BOTH images
            nodata_target = np.mean(patch_target == self.nodata_class)
            nodata_ref = np.mean(patch_ref == self.nodata_class)
            cloud_target = np.mean(np.isin(patch_target, self.cloud_classes))
            cloud_ref = np.mean(np.isin(patch_ref, self.cloud_classes))
            
            # Total bad ratio (lower is better)
            total_bad = nodata_target + nodata_ref + cloud_target + cloud_ref
            
            # Keep track of the best patch we've seen so far
            if total_bad < min_bad_ratio:
                min_bad_ratio = total_bad
                best_y, best_x = y1, x1
                
            # Reject if thresholds exceeded
            if nodata_target > self.max_nodata_pct or nodata_ref > self.max_nodata_pct:
                continue
            if cloud_target > self.max_cloud_pct or cloud_ref > self.max_cloud_pct:
                continue
                
            # Check for coastal dynamic elements
            has_water = np.any(patch_target == 6)
            has_land = np.any(np.isin(patch_target, [2, 4, 5, 7]))
            
            if has_water and has_land:
                return y1, x1, True, total_bad # Found a perfect patch!
            
        # Exhausted spatial retries, return the least bad one we found
        return best_y, best_x, False, min_bad_ratio

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
        for _ in range(10):
            site = random.choice(self.sites_list)
            entries = self.entries_by_site[site]

            # 1. Recuperar referencia desde RAM
            ref_entry = self.ref_image_by_site[site]
            ref_scl = ref_entry.scl_data

            # Select from valid tide-separated targets instead of silently
            # using the last random candidate when no candidate was found.
            candidates = [
                entry for entry in entries
                if abs(entry.norm_tide - ref_entry.norm_tide) > self.min_tide_diff
            ]
            if not candidates:
                continue
            target_entry = random.choice(candidates)
            target_scl = target_entry.scl_data

            # 3. Buscar parche válido (es ultra rápido porque solo recorta matrices en RAM)
            y, x, success, _ = self._get_valid_crop_coords(ref_scl, target_scl, max_retries=50)

            if success:
                # 4. Recuperar RGB desde RAM
                ref_img = ref_entry.rgb_data
                target_img = target_entry.rgb_data

                # Recortar y Pad
                ref_patch = self._extract_and_pad(ref_img, y, x)
                target_patch = self._extract_and_pad(target_img, y, x)

                # Aumentos
                if self.augment and random.random() > 0.5:
                    ref_patch = np.ascontiguousarray(ref_patch[:, ::-1, :])
                    target_patch = np.ascontiguousarray(target_patch[:, ::-1, :])

                if self.augment:
                    ref_patch, target_patch = self._color_jitter_pair(ref_patch, target_patch)

                # A tensores
                ref_tensor = self._to_tensor(ref_patch)
                target_tensor = self._to_tensor(target_patch)
                tide = torch.tensor([target_entry.norm_tide], dtype=torch.float32)
                condition = torch.cat([tide, self._site_one_hot(site, len(SITES))])

                return ref_tensor, target_tensor, condition

        raise RuntimeError(
            "No se pudo extraer un parche válido tras varios intentos. "
            "Revisa max_cloud_pct, max_nodata_pct y patch_size."
        )
