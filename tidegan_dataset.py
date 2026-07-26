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
        self.max_cloud_pct = max_cloud_pct
        self.max_nodata_pct = max_nodata_pct
        
        self.nodata_class = 0
        self.cloud_classes = [1, 3, 8, 9, 10]
        
        random.seed(seed)
        np.random.seed(seed)

        if sites is None: 
            sites = SITES

        self.entries_by_site = {}
        all_tides = []  # Lista temporal para guardar todas las mareas absolutas

        # 1. Cargar todos los datos y registrar las mareas
        for site in sites:
            csv_path = os.path.join(SITE_DIR, site, f"dataset_{site.lower()}.csv")
            rows = _read_csv(csv_path)
            # Ordenar las entradas de marea más baja a más alta
            entries = sorted([TideImageEntry(r, site) for r in rows], key=lambda e: e.tide)
            self.entries_by_site[site] = entries
            
            # Guardar las mareas para calcular los extremos globales
            all_tides.extend([e.tide for e in entries])

        # 2. Calcular extremos globales y normalizar a [-1, 1]
        self.global_min_tide = min(all_tides)
        self.global_max_tide = max(all_tides)
        print(f"Dataset cargado. Rango de marea global: [{self.global_min_tide}m, {self.global_max_tide}m]")

        for site, entries in self.entries_by_site.items():
            for entry in entries:
                entry.norm_tide = 2.0 * (entry.tide - self.global_min_tide) / (self.global_max_tide - self.global_min_tide) - 1.0

        # 3. Buscar UNA única imagen de referencia perfecta por sitio
        self.ref_image_by_site = {}
        for site, entries in self.entries_by_site.items():
            best_ref = None
            
            # Recorrer las imágenes empezando por la marea más baja
            for entry in entries:
                scl = _load_scl(entry.scl_path)
                cloud_ratio = np.mean(np.isin(scl, self.cloud_classes))
                nodata_ratio = np.mean(scl == self.nodata_class)
                
                # Tolerancia súper estricta para la referencia global de toda la imagen
                if cloud_ratio <= 0.01 and nodata_ratio <= 0.05:
                    best_ref = entry
                    print(f"[{site}] Referencia fija: {entry.date} (Marea: {entry.tide}m, Nubes: {cloud_ratio:.1%})")
                    break
            
            # Fallback de seguridad: si todas tienen nubes, cogemos la marea más baja
            if best_ref is None:
                print(f"[{site}] Aviso: No se encontró referencia sin nubes. Usando marea más baja por defecto.")
                best_ref = entries[0]
                
            self.ref_image_by_site[site] = best_ref

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

    def _get_valid_crop_coords(self, ref_scl: np.ndarray, target_scl: np.ndarray, max_retries=50):
        """
        Returns: (y, x, success_boolean, min_bad_ratio)
        """
        H, W = target_scl.shape
        if H <= self.patch_size or W <= self.patch_size:
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
        site = random.choice(self.sites_list)
        entries = self.entries_by_site[site]

        # 1. Coger la referencia única y limpia de este sitio
        ref_entry = self.ref_image_by_site[site]
        # Cargamos la máscara de referencia fuera del bucle de intentos (optimización de I/O)
        ref_scl = _load_scl(ref_entry.scl_path)

        best_overall_bad_ratio = float('inf')
        best_target_entry = None
        best_y, best_x = 0, 0

        # Intentar un máximo de 5 pares diferentes para evitar bucles infinitos en zonas muy nubladas
        for _ in range(5):
            # 2. Buscar un target con suficiente diferencia de marea
            for _ in range(20):
                target_entry = random.choice(entries)
                if abs(target_entry.norm_tide - ref_entry.norm_tide) > self.min_tide_diff:
                    break
                    
            # 3. Cargar la máscara del target
            target_scl = _load_scl(target_entry.scl_path)

            # 4. Buscar un recorte válido evaluando ambas máscaras juntas
            y, x, success, bad_ratio = self._get_valid_crop_coords(ref_scl, target_scl)

            # Si encontramos un parche perfecto (cumple todos los umbrales), paramos de buscar
            if success:
                best_target_entry = target_entry
                best_y, best_x = y, x
                break
                
            # Si no es perfecto, lo guardamos por si es el "menos malo" que hemos visto
            if bad_ratio < best_overall_bad_ratio:
                best_overall_bad_ratio = bad_ratio
                best_target_entry = target_entry
                best_y, best_x = y, x

        # 5. Cargar las imágenes RGB pesadas SOLO para el par ganador final
        ref_img = _load_image(ref_entry.rgb_path)
        target_img = _load_image(best_target_entry.rgb_path)

        # 6. Recortar y aplicar padding usando nuestras coordenadas ganadoras
        ref_patch = self._extract_and_pad(ref_img, best_y, best_x)
        target_patch = self._extract_and_pad(target_img, best_y, best_x)

        # 7. Aumentos espaciales sincronizados (flip horizontal)
        if self.augment and random.random() > 0.5:
            ref_patch = np.ascontiguousarray(ref_patch[:, ::-1, :])
            target_patch = np.ascontiguousarray(target_patch[:, ::-1, :])

        # 8. Aumentos de color independientes
        if self.augment:
            ref_patch = self._color_jitter(ref_patch)
            target_patch = self._color_jitter(target_patch)

        # 9. Convertir a tensores PyTorch
        ref_tensor = self._to_tensor(ref_patch)
        target_tensor = self._to_tensor(target_patch)
        tide = torch.tensor([best_target_entry.norm_tide], dtype=torch.float32)

        return ref_tensor, target_tensor, tide