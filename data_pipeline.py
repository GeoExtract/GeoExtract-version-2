"""
GeoExtract v2 — Data Pipeline
===============================
Parses SpaceNet 7 (GeoTIFF + GeoJSON), converts to YOLO bbox format,
applies augmentations, and creates train/val splits.
"""

import json
import random
import shutil
from pathlib import Path
from typing import List, Tuple, Dict, Optional

import cv2
import numpy as np
import rasterio
from rasterio.windows import Window
from shapely.geometry import shape, box
import geopandas as gpd
import albumentations as A
from tqdm import tqdm

from config import CFG, YOLO_DATA_DIR
from utils import log, Timer


# ─────────────────────────────────────────────
# 1. SpaceNet 7 Parser
# ─────────────────────────────────────────────
class SpaceNet7Parser:
    """
    Reads SpaceNet 7 dataset structure:
        root/
          train/
            AOI_<id>_<name>/
              images/         ← monthly GeoTIFF images
              labels/         ← GeoJSON per month
              images_masked/  ← (optional)
    """

    def __init__(self, cfg=CFG.data):
        self.cfg = cfg
        self.root = Path(cfg.root)
        self.image_size = cfg.image_size
        self._samples: List[Dict] = []

    def discover(self) -> List[Dict]:
        """
        Walk the SpaceNet 7 directory tree and pair images with labels.
        Returns list of dicts: {image_path, label_path, aoi, timestamp}.
        """
        train_dir = self.root / self.cfg.images_subdir
        if not train_dir.exists():
            log.error(f"[Data] Train directory not found: {train_dir}")
            log.info(f"[Data] Searching for alternative structures under {self.root} ...")
            # Fallback: flat structure or different naming
            train_dir = self.root
            if not any(train_dir.iterdir()):
                raise FileNotFoundError(f"No data found at {self.root}")

        samples = []
        aoi_dirs = sorted([d for d in train_dir.iterdir() if d.is_dir()])

        for aoi_dir in aoi_dirs:
            aoi_name = aoi_dir.name
            images_dir = aoi_dir / "images"
            labels_dir = aoi_dir / "labels"

            # Alternative structure: images_masked
            if not images_dir.exists():
                images_dir = aoi_dir / "images_masked"
            if not images_dir.exists():
                log.warning(f"[Data] No images dir in {aoi_dir}, skipping.")
                continue

            # Collect all tif images
            tif_files = sorted(images_dir.glob("*.tif"))
            for tif_path in tif_files:
                # Find matching GeoJSON label
                # SpaceNet 7 naming: global_monthly_YYYY_MM_mosaic_<aoi>.tif
                # Label naming: global_monthly_YYYY_MM_mosaic_<aoi>.geojson
                stem = tif_path.stem
                label_candidates = [
                    labels_dir / f"{stem}.geojson",
                    labels_dir / f"{stem}_Buildings.geojson",
                    labels_dir / f"Buildings_{stem}.geojson",
                ]
                label_path = None
                for lc in label_candidates:
                    if lc.exists():
                        label_path = lc
                        break

                # Also try: search for any geojson in labels dir with similar name
                if label_path is None and labels_dir.exists():
                    geojsons = list(labels_dir.glob("*.geojson"))
                    # Try partial matching
                    for gj in geojsons:
                        if stem in gj.stem or gj.stem in stem:
                            label_path = gj
                            break

                samples.append({
                    "image_path": tif_path,
                    "label_path": label_path,
                    "aoi": aoi_name,
                    "timestamp": stem,
                    "has_labels": label_path is not None,
                })

        # Apply max_samples limit for debugging
        if self.cfg.max_samples and len(samples) > self.cfg.max_samples:
            random.seed(self.cfg.seed)
            samples = random.sample(samples, self.cfg.max_samples)

        self._samples = samples
        log.info(f"[Data] Discovered {len(samples)} image-label pairs "
                 f"across {len(aoi_dirs)} AOIs.")
        labeled = sum(1 for s in samples if s["has_labels"])
        log.info(f"[Data] {labeled}/{len(samples)} have GeoJSON labels.")
        return samples

    def read_geotiff(self, path: Path) -> np.ndarray:
        """
        Read a GeoTIFF and return as (H, W, C) uint8 numpy array.
        Handles 3-band and 4-band (drops alpha) images.
        """
        with rasterio.open(path) as src:
            # Read all bands → (C, H, W)
            img = src.read()
            transform = src.transform
            crs = src.crs

        # Transpose to (H, W, C)
        img = np.transpose(img, (1, 2, 0))

        # Keep only RGB (first 3 channels)
        if img.shape[2] > 3:
            img = img[:, :, :3]

        # Normalize to uint8 if needed
        if img.dtype != np.uint8:
            # Clip to 0-255 range (SpaceNet images can have wider range)
            img = np.clip(img, 0, 255).astype(np.uint8)

        return img

    def read_geojson(self, path: Path) -> List[Dict]:
        """
        Parse a GeoJSON file and return list of building features
        with shapely geometries.
        """
        if path is None or not path.exists():
            return []

        try:
            gdf = gpd.read_file(path)
            buildings = []
            for _, row in gdf.iterrows():
                geom = row.geometry
                if geom is not None and geom.is_valid:
                    buildings.append({
                        "geometry": geom,
                        "bounds": geom.bounds,     # (minx, miny, maxx, maxy)
                        "area": geom.area,
                        "properties": {k: v for k, v in row.items()
                                       if k != "geometry"},
                    })
            return buildings
        except Exception as e:
            log.warning(f"[Data] Failed to read {path}: {e}")
            return []

    def get_image_metadata(self, path: Path) -> Dict:
        """Get spatial metadata from a GeoTIFF (CRS, transform, bounds)."""
        with rasterio.open(path) as src:
            return {
                "crs": str(src.crs),
                "transform": src.transform,
                "bounds": src.bounds,
                "width": src.width,
                "height": src.height,
            }


# ─────────────────────────────────────────────
# 2. GeoJSON → YOLO Bounding Box Converter
# ─────────────────────────────────────────────
class YOLOFormatConverter:
    """
    Converts GeoJSON polygon annotations to YOLO bounding box format.
    YOLO format: <class_id> <x_center> <y_center> <width> <height>
    All values normalized to [0, 1].
    """

    CLASS_BUILDING = 0  # Single-class detection

    def __init__(self, image_size: int = 640):
        self.image_size = image_size

    def polygon_to_yolo_bbox(
        self,
        geometry,
        img_width: int,
        img_height: int,
        geo_transform=None,
    ) -> Optional[Tuple[int, float, float, float, float]]:
        """
        Convert a shapely polygon to YOLO bbox format.
        If geo_transform is provided, convert from geo coords to pixel coords first.
        """
        minx, miny, maxx, maxy = geometry.bounds

        if geo_transform is not None:
            # Convert geographic coordinates to pixel coordinates
            from rasterio.transform import rowcol
            row_min, col_min = rowcol(geo_transform, minx, maxy)  # top-left
            row_max, col_max = rowcol(geo_transform, maxx, miny)  # bottom-right

            # Ensure correct ordering
            px_xmin = max(0, min(col_min, col_max))
            px_ymin = max(0, min(row_min, row_max))
            px_xmax = min(img_width, max(col_min, col_max))
            px_ymax = min(img_height, max(row_min, row_max))
        else:
            # Already in pixel coordinates
            px_xmin = max(0, minx)
            px_ymin = max(0, miny)
            px_xmax = min(img_width, maxx)
            px_ymax = min(img_height, maxy)

        # Calculate YOLO normalized format
        bw = px_xmax - px_xmin
        bh = px_ymax - px_ymin

        # Skip degenerate boxes
        if bw <= 2 or bh <= 2:
            return None

        x_center = (px_xmin + bw / 2) / img_width
        y_center = (px_ymin + bh / 2) / img_height
        w_norm = bw / img_width
        h_norm = bh / img_height

        # Clamp to [0, 1]
        x_center = np.clip(x_center, 0.0, 1.0)
        y_center = np.clip(y_center, 0.0, 1.0)
        w_norm = np.clip(w_norm, 0.0, 1.0)
        h_norm = np.clip(h_norm, 0.0, 1.0)

        return (self.CLASS_BUILDING, x_center, y_center, w_norm, h_norm)

    def convert_sample(
        self,
        image_path: Path,
        buildings: List[Dict],
        geo_transform=None,
    ) -> List[Tuple[int, float, float, float, float]]:
        """Convert all buildings in one image to YOLO format."""
        # Get image dimensions
        with rasterio.open(image_path) as src:
            img_w, img_h = src.width, src.height
            if geo_transform is None:
                geo_transform = src.transform

        bboxes = []
        for bld in buildings:
            bbox = self.polygon_to_yolo_bbox(
                bld["geometry"], img_w, img_h, geo_transform
            )
            if bbox is not None:
                bboxes.append(bbox)
        return bboxes


# ─────────────────────────────────────────────
# 3. Augmentation Pipeline
# ─────────────────────────────────────────────
def build_augmentation_pipeline(cfg=CFG.data) -> A.Compose:
    """
    Build an Albumentations pipeline compatible with YOLO bbox format.
    """
    transforms = []
    if cfg.augment:
        transforms.extend([
            A.HorizontalFlip(p=cfg.flip_prob),
            A.VerticalFlip(p=cfg.flip_prob * 0.5),
            A.RandomRotate90(p=0.3),
            A.Rotate(limit=cfg.rotation_limit, p=0.4, border_mode=cv2.BORDER_CONSTANT),
            A.ColorJitter(
                brightness=cfg.color_jitter,
                contrast=cfg.color_jitter,
                saturation=cfg.color_jitter * 0.5,
                hue=cfg.color_jitter * 0.2,
                p=0.5,
            ),
            A.RandomBrightnessContrast(p=0.3),
            A.GaussNoise(p=0.1),
        ])
    # Always resize to target size
    transforms.append(A.Resize(cfg.image_size, cfg.image_size))

    return A.Compose(
        transforms,
        bbox_params=A.BboxParams(
            format="yolo",
            label_fields=["class_labels"],
            min_visibility=0.3,
        ),
    )


# ─────────────────────────────────────────────
# 4. Full Dataset Builder
# ─────────────────────────────────────────────
class YOLODatasetBuilder:
    """
    End-to-end: SpaceNet 7 → YOLO-format dataset directory.
    Creates:
        yolo_dataset/
          images/
            train/
            val/
          labels/
            train/
            val/
          dataset.yaml
    """

    def __init__(self, cfg=CFG):
        self.cfg = cfg
        self.parser = SpaceNet7Parser(cfg.data)
        self.converter = YOLOFormatConverter(cfg.data.image_size)
        self.augmenter = build_augmentation_pipeline(cfg.data)
        self.output_dir = YOLO_DATA_DIR

    def build(self) -> Path:
        """Main entry point — builds the entire YOLO dataset."""
        with Timer("YOLO Dataset Build"):
            # 1. Discover samples
            samples = self.parser.discover()
            labeled_samples = [s for s in samples if s["has_labels"]]

            if not labeled_samples:
                log.error("[Data] No labeled samples found! Check dataset path.")
                raise ValueError(f"No labeled samples in {self.cfg.data.root}")

            # 2. Train/val split
            random.seed(self.cfg.data.seed)
            random.shuffle(labeled_samples)
            split_idx = int(len(labeled_samples) * (1 - self.cfg.data.val_split))
            train_samples = labeled_samples[:split_idx]
            val_samples = labeled_samples[split_idx:]

            log.info(f"[Data] Split: {len(train_samples)} train, {len(val_samples)} val")

            # 3. Create directory structure
            for split in ["train", "val"]:
                (self.output_dir / "images" / split).mkdir(parents=True, exist_ok=True)
                (self.output_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

            # 4. Process each split
            self._process_split(train_samples, "train", augment=True)
            self._process_split(val_samples, "val", augment=False)

            # 5. Write dataset YAML
            yaml_path = self._write_dataset_yaml()

            log.info(f"[Data] ✓ YOLO dataset ready at {self.output_dir}")
            return yaml_path

    def _process_split(self, samples: List[Dict], split: str, augment: bool) -> None:
        """Convert and save all samples for one split."""
        img_dir = self.output_dir / "images" / split
        lbl_dir = self.output_dir / "labels" / split

        for sample in tqdm(samples, desc=f"Processing {split}"):
            try:
                # Read image
                img = self.parser.read_geotiff(sample["image_path"])
                # Read labels
                buildings = self.parser.read_geojson(sample["label_path"])

                # Convert to YOLO bboxes
                meta = self.parser.get_image_metadata(sample["image_path"])
                bboxes = self.converter.convert_sample(
                    sample["image_path"], buildings, meta.get("transform")
                )

                if not bboxes:
                    continue  # skip images with no valid annotations

                # Prepare for augmentation
                yolo_bboxes = [(b[1], b[2], b[3], b[4]) for b in bboxes]
                class_labels = [b[0] for b in bboxes]

                # Apply augmentation (training only)
                if augment and self.cfg.data.augment:
                    try:
                        augmented = self.augmenter(
                            image=img,
                            bboxes=yolo_bboxes,
                            class_labels=class_labels,
                        )
                        img = augmented["image"]
                        yolo_bboxes = augmented["bboxes"]
                        class_labels = augmented["class_labels"]
                    except Exception as e:
                        # Fallback: just resize without bbox-aware augmentation
                        img = cv2.resize(img, (self.cfg.data.image_size, self.cfg.data.image_size))
                else:
                    img = cv2.resize(img, (self.cfg.data.image_size, self.cfg.data.image_size))

                if not yolo_bboxes:
                    continue  # augmentation removed all boxes

                # Save image as PNG
                stem = f"{sample['aoi']}_{sample['timestamp']}"
                img_path = img_dir / f"{stem}.png"
                cv2.imwrite(str(img_path), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

                # Save YOLO label txt
                lbl_path = lbl_dir / f"{stem}.txt"
                with open(lbl_path, "w") as f:
                    for cls_id, bbox in zip(class_labels, yolo_bboxes):
                        f.write(f"{cls_id} {bbox[0]:.6f} {bbox[1]:.6f} "
                                f"{bbox[2]:.6f} {bbox[3]:.6f}\n")

            except Exception as e:
                log.warning(f"[Data] Failed to process {sample['image_path'].name}: {e}")
                continue

    def _write_dataset_yaml(self) -> Path:
        """Generate the YOLO dataset.yaml config file."""
        yaml_content = {
            "path": str(self.output_dir),
            "train": "images/train",
            "val": "images/val",
            "nc": 1,
            "names": ["building"],
        }

        yaml_path = self.output_dir / "dataset.yaml"
        import yaml
        with open(yaml_path, "w") as f:
            yaml.dump(yaml_content, f, default_flow_style=False)

        log.info(f"[Data] Dataset YAML written to {yaml_path}")
        return yaml_path

    def get_stats(self) -> Dict:
        """Return dataset statistics after build."""
        stats = {}
        for split in ["train", "val"]:
            img_dir = self.output_dir / "images" / split
            lbl_dir = self.output_dir / "labels" / split
            n_images = len(list(img_dir.glob("*.png")))
            n_labels = len(list(lbl_dir.glob("*.txt")))

            # Count total bounding boxes
            total_boxes = 0
            for lbl_file in lbl_dir.glob("*.txt"):
                with open(lbl_file) as f:
                    total_boxes += len(f.readlines())

            stats[split] = {
                "images": n_images,
                "labels": n_labels,
                "total_bboxes": total_boxes,
                "avg_bboxes_per_image": round(total_boxes / max(n_images, 1), 1),
            }
        return stats
