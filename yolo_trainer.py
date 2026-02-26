"""
GeoExtract v2 — YOLO Trainer
==============================
YOLOv11-nano training for building footprint detection.
Includes bulletproof checkpointing and W&B integration.
"""

import os
import shutil
from pathlib import Path
from typing import Optional, Dict

import torch

from config import CFG, YOLO_CKPT_DIR
from utils import log, Timer, free_vram, log_vram, init_wandb, finish_wandb


class YOLOTrainer:
    """
    Wraps Ultralytics YOLO for building detection training.
    - Auto-downloads YOLOv11-nano pretrained weights
    - Trains on converted SpaceNet 7 data
    - Saves checkpoints compatible with Kaggle timeouts
    - Logs to Weights & Biases
    """

    def __init__(self, dataset_yaml: Path, cfg=CFG):
        self.cfg = cfg
        self.yolo_cfg = cfg.yolo
        self.dataset_yaml = dataset_yaml
        self.model = None
        self._wandb_run = None

    def train(self) -> Path:
        """
        Main training entry point.
        Returns path to best weights.
        """
        from ultralytics import YOLO

        with Timer("YOLO Training"):
            log_vram("before YOLO load")

            # ── 1. Initialize or resume model ──
            resume_weights = self._find_resume_weights()
            if resume_weights and self.yolo_cfg.resume:
                log.info(f"[YOLO] Resuming from: {resume_weights}")
                self.model = YOLO(str(resume_weights))
            else:
                log.info(f"[YOLO] Starting fresh with {self.yolo_cfg.model_variant}")
                self.model = YOLO(self.yolo_cfg.model_variant)

            log_vram("after YOLO load")

            # ── 2. Initialize W&B ──
            self._wandb_run = init_wandb(
                self.cfg, run_name="yolo-building-detector", tags=["yolo", "training"]
            )

            # ── 3. Train ──
            try:
                results = self.model.train(
                    data=str(self.dataset_yaml),
                    epochs=self.yolo_cfg.epochs,
                    batch=self.yolo_cfg.batch_size,
                    imgsz=self.yolo_cfg.image_size,
                    lr0=self.yolo_cfg.lr0,
                    lrf=self.yolo_cfg.lrf,
                    patience=self.yolo_cfg.patience,
                    save_period=self.yolo_cfg.save_period,
                    workers=self.yolo_cfg.workers,
                    device=self.yolo_cfg.device,
                    project=str(self.yolo_cfg.project),
                    name=self.yolo_cfg.name,
                    exist_ok=True,
                    pretrained=True,
                    verbose=True,
                    # Augmentations (Ultralytics built-in)
                    hsv_h=0.015,
                    hsv_s=0.4,
                    hsv_v=0.3,
                    flipud=0.3,
                    fliplr=0.5,
                    mosaic=0.8,
                    mixup=0.1,
                    # Save & logging
                    plots=True,
                    val=True,
                )
            except KeyboardInterrupt:
                log.warning("[YOLO] Training interrupted. Weights are saved.")
            finally:
                finish_wandb()

            # ── 4. Return best weights path ──
            best_weights = self._get_best_weights()
            log.info(f"[YOLO] ✓ Best weights: {best_weights}")
            log_vram("after YOLO training")

            return best_weights

    def validate(self) -> Dict:
        """Run validation and return metrics."""
        from ultralytics import YOLO

        best = self._get_best_weights()
        if not best.exists():
            log.error("[YOLO] No trained weights found for validation.")
            return {}

        model = YOLO(str(best))
        results = model.val(
            data=str(self.dataset_yaml),
            batch=self.yolo_cfg.batch_size,
            imgsz=self.yolo_cfg.image_size,
            device=self.yolo_cfg.device,
        )

        metrics = {
            "mAP50": results.box.map50 if hasattr(results.box, 'map50') else 0.0,
            "mAP50-95": results.box.map if hasattr(results.box, 'map') else 0.0,
            "precision": results.box.mp if hasattr(results.box, 'mp') else 0.0,
            "recall": results.box.mr if hasattr(results.box, 'mr') else 0.0,
            "f1": (
                2 * results.box.mp * results.box.mr /
                max(results.box.mp + results.box.mr, 1e-6)
                if hasattr(results.box, 'mp') else 0.0
            ),
        }
        log.info(f"[YOLO] Validation metrics: {metrics}")
        return metrics

    def export_for_deployment(self, format: str = "onnx") -> Path:
        """Export model for production deployment (FastAPI serving)."""
        from ultralytics import YOLO

        best = self._get_best_weights()
        model = YOLO(str(best))
        export_path = model.export(format=format, imgsz=self.yolo_cfg.image_size)
        log.info(f"[YOLO] Exported to {format}: {export_path}")
        return Path(export_path)

    # ─── Private helpers ───

    def _find_resume_weights(self) -> Optional[Path]:
        """Find the latest checkpoint for resuming training."""
        run_dir = self.yolo_cfg.project / self.yolo_cfg.name
        last_weights = run_dir / "weights" / "last.pt"
        if last_weights.exists():
            return last_weights

        # Check for numbered checkpoints
        weights_dir = run_dir / "weights"
        if weights_dir.exists():
            pts = sorted(weights_dir.glob("epoch*.pt"))
            if pts:
                return pts[-1]
        return None

    def _get_best_weights(self) -> Path:
        """Return path to best.pt from the latest run."""
        return self.yolo_cfg.project / self.yolo_cfg.name / "weights" / "best.pt"

    def cleanup(self) -> None:
        """Free YOLO model from GPU memory."""
        if self.model is not None:
            del self.model
            self.model = None
        free_vram()
        log.info("[YOLO] Model unloaded, VRAM freed.")
