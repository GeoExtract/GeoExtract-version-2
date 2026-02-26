"""
GeoExtract v2 — Utility Helpers
================================
VRAM monitoring, checkpoint management, GPU cleanup, logging.
"""

import os
import gc
import time
import logging
from pathlib import Path
from typing import Optional

import torch


# ─────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────
def setup_logger(name: str = "geoextract", level: int = logging.INFO) -> logging.Logger:
    """Create a consistent logger for all modules."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        fmt = logging.Formatter(
            "[%(asctime)s] %(levelname)s — %(name)s — %(message)s",
            datefmt="%H:%M:%S",
        )
        handler.setFormatter(fmt)
        logger.addHandler(handler)
    logger.setLevel(level)
    return logger


log = setup_logger()


# ─────────────────────────────────────────────
# VRAM Monitoring
# ─────────────────────────────────────────────
def get_vram_usage() -> dict:
    """Return current VRAM usage in GB (allocated / reserved / total)."""
    if not torch.cuda.is_available():
        return {"allocated_gb": 0, "reserved_gb": 0, "total_gb": 0}
    return {
        "allocated_gb": round(torch.cuda.memory_allocated() / 1e9, 2),
        "reserved_gb": round(torch.cuda.memory_reserved() / 1e9, 2),
        "total_gb": round(torch.cuda.get_device_properties(0).total_memory / 1e9, 2),
    }


def log_vram(tag: str = "") -> None:
    """Log VRAM usage with an optional tag."""
    v = get_vram_usage()
    log.info(
        f"[VRAM {tag}] Allocated: {v['allocated_gb']} GB | "
        f"Reserved: {v['reserved_gb']} GB | Total: {v['total_gb']} GB"
    )


def free_vram() -> None:
    """Aggressively free GPU memory between training phases."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    log.info("[VRAM] Cache cleared.")
    log_vram("after cleanup")


# ─────────────────────────────────────────────
# Checkpoint Management
# ─────────────────────────────────────────────
def find_latest_checkpoint(ckpt_dir: Path, prefix: str = "checkpoint-") -> Optional[Path]:
    """
    Scan a directory for HuggingFace-style checkpoint folders
    (e.g., checkpoint-500, checkpoint-1000) and return the latest one.
    Returns None if no checkpoints exist.
    """
    if not ckpt_dir.exists():
        return None

    ckpts = sorted(
        [d for d in ckpt_dir.iterdir() if d.is_dir() and d.name.startswith(prefix)],
        key=lambda p: int(p.name.split("-")[-1]) if p.name.split("-")[-1].isdigit() else 0,
    )
    if ckpts:
        log.info(f"[Checkpoint] Found {len(ckpts)} checkpoints. Latest: {ckpts[-1].name}")
        return ckpts[-1]
    return None


def count_parameters(model: torch.nn.Module) -> dict:
    """Count total and trainable parameters."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        "total": total,
        "trainable": trainable,
        "trainable_pct": round(100 * trainable / total, 2) if total > 0 else 0,
    }


# ─────────────────────────────────────────────
# Timer Context Manager
# ─────────────────────────────────────────────
class Timer:
    """Simple context manager for timing code blocks."""

    def __init__(self, label: str = "Block"):
        self.label = label
        self.start = 0.0
        self.elapsed = 0.0

    def __enter__(self):
        self.start = time.time()
        return self

    def __exit__(self, *args):
        self.elapsed = time.time() - self.start
        log.info(f"[Timer] {self.label} took {self.elapsed:.1f}s")


# ─────────────────────────────────────────────
# W&B Initialization Helper
# ─────────────────────────────────────────────
def init_wandb(config, run_name: str, tags: Optional[list] = None):
    """
    Safely initialize a W&B run.
    Returns the run object, or None if W&B is disabled / unavailable.
    """
    if not config.wandb.enabled:
        log.warning("[wandb] Disabled — skipping init.")
        return None
    try:
        import wandb
        os.environ["WANDB_API_KEY"] = config.wandb.api_key
        run = wandb.init(
            project=config.wandb.project,
            entity=config.wandb.entity,
            name=run_name,
            tags=tags or [],
            config={
                "yolo_model": config.yolo.model_variant,
                "vlm_model": config.vlm.model_id,
                "lora_r": config.vlm.lora_r,
            },
            reinit=True,
        )
        log.info(f"[wandb] Run '{run_name}' initialized.")
        return run
    except Exception as e:
        log.error(f"[wandb] Init failed: {e}. Continuing without logging.")
        return None


def finish_wandb():
    """Safely finish the current W&B run."""
    try:
        import wandb
        if wandb.run is not None:
            wandb.finish()
    except Exception:
        pass
