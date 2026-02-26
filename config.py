"""
GeoExtract v2 — Central Configuration
======================================
All paths are DYNAMIC: auto-detects Kaggle vs local environment.
No hardcoded API keys — set them via environment variables.
"""

import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Optional


# ─────────────────────────────────────────────
# 1. Environment Detection
# ─────────────────────────────────────────────
IS_KAGGLE = os.path.exists("/kaggle/working")


def _resolve_root() -> Path:
    """Return the workspace root depending on runtime."""
    if IS_KAGGLE:
        return Path("/kaggle/working")
    # Local dev: repo root (directory that contains this file)
    return Path(__file__).resolve().parent


PROJECT_ROOT = _resolve_root()


def _resolve_data_root() -> Path:
    """
    SpaceNet 7 dataset root.
    On Kaggle  → /kaggle/input/spacenet7  (added as a dataset)
    Locally    → PROJECT_ROOT / data      (symlink or download here)
    Override   → set env var SPACENET7_ROOT
    """
    env = os.environ.get("SPACENET7_ROOT")
    if env:
        return Path(env)
    if IS_KAGGLE:
        return Path("/kaggle/input/spacenet7")
    return PROJECT_ROOT / "data"


DATA_ROOT = _resolve_data_root()


# ─────────────────────────────────────────────
# 2. Output directories (always writable)
# ─────────────────────────────────────────────
OUTPUT_DIR       = PROJECT_ROOT / "outputs"
YOLO_DATA_DIR    = OUTPUT_DIR / "yolo_dataset"
VLM_DATA_DIR     = OUTPUT_DIR / "vlm_dataset"
CHECKPOINT_DIR   = OUTPUT_DIR / "checkpoints"
YOLO_CKPT_DIR    = CHECKPOINT_DIR / "yolo"
VLM_CKPT_DIR     = CHECKPOINT_DIR / "vlm"
EVAL_DIR         = OUTPUT_DIR / "evaluation"
EXPORT_DIR       = OUTPUT_DIR / "export"          # for FastAPI deployment

# Create all output dirs eagerly
for _d in [YOLO_DATA_DIR, VLM_DATA_DIR, YOLO_CKPT_DIR, VLM_CKPT_DIR, EVAL_DIR, EXPORT_DIR]:
    _d.mkdir(parents=True, exist_ok=True)


# ─────────────────────────────────────────────
# 3. Weights & Biases  (NO hardcoded keys)
# ─────────────────────────────────────────────
@dataclass
class WandbConfig:
    project: str = "GeoExtract-v2"
    entity: Optional[str] = None                     # set via WANDB_ENTITY env
    enabled: bool = True
    api_key: Optional[str] = field(default=None, repr=False)

    def __post_init__(self):
        # Pull from env — user sets WANDB_API_KEY before running
        self.api_key = os.environ.get("WANDB_API_KEY", self.api_key)
        self.entity = os.environ.get("WANDB_ENTITY", self.entity)
        if not self.api_key:
            print("[⚠ wandb] WANDB_API_KEY not set — logging disabled.")
            self.enabled = False


# ─────────────────────────────────────────────
# 4. SpaceNet 7 Dataset Config
# ─────────────────────────────────────────────
@dataclass
class DataConfig:
    root: Path = DATA_ROOT
    # SpaceNet 7 directory layout
    images_subdir: str = "train"                     # contains AOI folders
    geojson_subdir: str = "train"                    # same root, labels inside
    image_size: int = 640                            # resize target for YOLO
    val_split: float = 0.15                          # 15% validation
    seed: int = 42
    max_samples: Optional[int] = None                # None = use all; set for debugging
    # Augmentation toggles
    augment: bool = True
    rotation_limit: int = 30
    color_jitter: float = 0.3
    flip_prob: float = 0.5


# ─────────────────────────────────────────────
# 5. YOLO Training Config
# ─────────────────────────────────────────────
@dataclass
class YOLOConfig:
    model_variant: str = "yolo11n.pt"                # YOLOv11-nano
    epochs: int = 50
    batch_size: int = 16
    image_size: int = 640
    lr0: float = 1e-3                                # initial LR
    lrf: float = 0.01                                # final LR ratio
    patience: int = 10                               # early stopping
    save_period: int = 5                             # save every N epochs
    workers: int = 2
    device: str = "0"                                # GPU 0
    project: Path = YOLO_CKPT_DIR
    name: str = "building_detector"
    resume: bool = True                              # auto-resume
    # Checkpoint granularity (steps)
    checkpoint_every_n_steps: int = 500


# ─────────────────────────────────────────────
# 6. VLM Training Config
# ─────────────────────────────────────────────
@dataclass
class VLMConfig:
    # Model
    model_id: str = "Qwen/Qwen2-VL-2B-Instruct"
    # Quantization
    load_in_4bit: bool = True
    bnb_4bit_quant_type: str = "nf4"
    bnb_4bit_compute_dtype: str = "float16"
    # LoRA
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: List[str] = field(
        default_factory=lambda: ["q_proj", "v_proj", "k_proj", "o_proj"]
    )
    # Training
    epochs: int = 3
    batch_size: int = 2
    gradient_accumulation_steps: int = 8             # effective batch = 16
    learning_rate: float = 2e-4
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    lr_scheduler_type: str = "cosine"
    max_seq_length: int = 1024
    # Checkpointing
    save_steps: int = 500
    logging_steps: int = 50
    eval_steps: int = 500
    # Paths
    output_dir: Path = VLM_CKPT_DIR
    resume_from_checkpoint: bool = True              # auto-detect latest
    # Hardware
    fp16: bool = True
    bf16: bool = False                                # T4 supports bf16 via amp
    gradient_checkpointing: bool = True              # save VRAM
    device: str = "cuda:0"


# ─────────────────────────────────────────────
# 7. QA Generation Config
# ─────────────────────────────────────────────
@dataclass
class QAConfig:
    # Density classification thresholds (building count per image tile)
    sparse_max: int = 10
    moderate_max: int = 30
    dense_max: int = 50
    # Everything above dense_max → "Urban Core / Hyper-Dense"
    # Number of QA turns per image
    min_turns: int = 2
    max_turns: int = 4
    # System prompt for conversational template
    system_prompt: str = (
        "You are GeoExtract, an expert urban planning AI that analyzes "
        "satellite imagery. You provide detailed assessments of building "
        "density, green space coverage, urban heat island risk, and "
        "construction quality based on visual and spatial data."
    )


# ─────────────────────────────────────────────
# 8. Inference Config
# ─────────────────────────────────────────────
@dataclass
class InferenceConfig:
    yolo_weights: Path = YOLO_CKPT_DIR / "building_detector" / "weights" / "best.pt"
    vlm_adapter_dir: Path = VLM_CKPT_DIR
    confidence_threshold: float = 0.25
    iou_threshold: float = 0.45
    max_new_tokens: int = 512
    device: str = "cuda:0"


# ─────────────────────────────────────────────
# 9. Evaluation Config
# ─────────────────────────────────────────────
@dataclass
class EvalConfig:
    iou_threshold: float = 0.5                       # for mAP
    density_classes: List[str] = field(
        default_factory=lambda: ["Sparse", "Moderate", "Dense", "Urban Core"]
    )
    output_dir: Path = EVAL_DIR


# ─────────────────────────────────────────────
# 10. Master Config (aggregates everything)
# ─────────────────────────────────────────────
@dataclass
class GeoExtractConfig:
    data: DataConfig = field(default_factory=DataConfig)
    yolo: YOLOConfig = field(default_factory=YOLOConfig)
    vlm: VLMConfig = field(default_factory=VLMConfig)
    qa: QAConfig = field(default_factory=QAConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    evaluation: EvalConfig = field(default_factory=EvalConfig)
    wandb: WandbConfig = field(default_factory=WandbConfig)

    def summary(self) -> str:
        """Print a human-readable summary for notebook display."""
        lines = [
            "═" * 55,
            "  GeoExtract v2 — Configuration Summary",
            "═" * 55,
            f"  Environment    : {'Kaggle' if IS_KAGGLE else 'Local'}",
            f"  Project Root   : {PROJECT_ROOT}",
            f"  Data Root      : {DATA_ROOT}",
            f"  Output Dir     : {OUTPUT_DIR}",
            f"  W&B Enabled    : {self.wandb.enabled}",
            "─" * 55,
            f"  YOLO model     : {self.yolo.model_variant}",
            f"  YOLO epochs    : {self.yolo.epochs}",
            f"  YOLO batch     : {self.yolo.batch_size}",
            "─" * 55,
            f"  VLM model      : {self.vlm.model_id}",
            f"  VLM 4-bit      : {self.vlm.load_in_4bit}",
            f"  LoRA r/alpha   : {self.vlm.lora_r}/{self.vlm.lora_alpha}",
            f"  VLM epochs     : {self.vlm.epochs}",
            f"  VLM eff. batch : {self.vlm.batch_size * self.vlm.gradient_accumulation_steps}",
            "═" * 55,
        ]
        return "\n".join(lines)


# Singleton — import this everywhere
CFG = GeoExtractConfig()
