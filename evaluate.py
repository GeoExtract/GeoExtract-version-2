"""
GeoExtract v2 — Evaluation Module
====================================
Computes hard metrics for defense panel:
  - YOLO: Precision, Recall, F1-Score, mAP (IoU-based)
  - VLM:  Density classification accuracy, confusion matrix, F1
Logs everything to Weights & Biases.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    accuracy_score,
)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import CFG, EVAL_DIR, VLM_DATA_DIR
from utils import log, Timer, init_wandb, finish_wandb


# ─────────────────────────────────────────────
# 1. YOLO Detection Evaluator
# ─────────────────────────────────────────────
class YOLOEvaluator:
    """
    Evaluates YOLO building detection using IoU-based matching.
    Computes Precision, Recall, F1, and mAP at various IoU thresholds.
    """

    def __init__(self, iou_threshold: float = 0.5):
        self.iou_threshold = iou_threshold

    def evaluate(self, dataset_yaml: Path, weights_path: Path) -> Dict:
        """Run full YOLO evaluation using Ultralytics val()."""
        from ultralytics import YOLO

        log.info(f"[Eval-YOLO] Running validation with IoU={self.iou_threshold}...")
        model = YOLO(str(weights_path))
        results = model.val(
            data=str(dataset_yaml),
            iou=self.iou_threshold,
            conf=0.25,
            verbose=True,
            plots=True,
            save_dir=str(EVAL_DIR / "yolo_eval"),
        )

        # Extract metrics
        metrics = {
            "precision": float(results.box.mp) if hasattr(results.box, 'mp') else 0.0,
            "recall": float(results.box.mr) if hasattr(results.box, 'mr') else 0.0,
            "mAP50": float(results.box.map50) if hasattr(results.box, 'map50') else 0.0,
            "mAP50-95": float(results.box.map) if hasattr(results.box, 'map') else 0.0,
        }

        # Compute F1 from P and R
        p, r = metrics["precision"], metrics["recall"]
        metrics["f1"] = 2 * p * r / max(p + r, 1e-8)

        log.info(f"[Eval-YOLO] Results:")
        for k, v in metrics.items():
            log.info(f"  {k}: {v:.4f}")

        # Save to file
        self._save_metrics(metrics, "yolo_metrics.json")

        return metrics

    def evaluate_counting_accuracy(
        self,
        predictions: List[int],
        ground_truths: List[int],
    ) -> Dict:
        """
        Evaluate building counting accuracy.
        Useful for comparing YOLO counts vs ground truth counts.
        """
        predictions = np.array(predictions)
        ground_truths = np.array(ground_truths)

        # Counting metrics
        mae = float(np.mean(np.abs(predictions - ground_truths)))
        rmse = float(np.sqrt(np.mean((predictions - ground_truths) ** 2)))
        mape = float(np.mean(
            np.abs(predictions - ground_truths) / np.maximum(ground_truths, 1)
        ) * 100)

        # Exact match accuracy (within tolerance)
        exact_match = float(np.mean(predictions == ground_truths))
        within_5 = float(np.mean(np.abs(predictions - ground_truths) <= 5))
        within_10 = float(np.mean(np.abs(predictions - ground_truths) <= 10))

        metrics = {
            "mae": mae,
            "rmse": rmse,
            "mape_pct": mape,
            "exact_match": exact_match,
            "within_5": within_5,
            "within_10": within_10,
        }

        log.info(f"[Eval-YOLO] Counting accuracy:")
        for k, v in metrics.items():
            log.info(f"  {k}: {v:.4f}")

        self._save_metrics(metrics, "yolo_counting_metrics.json")
        return metrics

    def _save_metrics(self, metrics: Dict, filename: str) -> None:
        out_path = EVAL_DIR / filename
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(metrics, f, indent=2)
        log.info(f"[Eval-YOLO] Metrics saved to {out_path}")


# ─────────────────────────────────────────────
# 2. VLM Density Classification Evaluator
# ─────────────────────────────────────────────
class VLMEvaluator:
    """
    Evaluates VLM on density classification task.
    Ground truth comes from GeoJSON building counts.
    Predictions come from VLM text output (parsed for density class).
    """

    DENSITY_CLASSES = ["Sparse", "Moderate", "Dense", "Urban Core"]

    def __init__(self, cfg=CFG):
        self.cfg = cfg
        self.class_to_idx = {c: i for i, c in enumerate(self.DENSITY_CLASSES)}

    def evaluate(
        self,
        gt_classes: List[str],
        pred_classes: List[str],
    ) -> Dict:
        """
        Compute classification metrics for density prediction.

        Args:
            gt_classes: Ground truth density class names.
            pred_classes: Predicted density class names from VLM.
        """
        log.info(f"[Eval-VLM] Evaluating {len(gt_classes)} samples...")

        # Convert to indices
        gt_idx = [self.class_to_idx.get(c, -1) for c in gt_classes]
        pred_idx = [self.class_to_idx.get(c, -1) for c in pred_classes]

        # Filter out unknowns
        valid = [(g, p) for g, p in zip(gt_idx, pred_idx) if g >= 0 and p >= 0]
        if not valid:
            log.error("[Eval-VLM] No valid predictions to evaluate!")
            return {}

        gt_valid = [v[0] for v in valid]
        pred_valid = [v[1] for v in valid]

        # Compute metrics
        metrics = {
            "accuracy": float(accuracy_score(gt_valid, pred_valid)),
            "precision_macro": float(precision_score(
                gt_valid, pred_valid, average="macro", zero_division=0
            )),
            "recall_macro": float(recall_score(
                gt_valid, pred_valid, average="macro", zero_division=0
            )),
            "f1_macro": float(f1_score(
                gt_valid, pred_valid, average="macro", zero_division=0
            )),
            "precision_weighted": float(precision_score(
                gt_valid, pred_valid, average="weighted", zero_division=0
            )),
            "recall_weighted": float(recall_score(
                gt_valid, pred_valid, average="weighted", zero_division=0
            )),
            "f1_weighted": float(f1_score(
                gt_valid, pred_valid, average="weighted", zero_division=0
            )),
        }

        # Per-class metrics
        report = classification_report(
            gt_valid, pred_valid,
            target_names=self.DENSITY_CLASSES,
            output_dict=True,
            zero_division=0,
        )
        metrics["per_class"] = {
            cls: {
                "precision": report[cls]["precision"],
                "recall": report[cls]["recall"],
                "f1": report[cls]["f1-score"],
                "support": report[cls]["support"],
            }
            for cls in self.DENSITY_CLASSES
            if cls in report
        }

        # Confusion matrix
        cm = confusion_matrix(gt_valid, pred_valid, labels=list(range(len(self.DENSITY_CLASSES))))
        metrics["confusion_matrix"] = cm.tolist()

        log.info(f"[Eval-VLM] Results:")
        log.info(f"  Accuracy:     {metrics['accuracy']:.4f}")
        log.info(f"  F1 (macro):   {metrics['f1_macro']:.4f}")
        log.info(f"  F1 (weighted):{metrics['f1_weighted']:.4f}")

        # Print full report
        print("\n" + classification_report(
            gt_valid, pred_valid,
            target_names=self.DENSITY_CLASSES,
            zero_division=0,
        ))

        # Save
        self._save_metrics(metrics, "vlm_metrics.json")
        self._plot_confusion_matrix(cm, "vlm_confusion_matrix.png")

        return metrics

    def evaluate_from_pipeline(
        self,
        pipeline,
        val_jsonl: Optional[Path] = None,
        max_samples: int = 100,
    ) -> Dict:
        """
        End-to-end evaluation: run pipeline on validation set,
        extract density predictions, and compare to ground truth.
        """
        if val_jsonl is None:
            val_jsonl = VLM_DATA_DIR / "val.jsonl"

        if not val_jsonl.exists():
            log.error(f"[Eval-VLM] Validation data not found: {val_jsonl}")
            return {}

        # Load ground truth
        with open(val_jsonl) as f:
            val_data = [json.loads(l) for l in f.readlines()[:max_samples]]

        gt_classes = []
        pred_classes = []

        from tqdm import tqdm
        for sample in tqdm(val_data, desc="VLM Evaluation"):
            gt_class = sample.get("density_class", "Unknown")
            image_path = sample.get("image", "")

            if not Path(image_path).exists():
                continue

            try:
                result = pipeline.analyze(image_path)
                pred_class = result.get("context", {}).get("density_class", "Unknown")
                gt_classes.append(gt_class)
                pred_classes.append(pred_class)
            except Exception as e:
                log.warning(f"[Eval-VLM] Failed on {image_path}: {e}")
                continue

        return self.evaluate(gt_classes, pred_classes)

    def _save_metrics(self, metrics: Dict, filename: str) -> None:
        out_path = EVAL_DIR / filename
        out_path.parent.mkdir(parents=True, exist_ok=True)
        # Remove non-serializable items
        serializable = {
            k: v for k, v in metrics.items()
            if not isinstance(v, np.ndarray)
        }
        with open(out_path, "w") as f:
            json.dump(serializable, f, indent=2, default=str)
        log.info(f"[Eval-VLM] Metrics saved to {out_path}")

    def _plot_confusion_matrix(self, cm: np.ndarray, filename: str) -> None:
        """Plot and save confusion matrix."""
        fig, ax = plt.subplots(figsize=(8, 6))
        im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
        ax.figure.colorbar(im, ax=ax)

        ax.set(
            xticks=np.arange(cm.shape[1]),
            yticks=np.arange(cm.shape[0]),
            xticklabels=self.DENSITY_CLASSES,
            yticklabels=self.DENSITY_CLASSES,
            title="Density Classification — Confusion Matrix",
            ylabel="Ground Truth",
            xlabel="Predicted",
        )

        plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

        # Add text annotations
        thresh = cm.max() / 2.0
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(
                    j, i, format(cm[i, j], "d"),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black",
                )

        fig.tight_layout()
        out_path = EVAL_DIR / filename
        fig.savefig(str(out_path), dpi=150)
        plt.close(fig)
        log.info(f"[Eval-VLM] Confusion matrix saved to {out_path}")


# ─────────────────────────────────────────────
# 3. Combined Evaluator
# ─────────────────────────────────────────────
class GeoExtractEvaluator:
    """
    Orchestrates evaluation of both models and logs to W&B.
    """

    def __init__(self, cfg=CFG):
        self.cfg = cfg
        self.yolo_eval = YOLOEvaluator(cfg.evaluation.iou_threshold)
        self.vlm_eval = VLMEvaluator(cfg)

    def run_full_evaluation(
        self,
        dataset_yaml: Path,
        yolo_weights: Path,
        pipeline=None,
    ) -> Dict:
        """
        Run complete evaluation suite.
        Returns combined metrics dict for both models.
        """
        wandb_run = init_wandb(
            self.cfg,
            run_name="evaluation",
            tags=["eval", "metrics"],
        )

        all_metrics = {}

        # ── YOLO Evaluation ──
        with Timer("YOLO Evaluation"):
            yolo_metrics = self.yolo_eval.evaluate(dataset_yaml, yolo_weights)
            all_metrics["yolo"] = yolo_metrics

            if wandb_run:
                import wandb
                wandb.log({f"eval/yolo_{k}": v for k, v in yolo_metrics.items()})

        # ── VLM Evaluation ──
        if pipeline is not None:
            with Timer("VLM Evaluation"):
                vlm_metrics = self.vlm_eval.evaluate_from_pipeline(pipeline)
                all_metrics["vlm"] = vlm_metrics

                if wandb_run:
                    import wandb
                    log_metrics = {
                        k: v for k, v in vlm_metrics.items()
                        if isinstance(v, (int, float))
                    }
                    wandb.log({f"eval/vlm_{k}": v for k, v in log_metrics.items()})

                    # Log confusion matrix as image
                    cm_path = EVAL_DIR / "vlm_confusion_matrix.png"
                    if cm_path.exists():
                        wandb.log({
                            "eval/confusion_matrix": wandb.Image(str(cm_path))
                        })

        finish_wandb()

        # Save combined report
        report_path = EVAL_DIR / "full_evaluation_report.json"
        with open(report_path, "w") as f:
            json.dump(all_metrics, f, indent=2, default=str)
        log.info(f"[Eval] ✓ Full report saved to {report_path}")

        # Print summary
        self._print_summary(all_metrics)

        return all_metrics

    def _print_summary(self, metrics: Dict) -> None:
        """Print a formatted summary for defense presentation."""
        print("\n" + "═" * 60)
        print("  GeoExtract v2 — EVALUATION SUMMARY")
        print("═" * 60)

        if "yolo" in metrics:
            y = metrics["yolo"]
            print("\n  ┌── YOLO Building Detection ──┐")
            print(f"  │ Precision:  {y.get('precision', 0):.4f}          │")
            print(f"  │ Recall:     {y.get('recall', 0):.4f}          │")
            print(f"  │ F1-Score:   {y.get('f1', 0):.4f}          │")
            print(f"  │ mAP@50:     {y.get('mAP50', 0):.4f}          │")
            print(f"  │ mAP@50-95:  {y.get('mAP50-95', 0):.4f}          │")
            print("  └─────────────────────────────┘")

        if "vlm" in metrics:
            v = metrics["vlm"]
            print("\n  ┌── VLM Density Classification ──┐")
            print(f"  │ Accuracy:   {v.get('accuracy', 0):.4f}             │")
            print(f"  │ F1 (macro): {v.get('f1_macro', 0):.4f}             │")
            print(f"  │ F1 (wgt.):  {v.get('f1_weighted', 0):.4f}             │")
            print(f"  │ Precision:  {v.get('precision_macro', 0):.4f}             │")
            print(f"  │ Recall:     {v.get('recall_macro', 0):.4f}             │")
            print("  └────────────────────────────────┘")

        print("\n" + "═" * 60)
