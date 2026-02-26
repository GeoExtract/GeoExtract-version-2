"""
GeoExtract v2 — Agentic Inference Pipeline
=============================================
Dual-model pipeline: YOLO (Counter) → VLM (Reasoner).
Produces structured JSON output ready for FastAPI/frontend.
"""

import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import torch
from PIL import Image

from config import CFG, InferenceConfig
from utils import log, Timer, log_vram, free_vram


class GeoExtractPipeline:
    """
    Agentic inference pipeline:
    1. YOLO detects buildings → bounding boxes + count
    2. Context (count, density) is injected into the VLM prompt
    3. VLM reasons about the image + context → urban analysis
    4. Returns structured JSON (FastAPI-ready)
    """

    def __init__(self, cfg: InferenceConfig = CFG.inference, full_cfg=CFG):
        self.cfg = cfg
        self.full_cfg = full_cfg
        self.yolo_model = None
        self.vlm_model = None
        self.vlm_processor = None
        self._loaded = False

    def load(self) -> None:
        """Load both models into GPU memory."""
        log_vram("before pipeline load")

        # ── Load YOLO ──
        self._load_yolo()

        # ── Load VLM ──
        self._load_vlm()

        self._loaded = True
        log_vram("after pipeline load")
        log.info("[Pipeline] ✓ Both models loaded and ready.")

    def _load_yolo(self) -> None:
        """Load trained YOLO model."""
        from ultralytics import YOLO

        weights_path = self.cfg.yolo_weights
        if not weights_path.exists():
            # Try finding any best.pt in the checkpoint directory
            candidates = list(self.cfg.yolo_weights.parent.parent.rglob("best.pt"))
            if candidates:
                weights_path = candidates[0]
                log.info(f"[Pipeline] Found YOLO weights at {weights_path}")
            else:
                raise FileNotFoundError(
                    f"YOLO weights not found at {self.cfg.yolo_weights}. "
                    "Train the YOLO model first!"
                )

        self.yolo_model = YOLO(str(weights_path))
        log.info(f"[Pipeline] YOLO loaded from {weights_path}")

    def _load_vlm(self) -> None:
        """Load quantized VLM with LoRA adapter."""
        from transformers import AutoModelForCausalLM, AutoProcessor, BitsAndBytesConfig
        from peft import PeftModel

        vlm_cfg = self.full_cfg.vlm

        # Load base model in 4-bit
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )

        base_model = AutoModelForCausalLM.from_pretrained(
            vlm_cfg.model_id,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
        )

        # Load LoRA adapter
        adapter_dir = self.cfg.vlm_adapter_dir / "final_adapter"
        if not adapter_dir.exists():
            # Try finding any adapter_config.json
            candidates = list(self.cfg.vlm_adapter_dir.rglob("adapter_config.json"))
            if candidates:
                adapter_dir = candidates[0].parent
            else:
                log.warning("[Pipeline] No LoRA adapter found — using base model.")
                self.vlm_model = base_model
                self.vlm_processor = AutoProcessor.from_pretrained(
                    vlm_cfg.model_id, trust_remote_code=True
                )
                return

        self.vlm_model = PeftModel.from_pretrained(base_model, str(adapter_dir))
        self.vlm_model.eval()

        self.vlm_processor = AutoProcessor.from_pretrained(
            str(adapter_dir), trust_remote_code=True
        )
        if self.vlm_processor.tokenizer.pad_token is None:
            self.vlm_processor.tokenizer.pad_token = self.vlm_processor.tokenizer.eos_token

        log.info(f"[Pipeline] VLM loaded with adapter from {adapter_dir}")

    # ─────────────────────────────────────────
    # Core Inference
    # ─────────────────────────────────────────
    def analyze(
        self,
        image: Union[str, Path, Image.Image],
        question: Optional[str] = None,
    ) -> Dict:
        """
        Full agentic analysis pipeline.

        Args:
            image: Path to image or PIL Image.
            question: Optional custom question. If None, uses default analysis.

        Returns:
            Structured JSON dict with all analysis results.
        """
        if not self._loaded:
            self.load()

        start_time = time.time()

        # Ensure PIL Image
        if isinstance(image, (str, Path)):
            image_path = str(image)
            pil_image = Image.open(image_path).convert("RGB")
        else:
            pil_image = image
            image_path = "uploaded_image"

        # ── Step 1: YOLO Detection ──
        yolo_results = self._run_yolo(pil_image)

        # ── Step 2: Build context for VLM ──
        context = self._build_context(yolo_results)

        # ── Step 3: VLM Reasoning ──
        if question is None:
            question = (
                "Analyze this satellite image comprehensively. Assess the building "
                "density, urban heat island risk, green space availability, "
                "infrastructure stress, and provide urban planning recommendations."
            )

        vlm_response = self._run_vlm(pil_image, question, context)

        # ── Step 4: Structure output ──
        result = {
            "image": image_path,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "processing_time_s": round(time.time() - start_time, 2),
            "detection": yolo_results,
            "context": context,
            "analysis": {
                "question": question,
                "response": vlm_response,
            },
            "metadata": {
                "yolo_model": str(self.cfg.yolo_weights.name),
                "vlm_model": self.full_cfg.vlm.model_id,
                "confidence_threshold": self.cfg.confidence_threshold,
            },
        }

        return result

    def batch_analyze(
        self,
        images: List[Union[str, Path]],
        question: Optional[str] = None,
    ) -> List[Dict]:
        """Analyze multiple images."""
        results = []
        for img in images:
            try:
                result = self.analyze(img, question)
                results.append(result)
            except Exception as e:
                log.error(f"[Pipeline] Failed on {img}: {e}")
                results.append({"image": str(img), "error": str(e)})
        return results

    # ─────────────────────────────────────────
    # YOLO Detection
    # ─────────────────────────────────────────
    def _run_yolo(self, image: Image.Image) -> Dict:
        """Run YOLO detection and return structured results."""
        results = self.yolo_model(
            image,
            conf=self.cfg.confidence_threshold,
            iou=self.cfg.iou_threshold,
            verbose=False,
        )

        detections = []
        if results and len(results) > 0:
            boxes = results[0].boxes
            if boxes is not None:
                for i in range(len(boxes)):
                    det = {
                        "bbox": boxes.xyxy[i].cpu().tolist(),       # [x1, y1, x2, y2]
                        "confidence": float(boxes.conf[i].cpu()),
                        "class": int(boxes.cls[i].cpu()),
                        "class_name": "building",
                    }
                    detections.append(det)

        return {
            "building_count": len(detections),
            "detections": detections,
            "avg_confidence": round(
                np.mean([d["confidence"] for d in detections]), 3
            ) if detections else 0.0,
        }

    # ─────────────────────────────────────────
    # Context Builder (YOLO → VLM bridge)
    # ─────────────────────────────────────────
    def _build_context(self, yolo_results: Dict) -> Dict:
        """
        Build a context dict from YOLO results to inject into VLM prompt.
        This is the 'agentic' bridge between the two models.
        """
        count = yolo_results["building_count"]

        # Classify density using same thresholds as training
        qa_cfg = self.full_cfg.qa
        if count <= qa_cfg.sparse_max:
            density = "Sparse"
        elif count <= qa_cfg.moderate_max:
            density = "Moderate"
        elif count <= qa_cfg.dense_max:
            density = "Dense"
        else:
            density = "Urban Core"

        return {
            "building_count": count,
            "density_class": density,
            "avg_detection_confidence": yolo_results["avg_confidence"],
            "context_prompt": (
                f"The building detection model has identified {count} structures "
                f"in this image with an average confidence of "
                f"{yolo_results['avg_confidence']:.1%}. This area is classified "
                f"as '{density}' density."
            ),
        }

    # ─────────────────────────────────────────
    # VLM Reasoning
    # ─────────────────────────────────────────
    def _run_vlm(self, image: Image.Image, question: str, context: Dict) -> str:
        """Run VLM inference with YOLO context injected."""
        # Build multi-turn prompt with context
        messages = [
            {
                "role": "system",
                "content": self.full_cfg.qa.system_prompt,
            },
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {
                        "type": "text",
                        "text": (
                            f"Context from detection model: {context['context_prompt']}\n\n"
                            f"Question: {question}"
                        ),
                    },
                ],
            },
        ]

        # Apply chat template
        try:
            text = self.vlm_processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        except Exception:
            # Fallback
            text = (
                f"<|im_start|>system\n{self.full_cfg.qa.system_prompt}<|im_end|>\n"
                f"<|im_start|>user\n{context['context_prompt']}\n{question}<|im_end|>\n"
                f"<|im_start|>assistant\n"
            )

        # Tokenize
        inputs = self.vlm_processor(
            text=[text],
            images=[image],
            return_tensors="pt",
            padding=True,
        ).to(self.vlm_model.device)

        # Generate
        with torch.no_grad():
            outputs = self.vlm_model.generate(
                **inputs,
                max_new_tokens=self.cfg.max_new_tokens,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                repetition_penalty=1.1,
            )

        # Decode (skip input tokens)
        input_len = inputs["input_ids"].shape[1]
        response = self.vlm_processor.tokenizer.decode(
            outputs[0][input_len:],
            skip_special_tokens=True,
        ).strip()

        return response

    # ─────────────────────────────────────────
    # Multi-Turn Conversation
    # ─────────────────────────────────────────
    def chat(
        self,
        image: Union[str, Path, Image.Image],
        conversation_history: List[Dict],
        new_question: str,
    ) -> Dict:
        """
        Multi-turn conversation mode.
        Maintains context across follow-up questions.
        """
        if not self._loaded:
            self.load()

        if isinstance(image, (str, Path)):
            pil_image = Image.open(str(image)).convert("RGB")
        else:
            pil_image = image

        # If first turn, run YOLO first
        if not conversation_history:
            yolo_results = self._run_yolo(pil_image)
            context = self._build_context(yolo_results)
        else:
            # Extract context from previous turn
            context = conversation_history[0].get("context", {})

        # Build full conversation messages
        messages = [
            {"role": "system", "content": self.full_cfg.qa.system_prompt},
        ]

        # Add previous turns
        for turn in conversation_history:
            messages.append({"role": "user", "content": turn.get("question", "")})
            messages.append({"role": "assistant", "content": turn.get("response", "")})

        # Add new question
        messages.append({"role": "user", "content": new_question})

        # Format and generate
        text = self.vlm_processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
        inputs = self.vlm_processor(
            text=[text], images=[pil_image], return_tensors="pt", padding=True,
        ).to(self.vlm_model.device)

        with torch.no_grad():
            outputs = self.vlm_model.generate(
                **inputs,
                max_new_tokens=self.cfg.max_new_tokens,
                do_sample=True,
                temperature=0.7,
            )

        input_len = inputs["input_ids"].shape[1]
        response = self.vlm_processor.tokenizer.decode(
            outputs[0][input_len:], skip_special_tokens=True,
        ).strip()

        # Update conversation history
        conversation_history.append({
            "question": new_question,
            "response": response,
            "context": context,
        })

        return {
            "response": response,
            "conversation_history": conversation_history,
        }

    def cleanup(self) -> None:
        """Free all models from GPU."""
        if self.yolo_model is not None:
            del self.yolo_model
            self.yolo_model = None
        if self.vlm_model is not None:
            del self.vlm_model
            self.vlm_model = None
        if self.vlm_processor is not None:
            del self.vlm_processor
            self.vlm_processor = None
        self._loaded = False
        free_vram()
        log.info("[Pipeline] All models unloaded.")
