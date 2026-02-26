"""
GeoExtract v2 — VLM Trainer
==============================
Qwen2-VL-2B-Instruct fine-tuning with LoRA/PEFT.
- 4-bit NF4 quantization via BitsAndBytes
- Cosine LR scheduler with warmup
- Multi-turn conversational training (ChatML)
- Bulletproof checkpointing (every 500 steps, auto-resume)
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Optional

import torch
from torch.utils.data import Dataset

from config import CFG, VLM_CKPT_DIR, VLM_DATA_DIR
from utils import (
    log, Timer, free_vram, log_vram, find_latest_checkpoint,
    count_parameters, init_wandb, finish_wandb,
)


# ─────────────────────────────────────────────
# 1. Conversational Dataset
# ─────────────────────────────────────────────
class GeoExtractVLMDataset(Dataset):
    """
    Loads JSONL conversations and formats them for Qwen2-VL training.
    Each sample is a multi-turn conversation with an associated image.
    """

    def __init__(
        self,
        jsonl_path: Path,
        processor,
        max_length: int = 1024,
        include_images: bool = True,
    ):
        self.processor = processor
        self.max_length = max_length
        self.include_images = include_images
        self.conversations = []

        with open(jsonl_path, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    self.conversations.append(json.loads(line))

        log.info(f"[VLM Dataset] Loaded {len(self.conversations)} conversations "
                 f"from {jsonl_path.name}")

    def __len__(self) -> int:
        return len(self.conversations)

    def __getitem__(self, idx: int) -> Dict:
        conv = self.conversations[idx]
        messages = conv["messages"]
        image_path = conv.get("image")

        # Build the ChatML-formatted text
        # Qwen2-VL uses: <|im_start|>role\ncontent<|im_end|>
        formatted_messages = []
        for msg in messages:
            role = msg["role"]
            content = msg["content"]

            if role == "user" and self.include_images and image_path:
                # For the first user message, include image reference
                # Qwen2-VL expects image tokens in the content
                formatted_messages.append({
                    "role": role,
                    "content": [
                        {"type": "image", "image": image_path},
                        {"type": "text", "text": content},
                    ],
                })
                # Only include image in first user turn
                image_path = None
            else:
                formatted_messages.append({
                    "role": role,
                    "content": [{"type": "text", "text": content}],
                })

        return {
            "messages": formatted_messages,
            "id": conv.get("id", str(idx)),
        }


# ─────────────────────────────────────────────
# 2. Data Collator for Multi-Turn Chat
# ─────────────────────────────────────────────
class ChatMLCollator:
    """
    Collates multi-turn conversations into model inputs.
    Handles the Qwen2-VL chat template with proper masking
    so the model only learns to predict assistant responses.
    """

    def __init__(self, processor, max_length: int = 1024):
        self.processor = processor
        self.max_length = max_length
        self.tokenizer = processor.tokenizer if hasattr(processor, 'tokenizer') else processor

    def __call__(self, batch: List[Dict]) -> Dict:
        texts = []
        images = []

        for sample in batch:
            messages = sample["messages"]

            # Apply chat template
            try:
                text = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=False,
                )
                texts.append(text)
            except Exception as e:
                # Fallback: manual ChatML formatting
                text = self._manual_chatml(messages)
                texts.append(text)

            # Collect images
            for msg in messages:
                if isinstance(msg.get("content"), list):
                    for part in msg["content"]:
                        if isinstance(part, dict) and part.get("type") == "image":
                            img_path = part.get("image", "")
                            if img_path and Path(img_path).exists():
                                from PIL import Image
                                try:
                                    img = Image.open(img_path).convert("RGB")
                                    images.append(img)
                                except Exception:
                                    pass

        # Tokenize
        encoding = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        # Labels = input_ids (for causal LM), with padding masked to -100
        labels = encoding["input_ids"].clone()
        labels[labels == self.tokenizer.pad_token_id] = -100

        # Mask system and user turns (only train on assistant responses)
        labels = self._mask_non_assistant_tokens(texts, labels)

        encoding["labels"] = labels
        return encoding

    def _mask_non_assistant_tokens(self, texts: List[str], labels: torch.Tensor) -> torch.Tensor:
        """Mask tokens that are not part of assistant responses."""
        for i, text in enumerate(texts):
            # Find assistant response boundaries
            # Qwen2-VL uses <|im_start|>assistant and <|im_end|>
            assistant_start_token = "<|im_start|>assistant"
            assistant_end_token = "<|im_end|>"

            # Tokenize the full text to get token positions
            tokens = self.tokenizer.encode(text, add_special_tokens=False)

            # Simple heuristic: find assistant sections
            in_assistant = False
            char_pos = 0
            assistant_ranges = []
            start_pos = 0

            while True:
                start_idx = text.find(assistant_start_token, char_pos)
                if start_idx == -1:
                    break
                # Find content start (after the role line)
                content_start = text.find("\n", start_idx)
                if content_start == -1:
                    break
                content_start += 1

                # Find end
                end_idx = text.find(assistant_end_token, content_start)
                if end_idx == -1:
                    end_idx = len(text)

                assistant_ranges.append((content_start, end_idx))
                char_pos = end_idx + len(assistant_end_token)

            if assistant_ranges:
                # Create a mask for non-assistant tokens
                # Tokenize prefix to find token boundaries
                prefix_text = ""
                mask = torch.ones_like(labels[i], dtype=torch.bool)  # True = mask out

                for start, end in assistant_ranges:
                    # Rough token position mapping
                    prefix_tokens = self.tokenizer.encode(
                        text[:start], add_special_tokens=False
                    )
                    content_tokens = self.tokenizer.encode(
                        text[start:end], add_special_tokens=False
                    )
                    tok_start = len(prefix_tokens)
                    tok_end = tok_start + len(content_tokens)
                    # Clamp to sequence length
                    tok_start = min(tok_start, labels.shape[1] - 1)
                    tok_end = min(tok_end, labels.shape[1])
                    mask[tok_start:tok_end] = False  # Don't mask assistant tokens

                # Apply mask: set non-assistant tokens to -100
                labels[i][mask] = -100

        return labels

    def _manual_chatml(self, messages: List[Dict]) -> str:
        """Fallback ChatML formatter."""
        parts = []
        for msg in messages:
            role = msg["role"]
            if isinstance(msg["content"], list):
                content = " ".join(
                    p["text"] for p in msg["content"]
                    if isinstance(p, dict) and p.get("type") == "text"
                )
            else:
                content = msg["content"]
            parts.append(f"<|im_start|>{role}\n{content}<|im_end|>")
        return "\n".join(parts)


# ─────────────────────────────────────────────
# 3. VLM Trainer
# ─────────────────────────────────────────────
class VLMTrainer:
    """
    Full training pipeline for Qwen2-VL-2B with LoRA.
    - 4-bit quantization
    - PEFT/LoRA adapter
    - Cosine schedule with warmup
    - Checkpointing every 500 steps
    - Auto-resume from latest checkpoint
    """

    def __init__(self, cfg=CFG):
        self.cfg = cfg
        self.vlm_cfg = cfg.vlm
        self.model = None
        self.processor = None
        self.trainer = None

    def train(self) -> Path:
        """Main entry point — runs full training loop."""
        with Timer("VLM Training"):
            log_vram("before VLM load")

            # ── 1. Load quantized model ──
            self._load_model()
            log_vram("after VLM load")

            # ── 2. Apply LoRA ──
            self._apply_lora()

            # ── 3. Load datasets ──
            train_dataset, val_dataset = self._load_datasets()

            # ── 4. Initialize HF Trainer ──
            self._setup_trainer(train_dataset, val_dataset)

            # ── 5. Train (with auto-resume) ──
            self._run_training()

            # ── 6. Save final adapter ──
            final_path = self._save_final()

            log.info(f"[VLM] ✓ Training complete. Adapter saved to {final_path}")
            return final_path

    def _load_model(self) -> None:
        """Load Qwen2-VL-2B with 4-bit NF4 quantization."""
        from transformers import (
            AutoModelForCausalLM,
            AutoProcessor,
            BitsAndBytesConfig,
        )

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=self.vlm_cfg.load_in_4bit,
            bnb_4bit_quant_type=self.vlm_cfg.bnb_4bit_quant_type,
            bnb_4bit_compute_dtype=getattr(torch, self.vlm_cfg.bnb_4bit_compute_dtype),
            bnb_4bit_use_double_quant=True,
        )

        log.info(f"[VLM] Loading {self.vlm_cfg.model_id} in 4-bit NF4...")

        self.model = AutoModelForCausalLM.from_pretrained(
            self.vlm_cfg.model_id,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2"
            if torch.cuda.get_device_capability()[0] >= 8
            else "eager",
        )

        self.processor = AutoProcessor.from_pretrained(
            self.vlm_cfg.model_id,
            trust_remote_code=True,
        )

        # Ensure pad token exists
        if self.processor.tokenizer.pad_token is None:
            self.processor.tokenizer.pad_token = self.processor.tokenizer.eos_token
            self.model.config.pad_token_id = self.model.config.eos_token_id

        params = count_parameters(self.model)
        log.info(f"[VLM] Model loaded. Total params: {params['total']:,}, "
                 f"Trainable (pre-LoRA): {params['trainable']:,}")

    def _apply_lora(self) -> None:
        """Apply LoRA adapters to the model."""
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType

        # Prepare for k-bit training (handles gradient checkpointing quirks)
        self.model = prepare_model_for_kbit_training(
            self.model,
            use_gradient_checkpointing=self.vlm_cfg.gradient_checkpointing,
        )

        lora_config = LoraConfig(
            r=self.vlm_cfg.lora_r,
            lora_alpha=self.vlm_cfg.lora_alpha,
            lora_dropout=self.vlm_cfg.lora_dropout,
            target_modules=self.vlm_cfg.lora_target_modules,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )

        self.model = get_peft_model(self.model, lora_config)

        params = count_parameters(self.model)
        log.info(f"[VLM] LoRA applied. Trainable: {params['trainable']:,} "
                 f"({params['trainable_pct']}% of total)")
        self.model.print_trainable_parameters()

    def _load_datasets(self):
        """Load train and val datasets from generated JSONL."""
        train_path = VLM_DATA_DIR / "train.jsonl"
        val_path = VLM_DATA_DIR / "val.jsonl"

        if not train_path.exists():
            raise FileNotFoundError(
                f"[VLM] Training data not found at {train_path}. "
                "Run qa_generator.py first!"
            )

        train_dataset = GeoExtractVLMDataset(
            train_path,
            self.processor,
            max_length=self.vlm_cfg.max_seq_length,
        )
        val_dataset = None
        if val_path.exists():
            val_dataset = GeoExtractVLMDataset(
                val_path,
                self.processor,
                max_length=self.vlm_cfg.max_seq_length,
            )

        return train_dataset, val_dataset

    def _setup_trainer(self, train_dataset, val_dataset) -> None:
        """Configure HuggingFace Trainer with proper args."""
        from transformers import TrainingArguments, Trainer

        # Check for resume checkpoint
        resume_ckpt = None
        if self.vlm_cfg.resume_from_checkpoint:
            resume_ckpt = find_latest_checkpoint(
                Path(self.vlm_cfg.output_dir)
            )
            if resume_ckpt:
                log.info(f"[VLM] Will resume from: {resume_ckpt}")

        training_args = TrainingArguments(
            output_dir=str(self.vlm_cfg.output_dir),
            num_train_epochs=self.vlm_cfg.epochs,
            per_device_train_batch_size=self.vlm_cfg.batch_size,
            per_device_eval_batch_size=self.vlm_cfg.batch_size,
            gradient_accumulation_steps=self.vlm_cfg.gradient_accumulation_steps,
            learning_rate=self.vlm_cfg.learning_rate,
            weight_decay=self.vlm_cfg.weight_decay,
            warmup_ratio=self.vlm_cfg.warmup_ratio,
            lr_scheduler_type=self.vlm_cfg.lr_scheduler_type,
            # Precision
            fp16=self.vlm_cfg.fp16,
            bf16=self.vlm_cfg.bf16,
            # Checkpointing
            save_steps=self.vlm_cfg.save_steps,
            save_total_limit=3,                         # keep last 3 checkpoints
            save_strategy="steps",
            # Evaluation
            eval_strategy="steps" if val_dataset else "no",
            eval_steps=self.vlm_cfg.eval_steps if val_dataset else None,
            # Logging
            logging_steps=self.vlm_cfg.logging_steps,
            logging_first_step=True,
            report_to="wandb" if self.cfg.wandb.enabled else "none",
            run_name="vlm-geoextract",
            # Memory optimization
            gradient_checkpointing=self.vlm_cfg.gradient_checkpointing,
            optim="paged_adamw_8bit",                   # memory-efficient optimizer
            max_grad_norm=1.0,
            # Misc
            remove_unused_columns=False,
            dataloader_num_workers=2,
            seed=self.cfg.data.seed,
            load_best_model_at_end=True if val_dataset else False,
            metric_for_best_model="eval_loss" if val_dataset else None,
        )

        collator = ChatMLCollator(
            self.processor,
            max_length=self.vlm_cfg.max_seq_length,
        )

        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            data_collator=collator,
        )

        self._resume_checkpoint = resume_ckpt

    def _run_training(self) -> None:
        """Execute training with resume support."""
        # Initialize W&B
        wandb_run = init_wandb(
            self.cfg,
            run_name="vlm-geoextract-lora",
            tags=["vlm", "lora", "qwen2-vl"],
        )

        try:
            if self._resume_checkpoint:
                log.info(f"[VLM] Resuming training from {self._resume_checkpoint}")
                self.trainer.train(resume_from_checkpoint=str(self._resume_checkpoint))
            else:
                log.info("[VLM] Starting training from scratch.")
                self.trainer.train()
        except KeyboardInterrupt:
            log.warning("[VLM] Training interrupted. Saving checkpoint...")
            self.trainer.save_model(str(self.vlm_cfg.output_dir / "interrupted"))
        finally:
            finish_wandb()

    def _save_final(self) -> Path:
        """Save the final LoRA adapter weights."""
        final_dir = Path(self.vlm_cfg.output_dir) / "final_adapter"
        final_dir.mkdir(parents=True, exist_ok=True)

        # Save adapter weights (safetensors format)
        self.model.save_pretrained(str(final_dir))
        self.processor.save_pretrained(str(final_dir))

        log.info(f"[VLM] Final adapter saved to {final_dir}")

        # List saved files
        saved_files = list(final_dir.glob("*"))
        log.info(f"[VLM] Saved files: {[f.name for f in saved_files]}")

        return final_dir

    def cleanup(self) -> None:
        """Free VLM model from GPU memory."""
        if self.model is not None:
            del self.model
            self.model = None
        if self.processor is not None:
            del self.processor
            self.processor = None
        if self.trainer is not None:
            del self.trainer
            self.trainer = None
        free_vram()
        log.info("[VLM] Model unloaded, VRAM freed.")
