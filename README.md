# GeoExtract v2 — Agentic Satellite Image Analysis Pipeline

> **CS Final Year Project** — Dual-Model pipeline for urban planning analytics using SpaceNet 7 satellite imagery.

## Architecture

```
┌─────────────────────────────────────────────────────┐
│                  GeoExtract v2                      │
│                                                     │
│   ┌───────────┐         ┌────────────────────┐      │
│   │  YOLOv11  │ counts  │  Qwen2-VL 2B (4b)  │      │
│   │  (Counter)├────────►│    (Reasoner)       │      │
│   └─────┬─────┘  bbox   └────────┬───────────┘      │
│         │                        │                   │
│    Bounding Boxes           Urban Analysis           │
│    Building Counts          Density Class.            │
│                             Heat Island Risk          │
│                             Green Space Index         │
└─────────────────────────────────────────────────────┘
```

## Hardware Target

| Resource | Spec |
|----------|------|
| GPU | NVIDIA T4 16 GB (Kaggle) |
| Training | Sequential (YOLO → VLM) |
| Inference | Both models loaded (~8 GB) |

## Quick Start (Kaggle)

1. Upload this repo as a Kaggle dataset or use the notebook directly.
2. Add SpaceNet 7 as an input dataset.
3. Set your W&B API key:
   ```python
   import os
   os.environ["WANDB_API_KEY"] = "your-key-here"
   ```
4. Run `geoextract_notebook.ipynb` cells in order.

## Project Structure

| File | Purpose |
|------|---------|
| `config.py` | Central configuration (paths, hyperparams) |
| `data_pipeline.py` | SpaceNet 7 GeoTIFF/GeoJSON → YOLO format |
| `qa_generator.py` | Synthetic multi-turn QA for VLM training |
| `yolo_trainer.py` | YOLOv11 training with checkpointing |
| `vlm_trainer.py` | Qwen2-VL LoRA fine-tuning |
| `inference.py` | Dual-model agentic pipeline |
| `evaluate.py` | F1/Precision/Recall + W&B logging |
| `utils.py` | VRAM monitoring, checkpoint helpers |

## Deployment

Trained weights can be exported for:
- **FastAPI** backend serving
- **Vercel** frontend dashboard
- **Carto** geospatial visualization

## License

MIT
