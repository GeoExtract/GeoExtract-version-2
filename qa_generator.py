"""
GeoExtract v2 — Synthetic QA Generator
========================================
Generates multi-turn ChatML conversational QA pairs from GeoJSON metadata
for VLM fine-tuning. Produces rich urban-planning reasoning data.
"""

import json
import random
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import numpy as np
from tqdm import tqdm

from config import CFG, VLM_DATA_DIR
from data_pipeline import SpaceNet7Parser
from utils import log, Timer


# ─────────────────────────────────────────────
# 1. Density Classification Logic
# ─────────────────────────────────────────────
class DensityClassifier:
    """Classify building density from GeoJSON metadata."""

    def __init__(self, cfg=CFG.qa):
        self.cfg = cfg

    def classify(self, building_count: int, image_area_m2: Optional[float] = None) -> Dict:
        """
        Classify density based on building count per tile.
        Returns a dict with class label, risk scores, and descriptions.
        """
        if building_count <= self.cfg.sparse_max:
            density_class = "Sparse"
            density_desc = "Low-density suburban or rural area"
            heat_risk = "Low"
            green_space = "Abundant — large open and vegetated areas visible"
            livability = "High — spacious residential environment"
            construction_intensity = "Minimal"
        elif building_count <= self.cfg.moderate_max:
            density_class = "Moderate"
            density_desc = "Moderate suburban density with mixed land use"
            heat_risk = "Moderate"
            green_space = "Moderate — some green patches between structures"
            livability = "Good — balanced density with accessible open areas"
            construction_intensity = "Active — ongoing development likely"
        elif building_count <= self.cfg.dense_max:
            density_class = "Dense"
            density_desc = "High-density urban area with tightly packed structures"
            heat_risk = "High"
            green_space = "Limited — minimal vegetation corridors"
            livability = "Moderate — constrained but functional residential zones"
            construction_intensity = "High — significant built-up coverage"
        else:
            density_class = "Urban Core"
            density_desc = "Hyper-dense urban core with maximum building coverage"
            heat_risk = "Very High"
            green_space = "Severely depleted — critical lack of vegetation"
            livability = "Low — crowded environment with limited open space"
            construction_intensity = "Maximum — near-complete land coverage"

        # Compute derived metrics
        buildings_per_hectare = building_count / max(1, (image_area_m2 or 409600) / 10000)

        return {
            "building_count": building_count,
            "density_class": density_class,
            "density_description": density_desc,
            "heat_island_risk": heat_risk,
            "green_space_assessment": green_space,
            "livability_rating": livability,
            "construction_intensity": construction_intensity,
            "buildings_per_hectare": round(buildings_per_hectare, 1),
        }


# ─────────────────────────────────────────────
# 2. Question Templates
# ─────────────────────────────────────────────
# Each template is (question_func, answer_func) — both take the density_info dict.

def _q_density_analysis(info: Dict) -> Tuple[str, str]:
    return (
        "Analyze the building density in this satellite image. "
        "What type of urban zone does this represent?",
        f"This area shows a **{info['density_class']}** density pattern with "
        f"approximately {info['building_count']} buildings detected. "
        f"{info['density_description']}. The estimated building density is "
        f"{info['buildings_per_hectare']} buildings per hectare."
    )


def _q_heat_island(info: Dict) -> Tuple[str, str]:
    return (
        "What is the urban heat island risk for this area based on "
        "the visible building coverage?",
        f"The urban heat island risk is **{info['heat_island_risk']}**. "
        f"With {info['building_count']} structures detected, the built-up area "
        f"significantly {'increases' if info['building_count'] > 30 else 'modestly affects'} "
        f"surface temperature relative to surrounding undeveloped land. "
        f"Construction intensity is classified as: {info['construction_intensity']}."
    )


def _q_green_space(info: Dict) -> Tuple[str, str]:
    return (
        "Assess the green space availability and environmental health "
        "of this area.",
        f"Green space assessment: {info['green_space_assessment']}. "
        f"In this {info['density_class'].lower()}-density zone, vegetation coverage "
        f"{'provides adequate cooling and biodiversity corridors' if info['building_count'] <= 20 else 'is insufficient for effective microclimate regulation'}. "
        f"Recommendation: {'Maintain current balance' if info['building_count'] <= 20 else 'Prioritize urban greening initiatives and rooftop gardens'}."
    )


def _q_livability(info: Dict) -> Tuple[str, str]:
    return (
        "Rate the residential livability of this zone. Would you "
        "recommend it for new housing development?",
        f"Livability rating: {info['livability_rating']}. "
        f"With a {info['density_class'].lower()} building density of "
        f"{info['buildings_per_hectare']} structures per hectare, "
        f"{'this area has capacity for additional development while maintaining quality of life' if info['building_count'] <= 25 else 'further development should be carefully planned to avoid overcrowding and infrastructure strain'}."
    )


def _q_construction_trend(info: Dict) -> Tuple[str, str]:
    return (
        "What can you tell about the construction activity and "
        "urban growth pattern in this area?",
        f"Construction intensity: {info['construction_intensity']}. "
        f"The {info['building_count']} detected structures suggest "
        f"{'an early-stage development area with significant growth potential' if info['building_count'] <= 15 else 'a mature built environment' if info['building_count'] > 40 else 'an actively developing zone in mid-growth phase'}. "
        f"The spatial distribution indicates "
        f"{'organic/informal growth patterns' if info['building_count'] > 45 else 'planned development with identifiable street grids'}."
    )


def _q_infrastructure(info: Dict) -> Tuple[str, str]:
    return (
        "Based on the building density and layout, what infrastructure "
        "challenges might this area face?",
        f"With {info['building_count']} buildings in this tile, key "
        f"infrastructure considerations include: "
        f"{'Water and sewage — adequate capacity likely available' if info['building_count'] <= 20 else 'Water and sewage — systems may be at or near capacity'}. "
        f"{'Road network — sufficient for current density' if info['building_count'] <= 30 else 'Road network — congestion risk is elevated'}. "
        f"{'Power grid — standard residential load' if info['building_count'] <= 25 else 'Power grid — peak demand management needed'}. "
        f"Overall infrastructure stress level: "
        f"{'Low' if info['building_count'] <= 15 else 'Moderate' if info['building_count'] <= 35 else 'High' if info['building_count'] <= 50 else 'Critical'}."
    )


def _q_planning_recommendation(info: Dict) -> Tuple[str, str]:
    return (
        "If you were an urban planner, what would you recommend for "
        "this area's future development?",
        f"For this {info['density_class'].lower()}-density area ({info['building_count']} structures), "
        f"I recommend: "
        f"{'1) Controlled expansion with green buffer zones, 2) Mixed-use zoning to reduce commute distances, 3) Investment in public transit corridors' if info['building_count'] <= 25 else '1) Densification limits to prevent overcrowding, 2) Mandatory green space ratios for new permits, 3) Stormwater management infrastructure upgrades' if info['building_count'] <= 45 else '1) Immediate moratorium on new construction until infrastructure catches up, 2) Retrofitting existing buildings for energy efficiency, 3) Creating pocket parks and urban forests to combat heat island effects'}."
    )


def _q_environmental_impact(info: Dict) -> Tuple[str, str]:
    return (
        "What is the environmental footprint of this built-up area? "
        "Discuss carbon implications and ecological connectivity.",
        f"Environmental analysis for {info['density_class']} zone "
        f"({info['building_count']} structures): "
        f"Carbon footprint: {'Low — minimal impervious surface coverage' if info['building_count'] <= 10 else 'Moderate — significant impervious surfaces reducing natural carbon sequestration' if info['building_count'] <= 30 else 'High — extensive land sealing limiting ecological function'}. "
        f"Ecological connectivity: "
        f"{'Intact — wildlife corridors likely preserved' if info['building_count'] <= 15 else 'Fragmented — habitat patches isolated by development' if info['building_count'] <= 40 else 'Severely disrupted — near-complete habitat loss within tile'}. "
        f"Stormwater: {'Natural infiltration adequate' if info['building_count'] <= 20 else 'Engineered drainage required to prevent flooding'}."
    )


# Collect all templates
QA_TEMPLATES = [
    _q_density_analysis,
    _q_heat_island,
    _q_green_space,
    _q_livability,
    _q_construction_trend,
    _q_infrastructure,
    _q_planning_recommendation,
    _q_environmental_impact,
]


# ─────────────────────────────────────────────
# 3. Conversational QA Generator
# ─────────────────────────────────────────────
class SyntheticQAGenerator:
    """
    Generates multi-turn ChatML conversations from SpaceNet 7 metadata.
    Output format matches Qwen2-VL chat template.
    """

    def __init__(self, cfg=CFG):
        self.cfg = cfg
        self.qa_cfg = cfg.qa
        self.parser = SpaceNet7Parser(cfg.data)
        self.classifier = DensityClassifier(cfg.qa)
        self.output_dir = VLM_DATA_DIR

    def generate(self) -> Path:
        """
        Main entry point.
        1. Parse SpaceNet 7 samples.
        2. For each labeled image, count buildings and classify density.
        3. Generate multi-turn conversations.
        4. Save as JSONL (one conversation per line).
        Returns path to the output JSONL file.
        """
        with Timer("VLM QA Generation"):
            samples = self.parser.discover()
            labeled = [s for s in samples if s["has_labels"]]

            conversations = []
            for sample in tqdm(labeled, desc="Generating QA pairs"):
                try:
                    conv = self._generate_conversation(sample)
                    if conv:
                        conversations.append(conv)
                except Exception as e:
                    log.warning(f"[QA] Failed for {sample['image_path'].name}: {e}")
                    continue

            # Train/val split
            random.seed(self.cfg.data.seed)
            random.shuffle(conversations)
            split_idx = int(len(conversations) * (1 - self.cfg.data.val_split))
            train_convs = conversations[:split_idx]
            val_convs = conversations[split_idx:]

            # Save
            train_path = self._save_jsonl(train_convs, "train.jsonl")
            val_path = self._save_jsonl(val_convs, "val.jsonl")

            log.info(f"[QA] Generated {len(train_convs)} train, "
                     f"{len(val_convs)} val conversations.")
            return train_path

    def _generate_conversation(self, sample: Dict) -> Optional[Dict]:
        """Generate a multi-turn conversation for a single image."""
        # Count buildings
        buildings = self.parser.read_geojson(sample["label_path"])
        building_count = len(buildings)

        # Compute total building area if available
        total_area = sum(b.get("area", 0) for b in buildings) if buildings else None

        # Classify density
        density_info = self.classifier.classify(building_count, total_area)

        # Select random subset of QA templates
        n_turns = random.randint(self.qa_cfg.min_turns, self.qa_cfg.max_turns)
        selected_templates = random.sample(
            QA_TEMPLATES, min(n_turns, len(QA_TEMPLATES))
        )

        # Build ChatML messages
        messages = [
            {"role": "system", "content": self.qa_cfg.system_prompt}
        ]

        for template_fn in selected_templates:
            question, answer = template_fn(density_info)
            messages.append({"role": "user", "content": question})
            messages.append({"role": "assistant", "content": answer})

        return {
            "id": f"{sample['aoi']}_{sample['timestamp']}",
            "image": str(sample["image_path"]),
            "building_count": building_count,
            "density_class": density_info["density_class"],
            "messages": messages,
        }

    def _save_jsonl(self, conversations: List[Dict], filename: str) -> Path:
        """Save conversations as JSONL."""
        path = self.output_dir / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            for conv in conversations:
                f.write(json.dumps(conv, default=str) + "\n")
        log.info(f"[QA] Saved {len(conversations)} conversations to {path}")
        return path

    def get_stats(self) -> Dict:
        """Get statistics about generated data."""
        stats = {}
        for split in ["train", "val"]:
            path = self.output_dir / f"{split}.jsonl"
            if path.exists():
                with open(path) as f:
                    lines = f.readlines()
                convs = [json.loads(l) for l in lines]

                density_dist = {}
                total_turns = 0
                for c in convs:
                    dc = c.get("density_class", "Unknown")
                    density_dist[dc] = density_dist.get(dc, 0) + 1
                    total_turns += len([m for m in c["messages"] if m["role"] == "user"])

                stats[split] = {
                    "conversations": len(convs),
                    "total_qa_turns": total_turns,
                    "density_distribution": density_dist,
                }
        return stats
