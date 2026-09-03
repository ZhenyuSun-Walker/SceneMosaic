#!/usr/bin/env python3
"""Filter raw SAM3 text masks into a bounded, numbered top-zone review set."""

from __future__ import annotations

import json
import math
import os
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage


@dataclass(frozen=True)
class TopZoneConfig:
    top_boundary_fraction: float = 0.32
    minimum_top_zone_coverage: float = 0.60
    minimum_score: float = 0.30
    minimum_component_pixels: int = 30
    minimum_component_source_fraction: float = 0.05
    maximum_area_fraction: float = 0.03
    duplicate_iou: float = 0.65
    duplicate_containment: float = 0.85
    maximum_candidates: int = 8


def _safe_name(value: Any) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("_") or "item"


def _bool_mask(value: np.ndarray) -> np.ndarray:
    mask = np.asarray(value)
    if mask.ndim != 2:
        raise ValueError(f"mask must be 2D, got {mask.shape}")
    return mask.astype(bool, copy=False)


def load_mask(path: str | Path, expected_shape: tuple[int, int]) -> np.ndarray:
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    mask = np.array(Image.open(resolved).convert("L")) > 0
    if mask.shape != expected_shape:
        raise ValueError(f"mask shape mismatch: {resolved}: {mask.shape} != {expected_shape}")
    return mask


def mask_bbox(mask: np.ndarray) -> list[int]:
    ys, xs = np.nonzero(_bool_mask(mask))
    if not len(xs):
        raise ValueError("empty mask")
    return [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]


def mask_overlap(left: np.ndarray, right: np.ndarray) -> dict[str, float]:
    left = _bool_mask(left)
    right = _bool_mask(right)
    intersection = int((left & right).sum())
    left_area = int(left.sum())
    right_area = int(right.sum())
    union = left_area + right_area - intersection
    return {
        "iou": intersection / union if union else 0.0,
        "left_coverage": intersection / left_area if left_area else 0.0,
        "right_coverage": intersection / right_area if right_area else 0.0,
    }


def _split_components(mask: np.ndarray, config: TopZoneConfig) -> list[np.ndarray]:
    labels, count = ndimage.label(_bool_mask(mask), structure=np.ones((3, 3), bool))
    source_area = int(mask.sum())
    minimum = max(
        config.minimum_component_pixels,
        int(math.ceil(config.minimum_component_source_fraction * source_area)),
    )
    components = []
    for label_id in range(1, count + 1):
        component = labels == label_id
        if int(component.sum()) >= minimum:
            components.append(component)
    return components


def filter_top_zone_candidates(
    raw_candidates: Sequence[Mapping[str, Any]],
    *,
    image_size: tuple[int, int],
    config: TopZoneConfig | None = None,
) -> dict[str, Any]:
    """Split disconnected masks, filter geometrically and remove duplicates."""

    cfg = config or TopZoneConfig()
    width, height = int(image_size[0]), int(image_size[1])
    if width <= 0 or height <= 0:
        raise ValueError("image_size must be positive")
    image_area = width * height
    top_y = int(round(cfg.top_boundary_fraction * height))
    expanded: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []

    for raw in raw_candidates:
        source_id = str(raw.get("candidate_id") or "")
        score = float(raw.get("score") or 0.0)
        source_mask = load_mask(raw["mask_path"], (height, width))
        components = _split_components(source_mask, cfg)
        for component_index, component in enumerate(components, start=1):
            area = int(component.sum())
            bbox = mask_bbox(component)
            top_coverage = float(component[:top_y].sum() / area)
            center_y = (bbox[1] + bbox[3]) / (2 * height)
            area_fraction = area / image_area
            reasons = []
            if score < cfg.minimum_score:
                reasons.append("low_score")
            if top_coverage < cfg.minimum_top_zone_coverage:
                reasons.append("insufficient_top_zone_coverage")
            if center_y > cfg.top_boundary_fraction:
                reasons.append("center_below_top_zone")
            if area_fraction > cfg.maximum_area_fraction:
                reasons.append("area_too_large")
            record = {
                "source_candidate_id": source_id,
                "source_component_index": component_index,
                "prompt": str(raw.get("prompt") or ""),
                "score": score,
                "area_pixels": area,
                "area_fraction": area_fraction,
                "top_zone_coverage": top_coverage,
                "bbox_abs_xyxy": bbox,
                "bbox_norm_xyxy": [
                    bbox[0] / width,
                    bbox[1] / height,
                    bbox[2] / width,
                    bbox[3] / height,
                ],
                "mask": component,
            }
            if reasons:
                rejected.append({**{k: v for k, v in record.items() if k != "mask"}, "reasons": reasons})
            else:
                expanded.append(record)

    expanded.sort(key=lambda item: (-item["score"], item["area_pixels"]))
    kept: list[dict[str, Any]] = []
    duplicate_rejections: list[dict[str, Any]] = []
    for candidate in expanded:
        duplicate = None
        for existing in kept:
            metrics = mask_overlap(candidate["mask"], existing["mask"])
            if (
                metrics["iou"] >= cfg.duplicate_iou
                or max(metrics["left_coverage"], metrics["right_coverage"])
                >= cfg.duplicate_containment
            ):
                duplicate = (existing, metrics)
                break
        if duplicate is None:
            kept.append(candidate)
        else:
            duplicate_rejections.append(
                {
                    "source_candidate_id": candidate["source_candidate_id"],
                    "source_component_index": candidate["source_component_index"],
                    "duplicate_of_source_candidate_id": duplicate[0]["source_candidate_id"],
                    "overlap": {k: round(v, 6) for k, v in duplicate[1].items()},
                }
            )

    overflow = kept[cfg.maximum_candidates :]
    kept = kept[: cfg.maximum_candidates]
    kept.sort(key=lambda item: (item["bbox_abs_xyxy"][1], item["bbox_abs_xyxy"][0]))
    for index, candidate in enumerate(kept, start=1):
        candidate["candidate_id"] = f"LC_{index:03d}"
    return {
        "status": "candidates_found" if kept else "no_candidates",
        "image_size": [width, height],
        "config": cfg.__dict__,
        "selected": kept,
        "geometric_rejections": rejected,
        "duplicate_rejections": duplicate_rejections,
        "overflow_rejections": [
            {k: v for k, v in item.items() if k != "mask"} for item in overflow
        ],
    }


def _font(size: int):
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size)
    except OSError:
        return ImageFont.load_default()


def materialize_top_zone_review(
    image_path: str | Path,
    filtered: Mapping[str, Any],
    output_dir: str | Path,
    *,
    overwrite: bool = False,
    panel_title: str = "TOP-ZONE LOCAL CANDIDATES",
    schema_version: str = "scenemosaic_top_zone_candidate_review",
) -> dict[str, Any]:
    """Save bounded candidate masks, overlays, a numbered panel and manifest."""

    output = Path(output_dir).expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        if not overwrite:
            raise FileExistsError(output)
        shutil.rmtree(output)
    output.mkdir(parents=True, exist_ok=True)
    masks_dir = output / "masks"
    overlays_dir = output / "overlays"
    masks_dir.mkdir()
    overlays_dir.mkdir()
    image = Image.open(image_path).convert("RGB")
    image_np = np.array(image).astype(np.uint8)
    records = []
    colors = [
        (240, 55, 55), (40, 150, 245), (40, 190, 90), (235, 155, 30),
        (150, 75, 235), (245, 80, 175), (25, 185, 185), (185, 180, 35),
    ]
    overview = image.copy()
    overview_draw = ImageDraw.Draw(overview)
    label_font = _font(22)
    cells = []

    for index, raw in enumerate(filtered.get("selected") or []):
        candidate = dict(raw)
        mask = _bool_mask(candidate.pop("mask"))
        candidate_id = candidate["candidate_id"]
        base = _safe_name(candidate_id)
        mask_path = masks_dir / f"{base}_mask.png"
        overlay_path = overlays_dir / f"{base}_overlay.png"
        Image.fromarray(mask.astype(np.uint8) * 255).save(mask_path)
        overlay_np = image_np.copy()
        color = np.array(colors[index % len(colors)], dtype=np.uint8)
        overlay_np[mask] = (0.45 * overlay_np[mask] + 0.55 * color).astype(np.uint8)
        overlay = Image.fromarray(overlay_np)
        bbox = candidate["bbox_abs_xyxy"]
        draw = ImageDraw.Draw(overlay)
        draw.rectangle(tuple(bbox), outline=tuple(color.tolist()), width=4)
        draw.text((bbox[0] + 3, max(0, bbox[1] - 25)), candidate_id, fill=tuple(color.tolist()), font=label_font)
        overlay.save(overlay_path)

        overview_draw.rectangle(tuple(bbox), outline=tuple(color.tolist()), width=4)
        overview_draw.text((bbox[0] + 3, max(0, bbox[1] - 25)), candidate_id, fill=tuple(color.tolist()), font=label_font)
        cell = overlay.copy()
        cell.thumbnail((440, 260), Image.Resampling.LANCZOS)
        canvas = Image.new("RGB", (460, 310), "white")
        canvas.paste(cell, ((460 - cell.width) // 2, 38))
        ImageDraw.Draw(canvas).text(
            (8, 7),
            f"{candidate_id}  score={candidate['score']:.3f}",
            fill="black",
            font=_font(18),
        )
        cells.append(canvas)
        records.append(
            {
                **candidate,
                "mask_path": str(mask_path),
                "overlay_path": str(overlay_path),
                "review_status": "pending",
            }
        )

    overview.thumbnail((1380, 780), Image.Resampling.LANCZOS)
    overview_cell = Image.new("RGB", (1380, overview.height + 42), "white")
    overview_cell.paste(overview, (0, 42))
    ImageDraw.Draw(overview_cell).text(
        (8, 8), panel_title, fill="black", font=_font(22)
    )
    columns = 3
    rows = math.ceil(len(cells) / columns) if cells else 0
    panel = Image.new(
        "RGB",
        (1380, overview_cell.height + rows * 310),
        "white",
    )
    panel.paste(overview_cell, (0, 0))
    for index, cell in enumerate(cells):
        panel.paste(cell, ((index % columns) * 460, overview_cell.height + (index // columns) * 310))
    panel_path = output / "numbered_candidate_panel.jpg"
    panel.save(panel_path, quality=88)

    manifest = {
        "schema_version": schema_version,
        "status": filtered.get("status"),
        "source_image": str(Path(image_path).expanduser().resolve()),
        "candidate_count": len(records),
        "candidates": records,
        "numbered_candidate_panel": str(panel_path),
        "geometric_rejections": filtered.get("geometric_rejections") or [],
        "duplicate_rejections": filtered.get("duplicate_rejections") or [],
        "overflow_rejections": filtered.get("overflow_rejections") or [],
        "config": filtered.get("config") or {},
    }
    manifest_path = output / "candidate_manifest.json"
    temporary = manifest_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, manifest_path)
    return {"manifest": manifest, "manifest_path": str(manifest_path), "panel_path": str(panel_path)}


__all__ = [
    "TopZoneConfig",
    "filter_top_zone_candidates",
    "materialize_top_zone_review",
    "mask_overlap",
]
