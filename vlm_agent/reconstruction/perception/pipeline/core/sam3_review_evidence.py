#!/usr/bin/env python3
"""Build a bounded review panel from actual assigned SAM3 masks.

The panel is evidence for the existing Pass-B semantic audit. It does not
create, modify, or approve any mask and never writes to formal pipeline stages.
"""

from __future__ import annotations

import json
import math
import os
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


@dataclass(frozen=True)
class AssignedEvidenceConfig:
    maximum_objects: int = 12
    low_score_threshold: float = 0.55
    fragmented_component_count: int = 4
    minimum_component_pixels: int = 16
    crop_padding_fraction: float = 0.12
    tile_width: int = 360
    tile_height: int = 250
    panel_columns: int = 3


COMPLETENESS_TOKENS = {
    "bathtub", "bed", "bench", "cabinet", "cart", "counter", "desk", "faucet",
    "hutch", "island", "nightstand", "rack", "shelf", "shelves", "shelving",
    "sink", "table", "vanity", "workbench",
}

FIXED_COMPONENT_ASSEMBLY_TOKEN_GROUPS = (
    frozenset({"water", "dispenser"}),
)


def _font(size: int):
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size)
    except OSError:
        return ImageFont.load_default()


def _tokens(value: Any) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", str(value or "").lower().replace("_", " ")))


def _object_text(obj: Mapping[str, Any]) -> str:
    return " ".join(
        str(obj.get(key) or "")
        for key in ("category", "sam3_prompt", "instance_name", "description")
    )


def _mask_bbox(mask: np.ndarray) -> list[int] | None:
    ys, xs = np.where(mask)
    if not len(xs):
        return None
    return [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]


def _normalized_bbox(box: Sequence[int], width: int, height: int) -> list[float]:
    return [
        round(float(box[0]) / width, 6),
        round(float(box[1]) / height, 6),
        round(float(box[2] + 1) / width, 6),
        round(float(box[3] + 1) / height, 6),
    ]


def _bbox_from_object(obj: Mapping[str, Any], width: int, height: int) -> list[int]:
    bbox = obj.get("bbox_2d_norm_xyxy")
    if isinstance(bbox, list) and len(bbox) == 4:
        return [
            max(0, min(width - 1, int(math.floor(float(bbox[0]) * width)))),
            max(0, min(height - 1, int(math.floor(float(bbox[1]) * height)))),
            max(0, min(width - 1, int(math.ceil(float(bbox[2]) * width)) - 1)),
            max(0, min(height - 1, int(math.ceil(float(bbox[3]) * height)) - 1)),
        ]
    return [0, 0, width - 1, height - 1]


def _expand_bbox(
    box: Sequence[int],
    width: int,
    height: int,
    fraction: float,
) -> list[int]:
    x1, y1, x2, y2 = [int(value) for value in box]
    pad_x = max(8, int(round((x2 - x1 + 1) * fraction)))
    pad_y = max(8, int(round((y2 - y1 + 1) * fraction)))
    return [
        max(0, x1 - pad_x),
        max(0, y1 - pad_y),
        min(width - 1, x2 + pad_x),
        min(height - 1, y2 + pad_y),
    ]


def _component_count(mask: np.ndarray, minimum_pixels: int) -> int:
    count, _labels, stats, _centroids = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8), 8
    )
    return sum(
        int(stats[index, cv2.CC_STAT_AREA]) >= minimum_pixels
        for index in range(1, count)
    )


def select_assigned_sam3_evidence(
    pass_a_objects: Sequence[Mapping[str, Any]],
    sam3_registry: Mapping[str, Any],
    *,
    config: AssignedEvidenceConfig | None = None,
) -> dict[str, Any]:
    cfg = config or AssignedEvidenceConfig()
    if not 1 <= cfg.maximum_objects <= 24:
        raise ValueError("maximum_objects must be in [1, 24]")
    pass_a_by_id = {
        str(item.get("object_id") or ""): dict(item) for item in pass_a_objects
    }
    ranked: list[dict[str, Any]] = []
    hard_errors: list[dict[str, Any]] = []
    seen: set[str] = set()
    for raw in sam3_registry.get("objects") or []:
        sam3_obj = dict(raw)
        object_id = str(sam3_obj.get("object_id") or "")
        if not object_id or object_id in seen:
            hard_errors.append(
                {"type": "invalid_or_duplicate_assigned_object_id", "object_id": object_id}
            )
            continue
        seen.add(object_id)
        base = pass_a_by_id.get(object_id)
        if base is None:
            hard_errors.append(
                {"type": "assigned_object_missing_from_pass_a", "object_id": object_id}
            )
            continue
        sam3 = dict(sam3_obj.get("sam3") or {})
        status = str(sam3.get("status") or "unknown")
        merge_required = bool((sam3.get("merge_review") or {}).get("required"))
        try:
            score = float(sam3.get("score"))
        except (TypeError, ValueError):
            score = math.nan
        low_score = not math.isfinite(score) or score < cfg.low_score_threshold
        sam3_prompt = " ".join(str(base.get("sam3_prompt") or "").split())
        if not sam3_prompt:
            hard_errors.append(
                {
                    "type": "assigned_object_missing_sam3_prompt",
                    "object_id": object_id,
                }
            )
            continue
        semantic_tokens = _tokens(_object_text(base))
        completeness_sensitive = bool(semantic_tokens & COMPLETENESS_TOKENS)
        fixed_component_assembly = any(
            group <= semantic_tokens
            for group in FIXED_COMPONENT_ASSEMBLY_TOKEN_GROUPS
        )
        component_count = None
        mask_path = Path(str(sam3.get("mask_path") or "")).expanduser()
        if mask_path.is_file():
            try:
                mask = np.array(Image.open(mask_path).convert("L")) > 0
                component_count = _component_count(mask, cfg.minimum_component_pixels)
            except Exception as exc:
                hard_errors.append(
                    {
                        "type": "invalid_assigned_mask",
                        "object_id": object_id,
                        "path": str(mask_path),
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
        fragmented = (
            component_count is not None
            and component_count > cfg.fragmented_component_count
        )
        reasons = []
        priority = 0
        if status != "success":
            reasons.append("sam3_failed")
            priority += 100
        if merge_required:
            reasons.append("merge_review_required")
            priority += 90
        if fragmented:
            reasons.append("fragmented_mask")
            priority += 70
        if low_score:
            reasons.append("low_or_missing_score")
            priority += 50
        if fixed_component_assembly:
            reasons.append("fixed_component_assembly")
            priority += 40
        if completeness_sensitive:
            reasons.append("completeness_sensitive_category")
            priority += 30
        if not reasons:
            reasons.append("bounded_registry_coverage")
        bbox = base.get("bbox_2d_norm_xyxy")
        ranked.append(
            {
                "source_object_id": object_id,
                "sam3_prompt": sam3_prompt,
                "category": str(base.get("category") or ""),
                "status": status,
                "score": None if not math.isfinite(score) else round(score, 6),
                "merge_review_required": merge_required,
                "component_count": component_count,
                "selection_reasons": reasons,
                "priority": priority,
                "bbox_norm_xyxy": bbox,
                "mask_path": str(mask_path.resolve()) if mask_path.is_file() else None,
                "overlay_path": str(Path(str(sam3.get("overlay_path") or "")).expanduser()),
            }
        )
    ranked.sort(
        key=lambda item: (
            -int(item["priority"]),
            item["bbox_norm_xyxy"][1]
            if isinstance(item.get("bbox_norm_xyxy"), list)
            else 2.0,
            item["bbox_norm_xyxy"][0]
            if isinstance(item.get("bbox_norm_xyxy"), list)
            else 2.0,
            item["source_object_id"],
        )
    )
    selected = [dict(item) for item in ranked[: cfg.maximum_objects]]
    for index, item in enumerate(selected, start=1):
        item["evidence_id"] = f"E_{index:03d}"
    overflow = ranked[cfg.maximum_objects :]
    suspicious_overflow = [
        item for item in overflow if int(item["priority"]) > 0
    ]
    return {
        "schema_version": "scenemosaic_assigned_sam3_evidence_selection",
        "status": "blocked" if hard_errors or suspicious_overflow else "pass",
        "config": asdict(cfg),
        "selected": selected,
        "selected_count": len(selected),
        "overflow_count": len(overflow),
        "suspicious_overflow_count": len(suspicious_overflow),
        "suspicious_overflow_object_ids": [
            item["source_object_id"] for item in suspicious_overflow
        ],
        "hard_errors": hard_errors,
    }


def _fit(image: Image.Image, width: int, height: int) -> Image.Image:
    result = image.copy()
    result.thumbnail((width, height), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (width, height), "white")
    canvas.paste(
        result,
        ((width - result.width) // 2, (height - result.height) // 2),
    )
    return canvas


def materialize_assigned_sam3_review(
    *,
    image_path: str | Path,
    pass_a_objects: Sequence[Mapping[str, Any]],
    sam3_registry: Mapping[str, Any],
    output_dir: str | Path,
    config: AssignedEvidenceConfig | None = None,
) -> dict[str, Any]:
    cfg = config or AssignedEvidenceConfig()
    output = Path(output_dir).expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"assigned evidence output is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    selection = select_assigned_sam3_evidence(
        pass_a_objects, sam3_registry, config=cfg
    )
    source = Image.open(image_path).convert("RGB")
    width, height = source.size
    tile_width, tile_height = cfg.tile_width, cfg.tile_height
    header_height = 44
    visual_height = tile_height - header_height
    half_width = tile_width // 2
    tiles: list[Image.Image] = []
    records: list[dict[str, Any]] = []
    for item in selection["selected"]:
        mask = None
        mask_path = Path(str(item.get("mask_path") or ""))
        if mask_path.is_file():
            mask = np.array(Image.open(mask_path).convert("L")) > 0
            if mask.shape != (height, width):
                selection["hard_errors"].append(
                    {
                        "type": "assigned_mask_shape_mismatch",
                        "object_id": item["source_object_id"],
                        "actual": list(mask.shape),
                        "expected": [height, width],
                    }
                )
                mask = None
        base = next(
            obj for obj in pass_a_objects
            if str(obj.get("object_id") or "") == item["source_object_id"]
        )
        claimed_box = _bbox_from_object(base, width, height)
        mask_box = _mask_bbox(mask) if mask is not None else None
        if mask_box is None:
            evidence_box = claimed_box
            claimed_bbox_coverage_by_mask_bbox = 0.0
        else:
            evidence_box = [
                min(claimed_box[0], mask_box[0]),
                min(claimed_box[1], mask_box[1]),
                max(claimed_box[2], mask_box[2]),
                max(claimed_box[3], mask_box[3]),
            ]
            ix1 = max(claimed_box[0], mask_box[0])
            iy1 = max(claimed_box[1], mask_box[1])
            ix2 = min(claimed_box[2], mask_box[2])
            iy2 = min(claimed_box[3], mask_box[3])
            intersection = (
                max(0, ix2 - ix1 + 1) * max(0, iy2 - iy1 + 1)
            )
            claimed_area = (
                (claimed_box[2] - claimed_box[0] + 1)
                * (claimed_box[3] - claimed_box[1] + 1)
            )
            claimed_bbox_coverage_by_mask_bbox = intersection / claimed_area
        crop_box = _expand_bbox(
            evidence_box, width, height, cfg.crop_padding_fraction
        )
        x1, y1, x2, y2 = crop_box
        original_crop = source.crop((x1, y1, x2 + 1, y2 + 1))
        overlay_path = Path(str(item.get("overlay_path") or "")).expanduser()
        if overlay_path.is_file():
            overlay = Image.open(overlay_path).convert("RGB")
        else:
            overlay_np = np.array(source).copy()
            if mask is not None:
                overlay_np[mask] = (
                    0.55 * overlay_np[mask]
                    + 0.45 * np.array((255, 0, 0), dtype=np.float32)
                ).astype(np.uint8)
            overlay = Image.fromarray(overlay_np)
        overlay_crop = overlay.crop((x1, y1, x2 + 1, y2 + 1))
        tile = Image.new("RGB", (tile_width, tile_height), "white")
        tile.paste(_fit(original_crop, half_width, visual_height), (0, header_height))
        tile.paste(
            _fit(overlay_crop, tile_width - half_width, visual_height),
            (half_width, header_height),
        )
        draw = ImageDraw.Draw(tile)
        title = (
            f"{item['evidence_id']} {item['source_object_id']} "
            f"{item['sam3_prompt']} | original / mask"
        )
        draw.text((8, 5), title[:56], fill=(0, 0, 0), font=_font(15))
        draw.text(
            (8, 23),
            ",".join(item["selection_reasons"])[:70],
            fill=(90, 20, 20),
            font=_font(12),
        )
        tiles.append(tile)
        record = {
            key: value
            for key, value in item.items()
            if key not in {"priority", "mask_path", "overlay_path"}
        }
        record["claimed_bbox_abs_xyxy"] = claimed_box
        record["actual_mask_bbox_abs_xyxy"] = mask_box
        record["claimed_bbox_coverage_by_mask_bbox"] = round(
            claimed_bbox_coverage_by_mask_bbox, 6
        )
        record["crop_basis"] = "union_of_claimed_and_actual_mask_bbox"
        record["crop_bbox_abs_xyxy"] = crop_box
        record["crop_bbox_norm_xyxy"] = _normalized_bbox(
            crop_box, width, height
        )
        record["source_mask_path"] = str(mask_path.resolve()) if mask_path.is_file() else None
        records.append(record)

    columns = max(1, cfg.panel_columns)
    rows = max(1, math.ceil(len(tiles) / columns))
    panel = Image.new(
        "RGB",
        (columns * tile_width, rows * tile_height + 34),
        (238, 238, 238),
    )
    draw = ImageDraw.Draw(panel)
    draw.text(
        (8, 6),
        "Assigned SAM3 evidence: each tile shows original crop then actual mask overlay",
        fill=(0, 0, 0),
        font=_font(16),
    )
    for index, tile in enumerate(tiles):
        x = (index % columns) * tile_width
        y = 34 + (index // columns) * tile_height
        panel.paste(tile, (x, y))
    panel_path = output / "assigned_sam3_evidence_panel.jpg"
    panel.save(panel_path, quality=84, optimize=True)
    manifest = {
        **selection,
        "status": (
            "blocked"
            if selection["hard_errors"]
            or selection["suspicious_overflow_count"]
            else "pass"
        ),
        "image_path": str(Path(image_path).expanduser().resolve()),
        "records": records,
        "panel_path": str(panel_path),
        "formal_outputs_modified": False,
    }
    manifest_path = output / "assigned_sam3_evidence_manifest.json"
    temporary = manifest_path.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, manifest_path)
    return {
        "manifest": manifest,
        "manifest_path": str(manifest_path),
        "panel_path": str(panel_path),
    }
