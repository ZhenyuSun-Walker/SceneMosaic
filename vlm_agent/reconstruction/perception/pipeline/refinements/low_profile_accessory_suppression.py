#!/usr/bin/env python3
"""Suppress a proved low-profile accessory that is not reconstructable alone.

This is intentionally geometric rather than category-specific. It removes a
very small, thin mask only when a larger, vertically extended retained object
directly overlaps it as a local dependent. Typical examples are a saucer or
tray beneath an already-separated candle or cup. It never edits source masks
and keeps an audit explaining every suppression.
"""

from __future__ import annotations

import copy
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from PIL import Image


@dataclass(frozen=True)
class LowProfileAccessoryConfig:
    enabled: bool = True
    isolated_attempt_only: bool = True
    maximum_area_fraction: float = 0.0005
    maximum_height_fraction: float = 0.03
    minimum_width_to_height_ratio: float = 1.5
    minimum_dependent_area_ratio: float = 1.5
    minimum_dependent_height_ratio: float = 1.5
    minimum_dependent_horizontal_coverage: float = 0.75
    minimum_mask_intersection_pixels: int = 20
    vertical_tolerance_fraction: float = 0.004


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _load_mask(path_value: Any, expected_shape: tuple[int, int]) -> np.ndarray:
    path = Path(str(path_value or "")).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    mask = np.array(Image.open(path).convert("L")) > 0
    if mask.shape != expected_shape or not bool(mask.any()):
        raise ValueError("invalid_mask_shape_or_empty")
    return mask


def _bbox(mask: np.ndarray) -> list[int]:
    ys, xs = np.where(mask)
    if not len(xs):
        raise ValueError("empty_mask")
    return [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]


def _horizontal_coverage(base: Sequence[int], dependent: Sequence[int]) -> float:
    overlap = max(
        0,
        min(int(base[2]), int(dependent[2]))
        - max(int(base[0]), int(dependent[0])),
    )
    dependent_width = max(1, int(dependent[2]) - int(dependent[0]))
    return overlap / dependent_width


def run_low_profile_accessory_suppression(
    *,
    objects: Sequence[Mapping[str, Any]],
    image_shape: tuple[int, int],
    output_dir: str | Path,
    config: LowProfileAccessoryConfig | None = None,
) -> dict[str, Any]:
    """Remove only a small horizontal accessory proved to support a child."""

    cfg = config or LowProfileAccessoryConfig()
    if not cfg.isolated_attempt_only:
        raise ValueError("low-profile accessory suppression is isolated-attempt only")
    work = Path(output_dir).expanduser().resolve()
    work.mkdir(parents=True, exist_ok=True)
    copied = [
        copy.deepcopy(dict(obj)) for obj in objects if isinstance(obj, Mapping)
    ]
    height, width = image_shape
    audit: dict[str, Any] = {
        "schema_version": "scenemosaic_low_profile_accessory_suppression",
        "enabled": cfg.enabled,
        "isolated_attempt_only": cfg.isolated_attempt_only,
        "status": "disabled" if not cfg.enabled else "noop",
        "input_object_count": len(copied),
        "suppressed": [],
        "rejected": [],
        "hard_errors": [],
        "thresholds": asdict(cfg),
    }
    if not cfg.enabled:
        _atomic_json(work / "low_profile_accessory_suppression_audit.json", audit)
        return {"objects": copied, "audit": audit}

    masks: dict[str, np.ndarray] = {}
    boxes: dict[str, list[int]] = {}
    for obj in copied:
        object_id = str(obj.get("object_id") or "")
        sam3 = obj.get("sam3") or {}
        if sam3.get("status") != "success":
            continue
        try:
            mask = _load_mask(sam3.get("mask_path"), image_shape)
            masks[object_id] = mask
            boxes[object_id] = _bbox(mask)
        except (OSError, ValueError) as exc:
            audit["hard_errors"].append(
                {
                    "type": "invalid_success_mask_for_low_profile_accessory",
                    "object_id": object_id,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )

    removal_ids: set[str] = set()
    tolerance = max(1, int(round(height * cfg.vertical_tolerance_fraction)))
    for base in copied:
        base_id = str(base.get("object_id") or "")
        if base_id not in masks:
            continue
        base_mask = masks[base_id]
        base_area = int(base_mask.sum())
        x1, y1, x2, y2 = boxes[base_id]
        base_width, base_height = x2 - x1, y2 - y1
        if (
            base_area / max(1, height * width) > cfg.maximum_area_fraction
            or base_height / max(1, height) > cfg.maximum_height_fraction
            or base_width / max(1, base_height) < cfg.minimum_width_to_height_ratio
        ):
            continue

        dependents = []
        for child in copied:
            child_id = str(child.get("object_id") or "")
            if child_id == base_id or child_id not in masks:
                continue
            child_mask = masks[child_id]
            child_area = int(child_mask.sum())
            cx1, cy1, cx2, cy2 = boxes[child_id]
            child_height = cy2 - cy1
            intersection = int(np.logical_and(base_mask, child_mask).sum())
            horizontal_coverage = _horizontal_coverage(
                boxes[base_id], boxes[child_id]
            )
            directly_supported = cy1 < y2 and cy2 >= y1 - tolerance
            if (
                child_area >= base_area * cfg.minimum_dependent_area_ratio
                and child_height >= base_height * cfg.minimum_dependent_height_ratio
                and horizontal_coverage >= cfg.minimum_dependent_horizontal_coverage
                and directly_supported
                and intersection >= cfg.minimum_mask_intersection_pixels
            ):
                dependents.append(
                    {
                        "object_id": child_id,
                        "category": str(child.get("category") or ""),
                        "area_pixels": child_area,
                        "mask_intersection_pixels": intersection,
                        "horizontal_coverage": round(horizontal_coverage, 6),
                        "bbox_abs_xyxy": [cx1, cy1, cx2, cy2],
                    }
                )
        if not dependents:
            continue
        removal_ids.add(base_id)
        audit["suppressed"].append(
            {
                "removed_object_id": base_id,
                "removed_category": str(base.get("category") or ""),
                "policy": "low_profile_accessory_with_local_dependent",
                "area_pixels": base_area,
                "bbox_abs_xyxy": [x1, y1, x2, y2],
                "dependent_objects": dependents,
            }
        )

    result = [
        obj for obj in copied if str(obj.get("object_id") or "") not in removal_ids
    ]
    audit["status"] = "pass" if removal_ids else "noop"
    audit["output_object_count"] = len(result)
    _atomic_json(work / "low_profile_accessory_suppression_audit.json", audit)
    return {"objects": result, "audit": audit}


__all__ = ["LowProfileAccessoryConfig", "run_low_profile_accessory_suppression"]
