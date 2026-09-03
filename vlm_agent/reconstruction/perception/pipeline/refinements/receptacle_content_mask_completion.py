#!/usr/bin/env python3


from __future__ import annotations

import copy
import json
import math
import os
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from PIL import Image
from scipy import ndimage


CONTAINER_TOKENS = {"bowl", "basket", "dish", "tray", "container"}
FRUIT_TOKENS = {
    "fruit", "fruits", "apple", "apples", "banana", "bananas",
    "orange", "oranges", "pear", "pears", "grape", "grapes",
    "lemon", "lemons", "peach", "peaches", "mango", "mangoes",
    "berry", "berries", "strawberry", "strawberries", "kiwi",
    "melon", "melons",
}


@dataclass(frozen=True)
class ReceptacleContentMaskCompletionConfig:
    maximum_targets: int = 2
    maximum_candidates_per_target: int = 8
    minimum_score: float = 0.45
    minimum_area_pixels: int = 24
    minimum_candidate_inside_fraction: float = 0.65
    minimum_retained_fraction: float = 0.25
    minimum_already_covered_fraction: float = 0.75
    bbox_expansion_fraction: float = 0.08


def _tokens(*values: Any) -> set[str]:
    return {
        token
        for value in values
        for token in re.sub(r"[^a-z0-9]+", " ", str(value or "").lower()).split()
        if token
    }


def _valid_norm_bbox(value: Any) -> bool:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return False
    if len(value) != 4:
        return False
    try:
        x1, y1, x2, y2 = (float(item) for item in value)
    except (TypeError, ValueError):
        return False
    return 0.0 <= x1 < x2 <= 1.0 and 0.0 <= y1 < y2 <= 1.0


def _expanded_bbox(bbox: Sequence[float], fraction: float) -> list[float]:
    x1, y1, x2, y2 = (float(value) for value in bbox)
    dx, dy = (x2 - x1) * fraction, (y2 - y1) * fraction
    return [
        max(0.0, x1 - dx), max(0.0, y1 - dy),
        min(1.0, x2 + dx), min(1.0, y2 + dy),
    ]


def _center_in_bbox(candidate: Sequence[float], host: Sequence[float]) -> bool:
    cx = (float(candidate[0]) + float(candidate[2])) / 2.0
    cy = (float(candidate[1]) + float(candidate[3])) / 2.0
    return float(host[0]) <= cx <= float(host[2]) and float(host[1]) <= cy <= float(host[3])


def plan_receptacle_content_mask_completion(
    pass_a_objects: Sequence[Mapping[str, Any]],
    sam3_registry: Mapping[str, Any],
    *,
    config: ReceptacleContentMaskCompletionConfig | None = None,
) -> list[dict[str, Any]]:
    """Plan explicit missing fruit only; plants are deliberately out of scope."""

    cfg = config or ReceptacleContentMaskCompletionConfig()
    if cfg.maximum_targets < 0:
        raise ValueError("maximum_targets must be non-negative")
    registry_by_id = {
        str(item.get("object_id") or ""): item
        for item in sam3_registry.get("objects") or []
        if isinstance(item, Mapping)
    }
    objects = [dict(item) for item in pass_a_objects if isinstance(item, Mapping)]
    plans = []
    for obj in objects:
        object_id = str(obj.get("object_id") or "")
        sam3 = (registry_by_id.get(object_id) or {}).get("sam3") or {}
        bbox = obj.get("bbox_2d_norm_xyxy")
        if (
            not object_id
            or str(sam3.get("status") or "") != "success"
            or not _valid_norm_bbox(bbox)
            or not (_tokens(obj.get("category"), obj.get("sam3_prompt")) & CONTAINER_TOKENS)
            or not (
                _tokens(
                    obj.get("description"),
                    obj.get("category"),
                    obj.get("sam3_prompt"),
                    obj.get("instance_name"),
                )
                & FRUIT_TOKENS
            )
        ):
            continue
        expanded = _expanded_bbox(bbox, cfg.bbox_expansion_fraction)
        reliable_existing = False
        for other in objects:
            other_id = str(other.get("object_id") or "")
            other_bbox = other.get("bbox_2d_norm_xyxy")
            other_sam3 = (registry_by_id.get(other_id) or {}).get("sam3") or {}
            if (
                other_id != object_id
                and _valid_norm_bbox(other_bbox)
                and str(other_sam3.get("status") or "") == "success"
                and _tokens(
                    other.get("category"),
                    other.get("sam3_prompt"),
                    other.get("instance_name"),
                ) & FRUIT_TOKENS
                and _center_in_bbox(other_bbox, expanded)
            ):
                reliable_existing = True
                break
        if reliable_existing:
            continue
        plans.append(
            {
                "target_source_object_id": object_id,
                "target_category": str(obj.get("category") or ""),
                "target_bbox_norm_xyxy": [float(value) for value in bbox],
                "probe_bbox_norm_xyxy": expanded,
                "prompt": "fruit",
                "category_hint": "fruit",
                "trigger_description": str(obj.get("description") or ""),
                "target_mask_path": str(sam3.get("mask_path") or ""),
            }
        )
        if len(plans) >= cfg.maximum_targets:
            break
    return plans


def _load_mask(path: str | Path, expected_shape: tuple[int, int]) -> np.ndarray:
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    mask = np.array(Image.open(resolved).convert("L")) > 0
    if mask.shape != expected_shape:
        raise ValueError(f"mask shape mismatch: {mask.shape} != {expected_shape}")
    return mask


def _mask_bbox(mask: np.ndarray) -> list[int]:
    ys, xs = np.nonzero(mask)
    if not len(xs):
        raise ValueError("empty mask")
    return [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]


def _save_mask_overlay(
    image_np: np.ndarray,
    mask: np.ndarray,
    output_dir: Path,
    stem: str,
    *,
    color: tuple[int, int, int],
) -> tuple[str, str]:
    masks_dir, overlays_dir = output_dir / "masks", output_dir / "overlays"
    masks_dir.mkdir(parents=True, exist_ok=True)
    overlays_dir.mkdir(parents=True, exist_ok=True)
    mask_path = masks_dir / f"{stem}_mask.png"
    overlay_path = overlays_dir / f"{stem}_overlay.png"
    Image.fromarray(mask.astype(np.uint8) * 255).save(mask_path)
    overlay = image_np.copy()
    overlay_color = np.array(color, dtype=np.uint8)
    overlay[mask] = (0.45 * overlay[mask] + 0.55 * overlay_color).astype(np.uint8)
    Image.fromarray(overlay).save(overlay_path)
    return str(mask_path), str(overlay_path)


def _bbox_iou(left: Sequence[int], right: Sequence[int]) -> float:
    x1, y1 = max(int(left[0]), int(right[0])), max(int(left[1]), int(right[1]))
    x2, y2 = min(int(left[2]), int(right[2])), min(int(left[3]), int(right[3]))
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    left_area = max(0, int(left[2]) - int(left[0])) * max(0, int(left[3]) - int(left[1]))
    right_area = max(0, int(right[2]) - int(right[0])) * max(0, int(right[3]) - int(right[1]))
    union = left_area + right_area - intersection
    return float(intersection / union) if union else 0.0


def run_bounded_receptacle_content_mask_completion(
    *,
    session: Any,
    infer_text_candidates_fn: Any,
    image_path: str | Path,
    pass_a_objects: Sequence[Mapping[str, Any]],
    sam3_registry: Mapping[str, Any],
    output_dir: str | Path,
    config: ReceptacleContentMaskCompletionConfig | None = None,
) -> dict[str, Any]:
    """Run bounded fruit probes and complete target container registry masks."""

    cfg = config or ReceptacleContentMaskCompletionConfig()
    if not 0.0 <= cfg.minimum_score <= 1.0:
        raise ValueError("minimum_score must be in [0, 1]")
    output = Path(output_dir).expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(output)
    output.mkdir(parents=True, exist_ok=True)
    image = Image.open(image_path).convert("RGB")
    image_np = np.array(image).astype(np.uint8)
    width, height = image.size
    shape = (height, width)
    plans = plan_receptacle_content_mask_completion(pass_a_objects, sam3_registry, config=cfg)

    successful_union = np.zeros(shape, dtype=bool)
    mask_errors = []
    for obj in sam3_registry.get("objects") or []:
        sam3 = (obj or {}).get("sam3") or {}
        if str(sam3.get("status") or "") != "success":
            continue
        try:
            successful_union |= _load_mask(sam3.get("mask_path"), shape)
        except Exception as exc:
            mask_errors.append(
                {"object_id": str((obj or {}).get("object_id") or ""),
                 "error": f"{type(exc).__name__}: {exc}"}
            )
    if mask_errors:
        raise ValueError(f"cannot build successful-mask registry: {mask_errors}")

    completed_registry = copy.deepcopy(sam3_registry)
    completed_by_id = {
        str(item.get("object_id") or ""): item
        for item in completed_registry.get("objects") or []
        if isinstance(item, Mapping)
    }
    completed_objects, probe_audits = [], []
    for plan_index, plan in enumerate(plans, start=1):
        bx = plan["probe_bbox_norm_xyxy"]
        positive_bbox = [
            float(bx[0]) * width, float(bx[1]) * height,
            float(bx[2]) * width, float(bx[3]) * height,
        ]
        raw = infer_text_candidates_fn(
            session,
            image_path,
            "fruit",
            output / f"probe_{plan_index:03d}_fruit",
            score_threshold=cfg.minimum_score,
            min_area=cfg.minimum_area_pixels,
            positive_bbox_abs_xyxy=positive_bbox,
            maximum_candidates=cfg.maximum_candidates_per_target,
        )
        x1, y1 = max(0, int(math.floor(positive_bbox[0]))), max(0, int(math.floor(positive_bbox[1])))
        x2, y2 = min(width, int(math.ceil(positive_bbox[2]))), min(height, int(math.ceil(positive_bbox[3])))
        probe_region = np.zeros(shape, dtype=bool)
        probe_region[y1:y2, x1:x2] = True
        target_mask = _load_mask(plan["target_mask_path"], shape)
        retained_masks, retained_records, rejected = [], [], []
        already_covered_records = []
        for record in raw["registry"].get("candidates") or []:
            reasons = []
            score = float(record.get("score") or 0.0)
            source_mask = _load_mask(record.get("mask_path"), shape)
            source_area = int(source_mask.sum())
            inside_fraction = float((source_mask & probe_region).sum() / source_area) if source_area else 0.0
            target_overlap_fraction = (
                float((source_mask & target_mask).sum() / source_area)
                if source_area else 0.0
            )
            retained = source_mask & probe_region & ~successful_union
            retained_area = int(retained.sum())
            retained_fraction = retained_area / source_area if source_area else 0.0
            if score < cfg.minimum_score:
                reasons.append("score_below_threshold")
            if inside_fraction < cfg.minimum_candidate_inside_fraction:
                reasons.append("insufficient_probe_bbox_coverage")
            if retained_area < cfg.minimum_area_pixels:
                reasons.append("retained_area_too_small")
            if retained_fraction < cfg.minimum_retained_fraction:
                reasons.append("mostly_existing_registry_pixels")
            record_audit = {
                "raw_candidate_id": str(record.get("candidate_id") or ""),
                "score": score,
                "area_pixels": source_area,
                "inside_fraction": round(inside_fraction, 6),
                "retained_fraction": round(retained_fraction, 6),
                "target_mask_overlap_fraction": round(target_overlap_fraction, 6),
                "source_mask_path": str(record.get("mask_path") or ""),
            }
            already_covered = (
                score >= cfg.minimum_score
                and source_area >= cfg.minimum_area_pixels
                and inside_fraction >= cfg.minimum_candidate_inside_fraction
                and target_overlap_fraction >= cfg.minimum_already_covered_fraction
            )
            if already_covered:
                already_covered_records.append(record_audit)
            elif reasons:
                rejected.append({**record_audit, "reasons": reasons})
            else:
                retained_masks.append(retained)
                retained_records.append(record_audit)

        grouped_component_count = 0
        grouped = np.logical_or.reduce(retained_masks) if retained_masks else np.zeros(shape, dtype=bool)
        if grouped.any():
            labels, count = ndimage.label(grouped, structure=np.ones((3, 3), dtype=bool))
            filtered = np.zeros(shape, dtype=bool)
            for label_id in range(1, count + 1):
                component = labels == label_id
                if int(component.sum()) >= cfg.minimum_area_pixels:
                    filtered |= component
                    grouped_component_count += 1
            grouped = filtered

        fruit_evidence_id = f"FRUIT_{plan_index:03d}"
        completion = None
        if grouped.any():
            fruit_bbox = _mask_bbox(grouped)
            fruit_mask_path, fruit_overlay_path = _save_mask_overlay(
                image_np, grouped, output / "fruit_evidence",
                fruit_evidence_id, color=(255, 75, 35),
            )
            source_id = plan["target_source_object_id"]
            target = completed_by_id.get(source_id)
            if target is None:
                raise KeyError(f"planned container missing from registry: {source_id}")
            target_sam3 = dict(target.get("sam3") or {})
            original_mask_path = str(target_sam3.get("mask_path") or "")
            original_mask = _load_mask(original_mask_path, shape)
            completed_mask = original_mask | grouped
            completed_bbox = _mask_bbox(completed_mask)
            completed_mask_path, completed_overlay_path = _save_mask_overlay(
                image_np, completed_mask, output / "completed_containers",
                f"{source_id}_fruit_container_complete", color=(50, 220, 95),
            )
            completion = {
                "policy": "fruit_only_merge_into_container",
                "source_object_id": source_id,
                "original_mask_path": original_mask_path,
                "original_area_pixels": int(original_mask.sum()),
                "fruit_evidence_id": fruit_evidence_id,
                "fruit_evidence_mask_path": fruit_mask_path,
                "fruit_evidence_overlay_path": fruit_overlay_path,
                "fruit_area_pixels": int(grouped.sum()),
                "fruit_mask_bbox_abs_xyxy": fruit_bbox,
                "grouped_component_count": grouped_component_count,
                "completed_mask_path": completed_mask_path,
                "completed_overlay_path": completed_overlay_path,
                "completed_area_pixels": int(completed_mask.sum()),
                "completed_mask_bbox_abs_xyxy": completed_bbox,
                "raw_candidate_ids": [item["raw_candidate_id"] for item in retained_records],
            }
            target_sam3.update(
                {
                    "mask_path": completed_mask_path,
                    "overlay_path": completed_overlay_path,
                    "area_pixels": int(completed_mask.sum()),
                    "mask_bbox_abs_xyxy": completed_bbox,
                    "container_content_completion": completion,
                }
            )
            vlm_bbox = target_sam3.get("vlm_bbox_abs_xyxy")
            if isinstance(vlm_bbox, Sequence) and len(vlm_bbox) == 4:
                target_sam3["vlm_bbox_iou"] = _bbox_iou(vlm_bbox, completed_bbox)
            target["sam3"] = target_sam3
            completed_objects.append(completion)
        elif already_covered_records:
            source_id = plan["target_source_object_id"]
            target = completed_by_id.get(source_id)
            if target is None:
                raise KeyError(f"planned container missing from registry: {source_id}")
            target_sam3 = dict(target.get("sam3") or {})
            original_mask_path = str(target_sam3.get("mask_path") or "")
            original_mask = _load_mask(original_mask_path, shape)
            best = max(
                already_covered_records,
                key=lambda item: (
                    float(item["target_mask_overlap_fraction"]),
                    float(item["score"]),
                ),
            )
            evidence_mask_path = str(best["source_mask_path"])
            completion = {
                "policy": "fruit_already_in_target_container_mask",
                "source_object_id": source_id,
                "original_mask_path": original_mask_path,
                "original_area_pixels": int(original_mask.sum()),
                "fruit_evidence_id": fruit_evidence_id,
                "fruit_evidence_mask_path": evidence_mask_path,
                "fruit_area_pixels": int(best["area_pixels"]),
                "target_mask_overlap_fraction": float(
                    best["target_mask_overlap_fraction"]
                ),
                "completed_mask_path": original_mask_path,
                "completed_area_pixels": int(original_mask.sum()),
                "completed_mask_bbox_abs_xyxy": _mask_bbox(original_mask),
                "raw_candidate_ids": [best["raw_candidate_id"]],
                "mask_write_required": False,
            }
            target_sam3["container_content_completion"] = completion
            target["sam3"] = target_sam3
            completed_objects.append(completion)

        probe_audits.append(
            {
                **plan,
                "raw_registry": raw.get("registry_path"),
                "raw_candidate_count": int(raw["registry"].get("candidate_count") or 0),
                "retained_raw_candidates": retained_records,
                "already_covered_raw_candidates": already_covered_records,
                "rejected_raw_candidates": rejected,
                "grouped_component_count": grouped_component_count,
                "fruit_evidence_id": fruit_evidence_id if completion else None,
                "container_completed": completion is not None,
                "completion_mode": (
                    completion.get("policy") if completion is not None else None
                ),
            }
        )

    completed_registry_path = output / "completed_sam3_registry.json"
    registry_temporary = completed_registry_path.with_suffix(".json.tmp")
    registry_temporary.write_text(
        json.dumps(completed_registry, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(registry_temporary, completed_registry_path)
    output_files = [path for path in output.rglob("*") if path.is_file()]
    audit = {
        "schema_version": "scenemosaic_bounded_fruit_container_completion",
        "status": (
            "not_applicable" if not plans else
            "containers_completed" if len(completed_objects) == len(plans) else
            "unresolved"
        ),
        "config": asdict(cfg),
        "plan_count": len(plans),
        "plans": plans,
        "probe_count": len(probe_audits),
        "probes": probe_audits,
        "completed_count": len(completed_objects),
        "completed_source_object_ids": [item["source_object_id"] for item in completed_objects],
        "completed_objects": completed_objects,
        "completed_registry_path": str(completed_registry_path),
        "policy": {
            "content_type": "fruit_only",
            "preferred_resolution": "merge_into_container_if_missing",
            "preserve_reliable_independent_fruit": True,
            "plants_in_scope": False,
        },
        "additional_model_load_count": 0,
        "additional_vlm_calls": 0,
        "storage_bytes_before_audit": sum(path.stat().st_size for path in output_files),
        "formal_outputs_modified": False,
        "hard_errors": [],
    }
    audit_path = output / "container_content_candidate_recall_audit.json"
    temporary = audit_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, audit_path)
    return {
        "completed_registry": completed_registry,
        "completed_registry_path": str(completed_registry_path),
        "completed_objects": completed_objects,
        "audit": audit,
        "audit_path": str(audit_path),
    }


__all__ = [
    "ReceptacleContentMaskCompletionConfig",
    "plan_receptacle_content_mask_completion",
    "run_bounded_receptacle_content_mask_completion",
]
