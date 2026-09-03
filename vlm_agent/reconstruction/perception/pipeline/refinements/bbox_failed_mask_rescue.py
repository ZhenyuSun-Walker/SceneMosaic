#!/usr/bin/env python3
"""Bounded bbox-first spatial rescue for VLM objects whose text mask failed."""

from __future__ import annotations

import copy
import json
import math
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
from PIL import Image
from scipy import ndimage


@dataclass(frozen=True)
class BboxFailedMaskRescueConfig:
    enabled: bool = True
    isolated_attempt_only: bool = True
    maximum_targets: int = 5
    maximum_candidates_per_target: int = 4
    maximum_recessed_candidates_per_target: int = 12
    minimum_score: float = 0.20
    minimum_recessed_ceiling_light_score: float = 0.45
    minimum_target_bbox_coverage: float = 0.35
    minimum_candidate_bbox_coverage: float = 0.50
    minimum_mask_inside_expanded_roi: float = 0.90
    minimum_bbox_area_ratio: float = 0.25
    maximum_bbox_area_ratio: float = 3.00
    minimum_rank_margin: float = 0.15
    roi_expansion_fraction: float = 0.35
    maximum_existing_iou: float = 0.35
    maximum_existing_candidate_coverage: float = 0.50
    minimum_conflict_pixels: int = 30
    minimum_successful_replacement_foreign_overlap_pixels: int = 64
    minimum_successful_replacement_other_object_coverage: float = 0.05
    # Bbox rescue is a fallback for failed or missing masks only.  A valid
    # successful SAM3 mask is never replaced merely because its extent differs
    # from a VLM box.
    repair_successful_low_extent: bool = False
    maximum_successful_vlm_bbox_iou: float = 0.35
    minimum_successful_target_bbox_coverage: float = 0.75
    minimum_successful_replacement_score_gain: float = 0.02
    minimum_successful_replacement_target_coverage_gain: float = 0.05
    repair_successful_low_extent_lighting: bool = False
    repair_failed_lighting: bool = False
    repair_failed_recessed_ceiling_lights: bool = True
    # Source IDs shown in Pass-B's bounded evidence panel.  These are the
    # only successful-low-extent objects whose declared completeness has been
    # explicitly reviewed by the VLM, so they outrank incidental loose boxes.
    priority_source_object_ids: frozenset[str] = frozenset()
    reject_disconnected_generic_successful_replacement: bool = True
    disconnected_component_min_area_fraction: float = 0.20
    disconnected_component_min_pixels: int = 128
    prefer_compact_nested_isolated_vase: bool = True
    compact_vase_minimum_containment: float = 0.95
    compact_vase_minimum_area_ratio: float = 0.75
    compact_vase_minimum_target_coverage: float = 0.50
    compact_vase_maximum_width_ratio: float = 0.85
    compact_vase_minimum_height_ratio: float = 0.80
    compact_vase_minimum_removed_pixels: int = 128
    preserve_compact_isolated_vase_success: bool = True
    compact_vase_minimum_existing_bbox_iou: float = 0.45
    compact_vase_minimum_existing_target_coverage: float = 0.50


ELIGIBLE_REASONS = {
    "no_candidates_from_text_prompt",
    "no_local_candidate",
}
GENERIC_TYPES = {
    "",
    "unknown",
    "object",
    "thing",
}


def normalize_prompt(value: Any) -> str:
    return " ".join(re.sub(r"[^a-z0-9]+", " ", str(value or "").lower()).split())


def is_lighting_object(obj: Mapping[str, Any]) -> bool:
    tokens = set(
        (
            normalize_prompt(obj.get("category"))
            + " "
            + normalize_prompt(obj.get("object_type"))
        ).split()
    )
    return bool(tokens & {"light", "lighting", "lamp", "downlight"})


def is_recessed_ceiling_light(obj: Mapping[str, Any]) -> bool:
    tokens = set(
        (
            normalize_prompt(obj.get("category"))
            + " "
            + normalize_prompt(obj.get("sam3_prompt"))
            + " "
            + normalize_prompt(obj.get("description"))
        ).split()
    )
    return "recessed" in tokens and "ceiling" in tokens and "light" in tokens


def _vase_tokens(obj: Mapping[str, Any]) -> set[str]:
    return set(
        (
            normalize_prompt(obj.get("category"))
            + " "
            + normalize_prompt(obj.get("sam3_prompt"))
            + " "
            + normalize_prompt(obj.get("description"))
        ).split()
    )


def is_vase_botanical_assembly(obj: Mapping[str, Any]) -> bool:
    """Whether one declaration names both a vase and its botanical content."""

    tokens = _vase_tokens(obj)
    return "vase" in tokens and bool(
        tokens
        & {
            "branch",
            "branches",
            "flower",
            "flowers",
            "floral",
            "foliage",
            "greenery",
            "plant",
            "arrangement",
        }
    )


def is_isolated_vase(obj: Mapping[str, Any]) -> bool:
    """Return true only for a vase declaration without botanical contents."""

    return "vase" in _vase_tokens(obj) and not is_vase_botanical_assembly(obj)


def _bbox_axis_gap(left: int, right: int, other_left: int, other_right: int) -> int:
    return max(0, other_left - right, left - other_right)


def rescue_prompt(obj: Mapping[str, Any]) -> str | None:
    """Prefer a broad VLM object type so a possibly wrong category is not repeated."""

    if is_recessed_ceiling_light(obj):
        return "recessed ceiling light"
    if is_vase_botanical_assembly(obj):
        # Preserve both the vessel and botanical terms.  This is intentionally
        # not an ``only`` prompt: the declared instance is one assembly.
        return (
            normalize_prompt(obj.get("category"))
            or normalize_prompt(obj.get("description"))
            or normalize_prompt(obj.get("sam3_prompt"))
        )
    if is_isolated_vase(obj):
        description = (
            normalize_prompt(obj.get("description"))
            or normalize_prompt(obj.get("sam3_prompt"))
            or normalize_prompt(obj.get("category"))
        )
        return description if description.endswith(" only") else f"{description} only"
    object_type = normalize_prompt(obj.get("object_type"))
    if object_type not in GENERIC_TYPES:
        return object_type
    category = normalize_prompt(obj.get("category"))
    return category or None


def is_broad_object_type_prompt(obj: Mapping[str, Any], prompt: str) -> bool:
    """Whether rescue fell back from an instance prompt to a generic type."""

    object_type = normalize_prompt(obj.get("object_type"))
    category = normalize_prompt(obj.get("category"))
    return (
        object_type not in GENERIC_TYPES
        and normalize_prompt(prompt) == object_type
        and object_type != category
    )


def disconnected_generic_replacement_evidence(
    candidate: np.ndarray,
    existing: np.ndarray,
    cfg: BboxFailedMaskRescueConfig,
) -> dict[str, Any] | None:
    """Detect a second ordinary instance in a broad-prompt rescue candidate.

    This guard is deliberately restricted to a successful existing mask plus a
    generic object-type fallback.  It does not treat ordinary disconnected parts
    of a known functional assembly as evidence to reject a specific semantic
    prompt.
    """

    labels, count = ndimage.label(
        np.asarray(candidate, dtype=bool),
        structure=np.ones((3, 3), dtype=bool),
    )
    if int(count) < 2:
        return None
    candidate_area = int(candidate.sum())
    existing_area = max(1, int(existing.sum()))
    rows: list[dict[str, Any]] = []
    for index in range(1, int(count) + 1):
        component = labels == index
        area = int(component.sum())
        overlap = int(np.logical_and(component, existing).sum())
        rows.append({
            "component_index": index,
            "area_pixels": area,
            "candidate_area_fraction": round(area / max(1, candidate_area), 6),
            "existing_coverage": round(overlap / existing_area, 6),
            "existing_component_coverage": round(overlap / max(1, area), 6),
        })
    matched = max(rows, key=lambda item: float(item["existing_coverage"]))
    if float(matched["existing_coverage"]) < 0.35:
        return None
    minimum_area = max(
        int(cfg.disconnected_component_min_pixels),
        int(round(candidate_area * cfg.disconnected_component_min_area_fraction)),
    )
    foreign = [
        item
        for item in rows
        if item["component_index"] != matched["component_index"]
        and int(item["area_pixels"]) >= minimum_area
        and float(item["existing_component_coverage"]) < 0.10
    ]
    if not foreign:
        return None
    return {
        "matched_existing_component": matched,
        "foreign_components": foreign,
        "minimum_foreign_area_pixels": minimum_area,
    }


def norm_bbox_to_abs(raw: Any, width: int, height: int) -> list[int] | None:
    try:
        values = [float(value) for value in raw]
    except (TypeError, ValueError):
        return None
    if (
        len(values) != 4
        or not all(math.isfinite(value) for value in values)
        or not (0.0 <= values[0] < values[2] <= 1.0)
        or not (0.0 <= values[1] < values[3] <= 1.0)
    ):
        return None
    result = [
        int(round(values[0] * width)),
        int(round(values[1] * height)),
        int(round(values[2] * width)),
        int(round(values[3] * height)),
    ]
    result[0] = max(0, min(width - 1, result[0]))
    result[1] = max(0, min(height - 1, result[1]))
    result[2] = max(result[0] + 1, min(width, result[2]))
    result[3] = max(result[1] + 1, min(height, result[3]))
    return result


def expand_bbox(box: Sequence[int], width: int, height: int, fraction: float) -> list[int]:
    x1, y1, x2, y2 = [int(value) for value in box]
    px = int(round((x2 - x1) * fraction))
    py = int(round((y2 - y1) * fraction))
    return [max(0, x1 - px), max(0, y1 - py), min(width, x2 + px), min(height, y2 + py)]


def bbox_metrics(target: Sequence[int], candidate: Sequence[int]) -> dict[str, float]:
    tx1, ty1, tx2, ty2 = [int(value) for value in target]
    cx1, cy1, cx2, cy2 = [int(value) for value in candidate]
    ix1, iy1, ix2, iy2 = max(tx1, cx1), max(ty1, cy1), min(tx2, cx2), min(ty2, cy2)
    intersection = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    target_area = max(1, (tx2 - tx1) * (ty2 - ty1))
    candidate_area = max(1, (cx2 - cx1) * (cy2 - cy1))
    union = target_area + candidate_area - intersection
    return {
        "iou": intersection / union if union else 0.0,
        "target_coverage": intersection / target_area,
        "candidate_coverage": intersection / candidate_area,
        "bbox_area_ratio": candidate_area / target_area,
    }


def candidate_evidence(
    obj: Mapping[str, Any],
    mask: np.ndarray,
    score: float,
    target_bbox: Sequence[int],
    cfg: BboxFailedMaskRescueConfig,
) -> dict[str, Any]:
    height, width = mask.shape
    ys, xs = np.nonzero(mask)
    if not len(xs):
        return {"status": "reject", "reason": "empty_mask"}
    candidate_bbox = [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]
    metrics = bbox_metrics(target_bbox, candidate_bbox)
    expanded = expand_bbox(target_bbox, width, height, cfg.roi_expansion_fraction)
    inside = int(mask[expanded[1] : expanded[3], expanded[0] : expanded[2]].sum())
    mask_inside_roi = inside / max(1, int(mask.sum()))
    anchor_in_mask = False
    anchor = obj.get("anchor_2d_norm_xy")
    if isinstance(anchor, (list, tuple)) and len(anchor) == 2:
        try:
            ax = max(0, min(width - 1, int(round(float(anchor[0]) * width))))
            ay = max(0, min(height - 1, int(round(float(anchor[1]) * height))))
            anchor_in_mask = bool(mask[ay, ax])
        except (TypeError, ValueError):
            anchor_in_mask = False
    strong_bbox_without_anchor = (
        (obj.get("sam3") or {}).get("status") == "success"
        and score >= 0.85
        and metrics["target_coverage"] >= 0.75
        and metrics["candidate_coverage"] >= 0.75
        and mask_inside_roi >= cfg.minimum_mask_inside_expanded_roi
        and cfg.minimum_bbox_area_ratio <= metrics["bbox_area_ratio"] <= cfg.maximum_bbox_area_ratio
    )
    target_width = max(1, int(target_bbox[2]) - int(target_bbox[0]))
    target_height = max(1, int(target_bbox[3]) - int(target_bbox[1]))
    nearby_recessed_ceiling_light = (
        is_recessed_ceiling_light(obj)
        and score >= 0.65
        and _bbox_axis_gap(
            int(target_bbox[0]), int(target_bbox[2]),
            candidate_bbox[0], candidate_bbox[2],
        ) <= max(6, int(round(0.75 * target_width)))
        and _bbox_axis_gap(
            int(target_bbox[1]), int(target_bbox[3]),
            candidate_bbox[1], candidate_bbox[3],
        ) <= max(4, int(round(0.75 * target_height)))
        and 0.20 <= metrics["bbox_area_ratio"] <= 3.00
    )
    # A tiny recessed light often occupies only the bright center inside an
    # intentionally loose VLM box.  It is safe to recover only when the local
    # mask is almost wholly contained by a *micro* target box; this does not
    # relax generic lighting or permit a nearby row member to substitute.
    micro_contained_recessed_ceiling_light = (
        is_recessed_ceiling_light(obj)
        and score >= cfg.minimum_score
        and target_width * target_height <= 400
        and metrics["target_coverage"] >= 0.15
        and metrics["candidate_coverage"] >= 0.95
        and mask_inside_roi >= cfg.minimum_mask_inside_expanded_roi
        and 0.12 <= metrics["bbox_area_ratio"] <= 0.35
    )
    recessed_score_ok = (
        not is_recessed_ceiling_light(obj)
        or score >= cfg.minimum_recessed_ceiling_light_score
        or nearby_recessed_ceiling_light
        or micro_contained_recessed_ceiling_light
    )
    accepted = (
        score >= cfg.minimum_score
        and recessed_score_ok
        and (
            (
                anchor_in_mask
                and metrics["target_coverage"] >= cfg.minimum_target_bbox_coverage
                and metrics["candidate_coverage"] >= cfg.minimum_candidate_bbox_coverage
                and mask_inside_roi >= cfg.minimum_mask_inside_expanded_roi
                and cfg.minimum_bbox_area_ratio <= metrics["bbox_area_ratio"] <= cfg.maximum_bbox_area_ratio
            )
            or strong_bbox_without_anchor
            or nearby_recessed_ceiling_light
            or micro_contained_recessed_ceiling_light
        )
    )
    rank = (
        2.0 * metrics["iou"]
        + 1.5 * metrics["target_coverage"]
        + metrics["candidate_coverage"]
        + mask_inside_roi
        + 0.25 * score
    )
    return {
        "status": "accept" if accepted else "reject",
        "reason": (
            None
            if accepted
            else (
                "recessed_semantic_score_below_threshold"
                if not recessed_score_ok
                else "spatial_gate_failed"
            )
        ),
        "target_bbox_abs_xyxy": list(target_bbox),
        "candidate_bbox_abs_xyxy": candidate_bbox,
        "score": round(float(score), 6),
        "anchor_in_mask": anchor_in_mask,
        "strong_bbox_without_anchor": strong_bbox_without_anchor,
        "nearby_recessed_ceiling_light": nearby_recessed_ceiling_light,
        "micro_contained_recessed_ceiling_light": micro_contained_recessed_ceiling_light,
        "mask_inside_expanded_roi": round(mask_inside_roi, 6),
        "rank": round(rank, 6),
        **{key: round(value, 6) for key, value in metrics.items()},
    }


def overlap_conflict(
    candidate: np.ndarray,
    existing: np.ndarray,
    cfg: BboxFailedMaskRescueConfig,
) -> dict[str, Any] | None:
    intersection = int(np.logical_and(candidate, existing).sum())
    if intersection < cfg.minimum_conflict_pixels:
        return None
    ca, ea = int(candidate.sum()), int(existing.sum())
    union = ca + ea - intersection
    iou = intersection / union if union else 0.0
    coverage = intersection / ca if ca else 0.0
    if iou < cfg.maximum_existing_iou and coverage < cfg.maximum_existing_candidate_coverage:
        return None
    return {"intersection_pixels": intersection, "iou": round(iou, 6), "candidate_coverage": round(coverage, 6)}


def successful_replacement_foreign_overlap(
    candidate: np.ndarray,
    existing: np.ndarray,
    cfg: BboxFailedMaskRescueConfig,
) -> dict[str, Any] | None:
    """Reject a broad successful-mask replacement that steals a named peer."""

    intersection = int(np.logical_and(candidate, existing).sum())
    if intersection < cfg.minimum_successful_replacement_foreign_overlap_pixels:
        return None
    candidate_area = max(1, int(candidate.sum()))
    existing_area = max(1, int(existing.sum()))
    existing_coverage = intersection / existing_area
    if existing_coverage < cfg.minimum_successful_replacement_other_object_coverage:
        return None
    return {
        "intersection_pixels": intersection,
        "candidate_coverage": round(intersection / candidate_area, 6),
        "existing_object_coverage": round(existing_coverage, 6),
    }


def _load_mask(path_value: Any, shape: tuple[int, int], required_root: Path | None = None) -> tuple[np.ndarray, Path]:
    path = Path(str(path_value or "")).expanduser().resolve()
    if required_root is not None and not path.is_relative_to(required_root.resolve()):
        raise ValueError("candidate mask escapes bbox rescue root")
    mask = np.array(Image.open(path).convert("L")) > 0
    if mask.shape != shape or not mask.any():
        raise ValueError("invalid candidate mask")
    return mask, path


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def run_bbox_failed_mask_rescue(
    *,
    session: Any,
    image_path: str | Path,
    objects: Sequence[Mapping[str, Any]],
    work_dir: str | Path,
    staging_dir: str | Path,
    infer_candidates_fn: Callable[..., Mapping[str, Any]],
    config: BboxFailedMaskRescueConfig | None = None,
    unresolved_source_object_ids: Sequence[str] | None = None,
) -> dict[str, Any]:
    cfg = config or BboxFailedMaskRescueConfig()
    if not cfg.isolated_attempt_only:
        raise ValueError("bbox rescue is isolated-attempt only")
    copied = [copy.deepcopy(dict(item)) for item in objects]
    work = Path(work_dir).expanduser().resolve()
    if work.exists() and any(work.iterdir()):
        raise FileExistsError(f"bbox rescue work directory is not empty: {work}")
    work.mkdir(parents=True, exist_ok=True)
    audit = {
        "schema_version": "scenemosaic_bbox_failed_mask_rescue",
        "enabled": cfg.enabled,
        "status": "disabled" if not cfg.enabled else "pending",
        "maximum_targets": cfg.maximum_targets,
        "target_count": 0,
        "inference_call_count": 0,
        "accepted_count": 0,
        "accepted": [],
        "rejected": [],
        "skipped": [],
        "resolved_prior_retry_error_object_ids": [],
        "excluded_unmaterialized": [],
        "excluded_unmaterialized_source_object_ids": [],
        "nonblocking_errors": [],
        "hard_errors": [],
        "missing_success_mask_object_ids": [],
        "additional_vlm_calls": 0,
    }
    audit_path = work / "bbox_failed_mask_rescue_audit.json"
    if not cfg.enabled:
        _atomic_json(audit_path, audit)
        return {"objects": copied, "audit": audit, "audit_path": str(audit_path)}

    image = Path(image_path).expanduser().resolve()
    with Image.open(image) as source:
        image_np = np.array(source.convert("RGB"))
    height, width = image_np.shape[:2]
    shape = (height, width)
    existing_masks: dict[str, np.ndarray] = {}
    for obj in copied:
        sam3 = obj.get("sam3") or {}
        if sam3.get("status") != "success":
            continue
        try:
            existing_masks[str(obj.get("object_id") or "")], _ = _load_mask(sam3.get("mask_path"), shape)
        except (OSError, ValueError):
            # Treat a stale or invalid mask path as a missing mask.  It is a
            # rescue target, not a terminal error, provided the local rescue
            # can materialize a valid replacement.
            audit["missing_success_mask_object_ids"].append(
                str(obj.get("object_id") or "")
            )

    eligible = []
    objects_by_id = {
        str(obj.get("object_id") or ""): obj for obj in copied
    }
    suppressed_duplicate_addition_object_ids: set[str] = set()
    suppressed_failed_original_object_ids: set[str] = set()
    for obj in copied:
        sam3 = obj.get("sam3") or {}
        object_id = str(obj.get("object_id") or "")
        target_bbox = norm_bbox_to_abs(obj.get("bbox_2d_norm_xyxy"), width, height)
        prompt = rescue_prompt(obj)
        existing_mask = existing_masks.get(object_id)
        has_valid_success_mask = (
            sam3.get("status") == "success" and existing_mask is not None
        )
        failed_or_missing_mask = not has_valid_success_mask
        try:
            existing_bbox_iou = float(sam3.get("vlm_bbox_iou"))
        except (TypeError, ValueError):
            existing_bbox_iou = math.nan
        existing_target_bbox_coverage = math.nan
        if existing_mask is not None and target_bbox is not None:
            ys, xs = np.nonzero(existing_mask)
            if len(xs):
                existing_bbox = [
                    int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1,
                ]
                existing_target_bbox_coverage = bbox_metrics(
                    target_bbox, existing_bbox,
                )["target_coverage"]
        successful_low_iou = (
            math.isfinite(existing_bbox_iou)
            and existing_bbox_iou < cfg.maximum_successful_vlm_bbox_iou
        )
        preserve_compact_isolated_vase_success = (
            cfg.preserve_compact_isolated_vase_success
            and sam3.get("status") == "success"
            and is_isolated_vase(obj)
            and math.isfinite(existing_bbox_iou)
            and existing_bbox_iou
            >= cfg.compact_vase_minimum_existing_bbox_iou
            and math.isfinite(existing_target_bbox_coverage)
            and existing_target_bbox_coverage
            >= cfg.compact_vase_minimum_existing_target_coverage
        )
        successful_low_coverage = (
            math.isfinite(existing_target_bbox_coverage)
            and existing_target_bbox_coverage
            < cfg.minimum_successful_target_bbox_coverage
            and not preserve_compact_isolated_vase_success
        )
        successful_low_extent = (
            cfg.repair_successful_low_extent
            and sam3.get("status") == "success"
            and (successful_low_iou or successful_low_coverage)
            and (
                cfg.repair_successful_low_extent_lighting
                or not is_lighting_object(obj)
            )
        )
        failed_recessed_ceiling_light = (
            failed_or_missing_mask
            and is_recessed_ceiling_light(obj)
            and cfg.repair_failed_recessed_ceiling_lights
        )
        # A broad type such as "furniture" can locate a nearby object, but it
        # is not evidence that a successful instance mask is wrong. Keep the
        # verified mask rather than replacing it with a generic query merely
        # because the VLM bbox and mask bbox disagree. Failed masks remain
        # eligible for the bounded spatial rescue.
        successful_broad_replacement_forbidden = (
            successful_low_extent and is_broad_object_type_prompt(obj, prompt or "")
        )
        if (
            failed_or_missing_mask
            and target_bbox
            and prompt
        ):
            source_passes = {
                str(source.get("pass") or "").upper()
                for source in (obj.get("proposal_sources") or [])
                if isinstance(source, Mapping)
            }
            source_object_ids = {
                str(source.get("source_id") or "")
                for source in (obj.get("proposal_sources") or [])
                if isinstance(source, Mapping)
            }
            evidence_priority = bool(
                source_object_ids & cfg.priority_source_object_ids
            )
            # Prioritize explicitly reviewed evidence, then failed Pass-B
            # additions, then every other failed-or-missing object.  Valid
            # successful masks never enter this queue.
            priority = (
                0 if evidence_priority
                else 1 if "B" in source_passes
                else 2
            )
            eligible.append(
                (
                    priority,
                    obj,
                    object_id,
                    target_bbox,
                    prompt,
                    (
                        "failed_recessed_ceiling_light"
                        if failed_recessed_ceiling_light
                        else (
                            "missing_success_mask"
                            if sam3.get("status") == "success"
                            else "failed_mask"
                        )
                    ),
                )
            )
        else:
            audit["skipped"].append({
                "object_id": object_id,
                "reason": (
                    "preserved_compact_isolated_vase_success"
                    if preserve_compact_isolated_vase_success
                    else "valid_success_mask"
                ),
            })
    eligible.sort(key=lambda item: (item[0], item[2]))
    audit["target_count"] = min(len(eligible), cfg.maximum_targets)
    for overflow in eligible[cfg.maximum_targets :]:
        audit["rejected"].append({"object_id": overflow[2], "reason": "target_budget_exceeded"})

    for _, obj, object_id, target_bbox, prompt, rescue_kind in eligible[: cfg.maximum_targets]:
        target_work = work / object_id
        source_passes = {
            str(source.get("pass") or "").upper()
            for source in (obj.get("proposal_sources") or [])
            if isinstance(source, Mapping)
        }
        is_pass_b_addition = "B" in source_passes and "A" not in source_passes
        duplicate_conflicts = []
        replacement_conflicts = []
        prior_target_mask = existing_masks.get(object_id)
        audit["inference_call_count"] += 1
        try:
            result = infer_candidates_fn(
                session,
                image,
                prompt,
                target_work,
                score_threshold=cfg.minimum_score,
                positive_bbox_abs_xyxy=target_bbox,
                maximum_candidates=(
                    max(
                        cfg.maximum_candidates_per_target,
                        cfg.maximum_recessed_candidates_per_target,
                    )
                    if rescue_kind == "failed_recessed_ceiling_light"
                    else cfg.maximum_candidates_per_target
                ),
            )
            registry = result.get("registry") or {}
            records = registry.get("candidates") or []
        except Exception as exc:
            audit["hard_errors"].append({"type": "bbox_rescue_inference_exception", "object_id": object_id, "error": f"{type(exc).__name__}: {exc}"})
            continue
        viable = []
        for record in records:
            try:
                mask, mask_path = _load_mask(record.get("mask_path"), shape, target_work)
                evidence = candidate_evidence(obj, mask, float(record.get("score") or 0.0), target_bbox, cfg)
            except (OSError, ValueError) as exc:
                audit["rejected"].append({"object_id": object_id, "candidate_id": record.get("candidate_id"), "reason": f"invalid_candidate:{exc}"})
                continue
            if evidence["status"] != "accept":
                audit["rejected"].append({"object_id": object_id, "candidate_id": record.get("candidate_id"), "reason": evidence["reason"], "evidence": evidence})
                continue
            if rescue_kind == "successful_low_extent":
                prior_sam3 = dict(obj.get("sam3") or {})
                try:
                    prior_score = float(prior_sam3.get("score"))
                except (TypeError, ValueError):
                    prior_score = math.nan
                prior_bbox = prior_sam3.get("mask_bbox_abs_xyxy")
                prior_metrics = None
                if (
                    isinstance(prior_bbox, (list, tuple))
                    and len(prior_bbox) == 4
                ):
                    try:
                        prior_metrics = bbox_metrics(target_bbox, prior_bbox)
                    except (TypeError, ValueError):
                        prior_metrics = None
                score_improved = (
                    math.isfinite(prior_score)
                    and float(evidence["score"])
                    >= prior_score + cfg.minimum_successful_replacement_score_gain
                )
                coverage_improved = (
                    prior_metrics is not None
                    and float(evidence["target_coverage"])
                    >= float(prior_metrics["target_coverage"])
                    + cfg.minimum_successful_replacement_target_coverage_gain
                )
                if not (score_improved and coverage_improved):
                    audit["rejected"].append({
                        "object_id": object_id,
                        "candidate_id": record.get("candidate_id"),
                        "reason": "successful_mask_replacement_not_proven_better",
                        "evidence": evidence,
                        "prior_score": (
                            round(prior_score, 6)
                            if math.isfinite(prior_score) else None
                        ),
                        "prior_target_bbox_coverage": (
                            round(float(prior_metrics["target_coverage"]), 6)
                            if prior_metrics is not None else None
                        ),
                    })
                    continue
            disconnected = None
            if (
                cfg.reject_disconnected_generic_successful_replacement
                and prior_target_mask is not None
                and is_broad_object_type_prompt(obj, prompt)
            ):
                disconnected = disconnected_generic_replacement_evidence(
                    mask,
                    prior_target_mask,
                    cfg,
                )
            if disconnected is not None:
                audit["rejected"].append({
                    "object_id": object_id,
                    "candidate_id": record.get("candidate_id"),
                    "reason": "generic_rescue_contains_significant_disconnected_foreign_component",
                    "evidence": evidence,
                    "disconnected_component_evidence": disconnected,
                })
                continue
            conflicts = []
            strict_foreign_conflicts = []
            for existing_id, existing_mask in existing_masks.items():
                if existing_id == object_id:
                    continue
                conflict = overlap_conflict(mask, existing_mask, cfg)
                if conflict is not None:
                    conflicts.append({"existing_object_id": existing_id, **conflict})
                if (
                    rescue_kind == "successful_low_extent"
                    and prior_target_mask is not None
                    and is_broad_object_type_prompt(obj, prompt)
                ):
                    strict_conflict = successful_replacement_foreign_overlap(
                        mask, existing_mask, cfg
                    )
                    if strict_conflict is not None:
                        strict_foreign_conflicts.append({
                            "existing_object_id": existing_id,
                            **strict_conflict,
                        })
            if conflicts or strict_foreign_conflicts:
                rejection = {
                    "object_id": object_id,
                    "candidate_id": record.get("candidate_id"),
                    "reason": (
                        "generic_rescue_overlaps_existing_success_object"
                        if strict_foreign_conflicts
                        else "overlaps_existing_success"
                    ),
                    "conflicts": conflicts,
                    "strict_foreign_conflicts": strict_foreign_conflicts,
                }
                audit["rejected"].append(rejection)
                if (
                    rescue_kind == "failed_recessed_ceiling_light"
                    and bool(evidence.get("nearby_recessed_ceiling_light"))
                ):
                    for item in conflicts:
                        replacement = objects_by_id.get(
                            str(item.get("existing_object_id") or "")
                        )
                        replacement_passes = {
                            str(source.get("pass") or "").upper()
                            for source in (replacement or {}).get("proposal_sources") or []
                            if isinstance(source, Mapping)
                        }
                        if (
                            replacement is not None
                            and is_recessed_ceiling_light(replacement)
                            and "B" in replacement_passes
                            and float(item.get("candidate_coverage") or 0.0) >= 0.90
                        ):
                            replacement_conflicts.append({
                                "replacement_object_id": replacement.get("object_id"),
                                "candidate_id": record.get("candidate_id"),
                                "candidate_coverage": item.get("candidate_coverage"),
                                "evidence": evidence,
                            })
                if (
                    is_pass_b_addition
                    and any(
                        float(item.get("candidate_coverage") or 0.0) >= 0.90
                        for item in conflicts
                    )
                ):
                    duplicate_conflicts.append(rejection)
                continue
            viable.append({"record": record, "mask": mask, "mask_path": mask_path, "evidence": evidence})
        viable.sort(key=lambda item: float(item["evidence"]["rank"]), reverse=True)
        if not viable:
            if replacement_conflicts:
                replacement_ids = sorted({
                    str(item.get("replacement_object_id") or "")
                    for item in replacement_conflicts
                    if str(item.get("replacement_object_id") or "")
                })
                if len(replacement_ids) != 1:
                    audit["hard_errors"].append({
                        "type": "bbox_spatial_replacement_ambiguous",
                        "object_id": object_id,
                        "replacement_object_ids": replacement_ids,
                    })
                    continue
                replacement_id = replacement_ids[0]
                replacement = objects_by_id[replacement_id]
                replacement_sources = list(replacement.get("proposal_sources") or [])
                source_keys = {
                    (str(source.get("pass") or ""), str(source.get("source_id") or ""))
                    for source in replacement_sources
                    if isinstance(source, Mapping)
                }
                for source in obj.get("proposal_sources") or []:
                    if not isinstance(source, Mapping):
                        continue
                    key = (str(source.get("pass") or ""), str(source.get("source_id") or ""))
                    if key in source_keys:
                        continue
                    carried = copy.deepcopy(dict(source))
                    carried["action"] = "nearby_spatial_replacement"
                    replacement_sources.append(carried)
                    source_keys.add(key)
                replacement["proposal_sources"] = replacement_sources
                replacement_sam3 = copy.deepcopy(dict(replacement.get("sam3") or {}))
                replacement_sam3["prior_selection_method"] = replacement_sam3.get(
                    "selection_method"
                )
                replacement_sam3["selection_method"] = "bbox_first_spatial_rescue"
                replacement_sam3["nearby_failed_original_replacement"] = {
                    "failed_object_id": object_id,
                    "replacement_object_id": replacement_id,
                    "prompt": prompt,
                    "candidate_id": replacement_conflicts[0].get("candidate_id"),
                    "evidence": replacement_conflicts[0].get("evidence"),
                }
                replacement["sam3"] = replacement_sam3
                suppressed_failed_original_object_ids.add(object_id)
                audit["accepted"].append({
                    "object_id": object_id,
                    "replacement_object_id": replacement_id,
                    "category_preserved": obj.get("category"),
                    "rescue_kind": "failed_recessed_light_reused_nearby_pass_b_success",
                    "evidence": replacement_conflicts[0].get("evidence"),
                })
                audit["resolved_prior_retry_error_object_ids"].append(object_id)
                continue
            if duplicate_conflicts:
                if prior_target_mask is not None:
                    audit["skipped"].append({
                        "object_id": object_id,
                        "reason": "successful_mask_preserved_after_conflicting_rescue",
                        "prompt": prompt,
                        "conflicts": duplicate_conflicts,
                    })
                    continue
                suppressed_duplicate_addition_object_ids.add(object_id)
                audit.setdefault("suppressed_duplicate_additions", []).append({
                    "object_id": object_id,
                    "reason": "failed_pass_b_addition_fully_covered_by_existing_success",
                    "conflicts": duplicate_conflicts,
                })
                continue
            if prior_target_mask is not None:
                audit["skipped"].append({
                    "object_id": object_id,
                    "reason": "successful_mask_preserved_after_unsuccessful_rescue",
                    "prompt": prompt,
                })
                continue
            unsafe_generic_rejection = any(
                item.get("object_id") == object_id
                and item.get("reason") in {
                    "generic_rescue_contains_significant_disconnected_foreign_component",
                    "generic_rescue_overlaps_existing_success_object",
                }
                for item in audit["rejected"]
                if isinstance(item, Mapping)
            )
            if prior_target_mask is not None and unsafe_generic_rejection:
                audit["skipped"].append({
                    "object_id": object_id,
                    "reason": "unsafe_generic_rescue_preserved_existing_success",
                    "prompt": prompt,
                })
                continue
            audit["hard_errors"].append({"type": "bbox_spatial_rescue_failed", "object_id": object_id, "prompt": prompt})
            continue
        if len(viable) > 1 and float(viable[0]["evidence"]["rank"]) - float(viable[1]["evidence"]["rank"]) < cfg.minimum_rank_margin:
            if prior_target_mask is not None:
                audit["skipped"].append({
                    "object_id": object_id,
                    "reason": "successful_mask_preserved_after_ambiguous_rescue",
                    "prompt": prompt,
                    "remaining_candidate_count": len(viable),
                    "remaining_rank_margin": round(
                        float(viable[0]["evidence"]["rank"])
                        - float(viable[1]["evidence"]["rank"]),
                        6,
                    ),
                })
                continue
            unsafe_generic_rejection = any(
                item.get("object_id") == object_id
                and item.get("reason") in {
                    "generic_rescue_contains_significant_disconnected_foreign_component",
                    "generic_rescue_overlaps_existing_success_object",
                }
                for item in audit["rejected"]
                if isinstance(item, Mapping)
            )
            if prior_target_mask is not None and unsafe_generic_rejection:
                audit["skipped"].append({
                    "object_id": object_id,
                    "reason": "unsafe_generic_rescue_preserved_existing_success",
                    "prompt": prompt,
                    "remaining_candidate_count": len(viable),
                    "remaining_rank_margin": round(
                        float(viable[0]["evidence"]["rank"])
                        - float(viable[1]["evidence"]["rank"]),
                        6,
                    ),
                })
                continue
            audit["hard_errors"].append({"type": "bbox_spatial_rescue_ambiguous", "object_id": object_id, "rank_margin": round(float(viable[0]["evidence"]["rank"]) - float(viable[1]["evidence"]["rank"]), 6)})
            continue
        selected = viable[0]
        compact_vase_selection = None
        if (
            cfg.prefer_compact_nested_isolated_vase
            and is_isolated_vase(obj)
            and len(viable) > 1
        ):
            broad = viable[0]
            broad_mask = np.asarray(broad["mask"], dtype=bool)
            broad_box = broad["evidence"]["candidate_bbox_abs_xyxy"]
            broad_area = max(1, int(broad_mask.sum()))
            broad_width = max(1, int(broad_box[2]) - int(broad_box[0]))
            broad_height = max(1, int(broad_box[3]) - int(broad_box[1]))
            compact_candidates = []
            for alternative in viable[1:]:
                compact_mask = np.asarray(alternative["mask"], dtype=bool)
                compact_area = int(compact_mask.sum())
                compact_box = alternative["evidence"]["candidate_bbox_abs_xyxy"]
                containment = int(
                    np.logical_and(compact_mask, broad_mask).sum()
                ) / max(1, compact_area)
                area_ratio = compact_area / broad_area
                width_ratio = (
                    int(compact_box[2]) - int(compact_box[0])
                ) / broad_width
                height_ratio = (
                    int(compact_box[3]) - int(compact_box[1])
                ) / broad_height
                removed_pixels = broad_area - int(
                    np.logical_and(compact_mask, broad_mask).sum()
                )
                if (
                    containment >= cfg.compact_vase_minimum_containment
                    and area_ratio >= cfg.compact_vase_minimum_area_ratio
                    and float(alternative["evidence"]["target_coverage"])
                    >= cfg.compact_vase_minimum_target_coverage
                    and width_ratio <= cfg.compact_vase_maximum_width_ratio
                    and height_ratio >= cfg.compact_vase_minimum_height_ratio
                    and removed_pixels >= cfg.compact_vase_minimum_removed_pixels
                ):
                    compact_candidates.append((
                        float(alternative["evidence"]["rank"]),
                        alternative,
                        {
                            "containment_by_broad_candidate": round(containment, 6),
                            "area_ratio_to_broad_candidate": round(area_ratio, 6),
                            "width_ratio_to_broad_candidate": round(width_ratio, 6),
                            "height_ratio_to_broad_candidate": round(height_ratio, 6),
                            "removed_broad_pixels": removed_pixels,
                            "broad_candidate_id": broad["record"].get("candidate_id"),
                        },
                    ))
            if compact_candidates:
                _, selected, compact_vase_selection = max(
                    compact_candidates,
                    key=lambda item: item[0],
                )
        staging = Path(staging_dir).expanduser().resolve()
        masks_dir, overlays_dir = staging / "masks", staging / "overlays"
        masks_dir.mkdir(parents=True, exist_ok=True)
        overlays_dir.mkdir(parents=True, exist_ok=True)
        base = f"{object_id}_{normalize_prompt(obj.get('category')).replace(' ', '_') or 'object'}"
        mask_path = masks_dir / f"{base}_mask.png"
        overlay_path = overlays_dir / f"{base}_overlay.png"
        Image.fromarray(selected["mask"].astype(np.uint8) * 255).save(mask_path)
        overlay = image_np.copy()
        overlay[selected["mask"]] = (0.55 * overlay[selected["mask"]] + 0.45 * np.array((255, 0, 0), dtype=np.uint8)).astype(np.uint8)
        Image.fromarray(overlay).save(overlay_path)
        prior = copy.deepcopy(dict(obj.get("sam3") or {}))
        candidate = selected["record"]
        obj["sam3"] = {
            "status": "success",
            "warning": "semantic_label_provisional_after_bbox_spatial_rescue",
            "prompt_used": prompt,
            "score": float(candidate.get("score") or 0.0),
            "area_pixels": int(selected["mask"].sum()),
            "vlm_bbox_abs_xyxy": list(target_bbox),
            "mask_bbox_abs_xyxy": list(selected["evidence"]["candidate_bbox_abs_xyxy"]),
            "mask_path": str(mask_path),
            "overlay_path": str(overlay_path),
            "selection_method": "bbox_first_spatial_rescue",
            "bbox_spatial_rescue": {
                "semantic_label_status": "provisional",
                "original_category": obj.get("category"),
                "original_prompt": prior.get("prompt_used") or obj.get("sam3_prompt"),
                "broad_prompt": prompt,
                "positive_bbox_abs_xyxy": list(target_bbox),
                "candidate_id": candidate.get("candidate_id"),
                "evidence": selected["evidence"],
                "rescue_kind": rescue_kind,
                "compact_nested_vase_selection": compact_vase_selection,
            },
        }
        existing_masks[object_id] = selected["mask"]
        accepted = {
            "object_id": object_id,
            "category_preserved": obj.get("category"),
            "semantic_label_status": "provisional",
            "broad_prompt": prompt,
            "candidate_id": candidate.get("candidate_id"),
            "mask_path": str(mask_path),
            "overlay_path": str(overlay_path),
            "prior_sam3": prior,
            "rescue_kind": rescue_kind,
            "evidence": selected["evidence"],
        }
        audit["accepted"].append(accepted)
        audit["resolved_prior_retry_error_object_ids"].append(object_id)

    suppressed_object_ids = (
        suppressed_duplicate_addition_object_ids
        | suppressed_failed_original_object_ids
    )
    unresolved_sources = {
        str(source_id)
        for source_id in unresolved_source_object_ids or []
        if str(source_id)
    }
    for obj in copied:
        object_id = str(obj.get("object_id") or "")
        sam3 = obj.get("sam3") or {}
        source_passes = {
            str(source.get("pass") or "").upper()
            for source in obj.get("proposal_sources") or []
            if isinstance(source, Mapping)
        }
        source_ids = {
            str(source.get("source_id") or "")
            for source in obj.get("proposal_sources") or []
            if isinstance(source, Mapping)
            and str(source.get("pass") or "").upper() == "A"
            and str(source.get("source_id") or "")
        }
        pass_b_source_ids = sorted(
            {
                str(source.get("source_id") or "")
                for source in obj.get("proposal_sources") or []
                if isinstance(source, Mapping)
                and str(source.get("pass") or "").upper() == "B"
                and str(source.get("source_id") or "")
            }
        )
        unresolved_existing_claim = bool(source_ids & unresolved_sources)
        unmaterialized_pass_b_addition = (
            "B" in source_passes and "A" not in source_passes
        )
        if sam3.get("status") != "success" and (
            unresolved_existing_claim or unmaterialized_pass_b_addition
        ):
            suppressed_object_ids.add(object_id)
            audit["excluded_unmaterialized"].append(
                {
                    "object_id": object_id,
                    "source_object_ids": sorted(source_ids & unresolved_sources),
                    "pass_b_source_ids": pass_b_source_ids,
                    "reason": (
                        "failed_sam3_and_spatial_rescue_after_unresolved_review"
                        if unresolved_existing_claim
                        else "unmaterialized_pass_b_addition_excluded"
                    ),
                }
            )
    if suppressed_object_ids:
        copied = [
            obj for obj in copied
            if str(obj.get("object_id") or "") not in suppressed_object_ids
        ]
    audit["suppressed_duplicate_addition_object_ids"] = sorted(
        suppressed_duplicate_addition_object_ids
    )
    audit["suppressed_failed_original_object_ids"] = sorted(
        suppressed_failed_original_object_ids
    )
    excluded_object_ids = {
        str(item.get("object_id") or "")
        for item in audit["excluded_unmaterialized"]
    }
    audit["excluded_unmaterialized_source_object_ids"] = sorted(
        {
            source_id
            for item in audit["excluded_unmaterialized"]
            for source_id in item["source_object_ids"]
        }
    )
    successful_object_ids = {
        str(obj.get("object_id") or "")
        for obj in copied
        if str((obj.get("sam3") or {}).get("status") or "") == "success"
    }
    rescue_error_types = {
        "bbox_rescue_inference_exception",
        "bbox_spatial_rescue_ambiguous",
        "bbox_spatial_rescue_failed",
    }
    blocking_errors = []
    for error in audit["hard_errors"]:
        object_id = str(error.get("object_id") or "")
        if object_id in excluded_object_ids:
            audit["nonblocking_errors"].append(
                {**error, "resolution": "unmaterialized_object_excluded"}
            )
        elif object_id in successful_object_ids and error.get("type") in rescue_error_types:
            audit["nonblocking_errors"].append(
                {**error, "resolution": "successful_object_retained"}
            )
        else:
            blocking_errors.append(error)
    audit["hard_errors"] = blocking_errors
    audit["accepted_count"] = len(audit["accepted"])
    if audit["hard_errors"]:
        audit["status"] = "failed"
    elif audit["target_count"] == 0:
        audit["status"] = "noop"
    elif audit["accepted_count"]:
        audit["status"] = "pass"
    else:
        audit["status"] = "no_rescue"
    _atomic_json(audit_path, audit)
    return {"objects": copied, "audit": audit, "audit_path": str(audit_path)}
