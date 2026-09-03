#!/usr/bin/env python3
"""Fail-closed closure for distinct Pass-A claims sharing one SAM3 mask.

SAM3 occasionally returns one broad region for two nearby Pass-A claims.  A
near-identical mask alone is not proof that the claims are duplicates: the
scene may contain two adjacent physical units.  This helper therefore only
detects and reports the conflict before Pass B.  After Pass B, it accepts an
explicit removal as a same-instance decision; otherwise it attempts two
bounded positive-box SAM3 probes and accepts them only when they are clearly
separate.  An unresolved pair remains unchanged for the normal quality gates
to reject, rather than silently deleting an object.
"""

from __future__ import annotations

import copy
import json
import math
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from PIL import Image


@dataclass(frozen=True)
class SameMaskConflictConfig:
    maximum_conflicts: int = 4
    minimum_intersection_pixels: int = 32
    minimum_mask_iou: float = 0.995
    minimum_mask_coverage: float = 0.995
    maximum_bbox_gap_norm: float = 0.04
    minimum_orthogonal_overlap_fraction: float = 0.20
    split_score_threshold: float = 0.20
    split_minimum_selected_score: float = 0.45
    split_minimum_bbox_coverage: float = 0.45
    split_maximum_center_distance_scale: float = 0.85
    split_maximum_pair_iou: float = 0.18
    split_maximum_candidates_per_claim: int = 8


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _load_mask(path_value: Any, shape: tuple[int, int]) -> np.ndarray:
    path = Path(str(path_value or "")).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"mask is missing: {path}")
    mask = np.array(Image.open(path).convert("L")) > 0
    if tuple(mask.shape) != shape:
        raise ValueError(f"mask has unexpected shape: {path}")
    return mask


def _mask_bbox(mask: np.ndarray) -> list[int]:
    ys, xs = np.where(mask)
    if not len(xs):
        raise ValueError("mask is empty")
    return [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]


def _norm_bbox(obj: Mapping[str, Any]) -> list[float]:
    try:
        box = [float(value) for value in obj.get("bbox_2d_norm_xyxy")]
    except (TypeError, ValueError):
        raise ValueError("invalid normalized bbox") from None
    if (
        len(box) != 4
        or not all(math.isfinite(value) for value in box)
        or not (0.0 <= box[0] < box[2] <= 1.0)
        or not (0.0 <= box[1] < box[3] <= 1.0)
    ):
        raise ValueError("invalid normalized bbox")
    return box


def _bbox_metrics(target: Sequence[float], candidate: Sequence[float]) -> dict[str, float]:
    tx1, ty1, tx2, ty2 = [float(value) for value in target]
    cx1, cy1, cx2, cy2 = [float(value) for value in candidate]
    ix1, iy1 = max(tx1, cx1), max(ty1, cy1)
    ix2, iy2 = min(tx2, cx2), min(ty2, cy2)
    intersection = max(0.0, ix2 - ix1 + 1.0) * max(0.0, iy2 - iy1 + 1.0)
    target_area = max(1.0, (tx2 - tx1 + 1.0) * (ty2 - ty1 + 1.0))
    candidate_area = max(1.0, (cx2 - cx1 + 1.0) * (cy2 - cy1 + 1.0))
    union = target_area + candidate_area - intersection
    target_center = ((tx1 + tx2) / 2.0, (ty1 + ty2) / 2.0)
    candidate_center = ((cx1 + cx2) / 2.0, (cy1 + cy2) / 2.0)
    scale = max(math.hypot(tx2 - tx1, ty2 - ty1), 1.0)
    return {
        "target_bbox_coverage": intersection / target_area,
        "candidate_bbox_coverage": intersection / candidate_area,
        "bbox_iou": intersection / union if union else 0.0,
        "center_distance_scale": math.dist(target_center, candidate_center) / scale,
    }


def _mask_iou(left: np.ndarray, right: np.ndarray) -> float:
    union = int(np.logical_or(left, right).sum())
    return int(np.logical_and(left, right).sum()) / union if union else 0.0


def _spatially_related(left: Sequence[float], right: Sequence[float], cfg: SameMaskConflictConfig) -> bool:
    lx1, ly1, lx2, ly2 = left
    rx1, ry1, rx2, ry2 = right
    horizontal_gap = max(0.0, max(lx1, rx1) - min(lx2, rx2))
    vertical_gap = max(0.0, max(ly1, ry1) - min(ly2, ry2))
    horizontal_overlap = max(0.0, min(lx2, rx2) - max(lx1, rx1))
    vertical_overlap = max(0.0, min(ly2, ry2) - max(ly1, ry1))
    left_width, right_width = lx2 - lx1, rx2 - rx1
    left_height, right_height = ly2 - ly1, ry2 - ry1
    if horizontal_gap <= cfg.maximum_bbox_gap_norm:
        return vertical_overlap >= cfg.minimum_orthogonal_overlap_fraction * min(left_height, right_height)
    if vertical_gap <= cfg.maximum_bbox_gap_norm:
        return horizontal_overlap >= cfg.minimum_orthogonal_overlap_fraction * min(left_width, right_width)
    return False


def _object_is_eligible(obj: Mapping[str, Any]) -> bool:
    """Exclude structural claims and known fixed-component assembly inputs."""

    text = " ".join(
        str(obj.get(key) or "")
        for key in ("category", "sam3_prompt", "instance_name")
    ).lower()
    excluded = {
        "wall", "floor", "ceiling", "window", "door", "curtain",
        "faucet", "sconce", "lamp", "light", "handle", "knob",
    }
    return not any(token in text for token in excluded)


def detect_pass_a_same_mask_conflicts(
    *,
    image_path: str | Path,
    registry: Mapping[str, Any],
    output_dir: str | Path,
    config: SameMaskConflictConfig | None = None,
) -> dict[str, Any]:
    """Detect only ambiguous same-mask pairs; this does not change registry."""

    cfg = config or SameMaskConflictConfig()
    output = Path(output_dir).expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"same-mask conflict output is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    with Image.open(Path(image_path).expanduser().resolve()) as source:
        shape = (source.height, source.width)
    objects = list(registry.get("objects") or [])
    records: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    candidates: list[tuple[Mapping[str, Any], np.ndarray, list[float]]] = []
    for obj in objects:
        object_id = str(obj.get("object_id") or "")
        sam3 = obj.get("sam3") or {}
        if not object_id or sam3.get("status") != "success" or not _object_is_eligible(obj):
            continue
        try:
            candidates.append((obj, _load_mask(sam3.get("mask_path"), shape), _norm_bbox(obj)))
        except Exception as exc:
            errors.append({"type": "same_mask_conflict_invalid_source_mask", "object_id": object_id, "error": f"{type(exc).__name__}: {exc}"})
    for left_index, (left, left_mask, left_box) in enumerate(candidates):
        for right, right_mask, right_box in candidates[left_index + 1 :]:
            if not _spatially_related(left_box, right_box, cfg):
                continue
            intersection = int(np.logical_and(left_mask, right_mask).sum())
            if intersection < cfg.minimum_intersection_pixels:
                continue
            left_area, right_area = int(left_mask.sum()), int(right_mask.sum())
            coverage_left = intersection / max(left_area, 1)
            coverage_right = intersection / max(right_area, 1)
            iou = _mask_iou(left_mask, right_mask)
            if (
                iou < cfg.minimum_mask_iou
                or coverage_left < cfg.minimum_mask_coverage
                or coverage_right < cfg.minimum_mask_coverage
            ):
                continue
            owner_ids = sorted((str(left["object_id"]), str(right["object_id"])))
            left_by_id = {str(left["object_id"]): (left, left_box), str(right["object_id"]): (right, right_box)}
            records.append({
                "conflict_id": "",
                "object_ids": owner_ids,
                "objects": [
                    {
                        "object_id": object_id,
                        "category": str(left_by_id[object_id][0].get("category") or "unknown"),
                        "sam3_prompt": str(left_by_id[object_id][0].get("sam3_prompt") or "object"),
                        "vlm_bbox_norm_xyxy": [round(value, 6) for value in left_by_id[object_id][1]],
                    }
                    for object_id in owner_ids
                ],
                "mask_iou": round(iou, 6),
                "intersection_pixels": intersection,
                "coverage_by_object_id": {
                    str(left["object_id"]): round(coverage_left, 6),
                    str(right["object_id"]): round(coverage_right, 6),
                },
                "resolution_policy": "pass_b_remove_one_or_local_bbox_split",
            })
    records.sort(key=lambda item: (-float(item["mask_iou"]), item["object_ids"]))
    if len(records) > cfg.maximum_conflicts:
        errors.append({"type": "same_mask_conflict_capacity_overflow", "actual": len(records), "maximum": cfg.maximum_conflicts})
        records = records[: cfg.maximum_conflicts]
    for index, record in enumerate(records, start=1):
        record["conflict_id"] = f"SMC_{index:03d}"
    audit = {
        "schema_version": "scenemosaic_pass_a_same_mask_conflict",
        "status": "blocked" if errors else ("pass" if records else "noop"),
        "config": asdict(cfg),
        "conflict_count": len(records),
        "records": records,
        "additional_vlm_calls": 0,
        "additional_sam3_inference_count": 0,
        "formal_outputs_modified": False,
        "hard_errors": errors,
    }
    _atomic_json(output / "pass_a_same_mask_conflicts_audit.json", audit)
    return {"conflicts": records, "audit": audit}


def _abs_box(box: Sequence[float], width: int, height: int) -> list[int]:
    x1 = max(0, min(width - 1, int(round(float(box[0]) * width))))
    y1 = max(0, min(height - 1, int(round(float(box[1]) * height))))
    x2 = max(x1 + 1, min(width, int(round(float(box[2]) * width))))
    y2 = max(y1 + 1, min(height, int(round(float(box[3]) * height))))
    return [x1, y1, x2, y2]


def _select_bbox_candidate(
    *,
    candidates: Sequence[Mapping[str, Any]],
    target_box: Sequence[int],
    shape: tuple[int, int],
    cfg: SameMaskConflictConfig,
) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
    evaluations = []
    for candidate in candidates:
        try:
            mask = _load_mask(candidate.get("mask_path"), shape)
            candidate_box = _mask_bbox(mask)
        except Exception as exc:
            evaluations.append({"candidate_id": candidate.get("candidate_id"), "eligible": False, "reason": f"invalid_candidate: {type(exc).__name__}: {exc}"})
            continue
        metrics = _bbox_metrics(target_box, candidate_box)
        score = float(candidate.get("score") or 0.0)
        eligible = (
            score >= cfg.split_minimum_selected_score
            and metrics["target_bbox_coverage"] >= cfg.split_minimum_bbox_coverage
            and metrics["candidate_bbox_coverage"] >= cfg.split_minimum_bbox_coverage
            and metrics["center_distance_scale"] <= cfg.split_maximum_center_distance_scale
        )
        quality = score + metrics["target_bbox_coverage"] + 0.25 * metrics["candidate_bbox_coverage"] - 0.20 * metrics["center_distance_scale"]
        evaluations.append({
            "candidate_id": str(candidate.get("candidate_id") or ""),
            "mask_path": str(candidate.get("mask_path") or ""),
            "score": score,
            "mask": mask,
            "mask_bbox_abs_xyxy": candidate_box,
            "target_bbox_coverage": round(metrics["target_bbox_coverage"], 6),
            "candidate_bbox_coverage": round(metrics["candidate_bbox_coverage"], 6),
            "center_distance_scale": round(metrics["center_distance_scale"], 6),
            "quality": quality,
            "eligible": eligible,
            "reason": None if eligible else "bounded_split_geometry_or_score_gate",
        })
    eligible = sorted((item for item in evaluations if item["eligible"]), key=lambda item: (-float(item["quality"]), item["candidate_id"]))
    public = [{key: value for key, value in item.items() if key != "mask"} for item in evaluations]
    return (eligible[0] if eligible else None), public


def resolve_pass_a_same_mask_conflicts(
    *,
    session: Any,
    infer_text_candidates_fn: Any,
    image_path: str | Path,
    registry: Mapping[str, Any],
    actions: Mapping[str, Any],
    conflicts: Sequence[Mapping[str, Any]],
    output_dir: str | Path,
    config: SameMaskConflictConfig | None = None,
) -> dict[str, Any]:
    """Apply Pass-B decision or bounded local split without changing actions."""

    cfg = config or SameMaskConflictConfig()
    output = Path(output_dir).expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"same-mask resolution output is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    copied = copy.deepcopy(dict(registry))
    objects = list(copied.get("objects") or [])
    by_id = {str(obj.get("object_id") or ""): obj for obj in objects}
    removed = {str(value) for value in (actions.get("removals") or [])}
    updated = {
        str(item.get("pass_a_id") or "")
        for item in (actions.get("updates") or [])
        if isinstance(item, Mapping)
    }
    resolved_image = Path(image_path).expanduser().resolve()
    image = np.array(Image.open(resolved_image).convert("RGB"), dtype=np.uint8)
    shape = tuple(image.shape[:2])
    height, width = shape
    records = []
    errors: list[dict[str, Any]] = []
    inference_before = int(getattr(session, "inference_run_count", 0))
    for conflict in conflicts:
        conflict_id = str(conflict.get("conflict_id") or "")
        owner_ids = [str(value) for value in (conflict.get("object_ids") or [])]
        if len(owner_ids) != 2 or any(owner_id not in by_id for owner_id in owner_ids):
            errors.append({"type": "same_mask_conflict_invalid_owners", "conflict_id": conflict_id, "object_ids": owner_ids})
            continue
        selected_removals = sorted(set(owner_ids) & removed)
        selected_updates = sorted(set(owner_ids) & updated)
        if selected_updates:
            errors.append({
                "type": "same_mask_conflict_local_split_blocked_by_pass_b_update",
                "conflict_id": conflict_id,
                "object_ids": owner_ids,
                "updated_object_ids": selected_updates,
            })
            records.append({
                "conflict_id": conflict_id,
                "object_ids": owner_ids,
                "status": "unresolved",
                "reason": "pass_b_update_requires_delta_resegmentation",
                "updated_object_ids": selected_updates,
            })
            continue
        if len(selected_removals) == 1:
            records.append({"conflict_id": conflict_id, "object_ids": owner_ids, "status": "same_instance_removed", "removed_object_id": selected_removals[0], "additional_sam3_inference_count": 0})
            continue
        if len(selected_removals) == 2:
            errors.append({"type": "same_mask_conflict_removed_both", "conflict_id": conflict_id, "object_ids": owner_ids})
            records.append({"conflict_id": conflict_id, "object_ids": owner_ids, "status": "blocked", "reason": "both_claims_removed"})
            continue
        winners: dict[str, dict[str, Any]] = {}
        evaluations_by_owner: dict[str, list[dict[str, Any]]] = {}
        for index, owner_id in enumerate(owner_ids, start=1):
            obj = by_id[owner_id]
            target_box = _abs_box(_norm_bbox(obj), width, height)
            raw_dir = output / "raw_candidates" / conflict_id.lower() / owner_id.lower()
            result = infer_text_candidates_fn(
                session,
                resolved_image,
                str(obj.get("sam3_prompt") or obj.get("category") or "object"),
                raw_dir,
                score_threshold=cfg.split_score_threshold,
                positive_bbox_abs_xyxy=target_box,
                maximum_candidates=cfg.split_maximum_candidates_per_claim,
                overwrite=False,
            )
            winner, public_evaluations = _select_bbox_candidate(
                candidates=list((result.get("registry") or {}).get("candidates") or []),
                target_box=target_box,
                shape=shape,
                cfg=cfg,
            )
            evaluations_by_owner[owner_id] = public_evaluations
            if winner is not None:
                winners[owner_id] = winner
        if len(winners) != 2:
            errors.append({"type": "same_mask_conflict_local_split_failed", "conflict_id": conflict_id, "object_ids": owner_ids, "reason": "missing_eligible_bounded_candidate"})
            records.append({"conflict_id": conflict_id, "object_ids": owner_ids, "status": "unresolved", "reason": "missing_eligible_bounded_candidate", "evaluations": evaluations_by_owner})
            continue
        pair_iou = _mask_iou(winners[owner_ids[0]]["mask"], winners[owner_ids[1]]["mask"])
        if pair_iou > cfg.split_maximum_pair_iou:
            errors.append({"type": "same_mask_conflict_local_split_failed", "conflict_id": conflict_id, "object_ids": owner_ids, "reason": "local_candidates_still_overlap", "pair_mask_iou": round(pair_iou, 6)})
            records.append({"conflict_id": conflict_id, "object_ids": owner_ids, "status": "unresolved", "reason": "local_candidates_still_overlap", "pair_mask_iou": round(pair_iou, 6), "evaluations": evaluations_by_owner})
            continue
        selected_dir = output / "selected" / conflict_id.lower()
        selected_dir.mkdir(parents=True, exist_ok=True)
        selected_objects = []
        for owner_id in owner_ids:
            winner = winners[owner_id]
            mask_path = selected_dir / f"{owner_id.lower()}_mask.png"
            overlay_path = selected_dir / f"{owner_id.lower()}_overlay.png"
            Image.fromarray(winner["mask"].astype(np.uint8) * 255).save(mask_path)
            canvas = image.astype(np.float32).copy()
            canvas[winner["mask"]] = 0.55 * canvas[winner["mask"]] + 0.45 * np.array((255, 0, 0), dtype=np.float32)
            Image.fromarray(np.clip(canvas, 0, 255).astype(np.uint8)).save(overlay_path)
            previous = copy.deepcopy(dict(by_id[owner_id].get("sam3") or {}))
            by_id[owner_id]["sam3"] = {
                **previous,
                "status": "success",
                "mask_path": str(mask_path),
                "overlay_path": str(overlay_path),
                "area_pixels": int(winner["mask"].sum()),
                "mask_bbox_abs_xyxy": list(winner["mask_bbox_abs_xyxy"]),
                "score": float(winner["score"]),
                "selection_method": "pass_a_same_mask_local_bbox_split",
                "pass_a_same_mask_local_bbox_split": {
                    "conflict_id": conflict_id,
                    "candidate_id": winner["candidate_id"],
                    "target_bbox_coverage": winner["target_bbox_coverage"],
                    "candidate_bbox_coverage": winner["candidate_bbox_coverage"],
                    "center_distance_scale": winner["center_distance_scale"],
                    "pair_mask_iou": round(pair_iou, 6),
                },
            }
            selected_objects.append(
                {
                    "object_id": owner_id,
                    "selected_mask_path": str(mask_path),
                    "candidate_id": winner["candidate_id"],
                }
            )
        records.append({"conflict_id": conflict_id, "object_ids": owner_ids, "status": "locally_split", "pair_mask_iou": round(pair_iou, 6), "selected": selected_objects, "evaluations": evaluations_by_owner})
    inference_after = int(getattr(session, "inference_run_count", inference_before))
    audit = {
        "schema_version": "scenemosaic_pass_a_same_mask_conflict_resolution",
        "status": "blocked" if errors else ("pass" if records else "noop"),
        "conflict_count": len(conflicts),
        "records": records,
        "additional_vlm_calls": 0,
        "additional_sam3_inference_count": max(0, inference_after - inference_before),
        "formal_outputs_modified": False,
        "hard_errors": errors,
    }
    _atomic_json(output / "pass_a_same_mask_conflict_resolution_audit.json", audit)
    return {"registry": copied, "audit": audit}
