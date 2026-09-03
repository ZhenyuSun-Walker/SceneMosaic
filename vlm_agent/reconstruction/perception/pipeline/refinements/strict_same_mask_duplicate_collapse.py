#!/usr/bin/env python3
"""Fail-closed removal of a Pass-B addition proved to duplicate an object."""

from __future__ import annotations

import copy
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from PIL import Image


@dataclass(frozen=True)
class StrictSameMaskDuplicateCollapseConfig:
    enabled: bool = True
    isolated_attempt_only: bool = True
    minimum_intersection_pixels: int = 30
    minimum_iou: float = 0.995
    minimum_coverage: float = 0.995
    minimum_aggregate_member_count: int = 2
    minimum_aggregate_coverage_by_members: float = 0.85
    minimum_member_coverage_by_aggregate: float = 0.95
    # Cross-category Pass-B additions need slightly more tolerance than the
    # same-prompt retry path: prompt-specific SAM3 masks can differ by a thin
    # boundary while still referring to exactly the same existing instance.
    minimum_pass_b_preexisting_iou: float = 0.99
    minimum_pass_b_preexisting_coverage: float = 0.995


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _category_key(obj: Mapping[str, Any]) -> str:
    return " ".join(
        re.findall(r"[a-z0-9]+", str(obj.get("category") or "").lower())
    )


def _category_tokens(obj: Mapping[str, Any]) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", str(obj.get("category") or "").lower()))


def _aggregate_member_tokens(obj: Mapping[str, Any]) -> set[str]:
    aggregate_markers = {"bundle", "cluster", "collection", "group", "pair", "set"}
    return _category_tokens(obj) - aggregate_markers


def _pass_b_only_addition(obj: Mapping[str, Any]) -> bool:
    sources = obj.get("proposal_sources") or []
    if not isinstance(sources, list) or not sources:
        return False
    return all(
        isinstance(source, Mapping)
        and str(source.get("pass") or "").upper() == "B"
        and str(source.get("action") or "").lower() == "added"
        for source in sources
    )


def _load_mask(path_value: Any, expected_shape: tuple[int, int]) -> np.ndarray:
    path = Path(str(path_value or "")).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    mask = np.array(Image.open(path).convert("L")) > 0
    if mask.shape != expected_shape or not mask.any():
        raise ValueError("invalid_mask_shape_or_empty")
    return mask


def _mask_metrics(left: np.ndarray, right: np.ndarray) -> dict[str, float | int]:
    intersection = int(np.logical_and(left, right).sum())
    left_area = int(left.sum())
    right_area = int(right.sum())
    union = left_area + right_area - intersection
    return {
        "intersection_pixels": intersection,
        "iou": intersection / max(union, 1),
        "left_coverage": intersection / max(left_area, 1),
        "right_coverage": intersection / max(right_area, 1),
    }


def _retry_object_by_id(
    retry_registry: Mapping[str, Any] | None,
) -> dict[str, Mapping[str, Any]]:
    objects = (retry_registry or {}).get("objects") or []
    return {
        str(obj.get("object_id") or ""): obj
        for obj in objects
        if isinstance(obj, Mapping) and str(obj.get("object_id") or "")
    }


def _remap_exact_proposal_ids(value: Any, mapping: Mapping[str, str]) -> Any:
    """Recursively rewrite exact ID values, never path substrings."""

    if isinstance(value, str):
        return mapping.get(value, value)
    if isinstance(value, list):
        return [_remap_exact_proposal_ids(item, mapping) for item in value]
    if isinstance(value, Mapping):
        return {
            key: _remap_exact_proposal_ids(item, mapping)
            for key, item in value.items()
        }
    return value


def run_strict_same_mask_duplicate_collapse(
    *,
    objects: Sequence[Mapping[str, Any]],
    retry_registry: Mapping[str, Any] | None,
    retry_audit: Mapping[str, Any] | None,
    image_shape: tuple[int, int],
    output_dir: str | Path,
    config: StrictSameMaskDuplicateCollapseConfig | None = None,
) -> dict[str, Any]:
    """Return copied objects and a strict audit without modifying source data."""

    cfg = config or StrictSameMaskDuplicateCollapseConfig()
    if not cfg.isolated_attempt_only:
        raise ValueError("strict same-mask duplicate collapse is isolated-attempt only")
    work = Path(output_dir).expanduser().resolve()
    work.mkdir(parents=True, exist_ok=True)
    copied = [copy.deepcopy(dict(obj)) for obj in objects if isinstance(obj, Mapping)]
    audit: dict[str, Any] = {
        "schema_version": "scenemosaic_strict_same_mask_duplicate_collapse",
        "enabled": cfg.enabled,
        "isolated_attempt_only": cfg.isolated_attempt_only,
        "status": "disabled" if not cfg.enabled else "noop",
        "input_object_count": len(copied),
        "collapsed": [],
        "rejected": [],
        "hard_errors": [],
        "thresholds": {
            "minimum_intersection_pixels": cfg.minimum_intersection_pixels,
            "minimum_iou": cfg.minimum_iou,
            "minimum_coverage": cfg.minimum_coverage,
            "minimum_aggregate_member_count": cfg.minimum_aggregate_member_count,
            "minimum_aggregate_coverage_by_members": (
                cfg.minimum_aggregate_coverage_by_members
            ),
            "minimum_member_coverage_by_aggregate": (
                cfg.minimum_member_coverage_by_aggregate
            ),
            "minimum_pass_b_preexisting_iou": (
                cfg.minimum_pass_b_preexisting_iou
            ),
            "minimum_pass_b_preexisting_coverage": (
                cfg.minimum_pass_b_preexisting_coverage
            ),
        },
    }
    if not cfg.enabled:
        _atomic_json(work / "strict_same_mask_duplicate_collapse_audit.json", audit)
        return {"objects": copied, "audit": audit}

    retry_by_id = _retry_object_by_id(retry_registry)
    rejected_by_id = {
        str(item.get("object_id") or ""): str(item.get("reason") or "")
        for item in ((retry_audit or {}).get("rejected") or [])
        if isinstance(item, Mapping)
    }
    success_masks: dict[str, np.ndarray] = {}
    for obj in copied:
        object_id = str(obj.get("object_id") or "")
        sam3 = obj.get("sam3") or {}
        if sam3.get("status") != "success":
            continue
        try:
            success_masks[object_id] = _load_mask(sam3.get("mask_path"), image_shape)
        except (OSError, ValueError) as exc:
            audit["hard_errors"].append(
                {
                    "type": "invalid_existing_success_mask_for_duplicate_collapse",
                    "object_id": object_id,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )

    removal_ids: set[str] = set()
    for obj in copied:
        object_id = str(obj.get("object_id") or "")
        sam3 = obj.get("sam3") or {}
        if sam3.get("status") != "failed" or not _pass_b_only_addition(obj):
            continue
        if rejected_by_id.get(object_id) != "mask_overlap_ambiguity":
            continue
        retry_obj = retry_by_id.get(object_id)
        retry_sam3 = (retry_obj or {}).get("sam3") or {}
        if retry_sam3.get("status") != "success":
            continue
        try:
            retry_mask = _load_mask(retry_sam3.get("mask_path"), image_shape)
        except (OSError, ValueError) as exc:
            audit["hard_errors"].append(
                {
                    "type": "invalid_retry_mask_for_duplicate_collapse",
                    "object_id": object_id,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
            continue
        category = _category_key(obj)
        best: tuple[str, dict[str, float | int]] | None = None
        for existing in copied:
            existing_id = str(existing.get("object_id") or "")
            if (
                existing_id == object_id
                or existing_id not in success_masks
                or _category_key(existing) != category
            ):
                continue
            metrics = _mask_metrics(retry_mask, success_masks[existing_id])
            if best is None or float(metrics["iou"]) > float(best[1]["iou"]):
                best = (existing_id, metrics)
        if best is None:
            audit["rejected"].append(
                {"object_id": object_id, "reason": "no_same_category_success"}
            )
            continue
        kept_id, metrics = best
        proven = (
            int(metrics["intersection_pixels"]) >= cfg.minimum_intersection_pixels
            and float(metrics["iou"]) >= cfg.minimum_iou
            and float(metrics["left_coverage"]) >= cfg.minimum_coverage
            and float(metrics["right_coverage"]) >= cfg.minimum_coverage
        )
        if not proven:
            audit["rejected"].append(
                {
                    "object_id": object_id,
                    "reason": "same_category_geometry_below_strict_threshold",
                    "kept_object_id": kept_id,
                    **{key: round(float(value), 6) for key, value in metrics.items()},
                }
            )
            continue
        removal_ids.add(object_id)
        audit["collapsed"].append(
            {
                "removed_object_id": object_id,
                "kept_object_id": kept_id,
                "category": category,
                "source_policy": "pass_b_only_addition_with_category_retry_exact_mask",
                **{key: round(float(value), 6) for key, value in metrics.items()},
            }
        )

    # A successful Pass-B-only addition can also duplicate an already
    # materialized Pass-A object.  If their masks are effectively identical,
    # provenance provides a deterministic owner even when the VLM assigned
    # incompatible category names: retain the pre-existing object and remove
    # the later addition.  This is stricter than semantic matching and never
    # chooses between two independently established objects.
    for addition in copied:
        addition_id = str(addition.get("object_id") or "")
        if (
            addition_id in removal_ids
            or addition_id not in success_masks
            or not _pass_b_only_addition(addition)
        ):
            continue
        best: tuple[str, dict[str, float | int]] | None = None
        for existing in copied:
            existing_id = str(existing.get("object_id") or "")
            if (
                existing_id == addition_id
                or existing_id in removal_ids
                or existing_id not in success_masks
                or _pass_b_only_addition(existing)
            ):
                continue
            metrics = _mask_metrics(
                success_masks[addition_id], success_masks[existing_id]
            )
            if best is None or float(metrics["iou"]) > float(best[1]["iou"]):
                best = (existing_id, metrics)
        if best is None:
            continue
        kept_id, metrics = best
        proven = (
            int(metrics["intersection_pixels"]) >= cfg.minimum_intersection_pixels
            and float(metrics["iou"]) >= cfg.minimum_pass_b_preexisting_iou
            and float(metrics["left_coverage"])
            >= cfg.minimum_pass_b_preexisting_coverage
            and float(metrics["right_coverage"])
            >= cfg.minimum_pass_b_preexisting_coverage
        )
        if not proven:
            continue
        removal_ids.add(addition_id)
        audit["collapsed"].append(
            {
                "removed_object_id": addition_id,
                "kept_object_id": kept_id,
                "removed_category": _category_key(addition),
                "kept_category": _category_key(
                    next(
                        item
                        for item in copied
                        if str(item.get("object_id") or "") == kept_id
                    )
                ),
                "source_policy": (
                    "pass_b_only_addition_exact_mask_prefer_preexisting_object"
                ),
                **{
                    key: round(float(value), 6)
                    for key, value in metrics.items()
                },
            }
        )

    aggregate_markers = {"bundle", "cluster", "collection", "group", "pair", "set"}
    for aggregate in copied:
        aggregate_id = str(aggregate.get("object_id") or "")
        aggregate_tokens = _category_tokens(aggregate)
        member_tokens = _aggregate_member_tokens(aggregate)
        if (
            aggregate_id in removal_ids
            or aggregate_id not in success_masks
            or not (aggregate_tokens & aggregate_markers)
            or not member_tokens
        ):
            continue
        aggregate_mask = success_masks[aggregate_id]
        members = []
        for candidate in copied:
            candidate_id = str(candidate.get("object_id") or "")
            if (
                candidate_id == aggregate_id
                or candidate_id in removal_ids
                or candidate_id not in success_masks
                or not (_category_tokens(candidate) & member_tokens)
                or bool(_category_tokens(candidate) & aggregate_markers)
            ):
                continue
            metrics = _mask_metrics(success_masks[candidate_id], aggregate_mask)
            if (
                int(metrics["intersection_pixels"]) >= cfg.minimum_intersection_pixels
                and float(metrics["left_coverage"])
                >= cfg.minimum_member_coverage_by_aggregate
            ):
                members.append((candidate_id, metrics))
        if len(members) < cfg.minimum_aggregate_member_count:
            continue
        member_union = np.logical_or.reduce(
            [success_masks[candidate_id] for candidate_id, _ in members]
        )
        aggregate_coverage = float(
            np.logical_and(aggregate_mask, member_union).sum()
            / max(int(aggregate_mask.sum()), 1)
        )
        if aggregate_coverage < cfg.minimum_aggregate_coverage_by_members:
            continue
        removal_ids.add(aggregate_id)
        audit["collapsed"].append(
            {
                "removed_object_id": aggregate_id,
                "kept_object_ids": [candidate_id for candidate_id, _ in members],
                "category": _category_key(aggregate),
                "source_policy": "aggregate_mask_explained_by_individual_members",
                "aggregate_coverage_by_members": round(aggregate_coverage, 6),
                "member_coverages_by_aggregate": {
                    candidate_id: round(float(metrics["left_coverage"]), 6)
                    for candidate_id, metrics in members
                },
            }
        )

    result = [obj for obj in copied if str(obj.get("object_id") or "") not in removal_ids]
    object_id_map = {
        str(obj.get("object_id") or ""): f"proposal_{index:03d}"
        for index, obj in enumerate(result, start=1)
    }
    result = [
        _remap_exact_proposal_ids(obj, object_id_map)
        for obj in result
    ]
    audit["object_id_map"] = [
        {"old_object_id": old_id, "new_object_id": new_id}
        for old_id, new_id in object_id_map.items()
        if old_id != new_id
    ]
    for record in audit["collapsed"]:
        if record.get("kept_object_id") is not None:
            kept_old_id = str(record.get("kept_object_id") or "")
            record["kept_final_object_id"] = object_id_map.get(
                kept_old_id, kept_old_id
            )
        else:
            record["kept_final_object_ids"] = [
                object_id_map.get(str(value), str(value))
                for value in record.get("kept_object_ids") or []
            ]
    audit["output_object_count"] = len(result)
    audit["collapsed_count"] = len(audit["collapsed"])
    audit["status"] = "fail" if audit["hard_errors"] else ("pass" if removal_ids else "noop")
    _atomic_json(work / "strict_same_mask_duplicate_collapse_audit.json", audit)
    return {"objects": result, "audit": audit}
