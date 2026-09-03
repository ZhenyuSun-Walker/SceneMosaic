#!/usr/bin/env python3
"""Always materialize the best usable SAM3D input package for a scene.

This module is deliberately downstream of perception.  It preserves source
perception facts and materializes derived, reconstruction-only masks when a
high-overlap conflict must be assigned to one owner.  It converts a published
isolated-attempt result into one of four delivery outcomes:

* ``sam3d_ready``: all source quality checks passed;
* ``sam3d_ready_degraded``: source quality passed but a usable object mask
  was excluded or ownership-derived;
* ``quality_failed``: a draft package was retained for audit, but source
  hard quality checks did not pass and SAM3D execution is forbidden;
* ``infrastructure_failed``: no technically consumable scene package can be
  produced (for example, unreadable image or zero valid masks).
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from PIL import Image


SAFE_NAME_RE = re.compile(r"[^A-Za-z0-9_-]+")
READY_STATUSES = {"sam3d_ready", "sam3d_ready_degraded"}
DUPLICATE_MINIMUM_INTERSECTION_PIXELS = 30
DUPLICATE_MINIMUM_IOU = 0.995
DUPLICATE_MINIMUM_COVERAGE = 0.995
RECONSTRUCTION_CONFLICT_MINIMUM_INTERSECTION_PIXELS = 30
RECONSTRUCTION_CONFLICT_MINIMUM_IOU = 0.55
RECONSTRUCTION_CONFLICT_MINIMUM_CONTAINMENT = 0.90
RECONSTRUCTION_MINIMUM_REMAINING_PIXELS = 64
RECONSTRUCTION_MINIMUM_REMAINING_FRACTION = 0.02


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_name(value: Any) -> str:
    text = str(value or "unknown").strip().lower().replace(" ", "_")
    text = SAFE_NAME_RE.sub("_", text)
    return re.sub(r"_+", "_", text).strip("_") or "unknown"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _load_binary_mask(path: Path, expected_shape: tuple[int, int]) -> np.ndarray:
    with Image.open(path) as source:
        mask = np.array(source.convert("L")) > 0
    if tuple(mask.shape) != tuple(expected_shape):
        raise ValueError(
            f"mask shape {tuple(mask.shape)} does not match image {expected_shape}"
        )
    if not bool(mask.any()):
        raise ValueError("mask is empty")
    return mask


def _score(obj: Mapping[str, Any]) -> tuple[float, float]:
    sam3 = obj.get("sam3") or {}
    try:
        score = float(sam3.get("score") or 0.0)
    except (TypeError, ValueError):
        score = 0.0
    try:
        bbox_iou = float(sam3.get("vlm_bbox_iou") or 0.0)
    except (TypeError, ValueError):
        bbox_iou = 0.0
    return score, bbox_iou


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


def _has_pass_a_provenance(obj: Mapping[str, Any]) -> bool:
    sources = obj.get("proposal_sources") or []
    return any(
        isinstance(source, Mapping)
        and str(source.get("pass") or "").upper() == "A"
        for source in sources
    )


def _duplicate_winner_key(
    item: Mapping[str, Any],
) -> tuple[int, float, float, int]:
    score, bbox_iou = item["rank"]
    return (
        int(_has_pass_a_provenance(item["source_object"])),
        score,
        bbox_iou,
        -int(item["source_index"]),
    )


def _is_reconstruction_conflict(metrics: Mapping[str, float | int]) -> bool:
    return (
        int(metrics["intersection_pixels"])
        >= RECONSTRUCTION_CONFLICT_MINIMUM_INTERSECTION_PIXELS
        and (
            float(metrics["iou"]) >= RECONSTRUCTION_CONFLICT_MINIMUM_IOU
            or max(
                float(metrics["left_coverage"]),
                float(metrics["right_coverage"]),
            )
            >= RECONSTRUCTION_CONFLICT_MINIMUM_CONTAINMENT
        )
    )


def _derive_reconstruction_ownership(
    items: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Derive non-overlapping SAM3D masks without changing perception masks.

    Only pairs already severe enough for the global quality gate are modified.
    Earlier, better-evidenced objects own shared pixels; a later object with no
    meaningful residual is excluded as a proved reconstruction duplicate.
    """

    ordered = sorted(
        [copy.deepcopy(dict(item)) for item in items],
        key=_duplicate_winner_key,
        reverse=True,
    )
    derived_by_source: dict[str, np.ndarray] = {}
    kept: list[dict[str, Any]] = []
    allocations: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []

    for priority_index, item in enumerate(ordered):
        source_id = str(item["source_object_id"])
        source_mask = np.asarray(item["mask"], dtype=bool)
        original_area = int(source_mask.sum())
        owned_by: list[dict[str, Any]] = []
        delivery_mask = source_mask.copy()
        for owner in kept:
            owner_id = str(owner["source_object_id"])
            metrics = _mask_metrics(source_mask, owner["mask"])
            if not _is_reconstruction_conflict(metrics):
                continue
            shared = np.logical_and(delivery_mask, derived_by_source[owner_id])
            shared_pixels = int(shared.sum())
            if not shared_pixels:
                continue
            delivery_mask &= ~derived_by_source[owner_id]
            owned_by.append(
                {
                    "source_object_id": owner_id,
                    "shared_pixels_removed": shared_pixels,
                    **{
                        key: round(float(value), 6)
                        for key, value in metrics.items()
                    },
                }
            )

        delivery_area = int(delivery_mask.sum())
        remaining_fraction = delivery_area / max(1, original_area)
        ownership = {
            "priority_index": priority_index,
            "mode": "copy" if not owned_by else "shared_pixels_subtracted",
            "owner_source_object_ids": [
                str(record["source_object_id"]) for record in owned_by
            ],
            "original_area_pixels": original_area,
            "delivery_area_pixels": delivery_area,
            "shared_pixels_removed": original_area - delivery_area,
            "remaining_fraction": round(remaining_fraction, 6),
            "conflicts": owned_by,
        }
        if (
            owned_by
            and (
                delivery_area < RECONSTRUCTION_MINIMUM_REMAINING_PIXELS
                or remaining_fraction < RECONSTRUCTION_MINIMUM_REMAINING_FRACTION
            )
        ):
            excluded.append(
                {
                    "source_object_id": source_id,
                    "reason": "high_overlap_fully_owned_by_higher_priority_object",
                    "ownership": ownership,
                }
            )
            continue

        item["reconstruction_mask_ownership"] = ownership
        item["delivery_mask"] = delivery_mask
        derived_by_source[source_id] = delivery_mask
        kept.append(item)
        if owned_by:
            allocations.append(
                {
                    "source_object_id": source_id,
                    **ownership,
                }
            )

    kept.sort(key=lambda item: int(item["source_index"]))
    audit = {
        "enabled": True,
        "policy": (
            "higher_evidence_owner_subtracts_only_global_high_overlap_pixels"
        ),
        "thresholds": {
            "minimum_intersection_pixels": (
                RECONSTRUCTION_CONFLICT_MINIMUM_INTERSECTION_PIXELS
            ),
            "minimum_iou": RECONSTRUCTION_CONFLICT_MINIMUM_IOU,
            "minimum_containment": RECONSTRUCTION_CONFLICT_MINIMUM_CONTAINMENT,
            "minimum_remaining_pixels": RECONSTRUCTION_MINIMUM_REMAINING_PIXELS,
            "minimum_remaining_fraction": RECONSTRUCTION_MINIMUM_REMAINING_FRACTION,
        },
        "allocation_count": len(allocations),
        "allocations": allocations,
        "excluded_count": len(excluded),
        "excluded_objects": excluded,
    }
    return kept, audit


def _materialize_derived_delivery_masks(
    objects: Sequence[Mapping[str, Any]],
    expected_shape: tuple[int, int],
) -> dict[str, np.ndarray]:
    """Rebuild the audited derived masks from immutable source masks."""

    by_source: dict[str, Mapping[str, Any]] = {}
    raw_masks: dict[str, np.ndarray] = {}
    for item in objects:
        source_id = str(item.get("source_object_id") or "")
        if not source_id or source_id in by_source:
            raise ValueError("delivery source object IDs must be unique")
        source_path = Path(str(item.get("source_mask_path") or "")).resolve()
        if not source_path.is_file():
            raise FileNotFoundError(source_path)
        expected_hash = str(item.get("source_mask_sha256") or "")
        if len(expected_hash) != 64 or _sha256(source_path) != expected_hash:
            raise ValueError(f"source mask hash mismatch: {source_id}")
        by_source[source_id] = item
        raw_masks[source_id] = _load_binary_mask(source_path, expected_shape)

    def priority(item: Mapping[str, Any]) -> int:
        ownership = item.get("reconstruction_mask_ownership") or {}
        value = ownership.get("priority_index")
        if not isinstance(value, int) or value < 0:
            raise ValueError("delivery ownership priority is invalid")
        return value

    ordered = sorted(objects, key=priority)
    derived: dict[str, np.ndarray] = {}
    for item in ordered:
        source_id = str(item["source_object_id"])
        ownership = item.get("reconstruction_mask_ownership") or {}
        owners = ownership.get("owner_source_object_ids") or []
        if not isinstance(owners, list):
            raise ValueError("delivery ownership owners must be a list")
        mask = raw_masks[source_id].copy()
        for owner_id in owners:
            owner_key = str(owner_id)
            if owner_key not in derived:
                raise ValueError(
                    f"delivery ownership owner is unavailable: {owner_key}"
                )
            mask &= ~derived[owner_key]
        expected_area = ownership.get("delivery_area_pixels")
        if not isinstance(expected_area, int) or int(mask.sum()) != expected_area:
            raise ValueError(
                f"derived delivery mask area mismatch for {source_id}"
            )
        if not bool(mask.any()):
            raise ValueError(f"derived delivery mask is empty: {source_id}")
        derived[source_id] = mask
    return derived


def _contains_value(value: Any, target: str) -> bool:
    if isinstance(value, Mapping):
        return any(_contains_value(item, target) for item in value.values())
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return any(_contains_value(item, target) for item in value)
    return str(value) == target


def _object_annotations(
    quality: Mapping[str, Any], source_object_id: str
) -> dict[str, list[dict[str, Any]]]:
    return {
        "hard_errors": [
            copy.deepcopy(dict(item))
            for item in quality.get("hard_errors") or []
            if isinstance(item, Mapping)
            and _contains_value(item, source_object_id)
        ],
        "warnings": [
            copy.deepcopy(dict(item))
            for item in quality.get("warnings") or []
            if isinstance(item, Mapping)
            and _contains_value(item, source_object_id)
        ],
    }


def build_delivery_plan(
    *,
    scene_name: str,
    image_path: str | Path,
    objects: Sequence[Mapping[str, Any]],
    quality: Mapping[str, Any],
) -> dict[str, Any]:
    """Classify usable objects and assign continuous downstream identities."""

    resolved_image = Path(image_path).expanduser().resolve()
    infrastructure_errors: list[dict[str, Any]] = []
    try:
        with Image.open(resolved_image) as source:
            source.verify()
        with Image.open(resolved_image) as source:
            width, height = source.size
    except (OSError, ValueError) as exc:
        width, height = 0, 0
        infrastructure_errors.append(
            {
                "type": "unreadable_source_image",
                "path": str(resolved_image),
                "error": f"{type(exc).__name__}: {exc}",
            }
        )

    usable: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []
    for source_index, raw in enumerate(objects):
        obj = copy.deepcopy(dict(raw))
        source_id = str(obj.get("object_id") or f"source_{source_index:03d}")
        sam3 = obj.get("sam3") or {}
        if sam3.get("status") != "success":
            excluded.append(
                {
                    "source_object_id": source_id,
                    "reason": "sam3_not_success",
                    "details": sam3.get("reason"),
                }
            )
            continue
        mask_path = Path(str(sam3.get("mask_path") or "")).expanduser().resolve()
        if not mask_path.is_file():
            excluded.append(
                {
                    "source_object_id": source_id,
                    "reason": "missing_mask_file",
                    "path": str(mask_path),
                }
            )
            continue
        if width <= 0 or height <= 0:
            excluded.append(
                {
                    "source_object_id": source_id,
                    "reason": "source_image_unavailable_for_mask_validation",
                }
            )
            continue
        try:
            mask = _load_binary_mask(mask_path, (height, width))
        except (OSError, ValueError) as exc:
            excluded.append(
                {
                    "source_object_id": source_id,
                    "reason": "invalid_mask",
                    "path": str(mask_path),
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
            continue
        usable.append(
            {
                "source_index": source_index,
                "source_object_id": source_id,
                "source_object": obj,
                "source_mask_path": str(mask_path),
                "mask_sha256": _sha256(mask_path),
                "area_pixels": int(mask.sum()),
                "mask": mask,
                "rank": _score(obj),
            }
        )

    parents = list(range(len(usable)))

    def find(index: int) -> int:
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    def union(left: int, right: int) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root != right_root:
            parents[right_root] = left_root

    for left_index, left in enumerate(usable):
        for right_index in range(left_index + 1, len(usable)):
            right = usable[right_index]
            metrics = _mask_metrics(left["mask"], right["mask"])
            identical_mask = left["mask_sha256"] == right["mask_sha256"]
            if identical_mask or (
                int(metrics["intersection_pixels"])
                >= DUPLICATE_MINIMUM_INTERSECTION_PIXELS
                and float(metrics["iou"]) >= DUPLICATE_MINIMUM_IOU
                and float(metrics["left_coverage"])
                >= DUPLICATE_MINIMUM_COVERAGE
                and float(metrics["right_coverage"])
                >= DUPLICATE_MINIMUM_COVERAGE
            ):
                union(left_index, right_index)

    components: dict[int, list[dict[str, Any]]] = {}
    for index, item in enumerate(usable):
        components.setdefault(find(index), []).append(item)
    duplicate_collapses: list[dict[str, Any]] = []
    duplicate_losers: set[str] = set()
    for group in components.values():
        if len(group) < 2:
            continue
        winner = max(group, key=_duplicate_winner_key)
        loser_items = [item for item in group if item is not winner]
        loser_ids = [item["source_object_id"] for item in loser_items]
        duplicate_losers.update(loser_ids)
        identical = len({item["mask_sha256"] for item in group}) == 1
        collapse = {
            "kind": "identical_mask" if identical else "near_identical_mask",
            "kept_source_object_id": winner["source_object_id"],
            "excluded_source_object_ids": loser_ids,
            "resolution": (
                "prefer_pass_a_provenance_then_score_then_bbox_iou_then_"
                "visual_order"
            ),
            "pair_metrics": [
                {
                    "excluded_source_object_id": item["source_object_id"],
                    **{
                        key: round(float(value), 6)
                        for key, value in _mask_metrics(
                            winner["mask"], item["mask"]
                        ).items()
                    },
                }
                for item in loser_items
            ],
        }
        if identical:
            collapse["mask_sha256"] = winner["mask_sha256"]
        duplicate_collapses.append(collapse)
    duplicate_collapses.sort(
        key=lambda item: str(item["kept_source_object_id"])
    )
    selected = [
        item
        for item in usable
        if item["source_object_id"] not in duplicate_losers
    ]
    selected.sort(key=lambda item: item["source_index"])
    collapse_by_loser = {
        source_id: collapse
        for collapse in duplicate_collapses
        for source_id in collapse["excluded_source_object_ids"]
    }
    for source_id, collapse in collapse_by_loser.items():
        excluded.append(
            {
                "source_object_id": source_id,
                "reason": (
                    "identical_mask_duplicate_collapsed"
                    if collapse["kind"] == "identical_mask"
                    else "near_identical_mask_duplicate_collapsed"
                ),
                "kept_source_object_id": collapse["kept_source_object_id"],
            }
        )

    selected, reconstruction_mask_ownership = (
        _derive_reconstruction_ownership(selected)
    )
    excluded.extend(reconstruction_mask_ownership["excluded_objects"])

    delivery_objects = []
    for order_index, item in enumerate(selected):
        source = item["source_object"]
        sam3 = source.get("sam3") or {}
        object_id = f"object_{order_index + 1:03d}"
        category = str(source.get("category") or "unknown")
        description = str(
            source.get("description")
            or source.get("instance_name")
            or source.get("sam3_prompt")
            or category
        ).strip()
        delivery_objects.append(
            {
                "order_index": order_index,
                "object_id": object_id,
                "sam3d_internal_name": object_id,
                "source_object_id": item["source_object_id"],
                "category": category,
                "description": description,
                "source_mask_path": item["source_mask_path"],
                "source_mask_sha256": item["mask_sha256"],
                # Materialization replaces this with the hash of the derived
                # Stage 04 mask; it is never a claim about the source mask.
                "mask_sha256": item["mask_sha256"],
                "area_pixels": int(item["delivery_mask"].sum()),
                "source_area_pixels": item["area_pixels"],
                "reconstruction_mask_ownership": item[
                    "reconstruction_mask_ownership"
                ],
                "bbox_2d_norm_xyxy": source.get("bbox_2d_norm_xyxy"),
                "mask_bbox_abs_xyxy": sam3.get("mask_bbox_abs_xyxy"),
                "sam3": {
                    "status": "success",
                    "score": sam3.get("score"),
                    "selection_method": sam3.get("selection_method"),
                },
                "quality_annotations": _object_annotations(
                    quality, item["source_object_id"]
                ),
            }
        )

    if not delivery_objects:
        infrastructure_errors.append(
            {
                "type": "no_usable_object_masks",
                "source_object_count": len(objects),
                "excluded_object_count": len(excluded),
            }
        )
    source_quality_passed = str(quality.get("status") or "") == "pass"
    advisory_errors = list(quality.get("advisory_errors") or [])
    if infrastructure_errors:
        status = "infrastructure_failed"
    elif not source_quality_passed:
        status = "quality_failed"
    elif not excluded and not duplicate_collapses and not advisory_errors:
        status = "sam3d_ready"
    else:
        status = "sam3d_ready_degraded"

    return {
        "schema_version": "scenemosaic_sam3d_delivery_plan",
        "scene_name": scene_name,
        "status": status,
        "sam3d_execution_allowed": (
            source_quality_passed and status in READY_STATUSES
        ),
        "source_quality_status": quality.get("status"),
        "source_hard_error_count": len(quality.get("hard_errors") or []),
        "source_advisory_error_count": len(advisory_errors),
        "source_warning_count": len(quality.get("warnings") or []),
        "source_object_count": len(objects),
        "usable_source_object_count": len(usable),
        "delivery_object_count": len(delivery_objects),
        "excluded_objects": excluded,
        "duplicate_collapses": duplicate_collapses,
        "duplicate_collapse_thresholds": {
            "minimum_intersection_pixels": (
                DUPLICATE_MINIMUM_INTERSECTION_PIXELS
            ),
            "minimum_iou": DUPLICATE_MINIMUM_IOU,
            "minimum_coverage": DUPLICATE_MINIMUM_COVERAGE,
        },
        "reconstruction_mask_ownership": reconstruction_mask_ownership,
        "infrastructure_errors": infrastructure_errors,
        "image_path": str(resolved_image),
        "image_size": [width, height] if width and height else None,
        "objects": delivery_objects,
    }


def materialize_delivery_package(
    *,
    publish_dir: str | Path,
    output_name: str = "sam3d_input",
) -> dict[str, Any]:
    """Build an atomic, standard SAM3D package inside a published result."""

    published = Path(publish_dir).expanduser().resolve()
    registry_path = published / "sam3_objects.json"
    quality_path = published / "quality_report.json"
    if not registry_path.is_file() or not quality_path.is_file():
        raise FileNotFoundError(
            f"published result lacks registry or quality report: {published}"
        )
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    quality = json.loads(quality_path.read_text(encoding="utf-8"))
    plan = build_delivery_plan(
        scene_name=str(registry.get("scene_name") or published.parent.name),
        image_path=str(registry.get("image_path") or ""),
        objects=registry.get("objects") or [],
        quality=quality,
    )

    output_dir = published / output_name
    temporary = published / f".{output_name}.tmp"
    if output_dir.exists():
        raise FileExistsError(f"SAM3D delivery already exists: {output_dir}")
    if temporary.exists():
        raise FileExistsError(f"stale SAM3D delivery staging exists: {temporary}")
    temporary.mkdir()
    masks_dir = temporary / "masks"
    masks_dir.mkdir()
    final_masks_dir = output_dir / "masks"
    image_output = output_dir / "image.png"

    delivery_masks: dict[str, np.ndarray] = {}
    if not plan["infrastructure_errors"]:
        shutil.copy2(plan["image_path"], temporary / "image.png")
        with Image.open(temporary / "image.png") as source_image:
            expected_shape = (source_image.height, source_image.width)
        delivery_masks = _materialize_derived_delivery_masks(
            plan["objects"], expected_shape
        )
    manifest_objects = []
    for item in plan["objects"]:
        copied = copy.deepcopy(item)
        filename = (
            f"{item['order_index']:03d}_{item['object_id']}_"
            f"{_safe_name(item['category'])}_mask.png"
        )
        mask_path = masks_dir / filename
        mask = delivery_masks[str(item["source_object_id"])]
        Image.fromarray(mask.astype(np.uint8) * 255).save(mask_path)
        delivery_hash = _sha256(mask_path)
        ownership = copied.get("reconstruction_mask_ownership") or {}
        copied["mask_mode"] = str(ownership.get("mode") or "copy")
        copied["mask_path"] = str(final_masks_dir / filename)
        copied["mask_sha256"] = delivery_hash
        item["mask_sha256"] = delivery_hash
        item["delivery_mask_path"] = str(final_masks_dir / filename)
        manifest_objects.append(copied)

    manifest = {
        "schema_version": "scenemosaic_sam3d_input_manifest",
        "scene_name": plan["scene_name"],
        "stage": "sam3d_input",
        "delivery_status": plan["status"],
        "sam3d_execution_allowed": plan["sam3d_execution_allowed"],
        "mask_mode": "derived_reconstruction_ownership",
        "image_path": str(image_output),
        "source_image_path": plan["image_path"],
        "source_registry": str(registry_path),
        "source_quality_report": str(quality_path),
        "object_count": len(manifest_objects),
        "objects": manifest_objects,
    }
    run_info = {
        "schema_version": "scenemosaic_sam3d_delivery_run_info",
        "scene_name": plan["scene_name"],
        "status": plan["status"],
        "sam3d_execution_allowed": plan["sam3d_execution_allowed"],
        "object_count": len(manifest_objects),
        "created_at": _utc_now(),
        "source_quality_status": plan["source_quality_status"],
        "excluded_object_count": len(plan["excluded_objects"]),
        "duplicate_collapse_count": len(plan["duplicate_collapses"]),
        "reconstruction_mask_ownership_allocation_count": (
            plan["reconstruction_mask_ownership"]["allocation_count"]
        ),
        "reconstruction_mask_ownership_excluded_count": (
            plan["reconstruction_mask_ownership"]["excluded_count"]
        ),
        "infrastructure_error_count": len(plan["infrastructure_errors"]),
    }
    scene_config = {
        "schema_version": "scenemosaic_sam3d_delivery_scene_config",
        "scene_name": plan["scene_name"],
        "run_root": str(published),
        "input_image": str(image_output),
        "source_input_image": plan["image_path"],
        "object_id_base": 1,
        "object_id_format": "object_{index:03d}",
        "conda_envs": {
            "sam3": os.environ.get("SCENEMOSAIC_SAM3_ENV", "sam3"),
            "sam3d": "sam3d-objects",
        },
        "stage_envs": {
            output_name: "sam3",
            "sam3d_output": "sam3d",
        },
        "stages": {
            "sam3d_input": output_name,
            "sam3d_output": "sam3d_output",
        },
        "delivery_status": plan["status"],
        "sam3d_execution_allowed": plan["sam3d_execution_allowed"],
        "created_at_utc": _utc_now(),
    }
    _atomic_json(temporary / "sam3d_input_manifest.json", manifest)
    _atomic_json(temporary / "sam3d_input_run_info.json", run_info)
    _atomic_json(temporary / "delivery_report.json", plan)
    _atomic_json(temporary / "sam3d_scene_config.json", scene_config)

    expected_ids = [
        f"object_{index + 1:03d}" for index in range(len(manifest_objects))
    ]
    actual_ids = [item["object_id"] for item in manifest_objects]
    if actual_ids != expected_ids:
        raise AssertionError(
            f"non-continuous delivery identities: {actual_ids} != {expected_ids}"
        )
    if plan["sam3d_execution_allowed"]:
        if not (temporary / "image.png").is_file() or not manifest_objects:
            raise AssertionError("ready delivery lacks image or objects")
        if len(list(masks_dir.glob("*.png"))) != len(manifest_objects):
            raise AssertionError("delivery mask count does not match manifest")
        if any(not item["description"] for item in manifest_objects):
            raise AssertionError("delivery object has empty description")
    os.replace(temporary, output_dir)
    return {
        "status": plan["status"],
        "sam3d_execution_allowed": plan["sam3d_execution_allowed"],
        "object_count": len(manifest_objects),
        "excluded_object_count": len(plan["excluded_objects"]),
        "duplicate_collapse_count": len(plan["duplicate_collapses"]),
        "reconstruction_mask_ownership_allocation_count": (
            plan["reconstruction_mask_ownership"]["allocation_count"]
        ),
        "reconstruction_mask_ownership_excluded_count": (
            plan["reconstruction_mask_ownership"]["excluded_count"]
        ),
        "output_dir": str(output_dir),
        "manifest_path": str(output_dir / "sam3d_input_manifest.json"),
        "scene_config_path": str(output_dir / "sam3d_scene_config.json"),
        "delivery_report_path": str(output_dir / "delivery_report.json"),
    }
