#!/usr/bin/env python3
"""Deterministic SEG-008 ownership cleanup for verified open-storage hosts."""

from __future__ import annotations

import hashlib
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from PIL import Image

MODULE_DIR = Path(__file__).resolve().parent
if str(MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(MODULE_DIR))

from . import repair_scope_router as SCOPE


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_mask(path: Any, shape: tuple[int, int]) -> tuple[Path, np.ndarray]:
    resolved = Path(str(path or "")).expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    mask = np.array(Image.open(resolved).convert("L")) > 0
    if mask.shape != shape:
        raise ValueError(f"mask shape mismatch: {mask.shape} != {shape}")
    return resolved, mask


def _bbox(mask: np.ndarray) -> list[int]:
    ys, xs = np.where(mask)
    if not len(xs):
        raise ValueError("empty mask")
    return [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]


def _safe(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(value or "").lower()).strip("_") or "object"


def resolve_open_storage_ownership(
    *,
    image_path: str | Path,
    objects: Sequence[Mapping[str, Any]],
    output_dir: str | Path,
) -> dict[str, Any]:
    """Subtract verified nested child pixels from open-storage structure masks."""

    resolved_image = Path(image_path).expanduser().resolve()
    image = np.array(Image.open(resolved_image).convert("RGB"), dtype=np.uint8)
    shape = tuple(image.shape[:2])
    output = Path(output_dir).expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"ownership output is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)

    copied = [dict(item) for item in objects]
    masks: dict[str, np.ndarray] = {}
    source_paths: dict[str, Path] = {}
    source_hashes: dict[str, str] = {}
    hard_errors: list[dict[str, Any]] = []
    for obj in copied:
        object_id = str(obj.get("object_id") or "")
        sam3 = dict(obj.get("sam3") or {})
        if sam3.get("status") != "success":
            continue
        try:
            path, mask = _load_mask(sam3.get("mask_path"), shape)
            if not mask.any():
                raise ValueError("empty mask")
        except Exception as exc:
            hard_errors.append(
                {
                    "type": "invalid_mask_for_ownership",
                    "object_id": object_id,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
            continue
        masks[object_id] = mask
        source_paths[object_id] = path
        source_hashes[object_id] = _sha256(path)

    records: list[dict[str, Any]] = []
    changed_ids: list[str] = []
    for host in copied:
        host_id = str(host.get("object_id") or "")
        if host_id not in masks:
            continue
        scope = SCOPE.classify_object_role(host)
        if scope.get("object_role") != SCOPE.OPEN_STORAGE_ROLE:
            continue
        host_mask = masks[host_id]
        host_box = _bbox(host_mask)
        hx1, hy1, hx2, hy2 = host_box
        child_ids: list[str] = []
        intersection_before: dict[str, int] = {}
        for child in copied:
            child_id = str(child.get("object_id") or "")
            if child_id == host_id or child_id not in masks:
                continue
            child_mask = masks[child_id]
            child_area = int(child_mask.sum())
            if not child_area:
                continue
            inside_fraction = (
                int(child_mask[hy1:hy2, hx1:hx2].sum()) / child_area
            )
            intersection = int(np.logical_and(host_mask, child_mask).sum())
            if inside_fraction < 0.70 or intersection <= 0:
                continue
            child_scope = SCOPE.classify_object_role(child)
            if (
                child_scope.get("object_role")
                in {SCOPE.OPEN_STORAGE_ROLE, SCOPE.SUPPORT_SURFACE_ROLE}
                and child_area > 0.25 * int(host_mask.sum())
            ):
                continue
            child_ids.append(child_id)
            intersection_before[child_id] = intersection

        if not child_ids:
            records.append(
                {
                    "host_object_id": host_id,
                    "repair_scope": scope,
                    "status": "no_overlapping_nested_contents",
                    "child_object_ids": [],
                    "removed_pixels": 0,
                }
            )
            continue
        child_union = np.zeros(shape, dtype=bool)
        for child_id in child_ids:
            child_union |= masks[child_id]
        resolved = host_mask & ~child_union
        if not resolved.any():
            hard_errors.append(
                {
                    "type": "host_subtraction_emptied_mask",
                    "host_object_id": host_id,
                    "child_object_ids": child_ids,
                }
            )
            continue
        removed = int(host_mask.sum()) - int(resolved.sum())
        masks_dir = output / "masks"
        overlays_dir = output / "overlays"
        masks_dir.mkdir(parents=True, exist_ok=True)
        overlays_dir.mkdir(parents=True, exist_ok=True)
        stem = f"{_safe(host_id)}_{_safe(host.get('category'))}"
        mask_path = masks_dir / f"{stem}_mask.png"
        overlay_path = overlays_dir / f"{stem}_overlay.png"
        Image.fromarray(resolved.astype(np.uint8) * 255).save(mask_path)
        canvas = image.astype(np.float32).copy()
        canvas[resolved] = (
            0.55 * canvas[resolved]
            + 0.45 * np.array((255, 0, 0), dtype=np.float32)
        )
        Image.fromarray(np.clip(canvas, 0, 255).astype(np.uint8)).save(
            overlay_path
        )
        sam3 = dict(host.get("sam3") or {})
        sam3.update(
            {
                "mask_path": str(mask_path),
                "overlay_path": str(overlay_path),
                "area_pixels": int(resolved.sum()),
                "mask_bbox_abs_xyxy": _bbox(resolved),
                "pixel_ownership_resolution": {
                    "action": "host_minus_contents",
                    "source_mask_path": str(source_paths[host_id]),
                    "child_object_ids": child_ids,
                    "removed_pixels": removed,
                },
            }
        )
        host["sam3"] = sam3
        changed_ids.append(host_id)
        masks[host_id] = resolved
        after = {
            child_id: int(np.logical_and(resolved, masks[child_id]).sum())
            for child_id in child_ids
        }
        if any(after.values()):
            hard_errors.append(
                {
                    "type": "host_content_overlap_remains",
                    "host_object_id": host_id,
                    "intersection_after": after,
                }
            )
        records.append(
            {
                "host_object_id": host_id,
                "repair_scope": scope,
                "status": "resolved",
                "child_object_ids": child_ids,
                "intersection_before": intersection_before,
                "intersection_after": after,
                "removed_pixels": removed,
                "output_mask_path": str(mask_path),
                "output_overlay_path": str(overlay_path),
            }
        )

    source_mutations = []
    for object_id, source_path in source_paths.items():
        current = _sha256(source_path)
        if current != source_hashes[object_id]:
            source_mutations.append(object_id)
    if source_mutations:
        hard_errors.append(
            {
                "type": "source_mask_mutated",
                "object_ids": source_mutations,
            }
        )
    audit = {
        "schema_version": "scenemosaic_open_storage_ownership_resolution",
        "status": "fail" if hard_errors else "pass",
        "changed_host_count": len(changed_ids),
        "changed_host_object_ids": changed_ids,
        "host_records": records,
        "source_masks_unchanged": not source_mutations,
        "formal_outputs_modified": False,
        "hard_errors": hard_errors,
    }
    audit_path = output / "ownership_resolution_audit.json"
    temporary = audit_path.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, audit_path)
    return {
        "objects": copied,
        "audit": audit,
        "audit_path": str(audit_path),
    }


__all__ = ["resolve_open_storage_ownership"]
