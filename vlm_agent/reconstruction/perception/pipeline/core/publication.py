from __future__ import annotations

import copy
import os
import re
import shutil
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from PIL import Image

def atomic_publish(staging: str | Path, final: str | Path) -> None:
    staging_path = Path(staging).expanduser().resolve()
    final_path = Path(final).expanduser().resolve()
    if not staging_path.is_dir():
        raise FileNotFoundError(staging_path)
    if final_path.exists():
        raise FileExistsError(final_path)
    final_path.parent.mkdir(parents=True, exist_ok=True)
    os.replace(staging_path, final_path)


def safe_name(value: Any) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("_") or "object"


def canonicalize_final_object_assets(
    *,
    objects: Sequence[Mapping[str, Any]],
    staging_dir: str | Path,
    image_np: np.ndarray,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    # Removal and replacement stages may renumber objects after their masks
    # were materialized. Build a clean two-phase asset tree so published
    # filenames match the final registry and superseded files cannot leak
    # into manual review.
    copied = [copy.deepcopy(dict(item)) for item in objects]
    staging = Path(staging_dir).expanduser().resolve()
    masks_dir = staging / "masks"
    overlays_dir = staging / "overlays"
    temporary_root = staging / ".canonical_final_assets"
    if temporary_root.exists():
        raise FileExistsError(
            f"canonical final asset directory already exists: {temporary_root}"
        )
    temporary_masks = temporary_root / "masks"
    temporary_overlays = temporary_root / "overlays"
    temporary_masks.mkdir(parents=True)
    temporary_overlays.mkdir(parents=True)

    old_mask_names = {
        path.name for path in masks_dir.glob("*") if path.is_file()
    } if masks_dir.is_dir() else set()
    old_overlay_names = {
        path.name for path in overlays_dir.glob("*") if path.is_file()
    } if overlays_dir.is_dir() else set()
    target_mask_names: set[str] = set()
    target_overlay_names: set[str] = set()
    path_updates: list[tuple[dict[str, Any], str, str]] = []
    external_source_count = 0

    for obj in copied:
        sam3 = copy.deepcopy(dict(obj.get("sam3") or {}))
        obj["sam3"] = sam3
        if sam3.get("status") != "success":
            continue
        object_id = str(obj.get("object_id") or "")
        if not re.fullmatch(r"proposal_\d{3}", object_id):
            raise ValueError(f"invalid final object ID for asset publication: {object_id}")
        category = safe_name(str(obj.get("category") or "object").lower())
        mask_name = f"{object_id}_{category}_mask.png"
        overlay_name = f"{object_id}_{category}_overlay.png"
        if mask_name in target_mask_names or overlay_name in target_overlay_names:
            raise ValueError(f"duplicate final asset name for {object_id}")
        source_mask = Path(str(sam3.get("mask_path") or "")).expanduser().resolve()
        if not source_mask.is_file():
            raise FileNotFoundError(
                f"missing final source mask for {object_id}: {source_mask}"
            )
        try:
            source_mask.relative_to(staging)
        except ValueError:
            external_source_count += 1
        with Image.open(source_mask) as source:
            mask = np.asarray(source.convert("L")) > 0
        if tuple(mask.shape) != tuple(image_np.shape[:2]):
            raise ValueError(
                f"final mask shape mismatch for {object_id}: "
                f"{tuple(mask.shape)} != {tuple(image_np.shape[:2])}"
            )
        Image.fromarray(mask.astype(np.uint8) * 255).save(
            temporary_masks / mask_name
        )
        overlay = np.asarray(image_np, dtype=np.uint8).copy()
        overlay[mask] = (
            0.55 * overlay[mask]
            + 0.45 * np.array((255, 0, 0), dtype=np.uint8)
        ).astype(np.uint8)
        Image.fromarray(overlay).save(temporary_overlays / overlay_name)
        target_mask_names.add(mask_name)
        target_overlay_names.add(overlay_name)
        path_updates.append((sam3, mask_name, overlay_name))

    if masks_dir.exists():
        shutil.rmtree(masks_dir)
    if overlays_dir.exists():
        shutil.rmtree(overlays_dir)
    os.replace(temporary_masks, masks_dir)
    os.replace(temporary_overlays, overlays_dir)
    temporary_root.rmdir()

    for sam3, mask_name, overlay_name in path_updates:
        sam3["mask_path"] = str(masks_dir / mask_name)
        sam3["overlay_path"] = str(overlays_dir / overlay_name)

    audit = {
        "schema_version": "scenemosaic_canonical_final_asset_publication",
        "status": "pass",
        "successful_object_count": len(path_updates),
        "published_mask_count": len(target_mask_names),
        "published_overlay_count": len(target_overlay_names),
        "external_source_count": external_source_count,
        "removed_stale_mask_names": sorted(old_mask_names - target_mask_names),
        "removed_stale_overlay_names": sorted(
            old_overlay_names - target_overlay_names
        ),
        "final_mask_names": sorted(target_mask_names),
        "final_overlay_names": sorted(target_overlay_names),
        "hard_errors": [],
        "additional_vlm_calls": 0,
        "additional_sam3_inference_calls": 0,
        "formal_outputs_modified": False,
    }
    return copied, audit
