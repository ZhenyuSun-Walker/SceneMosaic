#!/usr/bin/env python3
"""Fail-closed suppression of a proved part masquerading as an inventory item."""

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

from . import repair_scope_router as SCOPE


@dataclass(frozen=True)
class NestedPartMaskSuppressionConfig:
    enabled: bool = True
    isolated_attempt_only: bool = True
    minimum_child_area_pixels: int = 32
    minimum_child_containment: float = 0.95
    maximum_child_to_parent_area_ratio: float = 0.85
    minimum_explicit_assembly_containment: float = 0.98


ASSEMBLY_LINK_TOKENS = {"with", "including", "featuring"}
ATTACHED_BOTANICAL_TOKENS = {
    "blossom", "blossoms", "branch", "branches", "flower", "flowers",
    "foliage", "leaf", "leaves", "petal", "petals", "stem", "stems",
}
INDEPENDENT_CONTENT_HOST_TOKENS = {
    "basket", "bowl", "container", "cup", "dish", "glass", "mug",
    "plate", "platter", "receptacle", "tray",
}


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _tokens(value: Any) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", str(value or "").lower()))


def _is_complete_potted_plant(obj: Mapping[str, Any]) -> bool:
    return {"potted", "plant"} <= _tokens(obj.get("category"))


def _is_vase_like_component(obj: Mapping[str, Any]) -> bool:
    return bool(_tokens(obj.get("category")) & {"vase", "planter"})


def _semantic_tokens(obj: Mapping[str, Any]) -> set[str]:
    result: set[str] = set()
    for field in ("category", "sam3_prompt", "instance_name", "description"):
        result.update(_tokens(obj.get(field)))
    return result


def _explicit_attached_botanical_tokens(obj: Mapping[str, Any]) -> set[str]:
    result: set[str] = set()
    for field in ("sam3_prompt", "description"):
        words = re.findall(r"[a-z0-9]+", str(obj.get(field) or "").lower())
        for index, word in enumerate(words):
            if word in ASSEMBLY_LINK_TOKENS:
                result.update(words[index + 1 :])
    return result & ATTACHED_BOTANICAL_TOKENS


def _can_own_attached_botanical_component(
    parent: Mapping[str, Any],
    child: Mapping[str, Any],
) -> bool:
    parent_components = _explicit_attached_botanical_tokens(parent)
    child_components = _semantic_tokens(child) & ATTACHED_BOTANICAL_TOKENS
    if not parent_components or not parent_components & child_components:
        return False
    if str(parent.get("object_type") or "").strip().lower() == "furniture":
        return False
    if _semantic_tokens(parent) & INDEPENDENT_CONTENT_HOST_TOKENS:
        return False
    role = SCOPE.classify_object_role(parent).get("object_role")
    return role not in {
        SCOPE.OPEN_STORAGE_ROLE,
        SCOPE.SUPPORT_SURFACE_ROLE,
        SCOPE.AMBIGUOUS_ROLE,
    }


def _contained_metrics(
    child_mask: np.ndarray,
    parent_mask: np.ndarray,
) -> tuple[int, int, float, float]:
    child_area = int(child_mask.sum())
    parent_area = int(parent_mask.sum())
    intersection = int(np.logical_and(child_mask, parent_mask).sum())
    return (
        child_area,
        parent_area,
        intersection / max(child_area, 1),
        child_area / max(parent_area, 1),
    )


def _load_mask(path_value: Any, expected_shape: tuple[int, int]) -> np.ndarray:
    path = Path(str(path_value or "")).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    mask = np.array(Image.open(path).convert("L")) > 0
    if mask.shape != expected_shape or not mask.any():
        raise ValueError("invalid_mask_shape_or_empty")
    return mask


def _remap_exact_proposal_ids(value: Any, mapping: Mapping[str, str]) -> Any:
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


def run_nested_part_mask_suppression(
    *,
    objects: Sequence[Mapping[str, Any]],
    image_shape: tuple[int, int],
    output_dir: str | Path,
    config: NestedPartMaskSuppressionConfig | None = None,
) -> dict[str, Any]:
    """Remove only a uniquely proved component already owned by an assembly."""

    cfg = config or NestedPartMaskSuppressionConfig()
    if not cfg.isolated_attempt_only:
        raise ValueError("nested part mask suppression is isolated-attempt only")
    work = Path(output_dir).expanduser().resolve()
    work.mkdir(parents=True, exist_ok=True)
    copied = [copy.deepcopy(dict(obj)) for obj in objects if isinstance(obj, Mapping)]
    audit: dict[str, Any] = {
        "schema_version": "scenemosaic_contained_component_suppression",
        "enabled": cfg.enabled,
        "isolated_attempt_only": cfg.isolated_attempt_only,
        "status": "disabled" if not cfg.enabled else "noop",
        "input_object_count": len(copied),
        "suppressed": [],
        "rejected": [],
        "hard_errors": [],
        "thresholds": {
            "minimum_child_area_pixels": cfg.minimum_child_area_pixels,
            "minimum_child_containment": cfg.minimum_child_containment,
            "maximum_child_to_parent_area_ratio": cfg.maximum_child_to_parent_area_ratio,
            "minimum_explicit_assembly_containment": (
                cfg.minimum_explicit_assembly_containment
            ),
        },
    }
    if not cfg.enabled:
        _atomic_json(work / "contained_component_suppression_audit.json", audit)
        return {"objects": copied, "audit": audit}

    masks: dict[str, np.ndarray] = {}
    for obj in copied:
        object_id = str(obj.get("object_id") or "")
        sam3 = obj.get("sam3") or {}
        if str(sam3.get("status") or "") != "success":
            continue
        try:
            masks[object_id] = _load_mask(sam3.get("mask_path"), image_shape)
        except (OSError, ValueError) as exc:
            audit["hard_errors"].append(
                {
                    "type": "invalid_success_mask_for_contained_component_suppression",
                    "object_id": object_id,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )

    parent_objects = [
        obj for obj in copied
        if _is_complete_potted_plant(obj)
        and str(obj.get("object_id") or "") in masks
    ]
    removal_ids: set[str] = set()
    for child in copied:
        child_id = str(child.get("object_id") or "")
        if (
            not _is_vase_like_component(child)
            or child_id not in masks
            or _is_complete_potted_plant(child)
        ):
            continue
        child_mask = masks[child_id]
        child_area = int(child_mask.sum())
        if child_area < cfg.minimum_child_area_pixels:
            audit["rejected"].append(
                {
                    "object_id": child_id,
                    "reason": "child_area_below_minimum",
                    "child_area_pixels": child_area,
                }
            )
            continue
        matches: list[dict[str, Any]] = []
        for parent in parent_objects:
            parent_id = str(parent.get("object_id") or "")
            if parent_id == child_id:
                continue
            parent_mask = masks[parent_id]
            _, parent_area, child_containment, child_to_parent_area_ratio = (
                _contained_metrics(child_mask, parent_mask)
            )
            if (
                child_containment >= cfg.minimum_child_containment
                and child_to_parent_area_ratio <= cfg.maximum_child_to_parent_area_ratio
            ):
                matches.append(
                    {
                        "parent_object_id": parent_id,
                        "parent_category": str(parent.get("category") or ""),
                        "parent_area_pixels": parent_area,
                        "child_containment": child_containment,
                        "child_to_parent_area_ratio": child_to_parent_area_ratio,
                    }
                )
        if len(matches) != 1:
            audit["rejected"].append(
                {
                    "object_id": child_id,
                    "reason": (
                        "no_unique_complete_potted_plant_container"
                        if not matches
                        else "ambiguous_complete_potted_plant_container"
                    ),
                    "matching_parent_count": len(matches),
                }
            )
            continue
        match = matches[0]
        removal_ids.add(child_id)
        audit["suppressed"].append(
            {
                "removed_object_id": child_id,
                "removed_category": str(child.get("category") or ""),
                "kept_object_id": match["parent_object_id"],
                "kept_category": match["parent_category"],
                "policy": "complete_potted_plant_contains_vase_like_submask",
                "child_area_pixels": child_area,
                "parent_area_pixels": match["parent_area_pixels"],
                "child_containment": round(float(match["child_containment"]), 6),
                "child_to_parent_area_ratio": round(
                    float(match["child_to_parent_area_ratio"]), 6
                ),
            }
        )

    for child in copied:
        child_id = str(child.get("object_id") or "")
        if (
            child_id in removal_ids
            or child_id not in masks
            or not (_semantic_tokens(child) & ATTACHED_BOTANICAL_TOKENS)
        ):
            continue
        semantic_parents = [
            parent
            for parent in copied
            if str(parent.get("object_id") or "") != child_id
            and str(parent.get("object_id") or "") in masks
            and _can_own_attached_botanical_component(parent, child)
        ]
        if not semantic_parents:
            continue
        matches = []
        for parent in semantic_parents:
            parent_id = str(parent.get("object_id") or "")
            (
                child_area,
                parent_area,
                containment,
                area_ratio,
            ) = _contained_metrics(masks[child_id], masks[parent_id])
            if (
                child_area >= cfg.minimum_child_area_pixels
                and containment >= cfg.minimum_explicit_assembly_containment
                and area_ratio <= cfg.maximum_child_to_parent_area_ratio
            ):
                matches.append(
                    {
                        "parent_object_id": parent_id,
                        "parent_category": str(parent.get("category") or ""),
                        "parent_area_pixels": parent_area,
                        "child_area_pixels": child_area,
                        "child_containment": containment,
                        "child_to_parent_area_ratio": area_ratio,
                    }
                )
        if len(matches) != 1:
            audit["rejected"].append(
                {
                    "object_id": child_id,
                    "reason": (
                        "no_unique_explicit_assembly_owner"
                        if not matches
                        else "ambiguous_explicit_assembly_owner"
                    ),
                    "semantic_parent_count": len(semantic_parents),
                    "matching_parent_count": len(matches),
                }
            )
            continue
        match = matches[0]
        removal_ids.add(child_id)
        audit["suppressed"].append(
            {
                "removed_object_id": child_id,
                "removed_category": str(child.get("category") or ""),
                "kept_object_id": match["parent_object_id"],
                "kept_category": match["parent_category"],
                "policy": (
                    "explicit_complete_assembly_contains_attached_component"
                ),
                "child_area_pixels": match["child_area_pixels"],
                "parent_area_pixels": match["parent_area_pixels"],
                "child_containment": round(
                    float(match["child_containment"]), 6
                ),
                "child_to_parent_area_ratio": round(
                    float(match["child_to_parent_area_ratio"]), 6
                ),
            }
        )

    result = [
        obj for obj in copied if str(obj.get("object_id") or "") not in removal_ids
    ]
    object_id_map = {
        str(obj.get("object_id") or ""): f"proposal_{index:03d}"
        for index, obj in enumerate(result, start=1)
    }
    result = [_remap_exact_proposal_ids(obj, object_id_map) for obj in result]
    audit["object_id_map"] = [
        {"old_object_id": old_id, "new_object_id": new_id}
        for old_id, new_id in object_id_map.items()
        if old_id != new_id
    ]
    for record in audit["suppressed"]:
        record["kept_final_object_id"] = object_id_map.get(
            str(record["kept_object_id"]), str(record["kept_object_id"])
        )
    audit["output_object_count"] = len(result)
    audit["suppressed_count"] = len(audit["suppressed"])
    audit["status"] = (
        "fail" if audit["hard_errors"] else ("pass" if removal_ids else "noop")
    )
    _atomic_json(work / "contained_component_suppression_audit.json", audit)
    return {"objects": result, "audit": audit}
