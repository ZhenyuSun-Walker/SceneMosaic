"""Parse and normalize VLM object-inventory responses."""

from __future__ import annotations

import json
import re
from typing import Any


def parse_vlm_response_text(
    text: str,
    *,
    finish_reason: str | None = None,
) -> Any:
    """Parse a JSON response and distinguish truncation from malformed output."""

    text = text.strip()
    if text.startswith("```"):
        text = text.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        failure_kind = "provider-truncated" if finish_reason == "length" else "malformed"
        raise RuntimeError(
            "VLM returned "
            f"{failure_kind} JSON: finish_reason={finish_reason!r}, "
            f"json_error={exc}"
        ) from exc


def _format_object_id(index: int, object_id_format: str) -> str:
    try:
        object_id = object_id_format.format(index=index)
    except Exception as exc:
        raise ValueError(
            f"Invalid object_id_format {object_id_format!r}: {exc}"
        ) from exc

    if not object_id:
        raise ValueError("object_id_format produced an empty object_id.")
    return object_id


def _grid_location_from_anchor(anchor: list[float]) -> tuple[str, dict[str, int]]:
    x, y = anchor
    row = min(10, max(1, int(y * 10) + 1))
    col = min(10, max(1, int(x * 10) + 1))
    return f"grid_r{row}_c{col}", {"row": row, "col": col}


def _normalize_anchor(
    anchor: Any,
    bbox: list[float],
    object_id: str,
) -> tuple[list[float], str]:
    if anchor is None:
        return [
            round((float(bbox[0]) + float(bbox[2])) / 2, 3),
            round((float(bbox[1]) + float(bbox[3])) / 2, 3),
        ], "derived_bbox_center"
    if not isinstance(anchor, list) or len(anchor) != 2:
        raise ValueError(f"Invalid anchor for {object_id}: {anchor}")
    x, y = anchor
    if not (0 <= x <= 1 and 0 <= y <= 1):
        raise ValueError(f"Anchor out of range for {object_id}: {anchor}")
    if not (bbox[0] <= x <= bbox[2] and bbox[1] <= y <= bbox[3]):
        raise ValueError(f"Anchor lies outside bbox for {object_id}: {anchor}")
    return [round(float(x), 3), round(float(y), 3)], "vlm"


def normalize_objects(
    objects: list[dict[str, Any]],
    object_id_base: int,
    object_id_format: str,
) -> list[dict[str, Any]]:
    """Assign stable IDs and validate inventory geometry without changing meaning."""

    normalized = []
    seen_ids = set()

    for offset, obj in enumerate(objects):
        object_id = _format_object_id(object_id_base + offset, object_id_format)
        if object_id in seen_ids:
            raise ValueError(f"Duplicate object_id generated: {object_id}")
        seen_ids.add(object_id)

        normalized_obj = dict(obj)
        normalized_obj.pop("image_object_id", None)
        normalized_obj["object_id"] = object_id

        semantic_tokens = set(
            re.findall(
                r"[a-z0-9]+",
                " ".join(
                    str(normalized_obj.get(key) or "")
                    for key in (
                        "category",
                        "sam3_prompt",
                        "instance_name",
                        "description",
                    )
                ).lower(),
            )
        )
        if (
            {"water", "dispenser"} <= semantic_tokens
            and semantic_tokens & {"bottle", "bottled", "jug", "reservoir"}
        ):
            previous_prompt = str(normalized_obj.get("sam3_prompt") or "")
            normalized_obj["sam3_prompt"] = (
                "complete freestanding water dispenser with water bottle on top"
            )
            normalized_obj["sam3_prompt_policy"] = {
                "rule": "complete_mounted_water_bottle",
                "previous_prompt": previous_prompt,
            }

        bbox = normalized_obj.get("bbox_2d_norm_xyxy")
        if not isinstance(bbox, list) or len(bbox) != 4:
            raise ValueError(f"Invalid bbox for {object_id}: {bbox}")

        x1, y1, x2, y2 = bbox
        if not (0 <= x1 <= 1 and 0 <= y1 <= 1 and 0 <= x2 <= 1 and 0 <= y2 <= 1):
            raise ValueError(f"BBox out of range for {object_id}: {bbox}")
        if x2 <= x1 or y2 <= y1:
            raise ValueError(f"Invalid bbox order for {object_id}: {bbox}")
        normalized_obj["bbox_2d_norm_xyxy"] = [round(float(value), 3) for value in bbox]

        anchor, anchor_source = _normalize_anchor(
            normalized_obj.get("anchor_2d_norm_xy"),
            normalized_obj["bbox_2d_norm_xyxy"],
            object_id,
        )
        normalized_obj["anchor_2d_norm_xy"] = anchor
        normalized_obj["anchor_source"] = anchor_source

        derived_location, derived_order = _grid_location_from_anchor(anchor)
        if not isinstance(normalized_obj.get("coarse_location"), str) or not normalized_obj.get("coarse_location", "").strip():
            normalized_obj["coarse_location"] = derived_location
        if not isinstance(normalized_obj.get("visual_order"), dict):
            normalized_obj["visual_order"] = derived_order

        # Technical SAM3 candidate assignment belongs to the segmentation stage.
        normalized_obj.pop("segmentation_mode", None)

        fallback_prompts = normalized_obj.get("fallback_prompts", [])
        if fallback_prompts is None:
            fallback_prompts = []
        if not isinstance(fallback_prompts, list):
            raise ValueError(f"Invalid fallback_prompts for {object_id}: {fallback_prompts}")
        normalized_obj["fallback_prompts"] = [
            str(prompt) for prompt in fallback_prompts[:1] if str(prompt).strip()
        ]

        normalized.append(normalized_obj)

    return normalized
