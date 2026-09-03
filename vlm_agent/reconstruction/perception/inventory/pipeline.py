"""Production two-pass VLM object inventory.

Pass A inventories primary reconstruction units in the whole image. Pass B
receives the same image, compact Pass-A registry, and optional segmentation
evidence, then reports missing units, corrections, and removals. Local code
validates and fuses both responses into provisional reconstruction objects.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .image_assets import create_vlm_jpeg, encode_data_url
from .normalization import normalize_objects, parse_vlm_response_text
from .review_assets import render_bbox_review_assets


ALLOWED_TYPES = {
    "furniture",
    "decor",
    "small_object",
    "structural_surface",
    "opening",
    "textile",
    "lighting",
    "plant",
    "appliance",
    "unknown",
}

LOCAL_CANDIDATE_ACTIONS = {
    "bind_to_addition",
    "replace_existing",
    "merge_extend_existing",
    "duplicate_of_existing",
    "part_of_existing",
    "reject",
    "uncertain",
}
LOCAL_CANDIDATE_EXISTING_TARGET_ACTIONS = {
    "replace_existing",
    "merge_extend_existing",
    "duplicate_of_existing",
    "part_of_existing",
}
LOCAL_CANDIDATE_NULL_TARGET_ACTIONS = {"reject", "uncertain"}

ASSIGNED_EVIDENCE_DECISIONS = {
    "pass",
    "update_existing",
    "split_instances",
    "remove_existing",
    "uncertain",
}

_COMPACT_NUMBER = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
_COMPACT_BARE_BBOX_AFTER_PROMPT = re.compile(
    rf'("p"\s*:\s*"(?:[^"\\]|\\.)*")\s*,\s*('
    rf'\[\s*{_COMPACT_NUMBER}\s*,\s*{_COMPACT_NUMBER}\s*,\s*'
    rf'{_COMPACT_NUMBER}\s*,\s*{_COMPACT_NUMBER}\s*\])'
)


def _repair_compact_bare_bbox_after_prompt(text: str) -> tuple[str, int]:
    """Restore the only unambiguous omitted key seen in compact additions."""

    return _COMPACT_BARE_BBOX_AFTER_PROMPT.subn(r'\1,"b":\2', text)


PASS_A_PROMPT = r"""
You are Pass A of a visible-object registry for SAM3 + SAM3D reconstruction.
Inspect the entire image, which has a cyan global coordinate grid. This pass
builds the PRIMARY inventory, not the final exhaustive inventory.

Register each visible primary physical reconstruction unit once:
- furniture, supports, storage, large textiles and wall carriers;
- visible appliances, plants, lighting, fixtures and salient medium objects;
- functional assemblies that would be incomplete if separated.

Do not spend the response on truly tiny loose surface details. Register each visible
open-storage carrier once; its small movable contents are reviewed in Pass B with
high-resolution content-host evidence. This exception does not apply to objects on
ordinary tables, counters or other support surfaces.
Always enumerate every visible recessed ceiling light. Include a small object
when it is visually distinct or is itself a primary scene unit. Exclude bare
wall, floor, ceiling, windows, doors, curtains,
background, and objects depicted inside paintings, photos, mirrors, televisions
or screens. This is not
relationship extraction; output no relations, scene-graph edges or 3D pose.

Object-unit rules:
- Mere contact does not merge independent objects.
- A visible gap, different shelf/cubby/container, or different pose starts a
  different unit.
- A bed includes frame, mattress, fitted sheet and visible duvet/blanket, while
  each loose pillow is separate.
- A potted plant is one assembly: include both foliage and its visible pot or
  planter. Use c=potted_plant and make p name the plant together with its pot.
- Keep fitted or dedicated parts with their assembly when separation would
  produce an incomplete reconstruction: join a sink, faucet and dedicated
  supporting vanity/counter; a fitted undercounter dishwasher belongs to that
  dedicated sink-counter assembly rather than becoming a separate unit. A visible
  range/stove interrupts a counter run: never widen one counter bbox across it;
  register the independent counter section on the other side separately. Join an
  attached water bottle with its dispenser; join connected lamp heads/rods; join a
  coat rack, towel ladder/rack, drying
  rack, roll holder/rack or pegboard with its hanging
  or dedicated contents. Do not absorb independent loose objects merely due to
  contact.

For each unit provide one coarse bbox. Identity and coverage matter more than
exact bbox edges; never widen a bbox to hedge across neighboring units. Category
c is a generic machine identifier for the visible physical unit: use lowercase
snake_case only (for example vase, book_stack, ceiling_light), with no spaces,
hyphens, color, material, size or location terms. Field p is the SAM3 semantic
text prompt: it must name the visible object itself, optionally with color,
material or form. Never put a location, support, direction, relation, grid cell
or nearby object in c or p.

Return ONLY one minified JSON object:
{"o":[{"c":"snake_case category","p":"2-5 object-only English words","t":"furniture|decor|small_object|structural_surface|opening|textile|lighting|plant|appliance|unknown","b":[0.000,0.000,0.000,0.000]}]}

Use normalized whole-image coordinates. ``b`` is exactly
``[left, top, right, bottom]`` (also called ``[x1, y1, x2, y2]``): require
``left < right`` and ``top < bottom``. Never use ``[x, y, width, height]``.
Do not return a name, description or anchor: local code deterministically
derives those from category and bbox. Do not output any other keys or prose.
""".strip()




COMPACT_PASS_B_PROTOCOL = "compact_pass_b"






def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.expanduser().resolve().read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_scene_config(path: Path) -> dict[str, Any]:
    config_path = path.expanduser().resolve()
    config = read_json(config_path)
    image_path = Path(str(config["input_image"])).expanduser().resolve()
    run_root = Path(str(config["run_root"])).expanduser().resolve()
    if not image_path.is_file():
        raise FileNotFoundError(image_path)
    return {
        "scene_name": str(config.get("scene_name") or run_root.name),
        "run_root": run_root,
        "image_path": image_path,
        "config_path": config_path,
    }


def ensure_attempt_output(output_dir: Path, run_root: Path) -> None:
    resolved = output_dir.expanduser().resolve()
    if "perception_attempts" not in resolved.parts:
        raise ValueError("output-dir must be inside a perception_attempts directory")
    formal_stages = {
        (run_root / name).resolve()
        for name in ("input", "reconstruction", "evolution")
    }
    if any(resolved == stage or stage in resolved.parents for stage in formal_stages):
        raise ValueError("refusing to write inside a formal pipeline stage")
    if resolved.exists() and any(resolved.iterdir()):
        raise FileExistsError(f"perception attempt output is not empty: {resolved}")
    resolved.mkdir(parents=True, exist_ok=True)


def valid_bbox(value: Any) -> bool:
    return (
        isinstance(value, list)
        and len(value) == 4
        and all(isinstance(item, (int, float)) and math.isfinite(item) for item in value)
        and 0 <= value[0] < value[2] <= 1
        and 0 <= value[1] < value[3] <= 1
    )


def valid_anchor(value: Any, bbox: list[float]) -> bool:
    return (
        isinstance(value, list)
        and len(value) == 2
        and all(isinstance(item, (int, float)) and math.isfinite(item) for item in value)
        and bbox[0] <= value[0] <= bbox[2]
        and bbox[1] <= value[1] <= bbox[3]
    )


def _normalize_category_identifier(value: Any, *, field_name: str) -> str:
    """Return a stable category identifier without touching SAM3 prompt text."""

    if not isinstance(value, str):
        raise ValueError(f"{field_name} must be a string")
    # VLMs occasionally express an otherwise valid multi-word category as
    # prose (for example, "dark vase"). Normalize only harmless spelling
    # variants; punctuation other than hyphens remains a schema error.
    category = re.sub(r"[\s-]+", "_", value.strip().lower())
    category = re.sub(r"_+", "_", category).strip("_")
    if not re.fullmatch(r"[a-z][a-z0-9_]{0,63}", category):
        raise ValueError(f"invalid {field_name}: {value!r}")
    return category


def validate_semantic_record(record: dict[str, Any], *, anchor_key: str = "a") -> None:
    required = {"c", "p", "n", "t", "b", anchor_key, "d"}
    if set(record) != required:
        raise ValueError(
            f"object keys must be exactly {sorted(required)}, got {sorted(record)}"
        )
    record["c"] = _normalize_category_identifier(record["c"], field_name="category")
    if str(record["t"]) not in ALLOWED_TYPES:
        raise ValueError(f"invalid object type: {record['t']!r}")
    for key in ("p", "n", "d"):
        if not str(record[key]).strip():
            raise ValueError(f"empty semantic field: {key}")
    if not valid_bbox(record["b"]) or not valid_anchor(record[anchor_key], record["b"]):
        raise ValueError("invalid bbox or anchor")


def expand_record(record: dict[str, Any], *, anchor_key: str = "a") -> dict[str, Any]:
    record = dict(record)
    if str(record.get("t")) not in ALLOWED_TYPES:
        record["t"] = "unknown"
    validate_semantic_record(record, anchor_key=anchor_key)
    return {
        "category": str(record["c"]).strip(),
        "sam3_prompt": semantic_sam3_prompt(record["c"], record["p"]),
        "instance_name": str(record["n"]).strip(),
        "object_type": str(record["t"]).strip(),
        "bbox_2d_norm_xyxy": [round(float(value), 3) for value in record["b"]],
        "anchor_2d_norm_xy": [
            round(float(value), 3) for value in record[anchor_key]
        ],
        "description": str(record["d"]).strip(),
        "fallback_prompts": [],
    }


def _validate_compact_category(value: Any) -> str:
    return _normalize_category_identifier(value, field_name="compact category")


def _validate_compact_prompt(value: Any) -> str:
    if not isinstance(value, str):
        raise ValueError("compact SAM3 prompt must be a string")
    prompt = " ".join(value.split())
    if not prompt or len(prompt) > 80:
        raise ValueError("compact SAM3 prompt must contain 1-80 characters")
    return prompt


_SAM3_LOCATION_TOKENS = {
    "above",
    "across",
    "against",
    "at",
    "behind",
    "beneath",
    "beside",
    "between",
    "center",
    "central",
    "corner",
    "far",
    "front",
    "inside",
    "left",
    "middle",
    "near",
    "on",
    "outside",
    "rear",
    "right",
    "under",
}


def semantic_sam3_prompt(category: Any, prompt: Any) -> str:
    """Return an instance-level SAM3 prompt with required assembly semantics.

    The prompting contract asks models for object-only text, but retaining an
    occasional positional qualifier is safer than silently replacing a
    specific instance or group with its broad category. SAM3 assignment still
    uses the corresponding local bounding box to distinguish repeated items.
    """

    normalized_category = _validate_compact_category(category)
    normalized_prompt = _validate_compact_prompt(prompt)
    if normalized_category != "potted_plant":
        return normalized_prompt
    prompt_tokens = set(re.findall(r"[a-z0-9]+", normalized_prompt.lower()))
    if prompt_tokens & {"pot", "planter", "container"}:
        return normalized_prompt
    return f"{normalized_prompt} with pot"


def _bbox_center(bbox: list[float]) -> list[float]:
    return [
        round((float(bbox[0]) + float(bbox[2])) / 2.0, 6),
        round((float(bbox[1]) + float(bbox[3])) / 2.0, 6),
    ]


def _minimal_semantic_record(
    *,
    category: Any,
    prompt: Any,
    bbox: Any,
    anchor: Any | None = None,
    object_type: Any = "unknown",
) -> dict[str, Any]:
    normalized_category = _validate_compact_category(category)
    normalized_prompt = semantic_sam3_prompt(normalized_category, prompt)
    normalized_object_type = str(object_type).strip()
    if normalized_object_type not in ALLOWED_TYPES:
        raise ValueError(f"invalid compact object type: {object_type!r}")
    if not valid_bbox(bbox):
        raise ValueError("invalid compact bbox")
    normalized_bbox = [round(float(value), 6) for value in bbox]
    normalized_anchor = _bbox_center(normalized_bbox) if anchor is None else [
        round(float(value), 6) for value in anchor
    ]
    if not valid_anchor(normalized_anchor, normalized_bbox):
        raise ValueError("invalid locally derived compact anchor")
    center_x, center_y = normalized_anchor
    row = min(10, max(1, int(center_y * 10) + 1))
    column = min(10, max(1, int(center_x * 10) + 1))
    return {
        "category": normalized_category,
        "sam3_prompt": normalized_prompt,
        "instance_name": (
            f"{normalized_category.replace('_', ' ')} r{row}c{column}"
        ),
        "object_type": normalized_object_type,
        "bbox_2d_norm_xyxy": normalized_bbox,
        "anchor_2d_norm_xy": normalized_anchor,
        "description": normalized_prompt,
        "fallback_prompts": [],
    }


def _semantic_record_copy(record: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "category",
        "sam3_prompt",
        "instance_name",
        "object_type",
        "bbox_2d_norm_xyxy",
        "anchor_2d_norm_xy",
        "description",
        "fallback_prompts",
    )
    missing = [key for key in keys if key not in record]
    if missing:
        raise ValueError(f"Pass-A object lacks semantic fields: {missing}")
    return {key: copy.deepcopy(record[key]) for key in keys}


def _local_candidate_geometry(candidate: dict[str, Any]) -> dict[str, Any]:
    candidate_id = str(candidate.get("candidate_id") or "")
    mask_path = Path(str(candidate.get("mask_path") or "")).expanduser().resolve()
    if not mask_path.is_file():
        raise ValueError(f"compact LC mask is missing: {candidate_id}")

    from PIL import Image

    with Image.open(mask_path) as source:
        mask = source.convert("L")
        width, height = mask.size
        if width <= 0 or height <= 0:
            raise ValueError(f"compact LC mask has invalid shape: {candidate_id}")
        absolute_bbox = mask.getbbox()
        if absolute_bbox is None:
            raise ValueError(f"compact LC mask is empty: {candidate_id}")
        x1, y1, x2, y2 = absolute_bbox
        pixels = mask.load()
        center_x = (x1 + x2 - 1) / 2.0
        center_y = (y1 + y2 - 1) / 2.0
        best_pixel = None
        best_distance = None
        area_pixels = 0
        for y in range(y1, y2):
            for x in range(x1, x2):
                if pixels[x, y] <= 0:
                    continue
                area_pixels += 1
                distance = (x - center_x) ** 2 + (y - center_y) ** 2
                if best_distance is None or distance < best_distance:
                    best_distance = distance
                    best_pixel = (x, y)
        if best_pixel is None or area_pixels <= 0:
            raise ValueError(f"compact LC mask has no foreground: {candidate_id}")

    bbox = [
        round(x1 / width, 6),
        round(y1 / height, 6),
        round(x2 / width, 6),
        round(y2 / height, 6),
    ]
    anchor = [
        round((best_pixel[0] + 0.5) / width, 6),
        round((best_pixel[1] + 0.5) / height, 6),
    ]
    if not valid_bbox(bbox) or not valid_anchor(anchor, bbox):
        raise ValueError(f"compact LC mask geometry is invalid: {candidate_id}")
    reported_bbox = candidate.get("bbox_norm_xyxy")
    if not valid_bbox(reported_bbox):
        raise ValueError(f"compact LC reported bbox is invalid: {candidate_id}")
    maximum_bbox_error = max(
        abs(float(left) - float(right))
        for left, right in zip(bbox, reported_bbox)
    )
    if maximum_bbox_error > 0.01:
        raise ValueError(
            f"compact LC mask and registry bbox disagree: {candidate_id}"
        )
    return {
        "bbox": bbox,
        "anchor": anchor,
        "area_pixels": area_pixels,
        "mask_path": str(mask_path),
        "mask_sha256": sha256_file(mask_path),
        "mask_size": [width, height],
        "maximum_registry_bbox_error": round(maximum_bbox_error, 6),
    }



def _validate_assigned_evidence_decisions(
    raw_decisions: Any,
    *,
    assigned_evidence_metadata: list[dict[str, Any]],
    additions: list[dict[str, Any]],
    updates: list[dict[str, Any]],
    removals: list[str],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Require one grounded semantic verdict for every supplied SAM3 mask tile."""

    expected: dict[str, dict[str, Any]] = {}
    for raw in assigned_evidence_metadata:
        item = dict(raw)
        evidence_id = str(item.get("evidence_id") or "")
        owner_id = str(item.get("source_object_id") or "")
        if not re.fullmatch(r"E_\d{3}", evidence_id):
            raise ValueError(f"invalid assigned evidence ID: {evidence_id!r}")
        if not re.fullmatch(r"A_\d{3}", owner_id):
            raise ValueError(f"invalid assigned evidence owner: {owner_id!r}")
        if evidence_id in expected:
            raise ValueError(f"duplicate assigned evidence ID: {evidence_id}")
        bbox = item.get("bbox_norm_xyxy")
        if not valid_bbox(bbox):
            raise ValueError(f"invalid assigned evidence bbox: {evidence_id}")
        expected[evidence_id] = item
    continuous = [f"E_{index:03d}" for index in range(1, len(expected) + 1)]
    if list(expected) != continuous:
        raise ValueError("assigned evidence IDs must be continuous and ordered")
    if not isinstance(raw_decisions, list):
        raise ValueError("assigned evidence decisions must be a list")

    update_ids = {str(item.get("pass_a_id") or "") for item in updates}
    removal_ids = {str(value) for value in removals}
    addition_boxes = [
        (f"B_{index:03d}", item.get("bbox_2d_norm_xyxy"))
        for index, item in enumerate(additions, start=1)
        if valid_bbox(item.get("bbox_2d_norm_xyxy"))
    ]
    decisions: list[dict[str, Any]] = []
    seen: set[str] = set()
    unresolved: list[str] = []
    normalized_action_conflicts: list[dict[str, Any]] = []
    normalized_unmaterialized_repair_assertions: list[dict[str, Any]] = []
    for raw in raw_decisions:
        if not isinstance(raw, dict) or set(raw) != {"i", "d", "r"}:
            raise ValueError("assigned evidence decision must contain i/d/r")
        evidence_id = str(raw["i"])
        decision = str(raw["d"])
        reason = str(raw["r"]).strip()
        if evidence_id not in expected:
            raise ValueError(f"unknown assigned evidence ID: {evidence_id}")
        if evidence_id in seen:
            raise ValueError(f"duplicate assigned evidence decision: {evidence_id}")
        if decision not in ASSIGNED_EVIDENCE_DECISIONS:
            raise ValueError(f"invalid assigned evidence decision: {decision}")
        if not reason:
            raise ValueError("assigned evidence reason must be non-empty")
        seen.add(evidence_id)
        item = expected[evidence_id]
        owner_id = str(item["source_object_id"])
        has_update = owner_id in update_ids
        has_removal = owner_id in removal_ids
        if decision == "pass" and (has_update or has_removal):
            # The explicit action is the materializable instruction.  Preserve it
            # deterministically instead of discarding an otherwise valid Pass-B
            # response merely because its explanatory E verdict says ``pass``.
            # Removal takes precedence if an invalid response somehow contains both.
            normalized_decision = (
                "remove_existing" if has_removal else "update_existing"
            )
            normalized_action_conflicts.append(
                {
                    "evidence_id": evidence_id,
                    "source_object_id": owner_id,
                    "reported_decision": "pass",
                    "effective_decision": normalized_decision,
                    "action": "removal" if has_removal else "update",
                    "reported_reason": reason,
                    "rule": "materialized_owner_action_overrides_conflicting_pass",
                }
            )
            decision = normalized_decision
            reason = (
                "Pass B supplied a materialized owner repair action; normalized "
                f"conflicting pass verdict. Original reason: {reason}"
            )
        if decision == "pass" and str(item.get("status") or "") == "failed":
            normalized_unmaterialized_repair_assertions.append(
                {
                    "evidence_id": evidence_id,
                    "source_object_id": owner_id,
                    "reported_decision": "pass",
                    "effective_decision": "uncertain",
                    "reported_reason": reason,
                    "rule": "failed_sam3_evidence_cannot_be_accepted_without_repair",
                }
            )
            decision = "uncertain"
            reason = (
                "SAM3 supplied no mask for this claim; a pass verdict cannot "
                f"accept it. Original reason: {reason}"
            )
        if decision == "update_existing" and not has_update:
            # Preserve the source object and mark just this tile uncertain when
            # Pass B notices a defect but omits the canonical repair patch.
            normalized_unmaterialized_repair_assertions.append(
                {
                    "evidence_id": evidence_id,
                    "source_object_id": owner_id,
                    "reported_decision": "update_existing",
                    "effective_decision": "uncertain",
                    "reported_reason": reason,
                    "rule": "unmaterialized_evidence_update_becomes_uncertain",
                }
            )
            decision = "uncertain"
            reason = (
                "Pass B requested an update without a materialized canonical "
                f"patch. Original reason: {reason}"
            )
        if decision == "remove_existing" and not has_removal:
            normalized_unmaterialized_repair_assertions.append(
                {
                    "evidence_id": evidence_id,
                    "source_object_id": owner_id,
                    "reported_decision": "remove_existing",
                    "effective_decision": "uncertain",
                    "reported_reason": reason,
                    "rule": "unmaterialized_evidence_removal_becomes_uncertain",
                }
            )
            decision = "uncertain"
            reason = (
                "Pass B requested removal without a materialized canonical "
                f"action. Original reason: {reason}"
            )
        supporting_addition_ids: list[str] = []
        if decision == "split_instances":
            owner_bbox = item["bbox_norm_xyxy"]
            supporting_addition_ids = [
                addition_id
                for addition_id, box in addition_boxes
                if max(owner_bbox[0], box[0]) < min(owner_bbox[2], box[2])
                and max(owner_bbox[1], box[1]) < min(owner_bbox[3], box[3])
            ]
            if not has_update or not supporting_addition_ids:
                raise ValueError(
                    f"evidence {evidence_id} split requires owner update and "
                    "an overlapping addition"
                )
        if decision == "uncertain":
            unresolved.append(evidence_id)
        decisions.append(
            {
                "evidence_id": evidence_id,
                "source_object_id": owner_id,
                "decision": decision,
                "reason": reason,
                "supporting_addition_ids": supporting_addition_ids,
            }
        )
    missing = sorted(set(expected) - seen)
    if missing:
        raise ValueError(f"missing assigned evidence decisions: {missing}")
    audit = {
        "schema_version": "scenemosaic_assigned_sam3_semantic_gate",
        "expected_count": len(expected),
        "decision_count": len(decisions),
        "complete_partition": len(decisions) == len(expected),
        "unresolved_evidence_ids": unresolved,
        "normalized_action_conflicts": normalized_action_conflicts,
        "normalized_unmaterialized_repair_assertions": (
            normalized_unmaterialized_repair_assertions
        ),
        "status": "blocked" if unresolved else "pass",
    }
    return decisions, audit


def parse_compact_pass_b(
    text: str,
    *,
    pass_a_objects: list[dict[str, Any]],
    local_candidate_metadata: list[dict[str, Any]] | None = None,
    assigned_evidence_metadata: list[dict[str, Any]] | None = None,
    **parse_context: Any,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Expand compact Pass-B into the unchanged canonical action schema."""

    repaired_text, input_repair_count = _repair_compact_bare_bbox_after_prompt(text)
    payload = parse_vlm_response_text(repaired_text, **parse_context)
    expected_keys = {"v", "a", "l", "u", "x", "r"}
    if assigned_evidence_metadata is not None:
        expected_keys.add("e")
    if not isinstance(payload, dict) or set(payload) != expected_keys:
        raise ValueError(
            f"compact Pass B must contain exactly {sorted(expected_keys)}"
        )
    if payload["v"] != 2:
        raise ValueError("compact Pass B protocol version must be 2")

    ordinary_additions = payload["a"]
    local_accepts = payload["l"]
    updates = payload["u"]
    removals = payload["x"]
    rejections = payload["r"]
    if not isinstance(ordinary_additions, list):
        raise ValueError("compact Pass B additions must be a list")
    if not isinstance(local_accepts, list):
        raise ValueError("compact Pass B LC accepts must be a list")
    if len(ordinary_additions) + len(local_accepts) > 36:
        raise ValueError("compact Pass B addition budget exceeded")
    if not isinstance(updates, list) or len(updates) > 8:
        raise ValueError("invalid compact Pass B updates")
    if not isinstance(removals, list) or len(removals) > 8:
        raise ValueError("invalid compact Pass B removals")
    if not isinstance(rejections, list):
        raise ValueError("compact Pass B LC rejections must be a list")

    pass_a_by_id: dict[str, dict[str, Any]] = {}
    for raw_object in pass_a_objects:
        object_id = str(raw_object.get("object_id") or "")
        if not re.fullmatch(r"A_\d{3}", object_id):
            raise ValueError(f"invalid Pass-A object ID: {object_id!r}")
        if object_id in pass_a_by_id:
            raise ValueError(f"duplicate Pass-A object ID: {object_id}")
        pass_a_by_id[object_id] = dict(raw_object)

    candidate_by_id: dict[str, dict[str, Any]] = {}
    for raw_candidate in local_candidate_metadata or []:
        candidate = dict(raw_candidate)
        candidate_id = str(candidate.get("candidate_id") or "")
        if not re.fullmatch(r"LC_\d{3}", candidate_id):
            raise ValueError(f"invalid compact local candidate ID: {candidate_id!r}")
        if candidate_id in candidate_by_id:
            raise ValueError(f"duplicate compact local candidate ID: {candidate_id}")
        _validate_compact_prompt(candidate.get("prompt"))
        if not valid_bbox(candidate.get("bbox_norm_xyxy")):
            raise ValueError(f"invalid compact local candidate bbox: {candidate_id}")
        candidate_by_id[candidate_id] = candidate

    ignored_unavailable_local_candidate_protocol = None
    if not candidate_by_id and (local_accepts or rejections):
        # Without an LC panel, these fields cannot refer to any real mask or
        # object. They are occasionally copied from the response template, so
        # record and discard them rather than turning template text into a
        # pipeline failure.
        ignored_unavailable_local_candidate_protocol = {
            "accepted_entry_count": len(local_accepts),
            "rejection_ids": [str(value) for value in rejections],
        }
        local_accepts = []
        rejections = []

    additions = []
    for raw in ordinary_additions:
        if not isinstance(raw, dict) or set(raw) != {"c", "p", "b"}:
            raise ValueError("compact ordinary addition must contain c/p/b")
        additions.append(
            _minimal_semantic_record(
                category=raw["c"], prompt=raw["p"], bbox=raw["b"]
            )
        )

    accepted_ids = []
    accepted_categories: dict[str, str] = {}
    for raw in local_accepts:
        if not isinstance(raw, dict) or set(raw) != {"i", "c"}:
            raise ValueError("compact LC accept must contain i/c")
        candidate_id = str(raw["i"])
        if candidate_id not in candidate_by_id:
            raise ValueError(f"unknown compact LC accept ID: {candidate_id}")
        if candidate_id in accepted_categories:
            raise ValueError(f"duplicate compact LC accept ID: {candidate_id}")
        accepted_ids.append(candidate_id)
        accepted_categories[candidate_id] = _validate_compact_category(raw["c"])

    normalized_rejections = [str(value) for value in rejections]
    if len(normalized_rejections) != len(set(normalized_rejections)):
        raise ValueError("duplicate compact LC rejection ID")
    unknown_rejections = set(normalized_rejections) - set(candidate_by_id)
    if unknown_rejections:
        raise ValueError(
            f"unknown compact LC rejection IDs: {sorted(unknown_rejections)}"
        )
    overlap = set(accepted_ids) & set(normalized_rejections)
    if overlap:
        raise ValueError(f"compact LC accepted and rejected: {sorted(overlap)}")
    decided = set(accepted_ids) | set(normalized_rejections)
    expected_candidates = set(candidate_by_id)
    if decided != expected_candidates:
        missing = sorted(expected_candidates - decided)
        extra = sorted(decided - expected_candidates)
        raise ValueError(
            f"compact LC partition is incomplete: missing={missing}, extra={extra}"
        )

    candidate_evidence = []
    local_decisions = []
    locally_suppressed_duplicate_ids = []
    for candidate_id in accepted_ids:
        candidate = candidate_by_id[candidate_id]
        geometry = _local_candidate_geometry(candidate)
        accepted_category = accepted_categories[candidate_id]
        semantic_prompt = str(candidate["prompt"])
        addition = _minimal_semantic_record(
            category=accepted_category,
            prompt=semantic_prompt,
            bbox=geometry["bbox"],
            anchor=geometry["anchor"],
        )
        duplicate_pass_a_id = None
        for existing in pass_a_objects:
            duplicate, _ = likely_same_object(dict(existing), addition)
            if duplicate:
                duplicate_pass_a_id = existing.get("object_id")
                break
        if duplicate_pass_a_id is not None:
            locally_suppressed_duplicate_ids.append(candidate_id)
            local_decisions.append(
                {
                    "candidate_id": candidate_id,
                    "decision": "reject",
                    "target_object_id": None,
                    "reason": (
                        "VLM-accepted compact candidate suppressed by local "
                        "semantic-and-geometry duplicate check"
                    ),
                }
            )
            continue

        duplicate_addition_index = None
        for existing_index, existing in enumerate(additions, start=1):
            duplicate, _ = likely_same_object(dict(existing), addition)
            if duplicate:
                duplicate_addition_index = existing_index
                break
        if duplicate_addition_index is not None:
            addition_index = duplicate_addition_index
            local_decisions.append(
                {
                    "candidate_id": candidate_id,
                    "decision": "bind_to_addition",
                    "target_object_id": f"B_{addition_index:03d}",
                    "reason": (
                        "accepted compact candidate bound to matching ordinary "
                        "addition"
                    ),
                }
            )
            candidate_evidence.append(
                {
                    "candidate_id": candidate_id,
                    "category": accepted_category,
                    **geometry,
                }
            )
        else:
            addition_index = len(additions) + 1
            additions.append(addition)
            local_decisions.append(
                {
                    "candidate_id": candidate_id,
                    "decision": "bind_to_addition",
                    "target_object_id": f"B_{addition_index:03d}",
                    "reason": "accepted compact candidate",
                }
            )
            candidate_evidence.append(
                {
                    "candidate_id": candidate_id,
                    "category": accepted_category,
                    **geometry,
                }
            )

    for candidate_id in normalized_rejections:
        local_decisions.append(
            {
                "candidate_id": candidate_id,
                "decision": "reject",
                "target_object_id": None,
                "reason": "rejected by compact Pass B",
            }
        )

    expanded_updates = []
    locally_suppressed_noop_update_ids = []
    seen_update_ids = set()
    for raw in updates:
        if not isinstance(raw, dict) or set(raw) != {"i", "v"}:
            raise ValueError("compact update must contain i/v")
        object_id = str(raw["i"])
        if object_id not in pass_a_by_id:
            raise ValueError(f"unknown compact update ID: {object_id}")
        if object_id in seen_update_ids:
            raise ValueError(f"duplicate compact update ID: {object_id}")
        seen_update_ids.add(object_id)
        patch = raw["v"]
        if not isinstance(patch, dict) or set(patch) not in (
            {"b"},
            {"c", "p"},
            {"c", "p", "b"},
        ):
            raise ValueError(
                "compact update patch must be b, c+p, or c+p+b"
            )
        replacement = _semantic_record_copy(pass_a_by_id[object_id])
        if "c" in patch:
            category = _validate_compact_category(patch["c"])
            prompt = semantic_sam3_prompt(category, patch["p"])
            replacement["category"] = category
            replacement["sam3_prompt"] = prompt
            replacement["instance_name"] = category.replace("_", " ")
            replacement["object_type"] = "unknown"
            replacement["description"] = prompt
            replacement["fallback_prompts"] = []
        if "b" in patch:
            if not valid_bbox(patch["b"]):
                raise ValueError(f"invalid compact update bbox: {object_id}")
            bbox = [round(float(value), 6) for value in patch["b"]]
            replacement["bbox_2d_norm_xyxy"] = bbox
            replacement["anchor_2d_norm_xy"] = _bbox_center(bbox)
        original = pass_a_by_id[object_id]
        semantic_changed = "c" in patch and (
            replacement.get("category") != original.get("category")
            or replacement.get("sam3_prompt") != original.get("sam3_prompt")
        )
        bbox_changed = "b" in patch and any(
            abs(float(left) - float(right)) > 0.000500001
            for left, right in zip(
                replacement["bbox_2d_norm_xyxy"],
                original["bbox_2d_norm_xyxy"],
            )
        )
        if not semantic_changed and not bbox_changed:
            # Compact context rounds coordinates to three decimals. Treat a
            # model echo of that rounded record as no action, so an evidence
            # verdict of ``pass`` remains consistent with the action graph.
            locally_suppressed_noop_update_ids.append(object_id)
            continue
        expanded_updates.append(
            {"pass_a_id": object_id, "replacement": replacement}
        )

    normalized_removals = [str(value) for value in removals]
    locally_materialized_evidence_removal_ids: list[str] = []
    locally_materialized_evidence_removals: list[dict[str, str]] = []
    if assigned_evidence_metadata is not None and isinstance(payload.get("e"), list):
        evidence_owner_by_id = {
            str(item.get("evidence_id") or ""): str(
                item.get("source_object_id") or ""
            )
            for item in assigned_evidence_metadata
            if isinstance(item, dict)
        }
        for raw_decision in payload["e"]:
            if (
                not isinstance(raw_decision, dict)
                or set(raw_decision) != {"i", "d", "r"}
                or raw_decision.get("d") != "remove_existing"
            ):
                continue
            evidence_id = str(raw_decision.get("i") or "")
            owner_id = evidence_owner_by_id.get(evidence_id, "")
            if (
                re.fullmatch(r"A_\d{3}", owner_id)
                and owner_id in pass_a_by_id
                and owner_id not in normalized_removals
            ):
                normalized_removals.append(owner_id)
                locally_materialized_evidence_removal_ids.append(owner_id)
                locally_materialized_evidence_removals.append(
                    {"evidence_id": evidence_id, "source_object_id": owner_id}
                )
    if len(normalized_removals) != len(set(normalized_removals)):
        raise ValueError("duplicate compact removal ID")
    unknown_removals = set(normalized_removals) - set(pass_a_by_id)
    if unknown_removals:
        raise ValueError(
            f"unknown compact removal IDs: {sorted(unknown_removals)}"
        )
    update_remove_conflicts = seen_update_ids & set(normalized_removals)
    locally_resolved_update_removal_conflicts: list[dict[str, Any]] = []
    if update_remove_conflicts:
        # Retaining an existing object (and any materialized repair) is safer
        # than silently deleting it when one compact response contradicts
        # itself. The conflict remains auditable and matching evidence removals
        # are rewritten below.
        materialized_update_ids = {
            str(item.get("pass_a_id") or "") for item in expanded_updates
        }
        normalized_removals = [
            object_id
            for object_id in normalized_removals
            if object_id not in update_remove_conflicts
        ]
        locally_materialized_evidence_removal_ids = [
            object_id
            for object_id in locally_materialized_evidence_removal_ids
            if object_id not in update_remove_conflicts
        ]
        locally_materialized_evidence_removals = [
            item
            for item in locally_materialized_evidence_removals
            if str(item.get("source_object_id") or "")
            not in update_remove_conflicts
        ]
        for object_id in sorted(update_remove_conflicts):
            locally_resolved_update_removal_conflicts.append(
                {
                    "source_object_id": object_id,
                    "effective_action": (
                        "retain_materialized_update"
                        if object_id in materialized_update_ids
                        else "retain_existing_noop_update"
                    ),
                    "rule": "compact_update_over_conflicting_remove",
                }
            )

    actions = {
        "additions": additions,
        "updates": expanded_updates,
        "removals": normalized_removals,
    }
    evidence_audit = None
    effective_evidence_payload = payload.get("e")
    locally_suppressed_unmaterialized_boundary_updates = []
    locally_rewritten_conflicting_evidence_removals: list[dict[str, Any]] = []
    if assigned_evidence_metadata is not None:
        effective_evidence_payload = copy.deepcopy(payload["e"])
        metadata_by_evidence_id = {
            str(item.get("evidence_id") or ""): dict(item)
            for item in assigned_evidence_metadata
            if isinstance(item, dict)
        }
        materialized_update_ids = {
            str(item.get("pass_a_id") or "") for item in expanded_updates
        }
        materialized_removal_ids = set(normalized_removals)
        for raw_decision in effective_evidence_payload:
            if not isinstance(raw_decision, dict):
                continue
            evidence_id = str(raw_decision.get("i") or "")
            owner_id = str(
                metadata_by_evidence_id.get(evidence_id, {}).get(
                    "source_object_id"
                )
                or ""
            )
            if (
                owner_id in update_remove_conflicts
                and raw_decision.get("d") == "remove_existing"
            ):
                effective_decision = (
                    "update_existing"
                    if owner_id in materialized_update_ids
                    else "pass"
                )
                locally_rewritten_conflicting_evidence_removals.append(
                    {
                        "evidence_id": evidence_id,
                        "source_object_id": owner_id,
                        "reported_decision": "remove_existing",
                        "effective_decision": effective_decision,
                        "rule": "compact_update_over_conflicting_remove",
                    }
                )
                raw_decision["d"] = effective_decision
                raw_decision["r"] = (
                    "Conflicting compact removal was suppressed to preserve "
                    "the existing verified object."
                )
        for raw_decision in effective_evidence_payload:
            if (
                not isinstance(raw_decision, dict)
                or raw_decision.get("d") != "update_existing"
            ):
                continue
            evidence_id = str(raw_decision.get("i") or "")
            metadata = metadata_by_evidence_id.get(evidence_id, {})
            owner_id = str(metadata.get("source_object_id") or "")
            if (
                owner_id in materialized_update_ids
                or owner_id in materialized_removal_ids
            ):
                continue
            reason = str(raw_decision.get("r") or "").strip()
            try:
                coverage = float(
                    metadata.get("claimed_bbox_coverage_by_mask_bbox")
                )
                component_count = int(metadata.get("component_count"))
            except (TypeError, ValueError):
                continue
            is_minor_boundary_tightening = bool(
                re.search(
                    r"(?:thin\s+(?:(?:left|right|top|bottom|side)\s+){0,2}edge|"
                    r"boundary\s+slightly|slightly\s+loose|tighten.{0,24}rim|"
                    r"minor\s+boundary)",
                    reason.lower(),
                )
            )
            if not (
                metadata.get("status") == "success"
                and not bool(metadata.get("merge_review_required"))
                and component_count <= 2
                and coverage >= 0.93
                and is_minor_boundary_tightening
            ):
                continue
            raw_decision["d"] = "pass"
            raw_decision["r"] = (
                "local high-coverage boundary policy retained the verified mask"
            )
            locally_suppressed_unmaterialized_boundary_updates.append(
                {
                    "evidence_id": evidence_id,
                    "source_object_id": owner_id,
                    "previous_decision": "update_existing",
                    "effective_decision": "pass",
                    "previous_reason": reason,
                    "claimed_bbox_coverage_by_mask_bbox": round(coverage, 6),
                    "component_count": component_count,
                    "rule": (
                        "unmaterialized_minor_boundary_update_requires_no_action_"
                        "when_verified_coverage_ge_0_93"
                    ),
                }
            )
        evidence_decisions, evidence_audit = (
            _validate_assigned_evidence_decisions(
                effective_evidence_payload,
                assigned_evidence_metadata=assigned_evidence_metadata,
                additions=additions,
                updates=expanded_updates,
                removals=normalized_removals,
            )
        )
        actions["assigned_evidence_decisions"] = evidence_decisions
        actions["assigned_evidence_semantic_gate"] = evidence_audit
    if candidate_by_id:
        actions["local_candidate_decisions"] = local_decisions
    audit = {
        "schema_version": "scenemosaic_compact_pass_b_expansion",
        "protocol": COMPACT_PASS_B_PROTOCOL,
        "input_repair_count": input_repair_count,
        "ignored_unavailable_local_candidate_protocol": (
            ignored_unavailable_local_candidate_protocol
        ),
        "pass_a_object_count": len(pass_a_by_id),
        "ordinary_addition_count": len(ordinary_additions),
        "accepted_local_candidate_count": len(accepted_ids),
        "rejected_local_candidate_count": len(normalized_rejections),
        "locally_suppressed_duplicate_count": len(
            locally_suppressed_duplicate_ids
        ),
        "locally_suppressed_duplicate_ids": locally_suppressed_duplicate_ids,
        "update_count": len(expanded_updates),
        "locally_suppressed_noop_update_count": len(
            locally_suppressed_noop_update_ids
        ),
        "locally_suppressed_noop_update_ids": (
            locally_suppressed_noop_update_ids
        ),
        "removal_count": len(normalized_removals),
        "locally_resolved_update_removal_conflict_count": len(
            locally_resolved_update_removal_conflicts
        ),
        "locally_resolved_update_removal_conflicts": (
            locally_resolved_update_removal_conflicts
        ),
        "locally_rewritten_conflicting_evidence_removal_count": len(
            locally_rewritten_conflicting_evidence_removals
        ),
        "locally_rewritten_conflicting_evidence_removals": (
            locally_rewritten_conflicting_evidence_removals
        ),
        "locally_materialized_evidence_removal_count": len(
            locally_materialized_evidence_removal_ids
        ),
        "locally_materialized_evidence_removal_ids": (
            locally_materialized_evidence_removal_ids
        ),
        "locally_materialized_evidence_removals": locally_materialized_evidence_removals,
        "locally_suppressed_unmaterialized_boundary_update_count": len(
            locally_suppressed_unmaterialized_boundary_updates
        ),
        "locally_suppressed_unmaterialized_boundary_updates": (
            locally_suppressed_unmaterialized_boundary_updates
        ),
        "canonical_addition_count": len(additions),
        "candidate_partition_complete": decided == expected_candidates,
        "candidate_evidence": candidate_evidence,
        "assigned_evidence_semantic_gate": evidence_audit,
        "locally_derived_fields": [
            "anchor_2d_norm_xy",
            "instance_name",
            "object_type",
            "description",
            "fallback_prompts",
        ],
        "canonical_action_schema_unchanged": True,
    }
    return actions, audit


def parse_pass_a(text: str, **parse_context: Any) -> list[dict[str, Any]]:
    payload = parse_vlm_response_text(text, **parse_context)
    if not isinstance(payload, dict) or set(payload) != {"o"}:
        raise ValueError("Pass A response must contain only key 'o'")
    records = payload["o"]
    if not isinstance(records, list) or not records:
        raise ValueError("Pass A must return a non-empty object list")
    if len(records) > 80:
        raise ValueError("Pass A object budget exceeded")
    expanded = []
    for record in records:
        if not isinstance(record, dict):
            raise ValueError("Pass A object record must be an object")
        compact_keys = set(record)
        if compact_keys == {"c", "p", "t", "b"}:
            expanded.append(
                _minimal_semantic_record(
                    category=record["c"],
                    prompt=record["p"],
                    object_type=record["t"],
                    bbox=record["b"],
                )
            )
        elif compact_keys == {"c", "p", "b"}:
            expanded.append(
                _minimal_semantic_record(
                    category=record["c"],
                    prompt=record["p"],
                    bbox=record["b"],
                )
            )
        else:
            expanded.append(expand_record(record))
    return expanded





def normalize_pass_a(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return normalize_objects(
        records,
        object_id_base=1,
        object_id_format="A_{index:03d}",
    )


def compact_pass_a_context(objects: list[dict[str, Any]]) -> str:
    """Encode only the Pass-A fields that Pass B must inspect."""

    records = []
    seen = set()
    for item in objects:
        object_id = str(item.get("object_id") or "")
        if not re.fullmatch(r"A_\d{3}", object_id):
            raise ValueError(f"invalid Pass-A ID for compact protocol: {object_id!r}")
        if object_id in seen:
            raise ValueError(f"duplicate Pass-A ID for compact protocol: {object_id}")
        seen.add(object_id)
        sam3_prompt = _validate_compact_prompt(item.get("sam3_prompt"))
        bbox = item.get("bbox_2d_norm_xyxy")
        if not valid_bbox(bbox):
            raise ValueError(f"invalid Pass-A bbox for compact protocol: {object_id}")
        records.append(
            [
                object_id,
                sam3_prompt,
                [round(float(value), 3) for value in bbox],
            ]
        )
    return json.dumps(records, ensure_ascii=False, separators=(",", ":"))


def compact_local_candidate_context(
    candidates: list[dict[str, Any]],
) -> tuple[str, list[str]]:
    """Drop scores and repeated semantics while retaining visual identity."""

    records = []
    seen = set()
    for candidate in candidates:
        candidate_id = str(candidate.get("candidate_id") or "")
        if not re.fullmatch(r"LC_\d{3}", candidate_id):
            raise ValueError(f"invalid compact protocol local candidate ID: {candidate_id!r}")
        if candidate_id in seen:
            raise ValueError(f"duplicate compact protocol local candidate ID: {candidate_id}")
        seen.add(candidate_id)
        prompt = _validate_compact_prompt(candidate.get("prompt"))
        namespace = str(candidate.get("source_namespace") or "other")
        if namespace not in {
            "top_zone",
            "sam3_a_unassigned",
            "other",
        }:
            raise ValueError(
                f"invalid compact protocol candidate namespace: {namespace!r}"
            )
        bbox = candidate.get("bbox_norm_xyxy")
        if not valid_bbox(bbox):
            raise ValueError(f"invalid compact protocol local candidate bbox: {candidate_id}")
        record = [
            candidate_id,
            namespace,
            prompt,
            [round(float(value), 3) for value in bbox],
        ]
        alternate_prompts = []
        for evidence in candidate.get("prompt_evidence") or []:
            alternate = _validate_compact_prompt(evidence.get("prompt"))
            if alternate != prompt and alternate not in alternate_prompts:
                alternate_prompts.append(alternate)
        if alternate_prompts:
            record.append(alternate_prompts[:3])
        records.append(record)
    return (
        json.dumps(records, ensure_ascii=False, separators=(",", ":")),
        [item[0] for item in records],
    )


def compact_content_host_context(hosts: list[dict[str, Any]]) -> str:
    """Encode H tiles without repeating their owner category."""

    records = []
    seen = set()
    for host in hosts:
        host_id = str(host.get("host_id") or "")
        source_id = str(host.get("source_object_id") or "")
        if not re.fullmatch(r"H_\d{3}", host_id):
            raise ValueError(f"invalid compact protocol content-host ID: {host_id!r}")
        if host_id in seen:
            raise ValueError(f"duplicate compact protocol content-host ID: {host_id}")
        if not re.fullmatch(r"A_\d{3}", source_id):
            raise ValueError(f"invalid compact protocol content-host owner: {source_id!r}")
        mode = str(host.get("host_mode") or "")
        if mode not in {"open_storage", "support_surface"}:
            raise ValueError(f"invalid compact protocol content-host mode: {host_id}")
        bbox = host.get("crop_norm_xyxy")
        if not valid_bbox(bbox):
            raise ValueError(f"invalid compact protocol content-host bbox: {host_id}")
        seen.add(host_id)
        records.append(
            [
                host_id,
                source_id,
                mode,
                [round(float(value), 3) for value in bbox],
            ]
        )
    expected = [f"H_{index:03d}" for index in range(1, len(records) + 1)]
    if [item[0] for item in records] != expected:
        raise ValueError("compact protocol content-host IDs must be continuous")
    return json.dumps(records, ensure_ascii=False, separators=(",", ":"))


def build_compact_pass_b_prompt(
    registry: str,
    local_candidate_registry: str | None = None,
    *,
    content_host_registry: str | None = None,
    assigned_evidence_registry: str | None = None,
    same_mask_conflict_registry: str | None = None,
    content_closure_only: bool = False,
) -> str:
    """Build the bounded, low-output Pass-B request."""

    if local_candidate_registry is None:
        local_response_shape = '"l":[],"r":[]'
        local_candidate_instruction = (
            "No LC candidate panel is supplied. Return l=[] and r=[] exactly; "
            "never invent an LC identifier. "
        )
    else:
        local_response_shape = (
            '"l":[{"i":"LC_001","c":"category"}],'
            '"r":["LC_002"]'
        )
        local_candidate_instruction = "l=accepted LC and r=rejected LC IDs. "
    response_shape = (
        '{"v":2,"a":[{"c":"category","p":"SAM3 prompt","b":[0,0,0,0]}],'
        + local_response_shape
        + ',"u":[{"i":"A_001","v":{"b":[0,0,0,0}}],'
        + '"x":[]}'
    )
    if assigned_evidence_registry is not None:
        response_shape = response_shape[:-1] + (
            ',"e":[{"i":"E_001","d":"pass|update_existing|split_instances|'
            'remove_existing|uncertain","r":"short reason"}]}'
        )

    audit_scope = (
        "Audit the whole image; do not repeat correct claims. Find visible "
        "missing physical units, especially small items on supports/shelves/"
        "cubbies/media furniture, loose pillows or cushions on sofas, chairs, "
        "or beds, edge or high fixtures, plants, lighting, "
        "repeated or symmetric locations. "
        if not content_closure_only
        else
        "Perform content closure only. The current inventory already includes "
        "the main Pass-B additions: preserve their selected object granularity. "
        "Inspect every supplied H open-storage window, each exposed level and "
        "both lateral edges only for a visually independent content group not "
        "already claimed there. Never subdivide, duplicate, or enumerate the "
        "members of a current group. Return one grouped unit when touching or "
        "tightly adjacent contents form one shelf-level unit; an empty gap is "
        "not itself an object. Do not correct, remove, split, rename, or resize "
        "any current owner. "
    )
    sections = [
        "You are Pass B, the coverage/correction audit for visible-object "
        "SAM3+SAM3D reconstruction. Image 1 is the whole scene with a cyan "
        "coordinate grid. Current inventory claims are [ID,sam3_prompt,bbox]:",
        registry,
        "Each registry label is an instance-level SAM3 prompt, not a broad "
        "category. Before emitting a, compare its p and bbox to every current "
        "claim: if it names the same visible physical unit, retain that claim "
        "or correct its A ID through u; do not add a second entry. Emit a only "
        "when pixels show a distinct physical unit.",
        audit_scope
        + (
            "Also correct materially wrong category/bbox and remove duplicate "
            "or nonphysical claims. "
            if not content_closure_only
            else
            "Use a only for those independent missing content groups and leave "
            "l, u, x, and r empty. Give a group an object-only p such as \'upright "
            "book group\' or \'stacked book group\', never an item-by-item list "
            "or a location phrase. "
        )
        + "Require visible pixels; symmetry is only a search cue. Exclude wall, floor, "
        "ceiling, windows, doors, curtains, background, glare, trim, and content "
        "depicted in art, mirrors, televisions or screens. Keep fitted assembly "
        "parts together: bed plus duvet; every visually independent loose pillow "
        "or cushion on a sofa, chair, or bed is a separate movable object. Sink plus "
        "faucet, dedicated support and fitted undercounter dishwasher. A visible "
        "range/stove splits counter sections; never keep one counter claim across "
        "both sides. Attached water bottle plus dispenser; "
        "a verified potted plant keeps its pot and foliage as one assembly, never "
        "replace it with only a vase or planter and never add its foliage as a "
        "second object. "
        "connected lamp heads/rods; dedicated coat/towel/drying/roll/pegboard "
        "racks plus attached contents. Mere "
        "contact does not merge independent objects. Field p is an "
        "object-only SAM3 semantic prompt: name the visible object itself, "
        "optionally with color/material/form; never use a location, support, "
        "direction, relation, grid cell or nearby object in p.",
    ]
    if (
        local_candidate_registry is not None
        or content_host_registry is not None
        or assigned_evidence_registry is not None
    ):
        sections.append(
            "One or more additional numbered review panels follow. All "
            "output bboxes remain normalized whole-image coordinates and use "
            "[left, top, right, bottom] (x1, y1, x2, y2), never "
            "[x, y, width, height]."
        )
    if local_candidate_registry is not None:
        sections.extend(
            [
                "LC claims [ID,source,prompt,bbox,optional aliases]:",
                local_candidate_registry,
                "Each LC mask is evidence; prompts are fallible hints. Relabel visually. Decide every LC "
                "exactly once: put a visible new reconstruction unit in l as "
                "{i,c}; put background, false, duplicate, part, or already "
                "registered candidates in r by ID. An accepted LC inherits its "
                "true mask and bbox; never duplicate it in a.",
            ]
        )
    if content_host_registry is not None:
        sections.extend(
            [
                "H tiles are [H_ID,owner_A_ID,open_storage|support_surface,"
                "crop_bbox]:",
                content_host_registry,
                "H tiles are high-resolution attention windows, not masks "
                "or objects. Scan every exposed level and both lateral "
                "edges. On fixed shelves/racks, report independent movable "
                "contents as separate units. On a mobile cart, group "
                "contents once per visible shelf level. Do not re-add the "
                "carrier. Report a missing physical unit from H normally "
                "in a. An H crop bbox is never an object bbox: do not copy "
                "it into u, and never update its owner because of H. Never "
                "emit an H ID.",
            ]
        )
    if assigned_evidence_registry is not None:
        sections.extend(
            [
                "E tiles are [E_ID,owner_A_ID,sam3_prompt,SAM3_status,mask_or_crop_bbox]:",
                assigned_evidence_registry,
                "Each E tile shows original crop left and the actual assigned "
                "SAM3 mask overlay right. The crop is the union of the claimed "
                "bbox and actual-mask bbox: any visible fixed assembly part "
                "inside the claim but outside the colored mask requires "
                "update_existing or uncertain, never pass. Decide every E "
                "exactly once in e: "
                "pass, update_existing, split_instances, remove_existing, or "
                "uncertain, with a short visible-pixel reason. If the original "
                "crop visibly contains no instance matching an E claim, choose "
                "remove_existing and put its owner in x; do not call visible "
                "absence uncertain. Use uncertain only when a likely visible "
                "instance remains but pixels cannot determine its correct mask "
                "or object unit. For an E with SAM3_status=failed, use "
                "update_existing only when the original claimed physical object "
                "is visibly present and the replacement preserves its object "
                "semantics. Never relabel a failed owner into a nearby different "
                "object; if its claimed object is absent or semantically mismatched, "
                "choose remove_existing. update_existing requires the owner A ID in u; "
                "remove_existing requires it in x; split_instances requires the "
                "owner in u plus the other physical "
                "unit in a. Tighten the existing bbox and add the missing object "
                "for true two-object merges. Do not split a single occluded object "
                "merely because visible pixels are disconnected.",
            ]
        )
    if same_mask_conflict_registry is not None:
        sections.extend(
            [
                "SMC records are [SMC_ID,A_ID_1,A_ID_2,sam3_prompt_1,sam3_prompt_2,"
                "bbox_1,bbox_2,mask_iou]. Their separate Pass-A claims received "
                "nearly identical SAM3 masks. This is ambiguous: they may be one "
                "physical unit or two adjacent units merged by SAM3. Inspect Image 1 "
                "and decide conservatively. If exactly one physical instance is "
                "visible, put exactly one non-retained A ID in x. If two distinct "
                "adjacent physical instances are visible, keep both out of x; a "
                "strict local bbox split will then be attempted. If pixels do not "
                "decide, keep both out of x. Never remove both SMC owners and never "
                "add a duplicate replacement for an SMC owner.",
                same_mask_conflict_registry,
            ]
        )
    sections.extend(
        [
            "Return ONLY this minified JSON shape with exact keys:",
            response_shape,
            "a=ordinary missing objects. " + local_candidate_instruction,
            "u=material A updates; each v must be exactly {b}, {c,p}, or {c,p,b}. "
            "x=removed A IDs. e=complete assigned-mask decisions when E "
            "tiles are supplied. c is a generic machine category identifier: "
            "use lowercase snake_case only (for example vase, book_stack, "
            "ceiling_light), never spaces, hyphens, color, material, size, or "
            "location terms. Put all instance-level visual detail in a short "
            "object-only English SAM3 prompt p. Use a tight normalized bbox "
            "[left, top, right, bottom] "
            "with left < right and top < bottom, and [] when empty. Combined a+l <=36; "
            "u<=8; x<=8. Output no prose and no other fields.",
        ]
    )
    return "\n\n".join(sections)




def compact_same_mask_conflict_context(
    conflicts: list[dict[str, Any]],
) -> str:
    """Return compact, auditable context for ambiguous Pass-A mask pairs."""

    compact = []
    for conflict in conflicts:
        conflict_id = str(conflict.get("conflict_id") or "")
        owners = [str(value) for value in (conflict.get("object_ids") or [])]
        objects = list(conflict.get("objects") or [])
        if not re.fullmatch(r"SMC_\d{3}", conflict_id):
            raise ValueError(f"invalid same-mask conflict ID: {conflict_id!r}")
        if len(owners) != 2 or len(set(owners)) != 2:
            raise ValueError(f"same-mask conflict must have two distinct owners: {conflict_id}")
        by_id = {str(item.get("object_id") or ""): item for item in objects}
        if set(by_id) != set(owners):
            raise ValueError(f"same-mask conflict owner metadata mismatch: {conflict_id}")
        iou = float(conflict.get("mask_iou") or 0.0)
        if not math.isfinite(iou) or not 0.0 <= iou <= 1.0:
            raise ValueError(f"invalid same-mask conflict IoU: {conflict_id}")
        compact.append(
            [
                conflict_id,
                owners[0],
                owners[1],
                _validate_compact_prompt(by_id[owners[0]].get("sam3_prompt")),
                _validate_compact_prompt(by_id[owners[1]].get("sam3_prompt")),
                [round(float(value), 3) for value in by_id[owners[0]]["vlm_bbox_norm_xyxy"]],
                [round(float(value), 3) for value in by_id[owners[1]]["vlm_bbox_norm_xyxy"]],
                round(iou, 3),
            ]
        )
    expected = [f"SMC_{index:03d}" for index in range(1, len(compact) + 1)]
    if [item[0] for item in compact] != expected:
        raise ValueError("same-mask conflict IDs must be continuous")
    return json.dumps(compact, ensure_ascii=False, separators=(",", ":"))









def bbox_metrics(left: list[float], right: list[float]) -> dict[str, float]:
    x1, y1 = max(left[0], right[0]), max(left[1], right[1])
    x2, y2 = min(left[2], right[2]), min(left[3], right[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    left_area = (left[2] - left[0]) * (left[3] - left[1])
    right_area = (right[2] - right[0]) * (right[3] - right[1])
    union = left_area + right_area - intersection
    return {
        "iou": intersection / union if union else 0.0,
        "left_coverage": intersection / left_area if left_area else 0.0,
        "right_coverage": intersection / right_area if right_area else 0.0,
    }


def category_key(value: Any) -> str:
    return "_".join(
        str(value or "").lower().replace("-", " ").replace("_", " ").split()
    )


def _semantic_label_tokens(value: Any) -> frozenset[str]:
    return frozenset(re.findall(r"[a-z0-9]+", str(value or "").lower()))


def _compatible_object_semantics(
    left: dict[str, Any], right: dict[str, Any]
) -> bool:
    """Match only prompt-level object identities without a category ontology."""

    left_prompt = _semantic_label_tokens(left.get("sam3_prompt"))
    right_prompt = _semantic_label_tokens(right.get("sam3_prompt"))
    if not left_prompt or not right_prompt:
        return False
    if "unknown" in left_prompt or "unknown" in right_prompt:
        return False
    return left_prompt <= right_prompt or right_prompt <= left_prompt


def likely_same_object(left: dict[str, Any], right: dict[str, Any]) -> tuple[bool, dict[str, float]]:
    metrics = bbox_metrics(
        left["bbox_2d_norm_xyxy"],
        right["bbox_2d_norm_xyxy"],
    )
    left_anchor = left["anchor_2d_norm_xy"]
    right_anchor = right["anchor_2d_norm_xy"]
    anchor_distance = math.dist(left_anchor, right_anchor)
    left_box = left["bbox_2d_norm_xyxy"]
    right_box = right["bbox_2d_norm_xyxy"]
    scale = max(
        0.02,
        min(
            math.hypot(left_box[2] - left_box[0], left_box[3] - left_box[1]),
            math.hypot(right_box[2] - right_box[0], right_box[3] - right_box[1]),
        ),
    )
    metrics["anchor_distance"] = anchor_distance
    metrics["relative_anchor_distance"] = anchor_distance / scale
    left_prompt = category_key(left.get("sam3_prompt"))
    right_prompt = category_key(right.get("sam3_prompt"))
    same_sam3_prompt = bool(left_prompt and left_prompt == right_prompt)
    metrics["same_sam3_prompt"] = float(same_sam3_prompt)
    if same_sam3_prompt:
        duplicate = (
            metrics["iou"] >= 0.55
            or (
                min(metrics["left_coverage"], metrics["right_coverage"]) >= 0.75
                and metrics["relative_anchor_distance"] <= 0.45
            )
        )
    else:
        duplicate = _compatible_object_semantics(left, right) and (
            metrics["iou"] >= 0.85
            or (
                min(metrics["left_coverage"], metrics["right_coverage"]) >= 0.90
                and metrics["relative_anchor_distance"] <= 0.15
            )
        )
    return duplicate, metrics


def fuse_passes(
    pass_a_objects: list[dict[str, Any]],
    pass_b: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    by_id = {str(item["object_id"]): dict(item) for item in pass_a_objects}
    known_ids = set(by_id)
    local_decisions = list(pass_b.get("local_candidate_decisions") or [])
    assigned_evidence_decisions = list(
        pass_b.get("assigned_evidence_decisions") or []
    )
    for decision in local_decisions:
        target = decision.get("target_object_id")
        action = decision.get("decision")
        if action in LOCAL_CANDIDATE_EXISTING_TARGET_ACTIONS and target not in known_ids:
            raise ValueError(f"local candidate references unknown Pass-A ID: {target}")
        if action == "bind_to_addition":
            addition_index = int(str(target).split("_")[1])
            if not 1 <= addition_index <= len(pass_b["additions"]):
                raise ValueError(f"local candidate references unknown Pass-B addition: {target}")
    update_ids = [item["pass_a_id"] for item in pass_b["updates"]]
    if len(update_ids) != len(set(update_ids)):
        raise ValueError("Pass B updates one Pass-A ID more than once")
    unknown = (set(update_ids) | set(pass_b["removals"])) - known_ids
    if unknown:
        raise ValueError(f"Pass B references unknown Pass-A IDs: {sorted(unknown)}")
    conflict = set(update_ids) & set(pass_b["removals"])
    if conflict:
        raise ValueError(f"Pass B both updates and removes IDs: {sorted(conflict)}")

    provenance: dict[str, list[dict[str, Any]]] = {
        object_id: [{"pass": "A", "source_id": object_id, "action": "created"}]
        for object_id in by_id
    }
    for update in pass_b["updates"]:
        object_id = update["pass_a_id"]
        replacement = dict(update["replacement"])
        replacement["object_id"] = object_id
        replacement = normalize_objects(
            [replacement],
            object_id_base=int(object_id.split("_")[1]),
            object_id_format="A_{index:03d}",
        )[0]
        by_id[object_id] = replacement
        provenance[object_id].append(
            {"pass": "B", "source_id": object_id, "action": "updated"}
        )
    for object_id in pass_b["removals"]:
        by_id.pop(object_id)
        provenance.pop(object_id)

    accepted_additions: list[tuple[str, dict[str, Any]]] = []
    rejected_additions: list[dict[str, Any]] = []
    for index, addition in enumerate(pass_b["additions"], start=1):
        source_id = f"B_{index:03d}"
        conflicts = []
        for existing_id, existing in [
            *by_id.items(),
            *accepted_additions,
        ]:
            duplicate, metrics = likely_same_object(existing, addition)
            if duplicate:
                conflicts.append(
                    {
                        "existing_id": existing_id,
                        **{key: round(value, 4) for key, value in metrics.items()},
                    }
                )
        if conflicts:
            rejected_additions.append(
                {
                    "source_id": source_id,
                    "category": addition["category"],
                    "reason": "semantic_and_geometry_duplicate",
                    "conflicts": conflicts,
                }
            )
            continue
        accepted_additions.append((source_id, dict(addition)))

    candidates: list[tuple[list[dict[str, Any]], dict[str, Any]]] = []
    for object_id, item in by_id.items():
        candidates.append((provenance[object_id], item))
    for source_id, item in accepted_additions:
        candidates.append(
            ([{"pass": "B", "source_id": source_id, "action": "added"}], item)
        )
    candidates.sort(
        key=lambda pair: (
            pair[1]["anchor_2d_norm_xy"][1],
            pair[1]["anchor_2d_norm_xy"][0],
            category_key(pair[1]["category"]),
        )
    )
    normalized = normalize_objects(
        [item for _, item in candidates],
        object_id_base=1,
        object_id_format="proposal_{index:03d}",
    )
    identity_records = []
    for output, (sources, _) in zip(normalized, candidates):
        output["proposal_sources"] = sources
        identity_records.append(
            {
                "proposal_id": output["object_id"],
                "category": output["category"],
                "description": output["description"],
                "instance_name": output["instance_name"],
                "anchor_2d_norm_xy": output["anchor_2d_norm_xy"],
                "bbox_2d_norm_xyxy": output["bbox_2d_norm_xyxy"],
                "sources": sources,
                "binding_status": "provisional_bbox_anchor_only",
            }
        )
    audit = {
        "pass_a_input_count": len(pass_a_objects),
        "pass_b_addition_count": len(pass_b["additions"]),
        "pass_b_update_count": len(pass_b["updates"]),
        "pass_b_removal_count": len(pass_b["removals"]),
        "accepted_addition_count": len(accepted_additions),
        "ordinary_addition_duplicate_rejection_count": len(rejected_additions),
        "rejected_additions": rejected_additions,
        "local_candidate_decision_count": len(local_decisions),
        "local_candidate_decisions": local_decisions,
        "assigned_evidence_decision_count": len(
            assigned_evidence_decisions
        ),
        "assigned_evidence_decisions": assigned_evidence_decisions,
        "assigned_evidence_semantic_gate": pass_b.get(
            "assigned_evidence_semantic_gate"
        ),
        "merged_proposal_count": len(normalized),
        "identity_records": identity_records,
    }
    return normalized, audit


def temperature_request_kwargs(model: str) -> dict[str, int]:
    """Use provider defaults for the repository-wide GPT-5.5 route."""

    canonical = str(model or "").lower().rsplit("/", 1)[-1]
    if canonical != "gpt-5.5":
        raise ValueError("SceneMosaic perception requires model gpt-5.5")
    return {}


def run_call(
    *,
    client: Any,
    prompt: str,
    image_data_url: str,
    model: str,
    reasoning_effort: str,
    output_dir: Path,
    additional_image_data_urls: list[str] | None = None,
) -> tuple[str, dict[str, Any]]:
    content = [
        {"type": "text", "text": prompt},
        {
            "type": "image_url",
            "image_url": {"url": image_data_url, "detail": "original"},
        },
    ]
    for additional_url in additional_image_data_urls or []:
        content.append(
            {
                "type": "image_url",
                "image_url": {"url": additional_url, "detail": "original"},
            }
        )
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": content}],
        reasoning_effort=reasoning_effort,
        **temperature_request_kwargs(model),
    )
    text = str(response.choices[0].message.content or "").strip()
    (output_dir / "response_raw.txt").write_text(text + "\n", encoding="utf-8")
    return text, {
        "finish_reason": getattr(response.choices[0], "finish_reason", None),
    }


def require_clean_finish(call_info: dict[str, Any], stage: str) -> None:
    """Reject any non-stop model finish before parsing a freeze-eligible response."""

    finish_reason = call_info.get("finish_reason")
    if finish_reason != "stop":
        raise RuntimeError(
            f"{stage} response did not finish cleanly: "
            f"finish_reason={finish_reason!r}"
        )


def run_pass_a(
    *,
    client: Any,
    image_data_url: str,
    model: str,
    reasoning_effort: str,
    output_dir: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Call, parse and normalize Pass A without running Pass B or fusion."""

    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "request_prompt.txt").write_text(
        PASS_A_PROMPT + "\n",
        encoding="utf-8",
    )
    response_text, call_info = run_call(
        client=client,
        prompt=PASS_A_PROMPT,
        image_data_url=image_data_url,
        model=model,
        reasoning_effort=reasoning_effort,
        output_dir=output_dir,
    )
    require_clean_finish(call_info, "Pass A")
    records = parse_pass_a(
        response_text,
        **parse_context(call_info),
    )
    return normalize_pass_a(records), call_info


def compact_assigned_sam3_evidence_context(
    records: list[dict[str, Any]],
) -> str:
    compact = []
    seen = set()
    for record in records:
        evidence_id = str(record.get("evidence_id") or "")
        source_id = str(record.get("source_object_id") or "")
        if not re.fullmatch(r"E_\d{3}", evidence_id):
            raise ValueError(f"invalid assigned evidence ID: {evidence_id!r}")
        if evidence_id in seen:
            raise ValueError(f"duplicate assigned evidence ID: {evidence_id}")
        if not re.fullmatch(r"A_\d{3}", source_id):
            raise ValueError(f"invalid assigned evidence owner: {source_id!r}")
        bbox = record.get("bbox_norm_xyxy") or record.get("crop_bbox_norm_xyxy")
        if not valid_bbox(bbox):
            raise ValueError(f"invalid assigned evidence bbox: {evidence_id}")
        seen.add(evidence_id)
        compact.append([
            evidence_id,
            source_id,
            _validate_compact_prompt(record.get("sam3_prompt")),
            str(record.get("status") or "unknown"),
            [round(float(value), 3) for value in bbox],
        ])
    expected = [f"E_{index:03d}" for index in range(1, len(compact) + 1)]
    if [item[0] for item in compact] != expected:
        raise ValueError("assigned evidence IDs must be continuous")
    return json.dumps(compact, ensure_ascii=False, separators=(",", ":"))


def run_pass_b(
    *,
    client: Any,
    image_data_url: str,
    pass_a_objects: list[dict[str, Any]],
    model: str,
    reasoning_effort: str,
    output_dir: Path,
    local_candidate_image_data_url: str | None = None,
    local_candidate_metadata: list[dict[str, Any]] | None = None,
    content_host_metadata: list[dict[str, Any]] | None = None,
    assigned_evidence_image_data_url: str | None = None,
    assigned_evidence_metadata: list[dict[str, Any]] | None = None,
    same_mask_conflicts: list[dict[str, Any]] | None = None,
    content_closure_only: bool = False,
) -> tuple[dict[str, Any], dict[str, Any], str]:
    """Call Pass B with at most one bounded local review panel."""

    panel_metadata_present = (
        local_candidate_metadata is not None
        or content_host_metadata is not None
    )
    if (local_candidate_image_data_url is not None) != panel_metadata_present:
        raise ValueError(
            "review panel image and candidate/host metadata must be supplied together"
        )
    if (assigned_evidence_image_data_url is not None) != (
        assigned_evidence_metadata is not None
    ):
        raise ValueError(
            "assigned evidence image and metadata must be supplied together"
        )
    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    candidate_context = None
    host_context = None
    evidence_context = None
    same_mask_conflict_context = None
    additional_images = None
    if local_candidate_metadata is not None:
        if not local_candidate_metadata:
            raise ValueError("local candidate metadata must not be empty")
        candidate_context, _ = compact_local_candidate_context(
            local_candidate_metadata
        )
    if content_host_metadata is not None:
        if not content_host_metadata:
            raise ValueError("content-host metadata must not be empty")
        pass_a_by_id = {
            str(item.get("object_id") or ""): item for item in pass_a_objects
        }
        for host in content_host_metadata:
            source_id = str(host.get("source_object_id") or "")
            if source_id not in pass_a_by_id:
                raise ValueError(
                    f"content-host references unknown Pass-A ID: {source_id}"
                )
            if category_key(host.get("category")) != category_key(
                pass_a_by_id[source_id].get("category")
            ):
                raise ValueError(
                    f"content-host category does not match {source_id}"
                )
        host_context = compact_content_host_context(
            content_host_metadata
        )
    if assigned_evidence_metadata is not None:
        if not assigned_evidence_metadata:
            raise ValueError("assigned evidence metadata must not be empty")
        pass_a_ids = {str(item.get("object_id") or "") for item in pass_a_objects}
        unknown = sorted(
            str(item.get("source_object_id") or "")
            for item in assigned_evidence_metadata
            if str(item.get("source_object_id") or "") not in pass_a_ids
        )
        if unknown:
            raise ValueError(f"assigned evidence references unknown Pass-A IDs: {unknown}")
        evidence_context = compact_assigned_sam3_evidence_context(
            assigned_evidence_metadata
        )
    if same_mask_conflicts is not None:
        if not same_mask_conflicts:
            raise ValueError("same-mask conflicts must be omitted when empty")
        pass_a_ids = {str(item.get("object_id") or "") for item in pass_a_objects}
        unknown = sorted(
            str(owner_id)
            for conflict in same_mask_conflicts
            for owner_id in (conflict.get("object_ids") or [])
            if str(owner_id) not in pass_a_ids
        )
        if unknown:
            raise ValueError(
                f"same-mask conflict references unknown Pass-A IDs: {unknown}"
            )
        same_mask_conflict_context = compact_same_mask_conflict_context(
            same_mask_conflicts
        )
    additional_images = []
    if panel_metadata_present:
        additional_images.append(str(local_candidate_image_data_url))
    if assigned_evidence_metadata is not None:
        additional_images.append(str(assigned_evidence_image_data_url))
    if not additional_images:
        additional_images = None
    prompt = build_compact_pass_b_prompt(
        compact_pass_a_context(pass_a_objects),
        candidate_context,
        content_host_registry=host_context,
        assigned_evidence_registry=evidence_context,
        same_mask_conflict_registry=same_mask_conflict_context,
        content_closure_only=content_closure_only,
    )
    (output_dir / "request_prompt.txt").write_text(
        prompt + "\n",
        encoding="utf-8",
    )
    response_text, call_info = run_call(
        client=client,
        prompt=prompt,
        image_data_url=image_data_url,
        model=model,
        reasoning_effort=reasoning_effort,
        output_dir=output_dir,
        additional_image_data_urls=additional_images,
    )
    require_clean_finish(call_info, "Pass B")
    actions, protocol_audit = parse_compact_pass_b(
        response_text,
        pass_a_objects=pass_a_objects,
        local_candidate_metadata=local_candidate_metadata,
        assigned_evidence_metadata=assigned_evidence_metadata,
        **parse_context(call_info),
    )
    call_info["response_protocol"] = COMPACT_PASS_B_PROTOCOL
    call_info["compact_protocol_audit"] = protocol_audit
    write_json(output_dir / "compact_protocol_audit.json", protocol_audit)
    return actions, call_info, prompt

def parse_context(call_info: dict[str, Any]) -> dict[str, Any]:
    return {"finish_reason": call_info.get("finish_reason")}
