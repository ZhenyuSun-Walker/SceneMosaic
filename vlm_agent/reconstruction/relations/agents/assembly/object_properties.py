from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from relations.agents.state import sorted_object_id_key, write_audit_pair
from relations.object_physics.core import local_upright_semantic_override

ROOM_MATERIAL_KEYS = ("walls", "floor", "ceiling")


def _is_ceiling_light_object(obj: dict[str, Any]) -> bool:
    category = str(obj.get("category") or "").lower()
    description = str(obj.get("description") or "").lower()
    return category == "ceiling_light" or ("ceiling" in description and any(token in description for token in ["light", "spotlight", "fixture"]))

def _snake_case(value: str | None) -> str:
    value = (value or "unknown_object").strip()
    value = re.sub(r"[^0-9A-Za-z]+", "_", value)
    value = re.sub(r"_+", "_", value).strip("_").lower()
    return value or "unknown_object"


def candidate_objects(scene_state: dict[str, Any]) -> list[dict[str, str]]:
    items = scene_state.get("object_candidates", {}).get("items", [])
    objects = []
    for item in items:
        object_id = item.get("object_id")
        if not object_id:
            continue
        objects.append(
            {
                "id": object_id,
                "category": _snake_case(item.get("category")),
                "description": item.get("description") or _snake_case(item.get("category")).replace("_", " "),
            }
        )
    objects.sort(key=lambda obj: sorted_object_id_key(obj["id"]))
    return objects


def _walk(value: Any):
    if isinstance(value, dict):
        yield value
        for child in value.values():
            yield from _walk(child)
    elif isinstance(value, list):
        for child in value:
            yield from _walk(child)


def _walk_with_path(value: Any, path: str = "$"):
    yield path, value
    if isinstance(value, dict):
        for key, child in value.items():
            yield from _walk_with_path(child, f"{path}.{key}")
    elif isinstance(value, list):
        for idx, child in enumerate(value):
            yield from _walk_with_path(child, f"{path}[{idx}]")


def normalize_room_materials(value: Any) -> tuple[dict[str, str] | None, list[str]]:
    if not isinstance(value, dict):
        return None, ["room_materials must be an object."]

    errors: list[str] = []
    keys = set(value.keys())
    missing = [key for key in ROOM_MATERIAL_KEYS if key not in keys]
    extra = sorted(keys - set(ROOM_MATERIAL_KEYS))
    if missing:
        errors.append(f"room_materials is missing required keys: {missing}.")
    if extra:
        errors.append(f"room_materials has unexpected keys: {extra}.")

    normalized: dict[str, str] = {}
    for key in ROOM_MATERIAL_KEYS:
        raw = value.get(key)
        if not isinstance(raw, str) or not raw.strip():
            errors.append(f"room_materials.{key} must be a non-empty string.")
            continue
        normalized[key] = re.sub(r"\s+", " ", raw).strip()

    return (normalized if not errors else None), errors


def collect_room_materials(
    task_results: list[dict[str, Any]],
    guard_result: dict[str, Any] | None = None,
) -> dict[str, Any]:
    roots: list[Any] = list(task_results)
    if guard_result:
        roots.append(guard_result)

    valid_candidates: list[dict[str, Any]] = []
    invalid_candidates: list[dict[str, Any]] = []
    for order, root in enumerate(roots):
        if not isinstance(root, dict) or "room_materials" not in root:
            continue
        provided = root.get("provided_capabilities")
        if isinstance(provided, str):
            provided_capabilities = [provided]
        elif isinstance(provided, list):
            provided_capabilities = [str(item) for item in provided]
        else:
            provided_capabilities = []
        record = {
            "source_task_id": root.get("task_id"),
            "source_task_type": root.get("canonical_task_kind") or root.get("task_type"),
            "declared_room_materials_provider": "room_materials_ready" in provided_capabilities,
            "source_order": order,
        }
        normalized, errors = normalize_room_materials(root.get("room_materials"))
        if normalized is None:
            invalid_candidates.append({
                **record,
                "errors": errors,
                "observed_keys": sorted(root["room_materials"].keys()) if isinstance(root.get("room_materials"), dict) else [],
            })
            continue
        valid_candidates.append({**record, "room_materials": normalized})

    declared = [item for item in valid_candidates if item["declared_room_materials_provider"]]
    selected = (declared or valid_candidates)[-1] if (declared or valid_candidates) else None
    return {
        "status": "ready" if selected else "missing_or_invalid",
        "room_materials": selected.get("room_materials") if selected else None,
        "source_task_id": selected.get("source_task_id") if selected else None,
        "source_task_type": selected.get("source_task_type") if selected else None,
        "selection_policy": "latest_declared_provider_else_latest_valid_candidate",
        "valid_candidates": valid_candidates,
        "invalid_candidates": invalid_candidates,
    }


_OBJECT_REVISION_CONFIDENCE_SCORES = {"low": 1, "medium": 2, "high": 3}
_OBJECT_REVISION_SOURCE_PRIORITIES = {
    "object_revision_check": 3,
    "schema_guard": 3,
    "support_contact_extraction": 2,
    "contain_relation_extraction": 2,
}


def _normalize_object_revision_confidence(value: Any) -> str:
    normalized = str(value or "medium").strip().lower()
    return normalized if normalized in _OBJECT_REVISION_CONFIDENCE_SCORES else "medium"


def _resolve_object_revision_patches(
    patches: list[dict[str, Any]],
    warnings: list[str],
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for patch in patches:
        for field_name in ("new_category", "new_description"):
            if patch.get(field_name) is not None:
                grouped.setdefault((patch["object_id"], field_name), []).append(patch)

    resolved_by_object: dict[str, dict[str, Any]] = {}
    for (object_id, field_name), candidates in grouped.items():
        best_score = max(
            (
                _OBJECT_REVISION_CONFIDENCE_SCORES[item["confidence"]],
                int(item.get("source_priority", 1)),
            )
            for item in candidates
        )
        top = [
            item
            for item in candidates
            if (
                _OBJECT_REVISION_CONFIDENCE_SCORES[item["confidence"]],
                int(item.get("source_priority", 1)),
            ) == best_score
        ]
        top_values = {str(item[field_name]) for item in top}
        if len(top_values) != 1:
            warnings.append(
                f"Skipped ambiguous object revision for {object_id}.{field_name}: "
                f"equally ranked tasks proposed {sorted(top_values)}."
            )
            continue
        chosen = top[0]
        discarded_values = {
            str(item[field_name])
            for item in candidates
            if str(item[field_name]) != str(chosen[field_name])
        }
        if discarded_values:
            warnings.append(
                f"Resolved object revision conflict for {object_id}.{field_name} in favor of "
                f"{chosen[field_name]!r} from {chosen.get('source_task_id') or chosen.get('source_task_kind')}; "
                f"discarded lower-ranked values {sorted(discarded_values)}."
            )
        resolved = resolved_by_object.setdefault(
            object_id,
            {
                "object_id": object_id,
                "new_category": None,
                "new_description": None,
                "reasons": [],
                "source_task_ids": [],
                "confidence": chosen["confidence"],
            },
        )
        resolved[field_name] = chosen[field_name]
        resolved["reasons"].append(chosen["reason"])
        source_id = chosen.get("source_task_id") or chosen.get("source_task_kind")
        if source_id:
            resolved["source_task_ids"].append(source_id)
        if _OBJECT_REVISION_CONFIDENCE_SCORES[chosen["confidence"]] > _OBJECT_REVISION_CONFIDENCE_SCORES[resolved["confidence"]]:
            resolved["confidence"] = chosen["confidence"]

    resolved_patches: list[dict[str, Any]] = []
    for object_id in sorted(resolved_by_object):
        item = resolved_by_object[object_id]
        reasons = list(dict.fromkeys(item.pop("reasons")))
        source_task_ids = list(dict.fromkeys(item.pop("source_task_ids")))
        item["reason"] = "; ".join(reasons)
        item["source_task_id"] = ",".join(source_task_ids) if source_task_ids else None
        resolved_patches.append(item)
    return resolved_patches


def collect_object_revision_patches(task_results: list[dict[str, Any]], guard_result: dict[str, Any] | None = None) -> tuple[list[dict[str, Any]], list[str]]:
    roots: list[tuple[Any, bool]] = [(item, False) for item in task_results]
    if guard_result:
        roots.append((guard_result, True))
    patches: list[dict[str, Any]] = []
    warnings: list[str] = []
    for root, is_guard_result in roots:
        root_kind = (
            "schema_guard"
            if is_guard_result
            else str(root.get("canonical_task_kind") or root.get("task_type") or "unknown")
            if isinstance(root, dict)
            else "unknown"
        )
        source_priority = _OBJECT_REVISION_SOURCE_PRIORITIES.get(root_kind, 1)
        source_task_id = root.get("task_id") if isinstance(root, dict) else None
        for node in _walk(root):
            candidates = []
            for key in ["object_revisions", "object_revision_candidates", "revision_patches", "object_revision_patches", "accepted_object_revision_patches"]:
                if isinstance(node.get(key), list):
                    candidates.extend(node[key])
            for patch in candidates:
                if not isinstance(patch, dict):
                    continue
                object_id = patch.get("object_id") or patch.get("id")
                attach_relevant_evidence = (
                    patch.get("image_or_topdown_evidence")
                    or patch.get("image_evidence")
                    or patch.get("mask_evidence")
                    or patch.get("topdown_evidence")
                    or patch.get("attach_relevant_conflict")
                )
                reason = patch.get("reason") or patch.get("rationale") or patch.get("evidence") or patch.get("reason_code")
                new_category = patch.get("new_category") or patch.get("final_category")
                new_description = patch.get("new_description") or patch.get("final_description")
                if new_category is None and patch.get("normalized_category") and patch.get("category"):
                    warnings.append(
                        f"Low-priority category alias normalization skipped for {object_id or 'unknown_object'}: "
                        f"{patch.get('category')} -> {patch.get('normalized_category')}."
                    )
                    continue
                if new_category is None and patch.get("category") and attach_relevant_evidence:
                    new_category = patch.get("category")
                if new_description is None and patch.get("description") and attach_relevant_evidence:
                    new_description = f"{patch.get('description')} [revision needed: {attach_relevant_evidence}]"
                if not reason and attach_relevant_evidence:
                    reason = f"Attach-relevant image/mask/topdown conflict: {attach_relevant_evidence}"
                if not object_id:
                    warnings.append(f"Skipped object revision patch without object_id: {patch}")
                    continue
                if not reason:
                    warnings.append(f"Skipped object revision patch for {object_id} because reason is missing.")
                    continue
                if new_category is None and new_description is None:
                    warnings.append(f"Skipped object revision patch for {object_id} because no category/description change was provided.")
                    continue
                patches.append(
                    {
                        "object_id": object_id,
                        "new_category": _snake_case(new_category) if new_category is not None else None,
                        "new_description": new_description,
                        "reason": str(reason),
                        "confidence": _normalize_object_revision_confidence(patch.get("confidence")),
                        "source_priority": source_priority,
                        "source_task_kind": root_kind,
                        "source_task_id": source_task_id,
                    }
                )
    return _resolve_object_revision_patches(patches, warnings), warnings

def apply_object_revision_patches(objects: list[dict[str, str]], patches: list[dict[str, Any]]) -> tuple[list[dict[str, str]], list[dict[str, Any]], list[str]]:
    by_id = {obj["id"]: obj for obj in objects}
    applied: list[dict[str, Any]] = []
    warnings: list[str] = []
    for patch in patches:
        object_id = patch["object_id"]
        if object_id not in by_id:
            warnings.append(f"Skipped object revision patch for unknown object_id {object_id}.")
            continue
        obj = by_id[object_id]
        changes: dict[str, dict[str, str]] = {}
        if patch.get("new_category") and patch["new_category"] != obj["category"]:
            changes["category"] = {"old": obj["category"], "new": patch["new_category"]}
            obj["category"] = patch["new_category"]
        if patch.get("new_description") and patch["new_description"] != obj["description"]:
            changes["description"] = {"old": obj["description"], "new": patch["new_description"]}
            obj["description"] = patch["new_description"]
        if changes:
            applied.append(
                {
                    "object_id": object_id,
                    "fields": list(changes.keys()),
                    "changes": changes,
                    "reason": patch["reason"],
                    "source_task_id": patch.get("source_task_id"),
                }
            )
    return objects, applied, warnings


def write_object_revision_audit(output_dir: Path, applied: list[dict[str, Any]], warnings: list[str], dry_run: bool) -> None:
    data = {
        "dry_run": dry_run,
        "status": "completed" if not warnings else "completed_with_warnings",
        "items": applied,
        "warnings": warnings,
        "description_policy": (
            "description is weak evidence; object revision should focus on attach-relevant semantic or positional "
            "conflicts grounded by image/mask evidence, with layout/topdown used only as coarse support. Minor category "
            "alias normalization is low priority and should not override attach-relevant conflicts."
        ),
    }
    write_audit_pair(output_dir, "objects", "object_revision_audit", data, "Object Revision Audit")


FLOOR_COVERING_TEXT_HINTS = {
    "area rug",
    "carpet",
    "floor covering",
    "floor mat",
    "mat",
    "rug",
    "tatami",
    "woven mat",
}

UPRIGHT_FALSE_TEXT_HINTS = {
    "artwork",
    "bean_bag",
    "box",
    "book",
    "carpet",
    "controller",
    "cooling_rack",
    "cushion",
    "floor_covering",
    "framed_art",
    "mat",
    "painting",
    "pillow",
    "rug",
    "slab",
    "wall_art",
    "wall_clock",
    "wall_decoration",
    "wall_mirror",
    "wall_panel",
    "wall_sign",
    "whiteboard",
}

UPRIGHT_IMAGE_AXIS_TEXT_HINTS = {
    "bottle",
    "canister",
    "cylinder",
    "paper_towel_roll",
    "roll",
    "thermos",
    "tumbler",
}

UPRIGHT_TRUE_TEXT_HINTS = {
    "appliance",
    "armchair",
    "basin",
    "basket",
    "bed",
    "bench",
    "book_shelf",
    "bookcase",
    "bookshelf",
    "bowl",
    "cabinet",
    "chair",
    "couch",
    "counter",
    "cup",
    "desk",
    "display_screen",
    "dresser",
    "figure",
    "figurine",
    "floor_lamp",
    "fridge",
    "lamp",
    "media_console",
    "monitor",
    "mug",
    "nightstand",
    "oven",
    "plate",
    "plant",
    "refrigerator",
    "shelf",
    "side_table",
    "sideboard",
    "sofa",
    "stand",
    "stool",
    "table",
    "television",
    "tray",
    "tv",
    "vase",
    "vessel",
    "wardrobe",
}

CONTAIN_UP_DIRECTION_HINTS = {
    "basket", "bin", "bowl", "box", "crate", "drawer", "open_top", "tray",
}
CONTAIN_FRONT_DIRECTION_HINTS = {
    "bookcase", "bookshelf", "bunk_bed", "cabinet", "case", "closet", "cubby", "display",
    "freezer", "fridge", "loft_bed", "multi_tier", "open_shelf", "rack", "refrigerator",
    "shelf", "shelving", "tiered", "wardrobe",
}


def _bool_or_none(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        text = value.strip().lower()
        if text in {"true", "yes", "1", "y", "positive", "accepted"}:
            return True
        if text in {"false", "no", "0", "n", "negative", "rejected"}:
            return False
    return None


def _scene_object_context(scene_state: dict[str, Any] | None) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    if not isinstance(scene_state, dict):
        return result
    items = scene_state.get("object_candidates", {}).get("items", []) if isinstance(scene_state.get("object_candidates"), dict) else []
    for item in items:
        if isinstance(item, dict) and item.get("object_id"):
            result[str(item["object_id"])] = item
    return result


def _object_text(obj: dict[str, Any], scene_obj: dict[str, Any] | None = None) -> str:
    scene_obj = scene_obj or {}
    return " ".join(
        str(value or "")
        for value in [
            obj.get("category"),
            obj.get("description"),
            scene_obj.get("category"),
            scene_obj.get("description"),
        ]
    ).lower()


def _bbox_extent_xyz(scene_obj: dict[str, Any] | None) -> list[float] | None:
    bbox = scene_obj.get("bbox_3d") if isinstance(scene_obj, dict) else None
    extent = bbox.get("extent") if isinstance(bbox, dict) else None
    if not (isinstance(extent, list) and len(extent) >= 3):
        return None
    values = []
    for item in extent[:3]:
        try:
            values.append(abs(float(item)))
        except Exception:
            return None
    return values


def _bbox_2d_height_width_ratio(scene_obj: dict[str, Any] | None) -> float | None:
    bbox = scene_obj.get("bbox_2d") if isinstance(scene_obj, dict) else None
    if not (isinstance(bbox, list) and len(bbox) >= 4):
        return None
    try:
        width = abs(float(bbox[2]) - float(bbox[0]))
        height = abs(float(bbox[3]) - float(bbox[1]))
    except (TypeError, ValueError):
        return None
    if width <= 1e-6 or height <= 1e-6:
        return None
    return height / width


def _local_floor_covering_hint(obj: dict[str, Any], scene_obj: dict[str, Any] | None = None) -> tuple[bool, str, str]:
    text = _object_text(obj, scene_obj)
    if _matches_semantic_hint(text, FLOOR_COVERING_TEXT_HINTS):
        return True, "high", "local_text_fallback: category/description indicates rug/carpet/mat/floor covering"
    extent = _bbox_extent_xyz(scene_obj)
    if extent:
        x, y, z = extent
        horizontal_area = x * z
        if y <= 0.035 and horizontal_area >= 0.04 and max(x, z) >= 0.25:
            return True, "medium", "local_geometry_fallback: object is very flat, broad, and floor-like"
    return False, "low", "no floor-covering evidence"


def _matches_semantic_hint(value: Any, hints: set[str]) -> bool:
    normalized = f"_{_snake_case(str(value or ''))}_"
    return any(f"_{_snake_case(hint)}_" in normalized for hint in hints)


def _local_upright_semantic_hint(
    obj: dict[str, Any],
    scene_obj: dict[str, Any] | None = None,
    is_floor_covering: bool = False,
) -> tuple[bool, str, str] | None:
    scene_obj = scene_obj or {}
    category_values = [str(value or "") for value in [obj.get("category"), scene_obj.get("category")]]
    category = " ".join(category_values)
    text = _object_text(obj, scene_obj)
    if is_floor_covering or _matches_semantic_hint(text, FLOOR_COVERING_TEXT_HINTS):
        return False, "local_semantic_policy", "floor covering is not an upright reconstruction constraint"
    for category_value in category_values:
        stable_category = local_upright_semantic_override({"category": category_value})
        if stable_category is not None:
            upright, reason = stable_category
            return upright, "local_semantic_policy", reason
    image_ratio = _bbox_2d_height_width_ratio(scene_obj)
    if (
        image_ratio is not None
        and image_ratio >= 1.35
        and _matches_semantic_hint(text, UPRIGHT_IMAGE_AXIS_TEXT_HINTS)
    ):
        return True, "local_image_geometry_policy", "axis-sensitive object is visibly vertical in the original-image bbox"
    if _matches_semantic_hint(category, UPRIGHT_TRUE_TEXT_HINTS):
        return True, "local_semantic_policy", "canonical category requires a stable local-up direction under gravity"
    if _matches_semantic_hint(category, UPRIGHT_FALSE_TEXT_HINTS):
        return False, "local_semantic_policy", "canonical category does not require a fixed local-up direction under gravity"
    if _matches_semantic_hint(text, UPRIGHT_TRUE_TEXT_HINTS):
        return True, "local_semantic_policy", "description indicates a stable local-up direction is required under gravity"
    if _matches_semantic_hint(text, UPRIGHT_FALSE_TEXT_HINTS):
        return False, "local_semantic_policy", "description indicates no fixed local-up direction is required under gravity"
    return None


def _local_upright_fallback(
    obj: dict[str, Any],
    scene_obj: dict[str, Any] | None = None,
    is_floor_covering: bool = False,
) -> tuple[bool, str, str]:
    semantic = _local_upright_semantic_hint(obj, scene_obj, is_floor_covering)
    if semantic is not None:
        return semantic
    extent = _bbox_extent_xyz(scene_obj)
    if extent:
        x, y, z = extent
        horizontal = max(x, z, 1e-6)
        flat = y <= min(x, z, 1e-6) * 0.45 or y <= 0.04
        tall = y >= horizontal * 0.65 and y >= 0.12
        if flat:
            return False, "local_geometry_fallback", "unknown-category object is geometrically flat/low"
        if tall:
            return True, "local_geometry_fallback", "unknown-category object is tall relative to footprint"
    return True, "local_default_fallback", "defaulting unknown non-wall object to upright for stable gravity-based reconstruction"

def collect_object_physical_properties(
    task_results: list[dict[str, Any]],
    guard_result: dict[str, Any] | None,
    objects: list[dict[str, Any]],
    scene_state: dict[str, Any] | None = None,
) -> dict[str, Any]:
    scene_by_id = _scene_object_context(scene_state)
    by_id: dict[str, dict[str, Any]] = {
        str(obj["id"]): {
            "object_id": str(obj["id"]),
            "agent": {},
            "fallback": {},
        }
        for obj in objects
        if isinstance(obj, dict) and obj.get("id")
    }
    roots: list[Any] = list(task_results)
    if guard_result:
        roots.append(guard_result)

    def apply_property(raw: dict[str, Any], source_task_id: str | None = None) -> None:
        object_id = raw.get("object_id") or raw.get("id") or raw.get("target_object_id")
        if not object_id or str(object_id) not in by_id:
            return
        entry = by_id[str(object_id)]
        agent = entry.setdefault("agent", {})
        floor_covering = _bool_or_none(
            raw.get("is_floor_covering")
            if "is_floor_covering" in raw
            else raw.get("floor_covering")
            if "floor_covering" in raw
            else raw.get("is_rug_or_floor_covering")
        )
        if floor_covering is not None:
            agent["is_floor_covering"] = floor_covering
            agent["floor_covering_confidence"] = raw.get("floor_covering_confidence") or raw.get("confidence") or "medium"
            agent["floor_covering_reason"] = raw.get("floor_covering_reason") or raw.get("reason") or raw.get("rationale")
            agent["floor_covering_source_task_id"] = source_task_id

    property_list_keys = [
        "object_physical_properties",
        "object_properties",
        "physical_properties",
        "object_semantic_properties",
        "object_property_judgements",
        "object_physical_property_reasoning",
    ]
    for root in roots:
        source_task_id = root.get("task_id") if isinstance(root, dict) else None
        for node in _walk(root):
            if not isinstance(node, dict):
                continue
            for key in property_list_keys:
                value = node.get(key)
                if isinstance(value, list):
                    for item in value:
                        if isinstance(item, dict):
                            apply_property(item, source_task_id)
                elif isinstance(value, dict):
                    if isinstance(value.get("items"), list):
                        for item in value["items"]:
                            if isinstance(item, dict):
                                apply_property(item, source_task_id)
                    else:
                        apply_property(value, source_task_id)
            if any(key in node for key in ["is_floor_covering", "floor_covering", "is_rug_or_floor_covering"]):
                apply_property(node, source_task_id)

    items = []
    floor_covering_ids: set[str] = set()
    for obj in objects:
        object_id = str(obj.get("id"))
        if object_id not in by_id:
            continue
        entry = by_id[object_id]
        agent = entry.get("agent", {})
        local_fc, local_fc_conf, local_fc_reason = _local_floor_covering_hint(obj, scene_by_id.get(object_id))
        is_floor_covering = bool(agent.get("is_floor_covering")) if "is_floor_covering" in agent else local_fc
        fc_source = "agent_property" if "is_floor_covering" in agent else ("local_text_or_geometry_fallback" if local_fc else "none")
        if is_floor_covering:
            floor_covering_ids.add(object_id)
        entry["resolved"] = {
            "object_id": object_id,
            "is_floor_covering": is_floor_covering,
            "floor_covering_confidence": agent.get("floor_covering_confidence") or local_fc_conf,
            "floor_covering_reason": agent.get("floor_covering_reason") or local_fc_reason,
            "floor_covering_source": fc_source,
        }
        items.append(entry["resolved"])
    return {
        "schema_version": "object_physical_properties",
        "items": items,
        "by_id": {item["object_id"]: item for item in items},
        "floor_covering_object_ids": sorted(floor_covering_ids, key=sorted_object_id_key),
    }


def write_object_physical_properties_audit(output_dir: Path, properties: dict[str, Any], dry_run: bool) -> None:
    data = {
        "dry_run": dry_run,
        "status": "completed",
        "schema_version": properties.get("schema_version"),
        "floor_covering_object_ids": properties.get("floor_covering_object_ids", []),
        "items": properties.get("items", []),
        "policy": "Relation Agents may supply floor-covering decisions only. Upright is owned by the independent Physics Sidecar and relation-aware final assembly.",
    }
    write_audit_pair(output_dir, "objects", "object_physical_properties_audit", data, "Object Physical Properties Audit")


def load_async_sidecar_upright_priors(
    output_dir: Path,
    object_ids: list[str],
) -> dict[str, Any]:
    sidecar_path = output_dir / "physics_sidecar" / "integrated_async" / "object_physics.json"
    audit_path = output_dir / "physics_sidecar" / "integrated_async" / "physics_audit.json"
    result: dict[str, Any] = {
        "schema_version": "sidecar_upright_priors",
        "sidecar_path": str(sidecar_path),
        "available": sidecar_path.exists(),
        "by_id": {},
        "missing_object_ids": list(object_ids),
        "invalid_items": [],
        "read_error": None,
    }
    if not sidecar_path.exists():
        return result
    try:
        payload = json.loads(sidecar_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        result["read_error"] = str(exc)
        return result

    provenance: dict[str, str] = {}
    if audit_path.exists():
        try:
            audit_payload = json.loads(audit_path.read_text(encoding="utf-8"))
            provenance = {
                str(item.get("object_id")): str(item.get("upright_source") or "unknown")
                for item in audit_payload.get("items", [])
                if isinstance(item, dict) and item.get("object_id")
            }
        except (OSError, json.JSONDecodeError):
            provenance = {}

    expected = set(object_ids)
    for item in payload.get("objects", []) if isinstance(payload, dict) else []:
        if not isinstance(item, dict):
            continue
        object_id = str(item.get("object_id") or "")
        if object_id not in expected:
            continue
        upright = item.get("upright")
        if isinstance(upright, bool):
            result["by_id"][object_id] = {
                "upright": upright,
                "sidecar_source": provenance.get(object_id, "agent"),
            }
        else:
            result["invalid_items"].append({
                "object_id": object_id,
                "upright": upright,
                "reason": "sidecar upright is missing or is not a JSON boolean",
            })
    result["missing_object_ids"] = [
        object_id for object_id in object_ids if object_id not in result["by_id"]
    ]
    return result


def apply_upright_properties(
    objects: list[dict[str, Any]],
    relations: list[dict[str, Any]],
    properties: dict[str, Any],
    scene_state: dict[str, Any] | None,
    output_dir: Path,
    dry_run: bool,
    upright_priors: dict[str, Any] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    scene_by_id = _scene_object_context(scene_state)
    props_by_id = properties.get("by_id", {}) if isinstance(properties, dict) else {}
    sidecar_by_id = upright_priors.get("by_id", {}) if isinstance(upright_priors, dict) else {}
    target_relation_anchors: dict[str, list[str]] = {}
    for rel in relations:
        if not (isinstance(rel, dict) and isinstance(rel.get("parameters"), dict)):
            continue
        target = rel.get("parameters", {}).get("object_target")
        anchor = rel.get("parameters", {}).get("object_anchor")
        if not target:
            continue
        target_relation_anchors.setdefault(str(target), []).append(str(anchor or ""))
    wall_attach_only_targets = {
        target
        for target, anchors in target_relation_anchors.items()
        if anchors and all(anchor.startswith("wall_") for anchor in anchors)
    }
    ceiling_attach_targets = {
        rel.get("parameters", {}).get("object_target")
        for rel in relations
        if isinstance(rel, dict)
        and rel.get("relation") == "attach"
        and isinstance(rel.get("parameters"), dict)
        and str(rel.get("parameters", {}).get("object_anchor", "")).lower() == "ceiling"
    }
    updated: list[dict[str, Any]] = []
    audit_items: list[dict[str, Any]] = []
    for obj in objects:
        new_obj = dict(obj)
        object_id = str(new_obj.get("id"))
        prop = props_by_id.get(object_id, {}) if isinstance(props_by_id, dict) else {}
        if object_id in wall_attach_only_targets:
            upright = False
            source = "wall_attach_only_override"
            reason = "object is covered only by wall attach relations; it is treated as wall-constrained rather than gravity-upright"
        elif object_id in ceiling_attach_targets:
            upright = False
            source = "ceiling_attach_override"
            reason = "object has final ceiling attach; ceiling-mounted objects are not gravity-upright reconstruction constraints"
        elif _is_ceiling_light_object(new_obj):
            upright = False
            source = "local_text_fallback"
            reason = "category/description indicates a ceiling light fixture, which is not a gravity-upright reconstruction constraint"
        elif bool(prop.get("is_floor_covering")):
            upright = False
            source = "floor_covering_override"
            reason = "floor covering is not an upright reconstruction constraint"
        elif isinstance(sidecar_by_id.get(object_id, {}).get("upright"), bool):
            upright = bool(sidecar_by_id[object_id]["upright"])
            sidecar_source = str(sidecar_by_id[object_id].get("sidecar_source") or "agent")
            source = "async_physics_sidecar_agent" if sidecar_source == "agent" else "async_physics_sidecar_prior"
            reason = "independent Physics Sidecar supplied the relation-agnostic canonical upright prior"
        else:
            upright, source, reason = _local_upright_fallback(new_obj, scene_by_id.get(object_id), bool(prop.get("is_floor_covering")))
        new_obj["upright"] = bool(upright)
        updated.append(new_obj)
        audit_items.append(
            {
                "object_id": object_id,
                "upright": bool(upright),
                "source": source,
                "reason": reason,
                "is_floor_covering": bool(prop.get("is_floor_covering")),
                "sidecar_upright": sidecar_by_id.get(object_id, {}).get("upright"),
                "sidecar_upright_source": sidecar_by_id.get(object_id, {}).get("sidecar_source"),
            }
        )
    audit = {
        "dry_run": dry_run,
        "status": "completed",
        "schema_version": "upright_audit",
        "items": audit_items,
        "sidecar_upright_priors": {
            "available": bool((upright_priors or {}).get("available")),
            "sidecar_path": (upright_priors or {}).get("sidecar_path"),
            "missing_object_ids": (upright_priors or {}).get("missing_object_ids", []),
            "invalid_items": (upright_priors or {}).get("invalid_items", []),
            "read_error": (upright_priors or {}).get("read_error"),
        },
        "policy": "wall-only, ceiling, and floor-covering relation overrides come first; the independent Physics Sidecar owns the general upright prior; local semantic/image/geometry fallback is used only when the sidecar prior is unavailable.",
    }
    write_audit_pair(output_dir, "objects", "upright_audit", audit, "Upright Audit")
    return updated, audit


def _normalize_contain_render_direction(value: Any) -> str | None:
    raw = str(value or "").strip().lower()
    if raw == "正面":
        return "front"
    if raw == "上面":
        return "up"
    normalized = _snake_case(raw)
    if normalized in {"front", "forward", "front_view"}:
        return "front"
    if normalized in {"up", "top", "top_view", "above"}:
        return "up"
    return None


def _semantic_contain_render_direction(obj: dict[str, Any], scene_obj: dict[str, Any] | None = None) -> str:
    text = _object_text(obj, scene_obj)
    front = _matches_semantic_hint(text, CONTAIN_FRONT_DIRECTION_HINTS)
    up = _matches_semantic_hint(text, CONTAIN_UP_DIRECTION_HINTS)
    if front:
        return "front"
    if up:
        return "up"
    extent = _bbox_extent_xyz(scene_obj)
    if extent:
        x, y, z = extent
        if y <= max(x, z, 1e-6) * 0.45:
            return "up"
    return "front"


def _contain_candidate_anchor_directions(scene_state: dict[str, Any] | None) -> tuple[dict[str, str], str | None, str | None]:
    if not isinstance(scene_state, dict):
        return {}, None, None
    evidence = scene_state.get("evidence", {}) if isinstance(scene_state.get("evidence"), dict) else {}
    relation_precompute = evidence.get("relation_precompute", {}) if isinstance(evidence.get("relation_precompute"), dict) else {}
    outputs = relation_precompute.get("outputs", {}) if isinstance(relation_precompute.get("outputs"), dict) else {}
    raw_path = outputs.get("contain_candidates_path")
    if not raw_path:
        return {}, None, None
    path = Path(str(raw_path))
    if not path.exists():
        return {}, str(path), "contain_candidates_path does not exist"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return {}, str(path), str(exc)
    directions: dict[str, str] = {}
    for item in payload.get("possible_containers", []) if isinstance(payload, dict) else []:
        if not isinstance(item, dict):
            continue
        object_id = item.get("object_id") or item.get("id")
        direction = _normalize_contain_render_direction(
            item.get("render_direction")
            or item.get("anchor_render_direction")
            or item.get("渲染方向")
        )
        if object_id and direction:
            directions[str(object_id)] = direction
    return directions, str(path), None


def _agent_contain_anchor_directions(
    task_results: list[dict[str, Any]],
    guard_result: dict[str, Any] | None,
) -> dict[str, dict[str, str]]:
    result: dict[str, dict[str, str]] = {}
    roots: list[Any] = list(task_results)
    if guard_result:
        roots.append(guard_result)
    for root in roots:
        source_task_id = str(root.get("task_id") or "schema_guard") if isinstance(root, dict) else "unknown"
        for node in _walk(root):
            if not isinstance(node, dict):
                continue
            parameters = node.get("parameters") if isinstance(node.get("parameters"), dict) else {}
            anchor = (
                node.get("object_anchor")
                or node.get("anchor_object_id")
                or parameters.get("object_anchor")
                or (node.get("object_id") if _bool_or_none(node.get("contain_anchor")) else None)
            )
            direction = _normalize_contain_render_direction(
                node.get("render_direction")
                or node.get("anchor_render_direction")
                or node.get("渲染方向")
            )
            if anchor and str(anchor).startswith("object_") and direction:
                result[str(anchor)] = {"render_direction": direction, "source_task_id": source_task_id}
    return result


def apply_contain_anchor_properties(
    objects: list[dict[str, Any]],
    relations: list[dict[str, Any]],
    task_results: list[dict[str, Any]],
    guard_result: dict[str, Any] | None,
    scene_state: dict[str, Any] | None,
    output_dir: Path,
    dry_run: bool,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    anchor_ids = {
        str(rel.get("parameters", {}).get("object_anchor"))
        for rel in relations
        if isinstance(rel, dict)
        and rel.get("relation") == "contain"
        and isinstance(rel.get("parameters"), dict)
        and str(rel.get("parameters", {}).get("object_anchor", "")).startswith("object_")
    }
    candidate_directions, candidate_path, candidate_read_error = _contain_candidate_anchor_directions(scene_state)
    agent_directions = _agent_contain_anchor_directions(task_results, guard_result)
    scene_by_id = _scene_object_context(scene_state)
    updated: list[dict[str, Any]] = []
    audit_items: list[dict[str, Any]] = []
    for obj in objects:
        new_obj = dict(obj)
        object_id = str(new_obj.get("id"))
        is_anchor = object_id in anchor_ids
        direction = None
        source = "not_a_contain_anchor"
        if is_anchor:
            if object_id in agent_directions:
                direction = agent_directions[object_id]["render_direction"]
                source = "agent_contain_property"
            elif object_id in candidate_directions:
                direction = candidate_directions[object_id]
                source = "local_contain_candidate"
            else:
                direction = _semantic_contain_render_direction(new_obj, scene_by_id.get(object_id))
                source = "local_semantic_or_geometry_fallback"
        new_obj.pop("contain_anchor", None)
        new_obj.pop("render_direction", None)
        if is_anchor:
            new_obj["contain_anchor"] = True
            new_obj["render_direction"] = direction
        updated.append(new_obj)
        audit_items.append({
            "object_id": object_id,
            "contain_anchor": is_anchor,
            "render_direction": direction,
            "source": source,
            "agent_direction": agent_directions.get(object_id),
            "candidate_direction": candidate_directions.get(object_id),
        })
    audit = {
        "dry_run": dry_run,
        "status": "completed",
        "schema_version": "contain_anchor_audit",
        "contain_anchor_object_ids": sorted(anchor_ids, key=sorted_object_id_key),
        "contain_candidates_path": candidate_path,
        "contain_candidates_read_error": candidate_read_error,
        "items": audit_items,
        "policy": "Only final contain-relation anchors receive contain_anchor=true and render_direction=front|up; non-anchors omit both fields. Direction priority is Agent visual judgement, local candidate hint, then semantic/geometry fallback.",
    }
    write_audit_pair(output_dir, "objects", "contain_anchor_audit", audit, "Contain Anchor Audit")
    return updated, audit
