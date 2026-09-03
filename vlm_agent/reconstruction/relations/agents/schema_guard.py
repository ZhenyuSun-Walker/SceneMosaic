from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from relations.agents.state import sorted_object_id_key, write_audit_pair
from relations.object_physics.core import FIELD_LIMITS, PHYSICS_FIELDS

FORBIDDEN_PARAM_KEYS = {"obj_anchor", "obj_tar", "obj_target", "object_tar"}
WALL_IDS = ["wall_001", "wall_002", "wall_003", "wall_004"]
ROOM_MATERIAL_KEYS = ("walls", "floor", "ceiling")
OBJECT_ID_RE = re.compile(r"^object_\d+$")
WALL_EPSILON = 1e-5
CLOCKWISE_SIGN_CONVENTION = "top_down_xz_standard_math: negative signed_area is interpreted as clockwise"


def is_number_pair(value: Any) -> bool:
    return isinstance(value, list) and len(value) == 2 and all(isinstance(x, (int, float)) and not isinstance(x, bool) for x in value)


def is_snake_case(value: Any) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[a-z0-9]+(?:_[a-z0-9]+)*", value) is not None


def _point_distance(a: list[float], b: list[float]) -> float:
    return math.sqrt((float(a[0]) - float(b[0])) ** 2 + (float(a[1]) - float(b[1])) ** 2)


def _polygon_signed_area(points: list[list[float]]) -> float:
    area2 = 0.0
    for idx, point in enumerate(points):
        nxt = points[(idx + 1) % len(points)]
        area2 += float(point[0]) * float(nxt[1]) - float(nxt[0]) * float(point[1])
    return area2 / 2.0


def wall_axis_alignment_summary(walls: list[Any], epsilon: float = WALL_EPSILON) -> dict[str, Any]:
    wall_by_id = {wall.get("id"): wall for wall in walls if isinstance(wall, dict)}
    edge_orientations: list[dict[str, Any]] = []
    usable_edges = 0
    all_axis_aligned = True
    for wall_id in WALL_IDS:
        wall = wall_by_id.get(wall_id)
        if not isinstance(wall, dict):
            all_axis_aligned = False
            edge_orientations.append({"id": wall_id, "orientation": "missing", "axis_aligned": False})
            continue
        first = wall.get("first_point")
        second = wall.get("second_point")
        if not is_number_pair(first) or not is_number_pair(second):
            all_axis_aligned = False
            edge_orientations.append({"id": wall_id, "orientation": "invalid", "axis_aligned": False})
            continue
        usable_edges += 1
        dx = abs(float(second[0]) - float(first[0]))
        dz = abs(float(second[1]) - float(first[1]))
        x_constant = dx <= epsilon
        z_constant = dz <= epsilon
        edge_axis_aligned = x_constant or z_constant
        if x_constant and z_constant:
            orientation = "degenerate"
        elif x_constant:
            orientation = "x_constant"
        elif z_constant:
            orientation = "z_constant"
        else:
            orientation = "oblique"
        all_axis_aligned = all_axis_aligned and edge_axis_aligned
        edge_orientations.append(
            {
                "id": wall_id,
                "orientation": orientation,
                "axis_aligned": edge_axis_aligned,
                "delta_x": dx,
                "delta_z": dz,
            }
        )
    complete = usable_edges == len(WALL_IDS)
    axis_aligned = bool(complete and all_axis_aligned)
    return {
        "complete": complete,
        "axis_aligned": axis_aligned,
        "oriented_boundary": bool(complete and not axis_aligned),
        "edge_orientations": edge_orientations,
        "definition": "axis_aligned means every wall edge is x-constant or z-constant within epsilon; oblique walls are allowed when the room footprint is oriented.",
    }


def extract_scene_state_candidates(scene_state: dict[str, Any] | None) -> dict[str, dict[str, Any]]:
    if not isinstance(scene_state, dict):
        return {}
    items = scene_state.get("object_candidates", {}).get("items", [])
    candidates = {}
    for item in items:
        if isinstance(item, dict) and item.get("object_id"):
            candidates[str(item["object_id"])] = item
    return candidates


def load_revision_audit(path: Path | None) -> dict[str, Any] | None:
    if path is None or not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return data if isinstance(data, dict) else None


def load_weak_wall_completion(output_dir: Path) -> dict[str, Any]:
    wall_audit_path = output_dir / "debug" / "walls" / "wall_audit.json"
    default = {
        "weak_wall_completion_used": False,
        "weak_wall_ids": [],
        "completion_methods": [],
        "completion_reasons": [],
        "wall_audit_path": str(wall_audit_path),
    }
    if not wall_audit_path.exists():
        return default
    try:
        wall_audit = json.loads(wall_audit_path.read_text(encoding="utf-8"))
    except Exception as exc:
        default["read_error"] = str(exc)
        return default
    completion = wall_audit.get("weak_wall_completion") if isinstance(wall_audit, dict) else None
    if not isinstance(completion, dict):
        return default
    return {
        "weak_wall_completion_used": bool(completion.get("used")),
        "weak_wall_ids": completion.get("weak_wall_ids", []),
        "completion_methods": completion.get("completion_methods", []),
        "completion_reasons": completion.get("completion_reasons", []),
        "items": completion.get("items", []),
        "wall_audit_path": str(wall_audit_path),
    }


def load_wall_final_adjudication(output_dir: Path) -> dict[str, Any]:
    wall_audit_path = output_dir / "debug" / "walls" / "wall_audit.json"
    default = {"executed": False, "decision": None, "accepted_final_walls": False, "wall_audit_path": str(wall_audit_path)}
    if not wall_audit_path.exists():
        return default
    try:
        wall_audit = json.loads(wall_audit_path.read_text(encoding="utf-8"))
    except Exception as exc:
        default["read_error"] = str(exc)
        return default
    adjudication = wall_audit.get("wall_final_adjudication") if isinstance(wall_audit, dict) else None
    if not isinstance(adjudication, dict):
        return default
    result = dict(adjudication)
    result.setdefault("executed", False)
    result.setdefault("decision", None)
    result.setdefault("accepted_final_walls", False)
    promotion = wall_audit.get("candidate_wall_promotion") if isinstance(wall_audit, dict) else None
    if isinstance(promotion, dict):
        result["candidate_modified"] = bool(promotion.get("candidate_modified"))
        result["revision_used"] = bool(promotion.get("revision_used") or promotion.get("candidate_modified"))
        result["revision_reasons"] = promotion.get("revision_reasons", [])
        result["candidate_wall_promotion"] = promotion
    else:
        result.setdefault("candidate_modified", False)
        result.setdefault("revision_used", False)
    result["wall_audit_path"] = str(wall_audit_path)
    if wall_audit.get("warnings"):
        result["wall_audit_warnings"] = wall_audit.get("warnings")
    return result


def _audit_items(data: dict[str, Any] | None) -> list[dict[str, Any]]:
    if not isinstance(data, dict):
        return []
    items = data.get("items")
    if isinstance(items, list):
        return [item for item in items if isinstance(item, dict)]
    revisions = data.get("revisions")
    if isinstance(revisions, list):
        return [item for item in revisions if isinstance(item, dict)]
    return []


def has_revision_reason(audit_data: dict[str, Any] | None, object_id: str, field_name: str) -> bool:
    for item in _audit_items(audit_data):
        item_object_id = item.get("object_id") or item.get("id")
        if item_object_id != object_id:
            continue
        fields = item.get("fields") or item.get("changed_fields") or item.get("changes") or []
        if isinstance(fields, dict):
            field_matches = field_name in fields
        elif isinstance(fields, list):
            field_matches = field_name in fields or any(isinstance(x, dict) and x.get("field") == field_name for x in fields)
        else:
            field_matches = False
        reason = item.get("reason") or item.get("rationale") or item.get("evidence") or item.get("audit_note")
        if field_matches and isinstance(reason, str) and reason.strip():
            return True
    return False


def validate_wall_geometry(walls: list[Any], epsilon: float = WALL_EPSILON) -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []
    wall_by_id = {wall.get("id"): wall for wall in walls if isinstance(wall, dict)}
    details: dict[str, Any] = {
        "epsilon": epsilon,
        "coordinate_format": "[x, z]",
        "coordinate_source_definition": "Wall geometry is validated in layout XZ coordinates; topdown/fixed views are verification and visual audit only.",
        "clockwise_sign_convention": CLOCKWISE_SIGN_CONVENTION,
        "signed_area": None,
        "interpreted_order": "unknown",
        "edge_connections": [],
        "axis_alignment_epsilon": epsilon,
        "axis_aligned": None,
        "oriented_boundary": None,
        "edge_orientations": [],
        "orientation_rule": "Final walls may be axis-aligned or oblique; geometry validation does not require x-constant/z-constant edges.",
    }

    if any(wall_id not in wall_by_id for wall_id in WALL_IDS):
        errors.append("Cannot validate wall geometry until wall_001 through wall_004 are all present.")
        return {"valid": False, "errors": errors, "warnings": warnings, "details": details}

    points: list[list[float]] = []
    for idx, wall_id in enumerate(WALL_IDS):
        wall = wall_by_id[wall_id]
        first_point = wall.get("first_point")
        second_point = wall.get("second_point")
        if not is_number_pair(first_point) or not is_number_pair(second_point):
            errors.append(f"Cannot validate geometry for `{wall_id}` because first_point/second_point is invalid.")
            continue
        points.append([float(first_point[0]), float(first_point[1])])
        next_wall = wall_by_id[WALL_IDS[(idx + 1) % len(WALL_IDS)]]
        next_first = next_wall.get("first_point")
        if is_number_pair(next_first):
            distance = _point_distance(second_point, next_first)
            connection = {
                "from": f"{wall_id}.second_point",
                "to": f"{WALL_IDS[(idx + 1) % len(WALL_IDS)]}.first_point",
                "distance": distance,
                "within_epsilon": distance <= epsilon,
            }
            details["edge_connections"].append(connection)
            if distance > epsilon:
                errors.append(
                    f"{connection['from']} does not connect to {connection['to']} within epsilon {epsilon}; distance={distance}."
                )

    alignment = wall_axis_alignment_summary(walls, epsilon=epsilon)
    details["axis_aligned"] = alignment["axis_aligned"]
    details["oriented_boundary"] = alignment["oriented_boundary"]
    details["edge_orientations"] = alignment["edge_orientations"]

    if len(points) == 4:
        signed_area = _polygon_signed_area(points)
        # In raw OpenGL XZ math, negative signed area is clockwise under the old
        # schema convention. In rendered PNG/screen coordinates the vertical axis
        # is mirrored, so the human-visible order is the opposite sign. Keep both
        # in the audit and let the canonical wall naming gate enforce screen order.
        screen_signed_area_y_down = -signed_area
        details["signed_area"] = signed_area
        details["signed_area_xz"] = signed_area
        details["signed_area_screen_y_down_fallback"] = screen_signed_area_y_down
        if abs(signed_area) <= epsilon:
            details["interpreted_order"] = "degenerate_or_too_small"
            details["interpreted_order_xz"] = "degenerate_or_too_small"
            details["interpreted_order_screen_y_down_fallback"] = "degenerate_or_too_small"
            errors.append("Wall polygon signed area is too small; cannot confirm a valid closed room polygon.")
        else:
            details["interpreted_order_xz"] = "clockwise" if signed_area < 0 else "counter_clockwise"
            details["interpreted_order_screen_y_down_fallback"] = "clockwise" if screen_signed_area_y_down > 0 else "counter_clockwise"
            details["interpreted_order"] = details["interpreted_order_xz"]
            if signed_area > 0:
                warnings.append(
                    "Wall polygon is counter-clockwise in raw layout XZ, but this may be intentional after screen-clockwise canonical wall naming; "
                    f"signed_area_xz={signed_area}."
                )
    else:
        errors.append("Wall polygon does not have four usable first_point vertices.")

    return {"valid": not errors, "errors": errors, "warnings": warnings, "details": details}


def validate_objects_payload(
    payload: Any,
    scene_state: dict[str, Any] | None = None,
    object_revision_audit: dict[str, Any] | None = None,
    wall_epsilon: float = WALL_EPSILON,
) -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []
    object_ids: list[str] = []
    if not isinstance(payload, dict):
        return {"valid": False, "errors": ["objects.json top-level must be an object."], "warnings": [], "object_ids": []}

    if "walls" not in payload:
        errors.append("objects.json is missing `walls`.")
    if "room_materials" not in payload:
        errors.append("objects.json is missing `room_materials`.")
    if "objects" not in payload:
        errors.append("objects.json is missing `objects`.")
    extra_top = sorted(set(payload.keys()) - {"walls", "room_materials", "objects"})
    if extra_top:
        warnings.append(f"objects.json has extra top-level keys: {extra_top}")

    walls = payload.get("walls")
    objects = payload.get("objects")
    if not isinstance(walls, list):
        errors.append("`walls` must be a list.")
        walls = []
    room_materials = payload.get("room_materials")
    if "room_materials" in payload:
        if not isinstance(room_materials, dict):
            errors.append("`room_materials` must be an object.")
        else:
            material_keys = set(room_materials.keys())
            missing_material_keys = [key for key in ROOM_MATERIAL_KEYS if key not in material_keys]
            extra_material_keys = sorted(material_keys - set(ROOM_MATERIAL_KEYS))
            if missing_material_keys:
                errors.append(f"`room_materials` is missing required keys: {missing_material_keys}.")
            if extra_material_keys:
                errors.append(f"`room_materials` has unexpected keys: {extra_material_keys}.")
            for key in ROOM_MATERIAL_KEYS:
                value = room_materials.get(key)
                if not isinstance(value, str) or not value.strip():
                    errors.append(f"`room_materials.{key}` must be a non-empty string.")
    if not isinstance(objects, list):
        errors.append("`objects` must be a list.")
        objects = []

    wall_id_set = {wall.get("id") for wall in walls if isinstance(wall, dict)}
    for wall_id in WALL_IDS:
        if wall_id not in wall_id_set:
            errors.append(f"Missing required wall id `{wall_id}`.")
    for wall in walls:
        if not isinstance(wall, dict):
            errors.append("Each wall must be an object.")
            continue
        keys = set(wall.keys())
        if keys != {"id", "first_point", "second_point"}:
            errors.append(f"Wall `{wall.get('id')}` must contain only id/first_point/second_point.")
        if wall.get("id") not in WALL_IDS:
            errors.append(f"Unexpected wall id `{wall.get('id')}`.")
        if not is_number_pair(wall.get("first_point")):
            errors.append(f"Wall `{wall.get('id')}` first_point must be a numeric [x, z] pair.")
        if not is_number_pair(wall.get("second_point")):
            errors.append(f"Wall `{wall.get('id')}` second_point must be a numeric [x, z] pair.")

    wall_geometry = validate_wall_geometry(walls, epsilon=wall_epsilon)
    errors.extend(wall_geometry["errors"])
    warnings.extend(wall_geometry["warnings"])

    previous_key = None
    seen_object_ids = set()
    final_by_id: dict[str, dict[str, Any]] = {}
    for obj in objects:
        if not isinstance(obj, dict):
            errors.append("Each object must be an object.")
            continue
        keys = set(obj.keys())
        base_object_keys = {"id", "category", "description", "upright", "physics"}
        contain_anchor_keys = {"contain_anchor", "render_direction"}
        if frozenset(keys) not in {frozenset(base_object_keys), frozenset(base_object_keys | contain_anchor_keys)}:
            errors.append(
                f"Object `{obj.get('id')}` must contain id/category/description/upright/physics, "
                "with contain_anchor/render_direction present together only for contain anchors."
            )
        object_id = obj.get("id")
        object_ids.append(object_id)
        if not isinstance(object_id, str) or not OBJECT_ID_RE.fullmatch(object_id):
            errors.append(f"Object id `{object_id}` must match object_<number>, for example object_001.")
        if object_id in seen_object_ids:
            errors.append(f"Duplicate object id `{object_id}` in final objects.json.")
        seen_object_ids.add(object_id)
        final_by_id[object_id] = obj
        key = sorted_object_id_key(object_id)
        if previous_key and key < previous_key:
            errors.append("Objects must be sorted by the numeric suffix of object id ascending.")
        previous_key = key
        if not is_snake_case(obj.get("category")):
            errors.append(f"Object `{object_id}` category must be snake_case.")
        if not isinstance(obj.get("description"), str):
            errors.append(f"Object `{object_id}` description must be a string.")
        if not isinstance(obj.get("upright"), bool):
            errors.append(f"Object `{object_id}` upright must be a boolean.")
        has_contain_fields = "contain_anchor" in obj or "render_direction" in obj
        if has_contain_fields:
            contain_anchor = obj.get("contain_anchor")
            render_direction = obj.get("render_direction")
            if contain_anchor is not True:
                errors.append(f"Object `{object_id}` may include contain_anchor only with value true.")
            if render_direction not in {"front", "up"}:
                errors.append(f"Contain anchor `{object_id}` render_direction must be `front` or `up`.")
        physics = obj.get("physics")
        if not isinstance(physics, dict):
            errors.append(f"Object `{object_id}` physics must be an object.")
        else:
            if set(physics.keys()) != set(PHYSICS_FIELDS):
                errors.append(
                    f"Object `{object_id}` physics must contain only: {list(PHYSICS_FIELDS)}."
                )
            for field in PHYSICS_FIELDS:
                value = physics.get(field)
                lower, upper = FIELD_LIMITS[field]
                if (
                    isinstance(value, bool)
                    or not isinstance(value, (int, float))
                    or not math.isfinite(float(value))
                    or not lower <= float(value) <= upper
                ):
                    errors.append(
                        f"Object `{object_id}` physics.{field} must be finite within [{lower}, {upper}]."
                    )
            static_friction = physics.get("static_friction")
            dynamic_friction = physics.get("dynamic_friction")
            if (
                isinstance(static_friction, (int, float))
                and not isinstance(static_friction, bool)
                and isinstance(dynamic_friction, (int, float))
                and not isinstance(dynamic_friction, bool)
                and float(dynamic_friction) > float(static_friction)
            ):
                errors.append(
                    f"Object `{object_id}` physics.dynamic_friction cannot exceed static_friction."
                )

    scene_candidates = extract_scene_state_candidates(scene_state)
    scene_candidate_ids = sorted(scene_candidates.keys(), key=sorted_object_id_key)
    if scene_candidates:
        final_id_set = set(object_ids)
        scene_id_set = set(scene_candidates.keys())
        unexpected = sorted(final_id_set - scene_id_set, key=sorted_object_id_key)
        missing = sorted(scene_id_set - final_id_set, key=sorted_object_id_key)
        if unexpected:
            errors.append(f"Final objects contain ids not present in scene_state candidates: {unexpected}.")
        if missing:
            errors.append(f"Final objects are missing scene_state candidate ids: {missing}.")
        if not unexpected and not missing:
            expected_order = scene_candidate_ids
            if object_ids != expected_order:
                errors.append(
                    "Final objects id set matches scene_state but order is not numeric-suffix ascending as inherited from scene_state."
                )
        for object_id in sorted(final_id_set & scene_id_set, key=sorted_object_id_key):
            final_obj = final_by_id.get(object_id, {})
            candidate = scene_candidates[object_id]
            for field_name in ["category", "description"]:
                if final_obj.get(field_name) != candidate.get(field_name):
                    if not has_revision_reason(object_revision_audit, object_id, field_name):
                        warnings.append(
                            f"Object `{object_id}` changed `{field_name}` from scene_state candidate but no reason was found in object_revision_audit."
                        )

    return {
        "valid": not errors,
        "errors": errors,
        "warnings": warnings,
        "object_ids": object_ids,
        "scene_state_candidate_ids": scene_candidate_ids,
        "wall_geometry": wall_geometry,
    }


def validate_relations_payload(payload: Any, object_ids: list[str], ceiling_confirmed: bool = False) -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []
    if not isinstance(payload, list):
        return {"valid": False, "errors": ["relations.json top-level must be a list."], "warnings": []}

    allowed_attach_anchors = {"floor", *WALL_IDS, *object_ids}
    if ceiling_confirmed:
        allowed_attach_anchors.add("ceiling")

    seen = set()
    seen_targets = set()
    last_target = None
    closed_targets = set()
    pair_relation_types: dict[tuple[Any, Any], set[str]] = {}
    for idx, rel in enumerate(payload):
        if not isinstance(rel, dict):
            errors.append(f"Relation at index {idx} must be an object.")
            continue
        relation_type = rel.get("relation")
        if relation_type not in {"attach", "contain"}:
            errors.append(f"Relation at index {idx} must have relation=`attach` or relation=`contain`.")
        params = rel.get("parameters")
        if not isinstance(params, dict):
            errors.append(f"Relation at index {idx} parameters must be an object.")
            continue
        forbidden = sorted(FORBIDDEN_PARAM_KEYS & set(params.keys()))
        if forbidden:
            errors.append(f"Relation at index {idx} uses forbidden parameter keys: {forbidden}.")
        if set(params.keys()) != {"object_anchor", "object_target"}:
            errors.append(f"Relation at index {idx} parameters must contain exactly object_anchor/object_target.")
        anchor = params.get("object_anchor")
        target = params.get("object_target")
        if target not in object_ids:
            errors.append(f"Relation at index {idx} object_target `{target}` is not in final objects.")
        else:
            seen_targets.add(target)
        if relation_type == "attach":
            if anchor == "ceiling" and not ceiling_confirmed:
                errors.append("Ceiling anchor is forbidden because ceiling was not confirmed present.")
            if anchor not in allowed_attach_anchors:
                errors.append(f"Relation at index {idx} object_anchor `{anchor}` is not allowed for attach.")
        elif relation_type == "contain":
            if anchor not in object_ids:
                errors.append(f"Relation at index {idx} contain object_anchor `{anchor}` must be a final object id.")
            if anchor == target:
                errors.append(f"Relation at index {idx} contain anchor and target cannot be the same object `{anchor}`.")
        triple = (relation_type, anchor, target)
        if triple in seen:
            errors.append(f"Duplicate relation triple found at index {idx}: {triple}.")
        seen.add(triple)
        pair_relation_types.setdefault((anchor, target), set()).add(str(relation_type))
        if target != last_target:
            if target in closed_targets:
                warnings.append(f"Relations for target `{target}` are not consecutive.")
            if last_target is not None:
                closed_targets.add(last_target)
            last_target = target

    for pair, relation_types in sorted(pair_relation_types.items(), key=lambda item: (str(item[0][1]), str(item[0][0]))):
        if "attach" in relation_types and "contain" in relation_types:
            errors.append(f"Pair {pair} has both attach and contain; resolve the pair to exactly one relation without a fixed relation-type priority.")

    missing_targets = sorted(set(object_ids) - seen_targets, key=sorted_object_id_key)
    for object_id in missing_targets:
        errors.append(f"Object {object_id} has no attach or contain relation as object_target.")

    return {
        "valid": not errors,
        "errors": errors,
        "warnings": warnings,
        "target_coverage": {
            "required_object_ids": object_ids,
            "covered_object_targets": sorted(seen_targets, key=sorted_object_id_key),
            "missing_object_targets": missing_targets,
            "complete": not missing_targets,
        },
    }


def validate_contain_anchor_annotations(objects_payload: Any, relations_payload: Any) -> dict[str, Any]:
    errors: list[str] = []
    objects = objects_payload.get("objects", []) if isinstance(objects_payload, dict) else []
    relations = relations_payload if isinstance(relations_payload, list) else []
    object_by_id = {
        str(obj.get("id")): obj
        for obj in objects
        if isinstance(obj, dict) and obj.get("id")
    }
    relation_anchor_ids = {
        str(rel.get("parameters", {}).get("object_anchor"))
        for rel in relations
        if isinstance(rel, dict)
        and rel.get("relation") == "contain"
        and isinstance(rel.get("parameters"), dict)
        and str(rel.get("parameters", {}).get("object_anchor", "")).startswith("object_")
    }
    annotated_ids = {
        object_id
        for object_id, obj in object_by_id.items()
        if "contain_anchor" in obj or "render_direction" in obj
    }
    for object_id in sorted(relation_anchor_ids):
        obj = object_by_id.get(object_id)
        if obj is None:
            continue
        if obj.get("contain_anchor") is not True or obj.get("render_direction") not in {"front", "up"}:
            errors.append(
                f"Contain relation anchor `{object_id}` must include contain_anchor=true and render_direction=front|up."
            )
    for object_id in sorted(annotated_ids - relation_anchor_ids):
        errors.append(
            f"Non-contain anchor `{object_id}` must omit contain_anchor and render_direction."
        )
    return {
        "valid": not errors,
        "errors": errors,
        "contain_relation_anchor_ids": sorted(relation_anchor_ids),
        "annotated_object_ids": sorted(annotated_ids),
    }


def validate_files(
    objects_path: Path,
    relations_path: Path,
    output_dir: Path,
    ceiling_confirmed: bool = False,
    dry_run: bool = False,
    scene_state: dict[str, Any] | None = None,
    scene_state_path: Path | None = None,
    object_revision_audit_path: Path | None = None,
    wall_epsilon: float = WALL_EPSILON,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "dry_run": dry_run,
        "objects_path": str(objects_path),
        "relations_path": str(relations_path),
        "ceiling_confirmed": ceiling_confirmed,
        "scene_state_path": str(scene_state_path) if scene_state_path else None,
        "objects": None,
        "relations": None,
        "valid": False,
    }
    if scene_state is None and scene_state_path and scene_state_path.exists():
        try:
            scene_state = json.loads(scene_state_path.read_text(encoding="utf-8"))
        except Exception as exc:
            result.setdefault("warnings", []).append(f"Failed to read scene_state for strict object id validation: {exc}")
    if object_revision_audit_path is None:
        object_revision_audit_path = output_dir / "debug" / "objects" / "object_revision_audit.json"
    object_revision_audit = load_revision_audit(object_revision_audit_path)
    result["object_revision_audit_path"] = str(object_revision_audit_path)
    result["wall_final_adjudication"] = load_wall_final_adjudication(output_dir)
    result["weak_wall_completion"] = load_weak_wall_completion(output_dir)

    if dry_run:
        result.update({"valid": False, "skipped": True, "reason": "Dry run does not validate or generate real final objects.json/relations.json."})
        write_audit_pair(output_dir, "final_validation", "schema_validation", result, "Schema Validation")
        return result

    try:
        objects_payload = json.loads(objects_path.read_text(encoding="utf-8"))
    except Exception as exc:
        result["objects"] = {"valid": False, "errors": [f"Failed to read objects.json: {exc}"], "warnings": [], "object_ids": []}
        objects_payload = None
    if objects_payload is not None:
        result["objects"] = validate_objects_payload(
            objects_payload,
            scene_state=scene_state,
            object_revision_audit=object_revision_audit,
            wall_epsilon=wall_epsilon,
        )
        wall_geometry = result.get("objects", {}).get("wall_geometry") if isinstance(result.get("objects"), dict) else None
        details = wall_geometry.get("details") if isinstance(wall_geometry, dict) else None
        if isinstance(details, dict):
            weak_completion = result.get("weak_wall_completion", {})
            details["weak_wall_completion_used"] = bool(weak_completion.get("weak_wall_completion_used"))
            details["weak_wall_ids"] = weak_completion.get("weak_wall_ids", [])
            details["completion_methods"] = weak_completion.get("completion_methods", [])
            details["completion_reasons"] = weak_completion.get("completion_reasons", [])

    try:
        relations_payload = json.loads(relations_path.read_text(encoding="utf-8"))
    except Exception as exc:
        result["relations"] = {"valid": False, "errors": [f"Failed to read relations.json: {exc}"], "warnings": []}
        relations_payload = None
    if relations_payload is not None:
        object_ids = result["objects"].get("object_ids", []) if result.get("objects") else []
        result["relations"] = validate_relations_payload(relations_payload, object_ids, ceiling_confirmed=ceiling_confirmed)
        coverage_audit_path = output_dir / "debug" / "attach" / "coverage_repair_audit.json"
        if coverage_audit_path.exists():
            try:
                coverage_audit = json.loads(coverage_audit_path.read_text(encoding="utf-8"))
            except Exception:
                coverage_audit = {}
            result["relations"]["coverage_repair_used"] = bool(coverage_audit.get("coverage_repair_used"))
            result["relations"]["coverage_repair_count"] = int(coverage_audit.get("coverage_repair_count") or 0)
            result["relations"]["coverage_repair_audit_path"] = str(coverage_audit_path)

    contain_anchor_validation = validate_contain_anchor_annotations(objects_payload, relations_payload)
    result["contain_anchor_annotations"] = contain_anchor_validation
    if not contain_anchor_validation["valid"]:
        if isinstance(result.get("objects"), dict):
            result["objects"].setdefault("errors", []).extend(contain_anchor_validation["errors"])
            result["objects"]["valid"] = False

    result["valid"] = bool(result.get("objects", {}).get("valid") and result.get("relations", {}).get("valid"))
    write_audit_pair(output_dir, "final_validation", "schema_validation", result, "Schema Validation")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate multi-agent scene delivery final outputs.")
    parser.add_argument("--objects_json", required=True, type=Path)
    parser.add_argument("--relations_json", required=True, type=Path)
    parser.add_argument("--output_dir", required=True, type=Path)
    parser.add_argument("--scene_state", type=Path, default=None)
    parser.add_argument("--object_revision_audit", type=Path, default=None)
    parser.add_argument("--wall_epsilon", type=float, default=WALL_EPSILON)
    parser.add_argument("--ceiling_confirmed", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()
    result = validate_files(
        args.objects_json,
        args.relations_json,
        args.output_dir,
        args.ceiling_confirmed,
        args.dry_run,
        scene_state_path=args.scene_state,
        object_revision_audit_path=args.object_revision_audit,
        wall_epsilon=args.wall_epsilon,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
