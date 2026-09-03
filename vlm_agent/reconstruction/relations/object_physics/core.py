"""Deterministic validation and fallback for object physics estimates."""

from __future__ import annotations

from functools import lru_cache
import math
from typing import Any


PHYSICS_FIELDS = (
    "mass",
    "linear_damping",
    "angular_damping",
    "static_friction",
    "dynamic_friction",
)

UPRIGHT_FIELD = "upright"

FIELD_LIMITS = {
    "mass": (0.01, 10000.0),
    "linear_damping": (0.0, 1.0),
    "angular_damping": (0.0, 1.0),
    "static_friction": (0.0, 2.0),
    "dynamic_friction": (0.0, 2.0),
}

DEFAULT_PROFILE = {
    "mass": 5.0,
    "linear_damping": 0.08,
    "angular_damping": 0.12,
    "static_friction": 0.55,
    "dynamic_friction": 0.4,
}

PROFILES = [
    (("rug", "carpet", "mat", "floor covering"), {"mass": 8.0, "linear_damping": 0.3, "angular_damping": 0.5, "static_friction": 0.9, "dynamic_friction": 0.75}),
    (("sofa", "couch", "bed"), {"mass": 55.0, "linear_damping": 0.12, "angular_damping": 0.2, "static_friction": 0.75, "dynamic_friction": 0.58}),
    (("cabinet", "wardrobe", "bookshelf", "refrigerator"), {"mass": 75.0, "linear_damping": 0.1, "angular_damping": 0.18, "static_friction": 0.72, "dynamic_friction": 0.55}),
    (("desk", "table", "counter", "console"), {"mass": 30.0, "linear_damping": 0.09, "angular_damping": 0.15, "static_friction": 0.68, "dynamic_friction": 0.5}),
    (("chair", "stool", "ottoman", "bench"), {"mass": 10.0, "linear_damping": 0.08, "angular_damping": 0.14, "static_friction": 0.62, "dynamic_friction": 0.46}),
    (("lamp", "floor light"), {"mass": 6.0, "linear_damping": 0.08, "angular_damping": 0.22, "static_friction": 0.7, "dynamic_friction": 0.48}),
    (("television", "projection screen", "painting", "panel", "whiteboard"), {"mass": 7.0, "linear_damping": 0.12, "angular_damping": 0.18, "static_friction": 0.55, "dynamic_friction": 0.4}),
    (("monitor", "speaker", "computer", "appliance", "equipment"), {"mass": 6.0, "linear_damping": 0.08, "angular_damping": 0.13, "static_friction": 0.58, "dynamic_friction": 0.42}),
    (("book", "remote", "figurine", "decoration", "box", "vase"), {"mass": 0.8, "linear_damping": 0.06, "angular_damping": 0.1, "static_friction": 0.5, "dynamic_friction": 0.36}),
    (("ceiling light", "recessed light"), {"mass": 2.5, "linear_damping": 0.12, "angular_damping": 0.18, "static_friction": 0.5, "dynamic_friction": 0.35}),
]


def _object_text(obj: dict[str, Any]) -> str:
    return f"{obj.get('category', '')} {obj.get('description', '')}".lower().replace("_", " ")


WALL_HANGING_UPRIGHT_FALSE_TERMS = {
    "artwork",
    "framed art",
    "hanging art",
    "painting",
    "wall art",
    "wall clock",
    "wall decoration",
    "wall mirror",
    "wall mounted",
    "wall panel",
    "wall sign",
    "whiteboard",
}

WALL_HANGING_UPRIGHT_PREFILTER_TERMS = ("wall", "art", "hang", "paint", "whiteboard")

STABLE_UPRIGHT_TRUE_CATEGORY_TERMINALS = {
    "basin",
    "basket",
    "stand",
    "vessel",
}

STABLE_UPRIGHT_FALSE_CATEGORY_TERMINALS = {
    "controller",
    "cushion",
    "pillow",
    "slab",
}

STABLE_UPRIGHT_TRUE_COMPOUND_CATEGORIES = {
    "slab_roller",
}
STABLE_UPRIGHT_TRUE_COMPOUND_SUFFIXES = tuple(f"_{value}" for value in STABLE_UPRIGHT_TRUE_COMPOUND_CATEGORIES)


@lru_cache(maxsize=512)
def _stable_upright_category_override(category_value: str) -> tuple[bool, str] | None:
    category = category_value.strip().lower().replace("-", "_").replace(" ", "_").strip("_")
    terminal = category.rsplit("_", 1)[-1]
    if category in STABLE_UPRIGHT_TRUE_COMPOUND_CATEGORIES or category.endswith(STABLE_UPRIGHT_TRUE_COMPOUND_SUFFIXES):
        return True, "rigid functional machine category has a gravity-defined functional up direction"
    if terminal in STABLE_UPRIGHT_TRUE_CATEGORY_TERMINALS:
        return True, "rigid container/support category has a gravity-defined functional up direction"
    if terminal == "box":
        if category == "box":
            return False, "generic loose box category has no required gravity-upright constraint"
        return True, "typed rigid box category has a gravity-defined functional up direction"
    if terminal in STABLE_UPRIGHT_FALSE_CATEGORY_TERMINALS:
        return False, "soft, handheld, or intentionally flat category has no gravity-upright constraint"
    if category in {"bean_bag", "cooling_rack"} or category.endswith(("_bean_bag", "_cooling_rack")):
        return False, "soft, handheld, or intentionally flat category has no gravity-upright constraint"
    return None


def local_upright_semantic_override(obj: dict[str, Any]) -> tuple[bool, str] | None:
    """Return only category-stable relation-agnostic upright overrides."""
    category_override = _stable_upright_category_override(str(obj.get("category") or ""))
    if category_override is not None:
        return category_override
    text = _object_text(obj)
    if any(marker in text for marker in WALL_HANGING_UPRIGHT_PREFILTER_TERMS):
        if any(term in text for term in WALL_HANGING_UPRIGHT_FALSE_TERMS):
            return False, "thin wall-hanging decorative/display object is not a gravity-upright constraint"
    return None


def local_physics_fallback(obj: dict[str, Any]) -> tuple[dict[str, float], str]:
    """Return a broad semantic prior, not a measured physical property."""
    category = str(obj.get("category") or "").lower().replace("_", " ")
    text = _object_text(obj)
    for keywords, profile in PROFILES:
        if any(keyword in category for keyword in keywords):
            return dict(profile), f"local semantic category profile matched: {category}"
    for keywords, profile in PROFILES:
        if any(keyword in text for keyword in keywords):
            return dict(profile), f"local semantic description profile matched: {', '.join(keyword for keyword in keywords if keyword in text)}"
    return dict(DEFAULT_PROFILE), "local generic object profile"


def _finite_number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def extract_agent_items(payload: Any) -> list[dict[str, Any]]:
    if not isinstance(payload, dict):
        return []
    for key in ("object_physics", "objects", "physics_properties", "items"):
        value = payload.get(key)
        if isinstance(value, list):
            return [item for item in value if isinstance(item, dict)]
        if isinstance(value, dict) and isinstance(value.get("items"), list):
            return [item for item in value["items"] if isinstance(item, dict)]
    findings = payload.get("findings")
    if isinstance(findings, dict):
        return extract_agent_items(findings)
    return []


def resolve_object_physics(
    objects_payload: dict[str, Any],
    agent_payload: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Resolve complete physics values and return sidecar, enriched objects, audit."""
    objects = objects_payload.get("objects") if isinstance(objects_payload, dict) else None
    if not isinstance(objects, list):
        raise ValueError("objects payload must contain an objects list")
    final_ids = {str(obj.get("id")) for obj in objects if isinstance(obj, dict) and obj.get("id")}
    agent_by_id: dict[str, dict[str, Any]] = {}
    ignored_agent_items: list[dict[str, Any]] = []
    for item in extract_agent_items(agent_payload or {}):
        object_id = str(item.get("object_id") or item.get("id") or "")
        if object_id in final_ids:
            agent_by_id[object_id] = item
        else:
            ignored_agent_items.append({"item": item, "reason": "object_id_not_in_final_objects"})

    sidecar_items: list[dict[str, Any]] = []
    enriched_objects: list[dict[str, Any]] = []
    audit_items: list[dict[str, Any]] = []
    invalid_agent_upright_object_ids: list[str] = []
    for obj in objects:
        if not isinstance(obj, dict) or not obj.get("id"):
            continue
        object_id = str(obj["id"])
        fallback, fallback_reason = local_physics_fallback(obj)
        agent = agent_by_id.get(object_id, {})
        nested = agent.get("physics") if isinstance(agent.get("physics"), dict) else {}
        resolved: dict[str, float] = {}
        sources: dict[str, str] = {}
        rejected: list[dict[str, Any]] = []
        for field in PHYSICS_FIELDS:
            raw = agent.get(field, nested.get(field))
            number = _finite_number(raw)
            lower, upper = FIELD_LIMITS[field]
            if number is not None and lower <= number <= upper:
                resolved[field] = round(number, 6)
                sources[field] = "agent"
            else:
                resolved[field] = float(fallback[field])
                sources[field] = "local_fallback"
                if raw is not None:
                    rejected.append({"field": field, "value": raw, "reason": f"must be finite and within [{lower}, {upper}]"})

        if resolved["dynamic_friction"] > resolved["static_friction"]:
            rejected.append({
                "field": "dynamic_friction",
                "value": resolved["dynamic_friction"],
                "reason": "dynamic_friction cannot exceed static_friction; clamped to static_friction",
            })
            resolved["dynamic_friction"] = resolved["static_friction"]
            sources["dynamic_friction"] = f"{sources['dynamic_friction']}_clamped"

        agent_upright = agent.get(UPRIGHT_FIELD)
        input_upright = obj.get(UPRIGHT_FIELD)
        resolved_upright: bool | None = None
        upright_source = "missing"
        upright_reason = None
        if agent_upright is not None and not isinstance(agent_upright, bool):
            invalid_agent_upright_object_ids.append(object_id)
            rejected.append({
                "field": UPRIGHT_FIELD,
                "value": agent_upright,
                "reason": "upright must be a JSON boolean",
            })
        semantic_upright_override = local_upright_semantic_override(obj)
        if semantic_upright_override is not None:
            resolved_upright, upright_reason = semantic_upright_override
            upright_source = "local_semantic_override"
        elif isinstance(agent_upright, bool):
            resolved_upright = agent_upright
            upright_source = "agent"
        elif isinstance(input_upright, bool):
            resolved_upright = input_upright
            upright_source = "input_object_fallback"

        sidecar_item = {
            "object_id": object_id,
            **resolved,
        }
        enriched_object = {**obj, "physics": resolved}
        if resolved_upright is not None:
            sidecar_item[UPRIGHT_FIELD] = resolved_upright
            enriched_object[UPRIGHT_FIELD] = resolved_upright
        sidecar_items.append(sidecar_item)
        enriched_objects.append(enriched_object)
        audit_items.append({
            "object_id": object_id,
            "resolved": resolved,
            "field_sources": sources,
            "upright": resolved_upright,
            "upright_source": upright_source,
            "upright_reason": upright_reason,
            "fallback_reason": fallback_reason,
            "agent_item_present": bool(agent),
            "rejected_agent_values": rejected,
        })

    sidecar = {
        "schema_version": "object_physics",
        "mass_unit": "kg_estimate",
        "damping_range": [0.0, 1.0],
        "friction_model": "dimensionless_coulomb_estimate",
        "objects": sidecar_items,
    }
    enriched = {**objects_payload, "objects": enriched_objects}
    audit = {
        "schema_version": "object_physics_audit",
        "status": "completed",
        "object_count": len(enriched_objects),
        "agent_item_count": len(agent_by_id),
        "fallback_only_object_count": sum(1 for item in audit_items if not item["agent_item_present"]),
        "invalid_agent_upright_object_ids": invalid_agent_upright_object_ids,
        "ignored_agent_items": ignored_agent_items,
        "items": audit_items,
        "policy": "Agent physics estimates are accepted only inside numeric bounds. Missing/invalid numeric fields use deterministic local semantic priors; dynamic friction cannot exceed static friction. Agent upright is retained only when it is a JSON boolean; relation-aware overrides remain the final assembler's responsibility.",
    }
    return sidecar, enriched, audit


def validate_object_physics(payload: dict[str, Any], expected_object_ids: list[str]) -> list[str]:
    errors: list[str] = []
    items = payload.get("objects") if isinstance(payload, dict) else None
    if not isinstance(items, list):
        return ["object_physics.json must contain an objects list"]
    ids = [str(item.get("object_id")) for item in items if isinstance(item, dict)]
    if ids != expected_object_ids:
        errors.append(f"physics object ids/order mismatch: expected {expected_object_ids}, got {ids}")
    for item in items:
        if not isinstance(item, dict):
            errors.append("each physics item must be an object")
            continue
        for field in PHYSICS_FIELDS:
            number = _finite_number(item.get(field))
            lower, upper = FIELD_LIMITS[field]
            if number is None or not lower <= number <= upper:
                errors.append(f"{item.get('object_id')} {field} must be finite within [{lower}, {upper}]")
        dynamic = _finite_number(item.get("dynamic_friction"))
        static = _finite_number(item.get("static_friction"))
        if dynamic is not None and static is not None and dynamic > static:
            errors.append(f"{item.get('object_id')} dynamic_friction exceeds static_friction")
    return errors

def validate_object_upright_priors(payload: dict[str, Any], expected_object_ids: list[str]) -> list[str]:
    """Validate the sidecar's relation-agnostic upright prior for each object."""
    errors: list[str] = []
    items = payload.get("objects") if isinstance(payload, dict) else None
    if not isinstance(items, list):
        return ["object_physics.json must contain an objects list"]
    by_id = {
        str(item.get("object_id")): item
        for item in items
        if isinstance(item, dict) and item.get("object_id")
    }
    for object_id in expected_object_ids:
        item = by_id.get(str(object_id))
        if item is None:
            errors.append(f"{object_id} is missing from sidecar upright priors")
        elif not isinstance(item.get(UPRIGHT_FIELD), bool):
            errors.append(f"{object_id} upright must be a JSON boolean")
    return errors

