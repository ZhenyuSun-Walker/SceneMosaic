from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any

from relations.agents.schema_guard import WALL_IDS
from relations.agents.assembly.containment import (
    CONTAIN_CONFLICT_MIN_LOCAL_CONFIDENCE,
    CONTAIN_INTERNAL_SURFACE_ROLE,
    CONTAIN_EXTERNAL_SURFACE_ROLES,
    _target_coverage,
    _contain_geometry_index,
    _contain_point_xz,
    _contain_center_y,
    _contain_y_bounds,
    _contain_hull_xz,
    _contain_rotate_xz,
    _contain_min_area_angle,
    _axis_outside_distance,
    _contain_geometry_decision,
    filter_contain_relations_by_geometry,
    _normalized_contain_surface_role,
    _contain_conflict_id,
    _contain_candidate_pairs_from_scene_state,
    _positive_containment_hypothesis,
    _contain_rejections_from_task_results,
    collect_multi_level_support_contain_promotions,
    _contain_uncertainties_from_task_results,
    _accepted_object_relation_hypotheses,
    _contain_conflict_object_metadata,
    collect_contain_conflicts,
    compact_contain_conflicts_for_guard,
    collect_contain_conflict_resolutions,
    resolve_attach_contain_conflicts,
    filter_redundant_floor_attaches_for_object_supported_targets,
    normalize_floor_covering_relations,
)
from relations.agents.assembly.relation_candidates import (
    _candidate_decision,
    _relation_from_candidate,
    _repair_inconsistent_support_candidate,
    _relation_from_accepted_support_candidate,
    collect_accepted_attach_candidates,
    collect_coverage_repair_candidates,
)
from relations.agents.assembly.wall_assembly import (
    FORMAL_OBB_WALL_METHOD,
    FORMAL_OBB_WALL_SOURCE,
    ALLOWED_FINAL_WALL_SOURCES,
    INTERMEDIATE_WALL_SOURCES,
    HINT_ONLY_WALL_SOURCES,
    DEFAULT_WALL_PROMOTION_EVIDENCE,
    WALL_SEMANTIC_CONFLICT_RESOLUTION_PRIORITY,
    _wall_source_from_context,
    _wall_verification_status,
    _truthy,
    _listify_strings,
    _candidate_promotion_audit,
    _normalize_wall_source,
    _wall_source_allowed,
    _normalize_wall_set,
    _is_wall_final_adjudication_node,
    _wall_final_adjudication_decision,
    summarize_wall_final_adjudication,
    collect_walls,
    _wall_semantic_source_priority,
    collect_wall_final_semantics,
    apply_wall_final_semantics,
    summarize_candidate_wall_promotion,
    summarize_weak_wall_completion,
    weak_or_inferred_wall_attach_blocks,
    shift_walls_from_attach_policy,
    _safe_float_pair,
    _load_xz_to_pixel_affine_from_wall_evidence,
    _xz_to_screen_xy,
    _signed_area_xy,
    _segment_midpoint_xy,
    _point_close,
    _match_old_wall_id,
    _record_by_old_wall_id,
    _wall_record_aliases,
    _wall_geometry_aliases,
    _record_by_wall_geometry_alias,
    build_wall_relation_id_remap,
    _visible_count_from_records,
    _formal_obb_primary_candidate,
    _authoritative_obb_walls_from_evidence,
    canonicalize_walls_for_screen_clockwise,
    write_wall_audit,
)
from relations.agents.assembly.object_properties import (
    ROOM_MATERIAL_KEYS,
    _OBJECT_REVISION_CONFIDENCE_SCORES,
    _OBJECT_REVISION_SOURCE_PRIORITIES,
    FLOOR_COVERING_TEXT_HINTS,
    UPRIGHT_FALSE_TEXT_HINTS,
    UPRIGHT_IMAGE_AXIS_TEXT_HINTS,
    UPRIGHT_TRUE_TEXT_HINTS,
    CONTAIN_UP_DIRECTION_HINTS,
    CONTAIN_FRONT_DIRECTION_HINTS,
    _snake_case,
    candidate_objects,
    _walk,
    _walk_with_path,
    normalize_room_materials,
    collect_room_materials,
    _normalize_object_revision_confidence,
    _resolve_object_revision_patches,
    collect_object_revision_patches,
    apply_object_revision_patches,
    write_object_revision_audit,
    _bool_or_none,
    _scene_object_context,
    _object_text,
    _bbox_extent_xyz,
    _bbox_2d_height_width_ratio,
    _local_floor_covering_hint,
    _matches_semantic_hint,
    _local_upright_semantic_hint,
    _local_upright_fallback,
    collect_object_physical_properties,
    write_object_physical_properties_audit,
    load_async_sidecar_upright_priors,
    apply_upright_properties,
    _normalize_contain_render_direction,
    _semantic_contain_render_direction,
    _contain_candidate_anchor_directions,
    _agent_contain_anchor_directions,
    apply_contain_anchor_properties,
    _is_ceiling_light_object,
)
from relations.agents.state import sorted_object_id_key, write_audit_pair, write_json, write_md
from relations.object_physics.core import resolve_object_physics, validate_object_physics

WALL_MOUNTED_KEEP_KEYWORDS = {
    "attached to wall",
    "embedded",
    "mounted",
    "wall mounted",
    "wall-mounted",
    "wall-hung",
}

WALL_MOUNTED_CATEGORIES = {
    "acoustic_panel",
    "art_panel",
    "bulletin_board",
    "painting",
    "projection_screen",
    "wall_art",
    "wall_panel",
    "whiteboard",
}










def _object_context_by_id(objects: list[dict[str, Any]] | None) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for obj in objects or []:
        if isinstance(obj, dict) and obj.get("id"):
            result[str(obj["id"])] = obj
    return result


def _alias_key(value: Any) -> str:
    return re.sub(r"[^0-9a-z]+", "_", str(value or "").lower()).strip("_")


def _object_anchor_alias_map(objects: list[dict[str, Any]] | None) -> dict[str, str]:
    alias_to_ids: dict[str, set[str]] = {}

    def add(alias: Any, object_id: str) -> None:
        key = _alias_key(alias)
        if not key or key.startswith("object_") or key in {"floor", "ceiling", *WALL_IDS}:
            return
        alias_to_ids.setdefault(key, set()).add(object_id)

    for obj in objects or []:
        if not isinstance(obj, dict) or not obj.get("id"):
            continue
        object_id = str(obj["id"])
        category = str(obj.get("category") or "")
        description = str(obj.get("description") or "")
        add(category, object_id)
        add(category.replace("_", " "), object_id)
        parts = [part for part in _alias_key(category).split("_") if part]
        if len(parts) > 1:
            add(parts[-1], object_id)
        for phrase in ["coffee table", "headphone stand", "built in cabinet", "built-in cabinet", "office chair", "mini fridge"]:
            if phrase in description.lower() or phrase.replace(" ", "_") == _alias_key(category):
                add(phrase, object_id)
    return {alias: next(iter(ids)) for alias, ids in alias_to_ids.items() if len(ids) == 1}




def _ceiling_light_object_ids(objects: list[dict[str, Any]] | None) -> list[str]:
    return [str(obj["id"]) for obj in objects or [] if isinstance(obj, dict) and obj.get("id") and _is_ceiling_light_object(obj)]


def _ceiling_confirmed_from_objects(objects: list[dict[str, Any]] | None) -> bool:
    return bool(_ceiling_light_object_ids(objects))


EXPLICIT_AGENT_WALL_ROLES = {"wall_mounted", "wall_embedded", "mounted", "embedded", "hung"}
FLOOR_AGENT_WALL_ROLES = EXPLICIT_AGENT_WALL_ROLES | {"against_wall"}


def _collect_high_confidence_wall_roles(
    task_results: list[dict[str, Any]],
    allowed_roles: set[str],
) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for root in task_results:
        for source_path, node in _walk_with_path(root, str(root.get("task_id") or "task") if isinstance(root, dict) else "task"):
            if not isinstance(node, dict):
                continue
            object_id = node.get("object_id")
            role = str(node.get("role") or "").strip().lower()
            confidence = str(node.get("confidence") or "").strip().lower()
            if (
                isinstance(object_id, str)
                and role in allowed_roles
                and confidence in {"high", "very_high", "very high"}
            ):
                result[object_id] = {
                    "object_id": object_id,
                    "role": role,
                    "confidence": confidence,
                    "source_path": source_path,
                }
    return result


def collect_high_confidence_explicit_wall_roles(
    task_results: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Collect roles that justify wall constraint independently of floor or object support."""
    return _collect_high_confidence_wall_roles(task_results, EXPLICIT_AGENT_WALL_ROLES)


def collect_high_confidence_floor_wall_roles(
    task_results: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Include high-confidence against-wall evidence only for the floor+wall filter."""
    return _collect_high_confidence_wall_roles(task_results, FLOOR_AGENT_WALL_ROLES)


SEMANTIC_WALL_CONTACT_MARKERS = (
    "built_in",
    "embedded",
    "fastened",
    "fixed_to_wall",
    "flush",
    "installed",
    "integrated",
    "leaning",
    "mounted",
    "rests_against_wall",
    "touching_wall",
    "visible_back_contact",
    "visible_rear_contact",
    "visible_wall_contact",
    "wall_contact",
    "wall_hung",
    "wall_mounted",
)
SEMANTIC_WALL_SEPARATION_MARKERS = (
    "clear_gap",
    "gap_from_wall",
    "no_contact",
    "not_in_contact",
    "not_touching",
    "separate_from_wall",
    "visible_gap",
)
SEMANTIC_WALL_PLACEMENT_ONLY_MARKERS = (
    "near_only",
    "position_only",
    "positional_only",
)


def _semantic_wall_text(node: dict[str, Any]) -> str:
    values: list[str] = []
    for key in (
        "evidence_code",
        "reason_code",
        "reason",
        "rationale",
        "wall_contact_type",
        "geometry_classification",
    ):
        if node.get(key) is not None:
            values.append(str(node.get(key)))
    for key in ("evidence_codes", "evidence"):
        value = node.get(key)
        if isinstance(value, list):
            values.extend(str(item) for item in value if item is not None)
        elif value is not None:
            values.append(str(value))
    return _alias_key(" ".join(values))


def _semantic_wall_signal_flags(text: str) -> dict[str, bool]:
    explicit_separation = any(
        marker in text for marker in SEMANTIC_WALL_SEPARATION_MARKERS
    )
    explicit_contact = (
        not explicit_separation
        and any(marker in text for marker in SEMANTIC_WALL_CONTACT_MARKERS)
    )
    placement_only = any(
        marker in text for marker in SEMANTIC_WALL_PLACEMENT_ONLY_MARKERS
    )
    return {
        "explicit_contact": explicit_contact,
        "explicit_separation": explicit_separation,
        "placement_only": placement_only,
    }


def _selected_wall_orientation_candidate_id(
    wall_evidence: dict[str, Any] | None,
) -> str | None:
    if not isinstance(wall_evidence, dict):
        return None
    containers = [wall_evidence]
    render_grid = wall_evidence.get("render_grid_wall_candidates")
    if isinstance(render_grid, dict):
        containers.append(render_grid)
    for container in containers:
        value = (
            container.get("selected_orientation_candidate_id")
            or container.get("selected_candidate_id")
        )
        if isinstance(value, str) and value:
            return value
    return None


def collect_semantic_wall_bindings(
    task_results: list[dict[str, Any]],
    wall_evidence: dict[str, Any] | None,
    wall_id_remap: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Resolve one medium-or-higher semantic wall binding per target.

    Only the already-selected orientation candidate is used when that id is
    available. Without a selected id, every qualifying binding must agree on
    one remapped wall. High confidence is still required for relation creation;
    medium confidence is retained only to protect an already-accepted pair.
    """
    wall_id_remap = wall_id_remap or {}
    selected_candidate_id = _selected_wall_orientation_candidate_id(wall_evidence)
    confidence_rank = {
        "medium": 1,
        "high": 2,
        "very_high": 3,
        "very high": 3,
    }
    role_rank = {role: 2 for role in EXPLICIT_AGENT_WALL_ROLES}
    role_rank["against_wall"] = 1
    role_records: dict[str, list[dict[str, Any]]] = {}
    binding_records: dict[str, list[dict[str, Any]]] = {}
    support_semantics: dict[str, list[str]] = {}

    for root in task_results:
        root_name = (
            str(root.get("task_id") or "task")
            if isinstance(root, dict)
            else "task"
        )
        for source_path, node in _walk_with_path(root, root_name):
            if not isinstance(node, dict):
                continue
            object_id = node.get("object_id")
            target_id = node.get("object_target")
            role = str(node.get("role") or "").strip().lower()
            confidence = str(node.get("confidence") or "").strip().lower()
            if (
                isinstance(object_id, str)
                and role in FLOOR_AGENT_WALL_ROLES
                and confidence in confidence_rank
            ):
                role_records.setdefault(object_id, []).append(
                    {
                        "object_id": object_id,
                        "role": role,
                        "confidence": confidence,
                        "evidence_text": _semantic_wall_text(node),
                        "source_path": source_path,
                    }
                )
            if (
                isinstance(target_id, str)
                and node.get("support_mode") is not None
            ):
                support_text = _semantic_wall_text(node)
                if support_text:
                    support_semantics.setdefault(target_id, []).append(
                        support_text
                    )

            bindings = node.get("candidate_wall_bindings")
            if not isinstance(bindings, list):
                continue
            for index, binding in enumerate(bindings):
                if not isinstance(binding, dict):
                    continue
                target = binding.get("object_id")
                candidate_id = str(binding.get("candidate_id") or "")
                if (
                    selected_candidate_id
                    and candidate_id
                    and candidate_id != selected_candidate_id
                ):
                    continue
                binding_role = str(binding.get("role") or "").strip().lower()
                binding_confidence = str(
                    binding.get("confidence") or ""
                ).strip().lower()
                provisional_wall = str(
                    binding.get("final_wall_id")
                    or binding.get("candidate_wall_id")
                    or binding.get("provisional_wall_id")
                    or binding.get("wall_id")
                    or ""
                )
                wall_id = wall_id_remap.get(provisional_wall, provisional_wall)
                if (
                    not isinstance(target, str)
                    or binding_role not in FLOOR_AGENT_WALL_ROLES
                    or binding_confidence not in confidence_rank
                    or wall_id not in WALL_IDS
                ):
                    continue
                binding_records.setdefault(target, []).append(
                    {
                        "object_id": target,
                        "wall_id": wall_id,
                        "provisional_wall_id": provisional_wall,
                        "candidate_id": candidate_id or None,
                        "role": binding_role,
                        "confidence": binding_confidence,
                        "evidence_text": _semantic_wall_text(binding),
                        "source_path": (
                            f"{source_path}.candidate_wall_bindings[{index}]"
                        ),
                    }
                )

    resolved: dict[str, dict[str, Any]] = {}
    ambiguous: list[dict[str, Any]] = []
    for target, records in binding_records.items():
        wall_ids = sorted({str(item["wall_id"]) for item in records})
        if len(wall_ids) != 1:
            ambiguous.append(
                {
                    "object_target": target,
                    "wall_ids": wall_ids,
                    "reason": "multiple_semantic_wall_bindings",
                    "bindings": records,
                }
            )
            continue
        all_role_records = records + role_records.get(target, [])
        best_role = max(
            all_role_records,
            key=lambda item: (
                confidence_rank.get(str(item.get("confidence") or ""), 0),
                role_rank.get(str(item.get("role") or ""), 0),
            ),
        )
        evidence_text = _alias_key(
            " ".join(
                [
                    *[
                        str(item.get("evidence_text") or "")
                        for item in all_role_records
                    ],
                    *support_semantics.get(target, []),
                ]
            )
        )
        resolved[target] = {
            "object_target": target,
            "wall_id": wall_ids[0],
            "role": best_role.get("role"),
            "confidence": best_role.get("confidence"),
            "evidence_text": evidence_text,
            "signal_flags": _semantic_wall_signal_flags(evidence_text),
            "selected_candidate_id": selected_candidate_id,
            "bindings": records,
            "role_evidence": role_records.get(target, []),
            "support_semantics": support_semantics.get(target, []),
        }
    return {
        "selected_candidate_id": selected_candidate_id,
        "bindings": resolved,
        "ambiguous": ambiguous,
    }


def reconcile_semantic_wall_contact_relations(
    relations: list[dict[str, Any]],
    accepted_candidates: list[dict[str, Any]],
    objects: list[dict[str, Any]] | None,
    task_results: list[dict[str, Any]],
    wall_evidence: dict[str, Any] | None,
    coverage_repairs: list[dict[str, Any]] | None = None,
    blocked_wall_attach_anchors: dict[str, str] | None = None,
    wall_id_remap: dict[str, str] | None = None,
) -> tuple[
    list[dict[str, Any]],
    dict[str, Any],
    dict[tuple[str, str], dict[str, Any]],
]:
    """Reconcile saved semantic wall binding with saved contact evidence.

    The rule is category-agnostic and conservative. A missing floor+wall
    relation is restored only when the Wall Agent's sole negative is the
    structured soft geometry outcome floor_supported_proximity_only with
    no flat-contact run, and independent high-confidence semantic plus contact
    evidence points to one wall. Non-soft rejection remains authoritative.
    """
    completed = list(relations)
    object_by_id = _object_context_by_id(objects)
    coverage_repairs = coverage_repairs or []
    blocked_wall_attach_anchors = blocked_wall_attach_anchors or {}
    wall_id_remap = wall_id_remap or {}
    binding_report = collect_semantic_wall_bindings(
        task_results,
        wall_evidence,
        wall_id_remap,
    )
    geometry = _geometry_by_object_wall(wall_evidence)
    floor_targets = {
        str(rel.get("parameters", {}).get("object_target"))
        for rel in completed
        if isinstance(rel, dict)
        and rel.get("relation") == "attach"
        and isinstance(rel.get("parameters"), dict)
        and rel.get("parameters", {}).get("object_anchor") == "floor"
    }

    accepted_wall_pairs: set[tuple[str, str]] = set()
    for item in accepted_candidates:
        rel = item.get("relation") if isinstance(item, dict) else None
        params = rel.get("parameters", {}) if isinstance(rel, dict) else {}
        anchor = str(params.get("object_anchor") or "")
        target = str(params.get("object_target") or "")
        mapped_anchor = wall_id_remap.get(anchor, anchor)
        if (
            isinstance(rel, dict)
            and rel.get("relation") == "attach"
            and mapped_anchor.startswith("wall_")
        ):
            accepted_wall_pairs.add((target, mapped_anchor))

    coverage_wall_pairs: set[tuple[str, str]] = set()
    for item in coverage_repairs:
        rel = item.get("relation") if isinstance(item, dict) else None
        params = rel.get("parameters", {}) if isinstance(rel, dict) else {}
        anchor = str(params.get("object_anchor") or "")
        target = str(params.get("object_target") or "")
        if item.get("coverage_repair") is True and anchor.startswith("wall_"):
            coverage_wall_pairs.add((target, anchor))

    pair_signals: dict[tuple[str, str], list[dict[str, Any]]] = {}
    reject_decisions = {"reject", "rejected", "false", "no"}
    for root in task_results:
        root_name = (
            str(root.get("task_id") or "task")
            if isinstance(root, dict)
            else "task"
        )
        for source_path, node in _walk_with_path(root, root_name):
            if not isinstance(node, dict) or node.get("decision") is None:
                continue
            target = str(node.get("object_target") or "")
            params = (
                node.get("parameters")
                if isinstance(node.get("parameters"), dict)
                else {}
            )
            provisional_wall = str(
                node.get("candidate_wall_id")
                or node.get("wall_id")
                or params.get("object_anchor")
                or ""
            )
            wall_id = wall_id_remap.get(provisional_wall, provisional_wall)
            if not target.startswith("object_") or wall_id not in WALL_IDS:
                continue
            text = _semantic_wall_text(node)
            flags = _semantic_wall_signal_flags(text)
            decision = str(node.get("decision") or "").lower()
            reason_code = _alias_key(
                node.get("reason_code") or node.get("reason")
            )
            evidence_values = (
                node.get("evidence_codes")
                if isinstance(node.get("evidence_codes"), list)
                else []
            )
            evidence_codes = {
                _alias_key(value)
                for value in evidence_values
                if value is not None
            }
            explicit_position_evidence = any(
                "position" in value for value in evidence_codes
            )
            strong_verified_contact = any(
                marker in text
                for marker in (
                    "valid_flat_contact",
                    "validated_flat_contact",
                    "flat_contact_run",
                    "flat_edge_contact",
                    "flush",
                    "mounted",
                    "wall_hung",
                    "wall_mounted",
                )
            )
            soft_geometry_rejection = (
                decision in reject_decisions
                and reason_code == "floor_supported_proximity_only"
                and "floor_supported" in evidence_codes
                and any(
                    "no_flat_contact" in value
                    for value in evidence_codes
                )
            )
            pair_signals.setdefault((target, wall_id), []).append(
                {
                    "decision": decision,
                    "reason_code": reason_code,
                    "source_path": source_path,
                    "evidence_text": text,
                    "explicit_separation": flags["explicit_separation"],
                    "placement_only": flags["placement_only"]
                    or (
                        explicit_position_evidence
                        and not strong_verified_contact
                    ),
                    "soft_geometry_rejection": soft_geometry_rejection,
                    "hard_rejection": (
                        decision in reject_decisions
                        and not soft_geometry_rejection
                    ),
                    "candidate_mount_surface_rejected": (
                        decision in reject_decisions
                        and reason_code
                        == "candidate_wall_not_verified_mount_surface"
                    ),
                }
            )

    added: list[dict[str, Any]] = []
    remapped: list[dict[str, Any]] = []
    retained: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    trusted: dict[tuple[str, str], dict[str, Any]] = {}
    high_confidences = {"high", "very_high", "very high"}
    for target, binding in binding_report["bindings"].items():
        wall_id = str(binding["wall_id"])
        role = str(binding.get("role") or "")
        confidence = str(binding.get("confidence") or "")
        metric = geometry.get((target, wall_id), {})
        distance = metric.get("footprint_min_distance_to_wall")
        local_possible = (
            metric.get("possible_against_wall_contact") is True
            or (
                isinstance(distance, (int, float))
                and float(distance) <= 0.035
            )
        )
        semantic_flags = binding.get("signal_flags", {})
        support_semantics = [
            _alias_key(value)
            for value in binding.get("support_semantics", [])
        ]
        independent_floor_support = any(
            marker in text
            for text in support_semantics
            for marker in (
                "appliance_on_floor",
                "floor_only",
                "free_standing",
                "freestanding",
                "standalone",
            )
        )
        signals = pair_signals.get((target, wall_id), [])
        soft_geometry_rejection = any(
            item.get("soft_geometry_rejection") for item in signals
        )
        hard_rejection = any(
            item.get("hard_rejection") for item in signals
        )
        explicit_separation = bool(
            semantic_flags.get("explicit_separation")
            or any(item.get("explicit_separation") for item in signals)
        )
        placement_only = any(item.get("placement_only") for item in signals)
        obj = object_by_id.get(target, {})
        object_text = _alias_key(
            f"{obj.get('category', '')} {obj.get('description', '')}"
        )
        freestanding = any(
            marker in object_text
            for marker in ("free_standing", "freestanding", "standalone")
        )
        existing_walls = {
            str(rel.get("parameters", {}).get("object_anchor") or "")
            for rel in completed
            if isinstance(rel, dict)
            and rel.get("relation") == "attach"
            and isinstance(rel.get("parameters"), dict)
            and str(
                rel.get("parameters", {}).get("object_target") or ""
            ) == target
            and str(
                rel.get("parameters", {}).get("object_anchor") or ""
            ).startswith("wall_")
        }
        pair = (target, wall_id)
        high_confidence = confidence in high_confidences
        explicit_role = role in EXPLICIT_AGENT_WALL_ROLES
        unreviewed_explicit_contact = (
            role == "against_wall"
            and target in floor_targets
            and not signals
            and semantic_flags.get("explicit_contact") is True
        )
        against_wall_contact = (
            role == "against_wall"
            and target in floor_targets
            and (
                (
                    high_confidence
                    and soft_geometry_rejection
                    and (
                        local_possible
                        or semantic_flags.get("explicit_contact") is True
                    )
                )
                or unreviewed_explicit_contact
            )
        )
        explicit_wall_contact = explicit_role and high_confidence
        already_accepted_contact = (
            pair in accepted_wall_pairs
            and wall_id in existing_walls
            and role == "against_wall"
            and semantic_flags.get("explicit_contact") is True
        )
        positive = (
            explicit_wall_contact
            or against_wall_contact
            or already_accepted_contact
        )
        base_record = {
            "object_target": target,
            "wall_id": wall_id,
            "semantic_wall_binding": binding,
            "local_possible_against_wall_contact": local_possible,
            "footprint_min_distance_to_wall": distance,
            "soft_geometry_rejection": soft_geometry_rejection,
            "unreviewed_explicit_contact": unreviewed_explicit_contact,
            "independent_floor_support": independent_floor_support,
            "support_semantics": support_semantics,
            "pair_signals": signals,
        }
        if wall_id in blocked_wall_attach_anchors:
            skipped.append(
                {
                    **base_record,
                    "reason": "semantic_wall_binding_is_weak_or_inferred",
                    "blocked_wall_anchor_reason": (
                        blocked_wall_attach_anchors[wall_id]
                    ),
                }
            )
            continue
        if (
            freestanding
            or independent_floor_support
            or explicit_separation
            or placement_only
            or hard_rejection
        ):
            skipped.append(
                {
                    **base_record,
                    "reason": (
                        "freestanding_semantics_veto_wall_contact"
                        if freestanding
                        else "independent_floor_support_semantics_veto_wall_contact"
                        if independent_floor_support
                        else "explicit_wall_separation_veto"
                        if explicit_separation
                        else "placement_only_wall_evidence_veto"
                        if placement_only
                        else "non_soft_wall_agent_rejection_veto"
                    ),
                }
            )
            continue
        if not positive:
            skipped.append(
                {
                    **base_record,
                    "reason": "insufficient_independent_contact_evidence",
                }
            )
            continue

        if wall_id in existing_walls:
            record = {
                **base_record,
                "reason": (
                    "accepted_semantic_contact_relation_retained"
                    if already_accepted_contact
                    else "existing_relation_matches_semantic_wall_contact"
                ),
            }
            retained.append(record)
            trusted[pair] = record
            continue
        if existing_walls:
            wrong_pairs = {
                (target, existing_wall) for existing_wall in existing_walls
            }
            coverage_only = all(
                wrong_pair in coverage_wall_pairs
                and wrong_pair not in accepted_wall_pairs
                for wrong_pair in wrong_pairs
            )
            wrong_coverage_rejected = all(
                any(
                    signal.get("candidate_mount_surface_rejected")
                    for signal in pair_signals.get(wrong_pair, [])
                )
                for wrong_pair in wrong_pairs
            )
            can_remap_coverage = (
                explicit_wall_contact
                and coverage_only
                and (local_possible or wrong_coverage_rejected)
            )
            if not can_remap_coverage:
                skipped.append(
                    {
                        **base_record,
                        "reason": (
                            "existing_non_coverage_or_unverified_wall_"
                            "relation_not_overridden"
                        ),
                        "existing_wall_ids": sorted(existing_walls),
                        "coverage_only": coverage_only,
                        "wrong_coverage_wall_rejected": (
                            wrong_coverage_rejected
                        ),
                    }
                )
                continue
            original_relations = [
                rel
                for rel in completed
                if isinstance(rel, dict)
                and rel.get("relation") == "attach"
                and isinstance(rel.get("parameters"), dict)
                and str(
                    rel.get("parameters", {}).get("object_target") or ""
                ) == target
                and str(
                    rel.get("parameters", {}).get("object_anchor") or ""
                ) in existing_walls
            ]
            completed = [
                rel for rel in completed if rel not in original_relations
            ]
            remap_record = {
                **base_record,
                "reason": (
                    "coverage_only_wall_choice_reconciled_to_unique_"
                    "semantic_binding"
                ),
                "original_wall_ids": sorted(existing_walls),
                "wrong_coverage_wall_rejected": wrong_coverage_rejected,
            }
            remapped.append(remap_record)
            for repair in coverage_repairs:
                repair_rel = (
                    repair.get("relation")
                    if isinstance(repair, dict)
                    else None
                )
                repair_params = (
                    repair_rel.get("parameters", {})
                    if isinstance(repair_rel, dict)
                    else {}
                )
                if (
                    str(repair_params.get("object_target") or "") == target
                    and str(
                        repair_params.get("object_anchor") or ""
                    ) in existing_walls
                ):
                    original_relation = json.loads(
                        json.dumps(repair_rel, ensure_ascii=False)
                    )
                    repair[
                        "original_relation_before_semantic_reconciliation"
                    ] = original_relation
                    repair["original_selected_anchor"] = (
                        repair_params.get("object_anchor")
                    )
                    repair["selected_anchor"] = wall_id
                    repair["relation"] = {
                        "relation": "attach",
                        "parameters": {
                            "object_anchor": wall_id,
                            "object_target": target,
                        },
                    }
                    repair["semantic_wall_contact_reconciliation"] = (
                        remap_record["reason"]
                    )

        relation = {
            "relation": "attach",
            "parameters": {
                "object_anchor": wall_id,
                "object_target": target,
            },
        }
        completed.append(relation)
        action = {
            **base_record,
            "relation": relation,
            "reason": (
                "coverage_only_wall_choice_reconciled_to_unique_semantic_binding"
                if existing_walls
                else "missing_wall_contact_restored_from_unique_semantic_binding"
            ),
        }
        added.append(action)
        trusted[pair] = action

    return completed, {
        "added_count": len(added),
        "added": added,
        "retained_count": len(retained),
        "retained": retained,
        "coverage_remap_count": len(remapped),
        "coverage_remaps": remapped,
        "skipped": skipped,
        "ambiguous_bindings": binding_report["ambiguous"],
        "selected_candidate_id": binding_report["selected_candidate_id"],
        "policy": (
            "A unique high-confidence semantic wall binding may restore a "
            "missing floor+wall contact only when the Wall Agent recorded the "
            "structured soft geometry rejection floor_supported_proximity_only "
            "with no flat-contact run, or when medium-or-higher semantic evidence "
            "explicitly states contact and that exact wall pair was never reviewed. "
            "Other rejections, blocked or ambiguous "
            "walls, freestanding or independently floor-supported objects, "
            "placement-only evidence, and explicit separation are vetoes. "
            "Coverage-only wall choices may be remapped "
            "only for explicit wall roles when the old anchor was itself "
            "rejected or saved local geometry supports the bound wall."
        ),
        "additional_api_calls": 0,
        "additional_geometry_or_render_passes": 0,
        "category_specific_gates": False,
    }, trusted


def _is_explicit_wall_mounted_object(obj: dict[str, Any]) -> bool:
    category = str(obj.get("category") or "").lower().strip()
    if category in WALL_MOUNTED_CATEGORIES:
        return True
    text = f"{obj.get('category', '')} {obj.get('description', '')}".lower()
    return any(keyword in text for keyword in WALL_MOUNTED_KEEP_KEYWORDS)


def filter_surface_supported_wall_attaches(
    relations: list[dict[str, Any]],
    objects: list[dict[str, Any]] | None,
    semantic_wall_roles: dict[str, dict[str, Any]] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Drop incidental wall proximity for targets already supported by another object.

    Object support and wall attachment coexist only when independent high-confidence
    wall-mount semantics exist. A flat rear edge or ordinary proximity is not enough.
    """
    object_by_id = _object_context_by_id(objects)
    semantic_wall_roles = semantic_wall_roles or {}
    object_supported_targets = {
        rel.get("parameters", {}).get("object_target")
        for rel in relations
        if isinstance(rel, dict)
        and rel.get("relation") == "attach"
        and isinstance(rel.get("parameters"), dict)
        and str(rel.get("parameters", {}).get("object_anchor", "")).startswith("object_")
    }
    if not object_supported_targets:
        return relations, []
    kept: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    for rel in relations:
        params = rel.get("parameters", {}) if isinstance(rel.get("parameters"), dict) else {}
        anchor = params.get("object_anchor")
        target = params.get("object_target")
        if not (str(anchor).startswith("wall_") and target in object_supported_targets):
            kept.append(rel)
            continue
        obj = object_by_id.get(str(target), {})
        explicit_agent_wall_mount = str(target) in semantic_wall_roles
        if _is_explicit_wall_mounted_object(obj) or explicit_agent_wall_mount:
            kept.append(rel)
            continue
        skipped.append(
            {
                "relation": rel,
                "reason": "surface_supported_object_wall_attach_filtered",
                "policy": "A target already supported by another object keeps wall attach only with explicit high-confidence wall-mounted/embedded/hung/rack/bracket evidence; proximity or a flat edge alone is insufficient.",
                "object": obj,
                "semantic_wall_role": semantic_wall_roles.get(str(target)),
            }
        )
    return kept, skipped


def collect_visual_reconstruction_wall_contact_overrides(
    task_results: list[dict[str, Any]],
) -> dict[tuple[str, str], dict[str, Any]]:
    """Collect exact Agent-approved wall pairs where image evidence overrides noisy SAM3D geometry."""
    overrides: dict[tuple[str, str], dict[str, Any]] = {}
    for root in task_results:
        if not isinstance(root, dict):
            continue
        kind = " ".join(
            str(root.get(key) or "")
            for key in ("task_id", "task_type", "canonical_task_kind")
        ).lower()
        capabilities = root.get("provided_capabilities", [])
        if "wall_attach" not in kind and not (
            isinstance(capabilities, list) and "wall_attach_decisions_ready" in capabilities
        ):
            continue
        for path, node in _walk_with_path(root, str(root.get("task_id") or "wall_attach_task")):
            values = node.get("wall_attach_candidates") if isinstance(node, dict) else None
            if not isinstance(values, list):
                continue
            for candidate in values:
                if not isinstance(candidate, dict):
                    continue
                target = str(candidate.get("object_target") or "")
                wall_id = str(candidate.get("candidate_wall_id") or candidate.get("wall_id") or "")
                geometry = candidate.get("geometry_evidence")
                geometry = geometry if isinstance(geometry, dict) else {}
                geometry_text = " ".join(
                    str(geometry.get(key) or "")
                    for key in ("geometry_reliability", "conflict", "reason")
                ).lower()
                structured_reconstruction_uncertainty = (
                    geometry.get("reconstruction_uncertainty_used") is True
                    and bool(str(geometry.get("conflict") or geometry.get("reason") or "").strip())
                )
                contact_type = re.sub(
                    r"[^a-z0-9]+",
                    "_",
                    str(candidate.get("wall_contact_type") or "").strip().lower(),
                ).strip("_")
                flat_back_or_side_contact = (
                    contact_type == "against_wall_horizontal_back_contact"
                    or (
                        contact_type.startswith("against_wall_")
                        and any(marker in contact_type for marker in ("back", "rear", "side"))
                        and any(marker in contact_type for marker in ("contact", "surface", "panel"))
                        and not any(marker in contact_type for marker in ("point", "corner", "tangent"))
                    )
                )
                qualifies = (
                    str(candidate.get("decision") or "").lower() in {"accept", "accepted", "true", "yes"}
                    and candidate.get("decision_is_final") is True
                    and flat_back_or_side_contact
                    and bool(str(candidate.get("visual_evidence") or "").strip())
                    and bool(str(candidate.get("mask_evidence") or "").strip())
                    and (
                        structured_reconstruction_uncertainty
                        or "reconstruction_override" in geometry_text
                        or "conflicts_with_reconstructed" in geometry_text
                        or "reconstructed_footprint" in geometry_text
                    )
                    and target.startswith("object_")
                    and wall_id.startswith("wall_")
                )
                if qualifies:
                    overrides[(target, wall_id)] = {
                        "object_id": target,
                        "wall_id": wall_id,
                        "source_task_id": root.get("task_id"),
                        "source_path": f"{path}.wall_attach_candidates",
                        "visual_evidence": candidate.get("visual_evidence"),
                        "mask_evidence": candidate.get("mask_evidence"),
                        "geometry_evidence": geometry,
                        "reason": candidate.get("reason"),
                    }
    return overrides


def collect_floor_wall_relation_policy_overrides(
    accepted_candidates: list[dict[str, Any]],
    objects: list[dict[str, Any]] | None,
    wall_evidence: dict[str, Any] | None,
    semantic_wall_roles: dict[str, dict[str, Any]] | None = None,
    wall_id_remap: dict[str, str] | None = None,
) -> dict[str, dict[tuple[str, str], dict[str, Any]]]:
    """Separate verified contact from proximity-only Agent language."""
    object_by_id = _object_context_by_id(objects)
    semantic_wall_roles = semantic_wall_roles or {}
    wall_id_remap = wall_id_remap or {}
    geometry = _geometry_by_object_wall(wall_evidence)
    trusted: dict[tuple[str, str], dict[str, Any]] = {}
    proximity_only: dict[tuple[str, str], dict[str, Any]] = {}
    strong_contact_markers = (
        "valid_flat_contact",
        "validated_flat_contact",
        "flat_contact_run",
        "flat_edge_contact",
        "flush",
    )
    installed_markers = (
        "built_in",
        "builtin",
        "embedded",
        "fastened",
        "fixed_to_wall",
        "installed",
        "integrated",
        "mounted",
        "wall_hung",
        "wall_mounted",
    )
    for item in accepted_candidates:
        rel = item.get("relation") if isinstance(item.get("relation"), dict) else {}
        params = rel.get("parameters", {}) if isinstance(rel.get("parameters"), dict) else {}
        target = str(params.get("object_target") or "")
        provisional_wall = str(params.get("object_anchor") or "")
        wall_id = wall_id_remap.get(provisional_wall, provisional_wall)
        if rel.get("relation") != "attach" or not wall_id.startswith("wall_"):
            continue
        obj = object_by_id.get(target, {})
        category_text = _alias_key(obj.get("category"))
        category_tokens = set(category_text.split("_"))
        object_text = _alias_key(f"{obj.get('category', '')} {obj.get('description', '')}")
        raw = item.get("raw_candidate") if isinstance(item.get("raw_candidate"), dict) else {}
        evidence_values = raw.get("evidence_codes")
        if not isinstance(evidence_values, list):
            evidence_values = []
        evidence_codes = [_alias_key(value) for value in evidence_values if value is not None]
        reason_text = _alias_key(
            " ".join(
                str(value or "")
                for value in (
                    raw.get("reason_code"),
                    raw.get("reason"),
                    item.get("reason"),
                )
            )
        )
        evidence_text = "_".join(evidence_codes + [reason_text])
        explicitly_installed = (
            _is_explicit_wall_mounted_object(obj)
            or any(marker in object_text or marker in evidence_text for marker in installed_markers)
        )
        freestanding_cooling = (
            bool({"refrigerator", "fridge", "freezer"}.intersection(category_tokens))
            and not explicitly_installed
        )
        strong_flat_contact = any(
            marker in evidence_text for marker in strong_contact_markers
        )
        positional_only = (
            any("position" in code for code in evidence_codes)
            and not strong_flat_contact
            and not explicitly_installed
        )
        pair = (target, wall_id)
        if freestanding_cooling or positional_only:
            proximity_only[pair] = {
                "object_id": target,
                "wall_id": wall_id,
                "reason": (
                    "freestanding_cooling_appliance_proximity_is_not_attachment"
                    if freestanding_cooling
                    else "position_only_wall_evidence_is_not_attachment"
                ),
                "category": obj.get("category"),
                "evidence_codes": evidence_values,
                "source_path": item.get("source_path"),
            }
            continue

        metric = geometry.get(pair, {})
        distance = metric.get("footprint_min_distance_to_wall")
        local_possible = (
            metric.get("possible_against_wall_contact") is True
            or (
                isinstance(distance, (int, float))
                and float(distance) <= 0.035
            )
        )
        role = semantic_wall_roles.get(target, {})
        high_against_wall = (
            str(role.get("role") or "") == "against_wall"
            and str(role.get("confidence") or "") in {"high", "very_high", "very high"}
        )
        tall_storage = (
            "tall" in object_text
            and any(
                marker in object_text
                for marker in ("armoire", "bookcase", "bookshelf", "cabinet", "shelving", "storage", "wardrobe")
            )
        )
        visible_rear_contact = any(
            marker in evidence_text
            for marker in ("back_wall", "rear_contact", "rear_wall", "visible_rear_contact")
        )
        if (
            (strong_flat_contact and local_possible)
            or (tall_storage and visible_rear_contact and high_against_wall)
        ):
            trusted[pair] = {
                "object_id": target,
                "wall_id": wall_id,
                "reason": (
                    "accepted_agent_flat_contact_with_local_possible_contact"
                    if strong_flat_contact and local_possible
                    else "high_confidence_tall_storage_visible_rear_contact"
                ),
                "category": obj.get("category"),
                "evidence_codes": evidence_values,
                "semantic_wall_role": role or None,
                "local_possible_against_wall_contact": local_possible,
                "source_path": item.get("source_path"),
            }
    for pair in trusted:
        proximity_only.pop(pair, None)
    return {
        "trusted_contact": trusted,
        "proximity_only": proximity_only,
    }


def filter_floor_supported_wall_attaches_without_flat_contact_edge(
    relations: list[dict[str, Any]],
    objects: list[dict[str, Any]] | None,
    wall_evidence: dict[str, Any] | None,
    semantic_wall_roles: dict[str, dict[str, Any]] | None = None,
    visual_reconstruction_overrides: dict[tuple[str, str], dict[str, Any]] | None = None,
    trusted_agent_contact_overrides: dict[tuple[str, str], dict[str, Any]] | None = None,
    proximity_only_wall_pairs: dict[tuple[str, str], dict[str, Any]] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Require real flat-edge contact, not point/corner proximity, for ordinary floor+wall contact."""
    object_by_id = _object_context_by_id(objects)
    semantic_wall_roles = semantic_wall_roles or {}
    visual_reconstruction_overrides = visual_reconstruction_overrides or {}
    trusted_agent_contact_overrides = trusted_agent_contact_overrides or {}
    proximity_only_wall_pairs = proximity_only_wall_pairs or {}
    floor_targets = {
        rel.get("parameters", {}).get("object_target")
        for rel in relations
        if isinstance(rel, dict)
        and rel.get("relation") == "attach"
        and isinstance(rel.get("parameters"), dict)
        and rel.get("parameters", {}).get("object_anchor") == "floor"
    }
    geometry: dict[tuple[str, str], dict[str, Any]] = {}
    if isinstance(wall_evidence, dict):
        for item in wall_evidence.get("wall_attach_geometry", []):
            if isinstance(item, dict) and item.get("object_id") and item.get("wall_id"):
                geometry[(str(item["object_id"]), str(item["wall_id"]))] = item

    kept: list[dict[str, Any]] = []
    filtered: list[dict[str, Any]] = []
    evaluated: list[dict[str, Any]] = []
    for rel in relations:
        params = rel.get("parameters", {}) if isinstance(rel, dict) and isinstance(rel.get("parameters"), dict) else {}
        anchor = str(params.get("object_anchor") or "")
        target = str(params.get("object_target") or "")
        if rel.get("relation") != "attach" or not anchor.startswith("wall_") or target not in floor_targets:
            kept.append(rel)
            continue
        obj = object_by_id.get(target, {})
        metric = geometry.get((target, anchor), {})
        analysis = metric.get("flat_contact_edge_analysis") if isinstance(metric.get("flat_contact_edge_analysis"), dict) else {}
        record = {
            "relation": rel,
            "object_id": target,
            "wall_id": anchor,
            "is_floor_supported": metric.get("is_floor_supported"),
            "analysis_available": analysis.get("analysis_available"),
            "has_valid_flat_contact_edge": analysis.get("has_valid_flat_contact_edge"),
            "flat_contact_edge_analysis": analysis,
        }
        visual_override = visual_reconstruction_overrides.get((target, anchor))
        if visual_override:
            record["decision"] = "preserve"
            record["reason"] = "accepted_visual_wall_contact_with_explicit_reconstruction_override"
            record["visual_reconstruction_override"] = visual_override
            evaluated.append(record)
            kept.append(rel)
            continue
        proximity_veto = proximity_only_wall_pairs.get((target, anchor))
        if proximity_veto:
            record.update({
                "decision": "filter",
                "reason": "floor_supported_wall_attach_proximity_only_evidence",
                "proximity_only_evidence": proximity_veto,
                "object": obj,
                "policy": (
                    "A freestanding cooling appliance or position-only wall candidate "
                    "does not gain attach from ordinary proximity, wall-run alignment, "
                    "or an against_wall label."
                ),
            })
            evaluated.append(record)
            filtered.append(record)
            continue
        trusted_contact = trusted_agent_contact_overrides.get((target, anchor))
        if trusted_contact:
            record["decision"] = "preserve"
            record["reason"] = "accepted_trusted_agent_floor_wall_contact"
            record["trusted_agent_contact_override"] = trusted_contact
            evaluated.append(record)
            kept.append(rel)
            continue
        semantic_role = semantic_wall_roles.get(target)
        semantic_role_has_geometry = bool(
            semantic_role
            and metric.get("possible_against_wall_contact") is True
            and isinstance(metric.get("footprint_min_distance_to_wall"), (int, float))
            and float(metric["footprint_min_distance_to_wall"]) <= 0.035
        )
        if _is_explicit_wall_mounted_object(obj) or semantic_role_has_geometry or metric.get("is_floor_supported") is not True:
            record["decision"] = "not_applicable"
            if semantic_role_has_geometry and str(semantic_role.get("role") or "") == "against_wall":
                record["reason"] = "high_confidence_against_wall_agent_verdict_within_reconstruction_uncertainty"
            elif semantic_role_has_geometry:
                record["reason"] = "high_confidence_wall_mounted_or_embedded_with_near_wall_geometry"
            else:
                record["reason"] = "explicit_wall_mounted_or_not_floor_supported"
            if semantic_role:
                record["semantic_wall_role"] = semantic_role
            evaluated.append(record)
            kept.append(rel)
            continue
        if analysis.get("analysis_available") is not True:
            record["decision"] = "preserve"
            record["reason"] = "mesh_floor_footprint_hull_unavailable"
            evaluated.append(record)
            kept.append(rel)
            continue
        if analysis.get("has_valid_flat_contact_edge") is True:
            record["decision"] = "preserve"
            record["reason"] = "valid_flat_mesh_hull_edge_contact_not_point_or_corner"
            evaluated.append(record)
            kept.append(rel)
            continue
        record.update({
            "decision": "filter",
            "reason": "floor_supported_wall_attach_without_valid_flat_contact_edge",
            "policy": "An ordinary floor-supported object may keep an additional wall attach when its real lower-mesh XZ hull has a finite wall-parallel flat contact edge. Consecutive near-collinear mesh segments are merged; isolated point/corner proximity and curved tangent fragments do not qualify.",
            "object": obj,
        })
        evaluated.append(record)
        filtered.append(record)
    return kept, filtered, evaluated


def enforce_single_wall_attach_per_target(
    relations: list[dict[str, Any]],
    preferred_wall_by_target: dict[str, str] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Keep at most one wall attach for each object target.

    Gravity/support relations such as floor->object and object->object are not
    affected. A wall selected by accepted Agent evidence is preferred; when no
    preference is available, canonical wall id order provides a deterministic
    fallback.
    """
    preferred_wall_by_target = preferred_wall_by_target or {}
    wall_anchors_by_target: dict[str, list[str]] = {}
    for relation in relations:
        parameters = relation.get("parameters", {}) if isinstance(relation, dict) else {}
        anchor = parameters.get("object_anchor")
        target = parameters.get("object_target")
        if relation.get("relation") != "attach" or not isinstance(anchor, str) or not anchor.startswith("wall_"):
            continue
        wall_anchors_by_target.setdefault(str(target), []).append(anchor)

    selected_wall_by_target: dict[str, str] = {}
    for target, anchors in wall_anchors_by_target.items():
        unique_anchors = sorted(set(anchors))
        if len(unique_anchors) <= 1:
            continue
        preferred = preferred_wall_by_target.get(target)
        selected_wall_by_target[target] = preferred if preferred in unique_anchors else unique_anchors[0]

    if not selected_wall_by_target:
        return relations, []

    kept: list[dict[str, Any]] = []
    removed: list[dict[str, Any]] = []
    for relation in relations:
        parameters = relation.get("parameters", {}) if isinstance(relation, dict) else {}
        anchor = parameters.get("object_anchor")
        target = str(parameters.get("object_target"))
        selected_wall = selected_wall_by_target.get(target)
        if (
            relation.get("relation") == "attach"
            and isinstance(anchor, str)
            and anchor.startswith("wall_")
            and selected_wall is not None
            and anchor != selected_wall
        ):
            removed.append(
                {
                    "relation": relation,
                    "reason": "multiple_wall_attach_same_target_deduped",
                    "object_target": target,
                    "removed_anchor": anchor,
                    "kept_anchor": selected_wall,
                }
            )
            continue
        kept.append(relation)
    return kept, removed


def filter_wall_mounted_floor_attaches(
    relations: list[dict[str, Any]],
    objects: list[dict[str, Any]] | None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Remove contradictory non-wall support for explicitly wall-mounted objects.

    A wall-mounted panel/painting/screen is physically covered by its wall
    attach. Keeping an additional floor or furniture attach for the same target
    creates a contradictory support path. This is intentionally limited to
    explicit wall-mounted semantics; leaning or freestanding furniture may still
    have both floor and wall contact.
    """
    object_by_id = _object_context_by_id(objects)
    wall_attached_targets = {
        str(rel.get("parameters", {}).get("object_target"))
        for rel in relations
        if isinstance(rel, dict)
        and rel.get("relation") == "attach"
        and isinstance(rel.get("parameters"), dict)
        and str(rel.get("parameters", {}).get("object_anchor", "")).startswith("wall_")
    }
    if not wall_attached_targets:
        return relations, []

    kept: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    for rel in relations:
        params = rel.get("parameters", {}) if isinstance(rel.get("parameters"), dict) else {}
        anchor = str(params.get("object_anchor") or "")
        target = str(params.get("object_target") or "")
        obj = object_by_id.get(target, {})
        contradictory_support = anchor == "floor" or anchor.startswith("object_")
        if contradictory_support and target in wall_attached_targets and _is_explicit_wall_mounted_object(obj):
            skipped.append(
                {
                    "relation": rel,
                    "reason": "explicit_wall_mounted_object_with_wall_attach_nonwall_support_filtered",
                    "policy": "An explicitly wall-mounted object with an accepted wall attach is not assigned a second floor or furniture support path.",
                    "object": obj,
                }
            )
            continue
        kept.append(rel)
    return kept, skipped


def _median_float(values: list[float]) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[middle]
    return (ordered[middle - 1] + ordered[middle]) / 2.0


def _object_bottom_y_from_wall_evidence(wall_evidence: dict[str, Any] | None) -> dict[str, float]:
    if not isinstance(wall_evidence, dict):
        return {}
    result: dict[str, float] = {}
    for key in ("render_grid_wall_attach_geometry", "wall_attach_geometry"):
        rows = wall_evidence.get(key)
        if not isinstance(rows, list):
            continue
        for item in rows:
            if not isinstance(item, dict) or not item.get("object_id"):
                continue
            try:
                bottom_y = float(item.get("bottom_y"))
            except (TypeError, ValueError):
                continue
            result.setdefault(str(item["object_id"]), bottom_y)
    return result


def complete_floor_support_for_structural_anchors(
    relations: list[dict[str, Any]],
    objects: list[dict[str, Any]] | None,
    wall_evidence: dict[str, Any] | None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Close omitted gravity support only for floor-level structural objects.

    The Agent remains responsible for semantic support selection. This local
    safety net addresses a narrower invariant: an object used as a support or
    contain anchor, or a freestanding object with a wall attach, cannot be left
    with no gravity-direction relation when its mesh bottom belongs to the
    scene's observed floor-height cluster.
    """
    object_by_id = _object_context_by_id(objects)
    object_ids = set(object_by_id)
    bottom_y_by_id = _object_bottom_y_from_wall_evidence(wall_evidence)
    relation_keys = {
        (
            rel.get("relation"),
            rel.get("parameters", {}).get("object_anchor"),
            rel.get("parameters", {}).get("object_target"),
        )
        for rel in relations
        if isinstance(rel, dict) and isinstance(rel.get("parameters"), dict)
    }

    candidate_sources: dict[str, set[str]] = {}
    gravity_resolved_targets: set[str] = set()
    floor_targets: set[str] = set()
    for rel in relations:
        if not isinstance(rel, dict) or not isinstance(rel.get("parameters"), dict):
            continue
        relation_type = rel.get("relation")
        params = rel.get("parameters", {})
        anchor = str(params.get("object_anchor") or "")
        target = str(params.get("object_target") or "")
        if anchor in object_ids and relation_type in {"attach", "contain"}:
            candidate_sources.setdefault(anchor, set()).add("object_relation_anchor")
        if relation_type == "attach" and anchor.startswith("wall_") and target in object_ids:
            candidate_sources.setdefault(target, set()).add("wall_attached_target")
        if target in object_ids and relation_type == "attach" and not anchor.startswith("wall_"):
            gravity_resolved_targets.add(target)
            if anchor == "floor":
                floor_targets.add(target)
        elif target in object_ids and relation_type == "contain":
            gravity_resolved_targets.add(target)

    all_bottoms = list(bottom_y_by_id.values())
    known_floor_bottoms = [
        bottom_y_by_id[target]
        for target in floor_targets
        if target in bottom_y_by_id
    ]
    floor_reference_y = _median_float(known_floor_bottoms)
    floor_reference_source = "median_existing_floor_attach_targets"
    if floor_reference_y is None and all_bottoms:
        ordered = sorted(all_bottoms)
        floor_reference_y = ordered[int((len(ordered) - 1) * 0.15)]
        floor_reference_source = "lower_15_percentile_object_bottoms"
    vertical_span = (max(all_bottoms) - min(all_bottoms)) if len(all_bottoms) >= 2 else 0.0
    floor_tolerance_y = max(0.025, min(0.060, vertical_span * 0.10))
    reconstruction_floor_tolerance_y = min(0.090, floor_tolerance_y * 1.5)

    completed = list(relations)
    added: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    for object_id in sorted(candidate_sources, key=sorted_object_id_key):
        sources = sorted(candidate_sources[object_id])
        obj = object_by_id.get(object_id, {})
        if object_id in gravity_resolved_targets:
            skipped.append({
                "object_id": object_id,
                "candidate_sources": sources,
                "reason": "gravity_or_containment_relation_already_present",
            })
            continue
        if _is_ceiling_light_object(obj) or _is_explicit_wall_mounted_object(obj):
            skipped.append({
                "object_id": object_id,
                "candidate_sources": sources,
                "reason": "explicit_wall_or_ceiling_mounted_object",
            })
            continue
        bottom_y = bottom_y_by_id.get(object_id)
        if bottom_y is None or floor_reference_y is None:
            skipped.append({
                "object_id": object_id,
                "candidate_sources": sources,
                "reason": "insufficient_mesh_bottom_evidence",
            })
            continue
        object_floor_tolerance_y = (
            reconstruction_floor_tolerance_y
            if "wall_attached_target" in sources
            else floor_tolerance_y
        )
        floor_threshold_y = floor_reference_y + object_floor_tolerance_y
        if bottom_y > floor_threshold_y:
            skipped.append({
                "object_id": object_id,
                "candidate_sources": sources,
                "bottom_y": bottom_y,
                "floor_threshold_y": floor_threshold_y,
                "floor_tolerance_y": object_floor_tolerance_y,
                "reconstruction_uncertainty_relaxation_used": "wall_attached_target" in sources,
                "reason": "object_bottom_above_observed_floor_cluster",
            })
            continue
        relation = {
            "relation": "attach",
            "parameters": {"object_anchor": "floor", "object_target": object_id},
        }
        key = ("attach", "floor", object_id)
        if key in relation_keys:
            continue
        completed.append(relation)
        relation_keys.add(key)
        added.append({
            "relation": relation,
            "object_id": object_id,
            "candidate_sources": sources,
            "bottom_y": bottom_y,
            "floor_reference_y": floor_reference_y,
            "floor_tolerance_y": object_floor_tolerance_y,
            "reconstruction_uncertainty_relaxation_used": "wall_attached_target" in sources,
            "reason": "floor_level_structural_anchor_missing_gravity_support",
        })

    return completed, {
        "schema_version": "gravity_support_completion_audit",
        "status": "completed",
        "policy": "Manager/Agent choose support semantics. Local completion adds floor attach only when an object is a structural relation anchor or wall-attached target, lacks any gravity/containment relation, is not explicitly wall/ceiling mounted, and its mesh bottom lies in the observed floor-height cluster.",
        "floor_reference_y": floor_reference_y,
        "floor_reference_source": floor_reference_source if floor_reference_y is not None else "unavailable",
        "floor_tolerance_y": floor_tolerance_y,
        "wall_attached_reconstruction_floor_tolerance_y": reconstruction_floor_tolerance_y,
        "candidate_object_ids": sorted(candidate_sources, key=sorted_object_id_key),
        "added_count": len(added),
        "added_relations": added,
        "skipped_items": skipped,
    }




def write_relation_normalization_audit(output_dir: Path, audit: dict[str, Any], dry_run: bool) -> dict[str, Any]:
    data = {"dry_run": dry_run, "status": "completed", **audit}
    write_audit_pair(output_dir, "relations", "relation_normalization_audit", data, "Relation Normalization Audit")
    return data


def write_flat_contact_edge_wall_attach_audit(
    output_dir: Path,
    merge_report: dict[str, Any],
    dry_run: bool,
) -> dict[str, Any]:
    report = merge_report.get("flat_contact_edge_wall_attach_filter", {})
    data = {
        "dry_run": dry_run,
        "status": "completed",
        "filtered_count": report.get("filtered_count", 0),
        "items": report.get("items", []),
        "evaluations": report.get("evaluations", []),
        "policy": report.get("policy"),
    }
    write_audit_pair(
        output_dir,
        "relations",
        "flat_contact_edge_wall_attach_audit",
        data,
        "Flat Contact Edge Wall Attach Audit",
    )
    return data


def write_semantic_wall_contact_reconciliation_audit(
    output_dir: Path,
    merge_report: dict[str, Any],
    dry_run: bool,
) -> dict[str, Any]:
    report = merge_report.get("semantic_wall_contact_reconciliation", {})
    data = {
        "dry_run": dry_run,
        "status": "completed",
        **report,
    }
    write_audit_pair(
        output_dir,
        "relations",
        "semantic_wall_contact_reconciliation_audit",
        data,
        "Semantic Wall Contact Reconciliation Audit",
    )
    return data


def write_weak_wall_attach_audit(
    output_dir: Path,
    merge_report: dict[str, Any],
    dry_run: bool,
    virtual_wall_shift_audit: dict[str, Any] | None = None,
) -> dict[str, Any]:
    items = [
        item
        for item in merge_report.get("skipped_candidates", [])
        if isinstance(item, dict) and item.get("reason") == "weak_or_inferred_wall_anchor_not_allowed_for_attach"
    ]
    data = {
        "dry_run": dry_run,
        "status": "completed",
        "filtered_count": len(items),
        "items": items,
        "blocked_wall_attach_anchors": merge_report.get("blocked_wall_attach_anchors", {}),
        "wall_id_remap_applied_to_relations": merge_report.get("wall_id_remap_applied_to_relations", {}),
        "wall_outward_shift": virtual_wall_shift_audit or {"status": "not_run"},
        "virtual_wall_outward_shift": virtual_wall_shift_audit or {"status": "not_run"},
        "policy": "Weak/hidden/inferred walls cannot anchor ordinary wall attach. After final relation filtering, every canonical wall without a retained wall attach moves outward once; a filtered weak-wall attach remains an explicit audit trigger.",
    }
    write_audit_pair(output_dir, "relations", "weak_wall_attach_audit", data, "Weak Wall Attach Audit")
    return data


def write_gravity_support_completion_audit(
    output_dir: Path,
    report: dict[str, Any],
    dry_run: bool,
) -> dict[str, Any]:
    data = {"dry_run": dry_run, **report}
    write_audit_pair(
        output_dir,
        "attach",
        "gravity_support_completion_audit",
        data,
        "Gravity Support Completion Audit",
    )
    return data


def write_coverage_repair_audit(output_dir: Path, coverage_report: dict[str, Any], dry_run: bool) -> dict[str, Any]:
    data = {
        "dry_run": dry_run,
        "status": "completed" if not coverage_report.get("missing_after") else "needs_repair",
        "missing_before": coverage_report.get("missing_before", []),
        "missing_after": coverage_report.get("missing_after", []),
        "coverage_repair_used": bool(coverage_report.get("used_repairs")),
        "coverage_repair_count": len(coverage_report.get("used_repairs", [])),
        "items": coverage_report.get("used_repairs", []),
        "skipped_candidates": coverage_report.get("skipped_candidates", []),
        "policy": "Coverage repair must choose the most plausible physical anchor and should prefer local physical contact over generic floor fallback.",
    }
    write_audit_pair(output_dir, "attach", "coverage_repair_audit", data, "Coverage Repair Audit")
    return data


def _ceiling_confirmed_from_guard(guard_result: dict[str, Any] | None) -> bool:
    if not isinstance(guard_result, dict):
        return False
    if guard_result.get("ceiling_confirmed") is True or guard_result.get("ceiling_anchor_allowed") is True:
        return True
    for node in _walk(guard_result):
        if isinstance(node, dict) and (node.get("ceiling_confirmed") is True or node.get("ceiling_anchor_allowed") is True or node.get("ceiling_present_for_attach") is True):
            return True
    return False


def _ceiling_confirmed_from_task_results(task_results: list[dict[str, Any]]) -> bool:
    accept_decisions = {"accept", "accepted", "true", "yes", "confirmed", "positive"}
    for root in task_results:
        if not isinstance(root, dict):
            continue
        for node in _walk(root):
            if not isinstance(node, dict):
                continue
            if node.get("relation") == "attach" and isinstance(node.get("parameters"), dict):
                if node["parameters"].get("object_anchor") == "ceiling":
                    decision_text = str(node.get("decision") or node.get("status") or node.get("result") or "accepted").lower()
                    if decision_text in accept_decisions or decision_text == "":
                        return True
            if node.get("object_anchor") == "ceiling" or node.get("anchor") == "ceiling":
                decision_text = str(node.get("decision") or node.get("status") or node.get("result") or "").lower()
                if decision_text in accept_decisions:
                    return True
            decision = node.get("ceiling_decision")
            if isinstance(decision, dict) and (
                decision.get("allow_ceiling_anchor") is True
                or decision.get("ceiling_confirmed") is True
                or decision.get("ceiling_anchor_allowed") is True
                or str(decision.get("status") or "").lower() in {"positive", "confirmed", "accepted", "yes", "true"}
            ):
                return True
            scene_level = node.get("scene_level_ceiling_present")
            if isinstance(scene_level, dict) and str(scene_level.get("judgement") or scene_level.get("status") or "").lower() in {"positive", "confirmed", "accepted", "yes", "true"}:
                return True
            if str(node.get("suggested_anchor") or "").lower() == "ceiling" and str(node.get("judgement") or node.get("status") or "").lower() in {"positive", "confirmed", "accepted", "yes", "true"}:
                return True
            judgements = node.get("ceiling_anchor_judgements")
            if isinstance(judgements, list):
                for item in judgements:
                    if isinstance(item, dict) and str(item.get("suggested_anchor") or "").lower() == "ceiling" and str(item.get("judgement") or item.get("status") or "").lower() in {"positive", "confirmed", "accepted", "yes", "true"}:
                        return True
            if (
                node.get("allow_ceiling_anchor") is True
                or node.get("allow_ceiling_attach") is True
                or node.get("ceiling_confirmed") is True
                or node.get("ceiling_anchor_allowed") is True
            ):
                return True
    return False


def collect_relations(
    guard_result: dict[str, Any] | None,
    task_results: list[dict[str, Any]],
    object_ids: list[str],
    blocked_wall_attach_anchors: dict[str, str] | None = None,
    wall_id_remap: dict[str, str] | None = None,
    objects: list[dict[str, Any]] | None = None,
    object_properties: dict[str, Any] | None = None,
    wall_evidence: dict[str, Any] | None = None,
    scene_state: dict[str, Any] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    collection = collect_accepted_attach_candidates(task_results, guard_result)
    coverage_collection = collect_coverage_repair_candidates(task_results, guard_result)
    ceiling_confirmed = (
        _ceiling_confirmed_from_guard(guard_result)
        or _ceiling_confirmed_from_task_results(task_results)
        or _ceiling_confirmed_from_objects(objects)
    )
    allowed_attach_anchors = {"floor", *WALL_IDS, *object_ids}
    if ceiling_confirmed:
        allowed_attach_anchors.add("ceiling")
    anchor_alias_map = _object_anchor_alias_map(objects)
    ceiling_light_ids = set(_ceiling_light_object_ids(objects))
    normalized: list[dict[str, Any]] = []
    skipped = list(collection["skipped"])
    blocked_wall_attach_anchors = blocked_wall_attach_anchors or {}
    wall_id_remap = wall_id_remap or {}
    seen = set()

    def target_already_covered(target_id: str) -> bool:
        return target_id in {
            rel.get("parameters", {}).get("object_target")
            for rel in normalized
            if rel.get("relation") in {"attach", "contain"}
        }

    def should_remap_wall_relation(item: dict[str, Any]) -> bool:
        source_path = str(item.get("source_path") or "").lower()
        raw = item.get("raw_candidate") if isinstance(item.get("raw_candidate"), dict) else {}
        raw_params = raw.get("parameters") if isinstance(raw.get("parameters"), dict) else {}
        # Wall attach recall runs before final wall canonicalization in the DAG, so
        # wall ids from that task are provisional candidate ids. Schema/supervisor
        # outputs, however, are written after final walls exist and may already
        # use canonical ids; remapping those again can turn a correct final wall id
        # back into a different provisional/final alias.
        if "wall_attach_candidate_recall" in source_path or "wall_attach_candidates" in source_path:
            return True
        if raw.get("candidate_wall_id") or raw.get("provisional_wall_id") or raw.get("geometry_wall_id"):
            return True
        if raw_params.get("candidate_wall_id") or raw_params.get("provisional_wall_id") or raw_params.get("geometry_wall_id"):
            return True
        return False

    def relation_remap(item: dict[str, Any], anchor: Any, target: Any) -> tuple[Any, Any]:
        if not should_remap_wall_relation(item):
            return anchor, target
        if anchor in wall_id_remap:
            anchor = wall_id_remap[anchor]
        if target in wall_id_remap:
            target = wall_id_remap[target]
        return anchor, target

    def raw_wall_aliases_for_candidate(item: dict[str, Any]) -> list[str]:
        raw = item.get("raw_candidate") if isinstance(item.get("raw_candidate"), dict) else {}
        aliases: list[str] = []

        def add(value: Any) -> None:
            if value is None:
                return
            if isinstance(value, str):
                for match in re.findall(r"wall_\d{3}", value):
                    if match not in aliases:
                        aliases.append(match)
            elif isinstance(value, dict):
                for child in value.values():
                    add(child)
            elif isinstance(value, list):
                for child in value:
                    add(child)

        for key in [
            "provisional_wall_id",
            "candidate_wall_id",
            "geometry_wall_id",
            "nearest_wall_id_by_footprint",
            "footprint_nearest_wall",
            "nearest_wall",
        ]:
            add(raw.get(key))
        geom = raw.get("geometry_evidence")
        if isinstance(geom, dict):
            for key in ["provisional_wall_id", "nearest_wall_id_by_footprint", "footprint_nearest_wall", "nearest_wall"]:
                add(geom.get(key))
        return aliases

    def remap_blocked_wall_anchor_from_raw_candidate(item: dict[str, Any], anchor: Any) -> Any:
        if not (isinstance(anchor, str) and anchor.startswith("wall_") and anchor in blocked_wall_attach_anchors):
            return anchor
        for alias in raw_wall_aliases_for_candidate(item):
            remapped = wall_id_remap.get(alias, alias)
            if (
                isinstance(remapped, str)
                and remapped.startswith("wall_")
                and remapped in allowed_attach_anchors
                and remapped not in blocked_wall_attach_anchors
            ):
                item.setdefault("wall_anchor_repair", {
                    "original_anchor": anchor,
                    "raw_wall_alias": alias,
                    "remapped_anchor": remapped,
                    "reason": "raw candidate geometry alias remapped away from weak/inferred wall anchor",
                })
                return remapped
        return anchor

    def try_add(item: dict[str, Any], source: str, require_missing_target: bool = False) -> bool:
        rel = json.loads(json.dumps(item["relation"], ensure_ascii=False))
        params = rel.get("parameters", {})
        anchor = params.get("object_anchor")
        target = params.get("object_target")
        anchor, target = relation_remap(item, anchor, target)
        params["object_anchor"] = anchor
        params["object_target"] = target
        relation_type = rel.get("relation")
        if relation_type == "attach" and isinstance(anchor, str) and anchor not in allowed_attach_anchors:
            mapped_anchor = anchor_alias_map.get(_alias_key(anchor))
            if mapped_anchor:
                params["object_anchor"] = mapped_anchor
                anchor = mapped_anchor
        if relation_type not in {"attach", "contain"}:
            skipped.append({**item, "reason": "relation_type_not_allowed", "merge_source": source})
            return False
        if target not in object_ids:
            skipped.append({**item, "reason": "object_target_not_in_final_objects", "merge_source": source})
            return False
        if relation_type == "attach":
            if anchor not in allowed_attach_anchors:
                skipped.append({**item, "reason": "object_anchor_not_allowed", "merge_source": source})
                return False
            repaired_anchor = remap_blocked_wall_anchor_from_raw_candidate(item, anchor)
            if repaired_anchor != anchor:
                params["object_anchor"] = repaired_anchor
                anchor = repaired_anchor
            if anchor in blocked_wall_attach_anchors:
                skipped.append(
                    {
                        **item,
                        "reason": "weak_or_inferred_wall_anchor_not_allowed_for_attach",
                        "blocked_wall_anchor_reason": blocked_wall_attach_anchors[anchor],
                        "merge_source": source,
                    }
                )
                return False
        elif relation_type == "contain":
            if anchor not in object_ids:
                skipped.append({**item, "reason": "contain_anchor_must_be_object", "merge_source": source})
                return False
            if anchor == target:
                skipped.append({**item, "reason": "contain_anchor_target_same_object", "merge_source": source})
                return False
        if require_missing_target and target_already_covered(target):
            skipped.append({**item, "reason": "target_already_covered", "merge_source": source})
            return False
        key = (relation_type, anchor, target)
        if key in seen:
            skipped.append({**item, "reason": "duplicate_relation", "merge_source": source})
            return False
        normalized.append({"relation": relation_type, "parameters": {"object_anchor": anchor, "object_target": target}})
        seen.add(key)
        return True

    for item in collection["accepted"]:
        try_add(item, "accepted_relation")

    contain_conflict_report = collect_contain_conflicts(scene_state, task_results)
    contain_conflict_resolutions = collect_contain_conflict_resolutions(guard_result)
    contain_conflict_applied: list[dict[str, Any]] = []
    contain_conflict_unresolved: list[dict[str, Any]] = []
    contain_conflict_candidate_removals: list[dict[str, Any]] = []
    schema_guard_support_pairs: set[tuple[str, str]] = set()

    # Remove both provisional verdicts before applying the single SchemaGuard
    # decision. This prevents insertion order or relation type from becoming a
    # hidden semantic priority.
    for conflict in contain_conflict_report.get("conflicts", []):
        if not isinstance(conflict, dict):
            continue
        conflict_pair = (conflict.get("object_anchor"), conflict.get("object_target"))
        kept_relations: list[dict[str, Any]] = []
        for relation in normalized:
            params = relation.get("parameters", {}) if isinstance(relation.get("parameters"), dict) else {}
            pair = (params.get("object_anchor"), params.get("object_target"))
            if pair == conflict_pair and relation.get("relation") in {"attach", "contain"}:
                removal = {
                    "relation": relation,
                    "reason": "schema_guard_attach_contain_arbitration",
                    "conflict_id": conflict.get("conflict_id"),
                    "merge_source": "contain_conflict_pre_arbitration",
                }
                contain_conflict_candidate_removals.append(removal)
                skipped.append(removal)
                continue
            kept_relations.append(relation)
        normalized = kept_relations
    seen = {
        (
            relation.get("relation"),
            relation.get("parameters", {}).get("object_anchor"),
            relation.get("parameters", {}).get("object_target"),
        )
        for relation in normalized
        if isinstance(relation, dict) and isinstance(relation.get("parameters"), dict)
    }
    trusted_multi_level_pairs: set[tuple[str, str]] = set()

    for conflict in contain_conflict_report.get("conflicts", []):
        if not isinstance(conflict, dict):
            continue
        conflict_id = str(conflict.get("conflict_id") or "")
        anchor = conflict.get("object_anchor")
        target = conflict.get("object_target")
        resolution = contain_conflict_resolutions.get(conflict_id)
        if not isinstance(resolution, dict):
            contain_conflict_unresolved.append({**conflict, "reason": "schema_guard_resolution_missing"})
            continue
        decision = str(resolution.get("decision") or resolution.get("verdict") or "").lower()
        accepted_attach = conflict.get("accepted_attach") if isinstance(conflict.get("accepted_attach"), dict) else {}
        accepted_contain = conflict.get("accepted_contain") if isinstance(conflict.get("accepted_contain"), dict) else {}
        support_anchor = (
            resolution.get("support_anchor")
            or resolution.get("object_support_anchor")
            or accepted_attach.get("object_anchor")
        )
        surface_role, surface_role_source = _normalized_contain_surface_role(
            resolution,
            conflict.get("anchor_category"),
        )
        effective_decision = decision
        policy_reason = None
        if surface_role == CONTAIN_INTERNAL_SURFACE_ROLE:
            effective_decision = "accept_contain"
            if decision not in {"accept_contain", "accept", "accepted"}:
                policy_reason = "internal shelf/cubby membership is contain under the shared ontology"
        elif surface_role in CONTAIN_EXTERNAL_SURFACE_ROLES:
            effective_decision = "reject_contain_use_support"
            if decision not in {"reject_contain_use_support", "reject_with_support", "use_support"}:
                policy_reason = "outermost top or exterior support is attach under the shared ontology"
        elif decision in {"reject_contain_use_support", "reject_with_support", "use_support"} and accepted_contain:
            effective_decision = "accept_contain"
            policy_reason = "unsubstantiated Guard rejection cannot override an accepted contain specialist verdict"

        if effective_decision in {"accept_contain", "accept", "accepted"}:
            item = {
                "relation": {"relation": "contain", "parameters": {"object_anchor": anchor, "object_target": target}},
                "source_path": "DeliverySchemaGuard.contain_conflict_resolutions",
                "raw_candidate": resolution,
                "decision": "accept",
            }
            if try_add(item, "contain_conflict_schema_guard"):
                if surface_role == CONTAIN_INTERNAL_SURFACE_ROLE:
                    trusted_multi_level_pairs.add((str(anchor), str(target)))
                contain_conflict_applied.append(
                    {
                        "conflict_id": conflict_id,
                        "decision": "accept_contain",
                        "schema_guard_decision": decision,
                        "support_surface_role": surface_role,
                        "support_surface_role_source": surface_role_source,
                        "policy_reason": policy_reason,
                        "relation": item["relation"],
                        "reason": resolution.get("reason") or resolution.get("reason_code") or resolution.get("explanation"),
                    }
                )
            continue
        if effective_decision in {"reject_contain_use_support", "reject_with_support", "use_support"} and support_anchor:
            schema_guard_support_pairs.add((str(support_anchor), str(target)))
            item = {
                "relation": {"relation": "attach", "parameters": {"object_anchor": support_anchor, "object_target": target}},
                "source_path": "DeliverySchemaGuard.contain_conflict_resolutions",
                "raw_candidate": resolution,
                "decision": "accept",
            }
            if try_add(item, "contain_conflict_schema_guard"):
                contain_conflict_applied.append(
                    {
                        "conflict_id": conflict_id,
                        "decision": "reject_contain_use_support",
                        "schema_guard_decision": decision,
                        "support_surface_role": surface_role,
                        "support_surface_role_source": surface_role_source,
                        "policy_reason": policy_reason,
                        "relation": item["relation"],
                        "reason": resolution.get("reason") or resolution.get("reason_code") or resolution.get("explanation"),
                    }
                )
            continue
        contain_conflict_unresolved.append(
            {
                **conflict,
                "reason": "schema_guard_resolution_not_actionable",
                "schema_guard_resolution": resolution,
                "support_surface_role": surface_role,
                "support_surface_role_source": surface_role_source,
            }
        )

    # A specific conflict resolution is more authoritative than the Guard's
    # provisional aggregate relation list. When support is selected, discard any
    # provisional contain for that exact support pair before later promotion.
    if schema_guard_support_pairs:
        kept_relations: list[dict[str, Any]] = []
        for relation in normalized:
            params = relation.get("parameters", {}) if isinstance(relation.get("parameters"), dict) else {}
            pair = (str(params.get("object_anchor") or ""), str(params.get("object_target") or ""))
            if pair in schema_guard_support_pairs and relation.get("relation") == "contain":
                removal = {
                    "relation": relation,
                    "reason": "schema_guard_selected_support_for_same_pair",
                    "merge_source": "contain_conflict_post_arbitration",
                }
                contain_conflict_candidate_removals.append(removal)
                skipped.append(removal)
                continue
            kept_relations.append(relation)
        normalized = kept_relations
        seen = {
            (
                relation.get("relation"),
                relation.get("parameters", {}).get("object_anchor"),
                relation.get("parameters", {}).get("object_target"),
            )
            for relation in normalized
            if isinstance(relation, dict) and isinstance(relation.get("parameters"), dict)
        }

    multi_level_contain_promotion = collect_multi_level_support_contain_promotions(task_results)
    conflict_pairs = {
        (str(item.get("object_anchor") or ""), str(item.get("object_target") or ""))
        for item in contain_conflict_report.get("conflicts", [])
        if isinstance(item, dict)
    }
    conflict_pairs.update(schema_guard_support_pairs)
    promotion_applied: list[dict[str, Any]] = []
    for item in multi_level_contain_promotion.get("accepted", []):
        pair = (str(item.get("object_anchor") or ""), str(item.get("object_target") or ""))
        if pair in conflict_pairs:
            multi_level_contain_promotion["skipped"].append({
                **item,
                "reason": (
                    "schema_guard_selected_support_for_same_pair"
                    if pair in schema_guard_support_pairs
                    else "attach_contain_conflict_already_delegated_to_schema_guard"
                ),
            })
            continue
        trusted_multi_level_pairs.add(pair)
        if try_add(item, "support_multi_level_contain_promotion"):
            promotion_applied.append({
                "relation": item.get("relation"),
                "source_task_id": item.get("source_task_id"),
                "source_path": item.get("source_path"),
                "support_surface_role": item.get("support_surface_role"),
                "reason_code": item.get("reason_code"),
            })
    multi_level_contain_promotion["applied"] = promotion_applied

    unresolved_contain_conflict_targets = {
        str(item.get("object_target"))
        for item in contain_conflict_unresolved
        if isinstance(item, dict) and item.get("object_target")
    }

    normalized, contain_geometry_validation = filter_contain_relations_by_geometry(
        normalized,
        scene_state,
        trusted_agent_pairs=trusted_multi_level_pairs,
    )
    for item in contain_geometry_validation.get("items", []):
        if item.get("decision") == "reject":
            skipped.append({
                "relation": item.get("relation"),
                "reason": "contain_target_clearly_outside_anchor_obb",
                "geometry_decision": item,
                "merge_source": "contain_geometry_policy",
            })

    coverage_before = _target_coverage(object_ids, normalized)
    used_repairs: list[dict[str, Any]] = []
    coverage_skipped = list(coverage_collection["skipped"])
    for target in coverage_before["missing_object_targets"]:
        if target not in ceiling_light_ids or not ceiling_confirmed:
            continue
        item = {
            "relation": {"relation": "attach", "parameters": {"object_anchor": "ceiling", "object_target": target}},
            "source_path": "local_ceiling_light_category_coverage",
            "reason": "object category/description indicates a ceiling light fixture; ceiling anchor is locally confirmed for coverage",
            "evidence_used": ["scene_state.object_candidates.category", "scene_state.object_candidates.description"],
            "coverage_repair": True,
        }
        if try_add(item, "local_ceiling_light_category_coverage", require_missing_target=True):
            used_repairs.append(
                {
                    "object_target": target,
                    "selected_anchor": "ceiling",
                    "relation": item["relation"],
                    "reason": item["reason"],
                    "evidence_used": item["evidence_used"],
                    "coverage_repair": True,
                    "source_path": item["source_path"],
                }
            )
    for item in coverage_collection["accepted"]:
        target = item["relation"].get("parameters", {}).get("object_target")
        if target in unresolved_contain_conflict_targets:
            coverage_skipped.append({**item, "reason": "contain_conflict_guard_missing_prevents_coverage_repair"})
            continue
        if target not in coverage_before["missing_object_targets"]:
            coverage_skipped.append({**item, "reason": "target_not_missing"})
            continue
        repair_anchor = str(item["relation"].get("parameters", {}).get("object_anchor") or "").lower()
        if repair_anchor == "ceiling" and target not in ceiling_light_ids:
            coverage_skipped.append(
                {
                    **item,
                    "reason": "non_fixture_ceiling_coverage_repair_requires_dedicated_ceiling_evidence",
                }
            )
            continue
        if try_add(item, "coverage_repair", require_missing_target=True):
            params = item["relation"].get("parameters", {})
            used_repairs.append(
                {
                    "object_target": params.get("object_target"),
                    "selected_anchor": params.get("object_anchor"),
                    "relation": item["relation"],
                    "reason": item.get("reason"),
                    "evidence_used": item.get("evidence_used"),
                    "coverage_repair": True,
                    "source_path": item.get("source_path"),
                }
            )
        else:
            coverage_skipped.append({**item, "reason": "coverage_relation_not_merged"})

    # Last-resort local coverage for freestanding / floor-supported objects.
    # This keeps fast async runs from failing schema when a support agent omits
    # obvious floor contacts. It deliberately skips ceiling fixtures and objects
    # explicitly described as wall-mounted/hung/embedded.
    object_by_id = _object_context_by_id(objects)
    explicit_semantic_wall_roles = collect_high_confidence_explicit_wall_roles(task_results)
    coverage_after_agent_repairs = _target_coverage(object_ids, normalized)
    for target in coverage_after_agent_repairs["missing_object_targets"]:
        if target in unresolved_contain_conflict_targets:
            coverage_skipped.append(
                {
                    "relation": {"relation": "attach", "parameters": {"object_anchor": "floor", "object_target": target}},
                    "reason": "contain_conflict_guard_missing_prevents_floor_fallback",
                    "coverage_repair": True,
                }
            )
            continue
        obj = object_by_id.get(str(target), {})
        if (
            target in ceiling_light_ids
            or _is_explicit_wall_mounted_object(obj)
            or target in explicit_semantic_wall_roles
        ):
            if target in explicit_semantic_wall_roles:
                coverage_skipped.append({
                    "relation": {
                        "relation": "attach",
                        "parameters": {"object_anchor": "floor", "object_target": target},
                    },
                    "reason": "high_confidence_wall_only_object_cannot_use_floor_coverage_fallback",
                    "semantic_wall_role": explicit_semantic_wall_roles[target],
                    "coverage_repair": True,
                })
            continue
        item = {
            "relation": {"relation": "attach", "parameters": {"object_anchor": "floor", "object_target": target}},
            "source_path": "local_floor_fallback_coverage",
            "reason": "No accepted attach/contain target coverage remained after relation merge; defaulting freestanding object to floor support for schema coverage.",
            "evidence_used": ["scene_state.object_candidates", "local_schema_coverage_fallback"],
            "coverage_repair": True,
        }
        if try_add(item, "local_floor_fallback_coverage", require_missing_target=True):
            used_repairs.append(
                {
                    "object_target": target,
                    "selected_anchor": "floor",
                    "relation": item["relation"],
                    "reason": item["reason"],
                    "evidence_used": item["evidence_used"],
                    "coverage_repair": True,
                    "source_path": item["source_path"],
                }
            )

    normalized, gravity_support_completion = complete_floor_support_for_structural_anchors(
        normalized,
        objects,
        wall_evidence,
    )

    floor_semantic_wall_roles = collect_high_confidence_floor_wall_roles(task_results)
    floor_wall_policy = collect_floor_wall_relation_policy_overrides(
        collection["accepted"],
        objects,
        wall_evidence,
        floor_semantic_wall_roles,
        wall_id_remap,
    )
    normalized, semantic_wall_contact_reconciliation, reconciled_contact_overrides = (
        reconcile_semantic_wall_contact_relations(
            normalized,
            collection["accepted"],
            objects,
            task_results,
            wall_evidence,
            used_repairs,
            blocked_wall_attach_anchors,
            wall_id_remap,
        )
    )
    trusted_agent_contact_overrides = dict(floor_wall_policy["trusted_contact"])
    trusted_agent_contact_overrides.update(reconciled_contact_overrides)
    raw_visual_reconstruction_overrides = collect_visual_reconstruction_wall_contact_overrides(task_results)
    visual_reconstruction_overrides: dict[tuple[str, str], dict[str, Any]] = {}
    for (target, wall_id), item in raw_visual_reconstruction_overrides.items():
        visual_reconstruction_overrides[(target, wall_id)] = item
        visual_reconstruction_overrides[(target, wall_id_remap.get(wall_id, wall_id))] = item
    normalized, flat_edge_filtered_wall_attaches, flat_edge_evaluations = filter_floor_supported_wall_attaches_without_flat_contact_edge(
        normalized,
        objects,
        wall_evidence,
        floor_semantic_wall_roles,
        visual_reconstruction_overrides,
        trusted_agent_contact_overrides,
        floor_wall_policy["proximity_only"],
    )
    for item in flat_edge_filtered_wall_attaches:
        skipped.append({**item, "merge_source": "flat_contact_edge_wall_attach_policy"})

    normalized, attach_contain_conflicts = resolve_attach_contain_conflicts(normalized)
    for item in attach_contain_conflicts:
        skipped.append({**item, "merge_source": "attach_contain_pair_policy"})

    filtered_surface_wall_attaches: list[dict[str, Any]] = []
    normalized, filtered_surface_wall_attaches = filter_surface_supported_wall_attaches(
        normalized,
        objects,
        explicit_semantic_wall_roles,
    )
    for item in filtered_surface_wall_attaches:
        skipped.append({**item, "merge_source": "surface_supported_wall_attach_policy"})

    preferred_wall_by_target: dict[str, str] = {}
    for item in collection["accepted"]:
        rel = item.get("relation") if isinstance(item.get("relation"), dict) else None
        params = rel.get("parameters", {}) if isinstance(rel, dict) else {}
        anchor = params.get("object_anchor")
        target = params.get("object_target")
        if not (isinstance(anchor, str) and anchor.startswith("wall_") and isinstance(target, str)):
            continue
        remapped_anchor, _ = relation_remap(item, anchor, target)
        if isinstance(remapped_anchor, str) and remapped_anchor.startswith("wall_"):
            preferred_wall_by_target.setdefault(target, remapped_anchor)

    normalized, removed_multi_wall_attaches = enforce_single_wall_attach_per_target(
        normalized,
        preferred_wall_by_target,
    )
    for item in removed_multi_wall_attaches:
        skipped.append({**item, "merge_source": "single_wall_attach_per_target_policy"})

    filtered_wall_mounted_floor_attaches: list[dict[str, Any]] = []
    normalized, filtered_wall_mounted_floor_attaches = filter_wall_mounted_floor_attaches(normalized, objects)
    for item in filtered_wall_mounted_floor_attaches:
        skipped.append({**item, "merge_source": "wall_mounted_floor_attach_policy"})

    normalized, relation_normalization_audit = normalize_floor_covering_relations(normalized, object_properties)
    for item in relation_normalization_audit.get("items", []):
        if isinstance(item, dict) and item.get("original_relation"):
            skipped.append({
                "relation": item.get("original_relation"),
                "normalized_relation": item.get("normalized_relation"),
                "reason": item.get("reason"),
                "merge_source": "floor_covering_relation_normalization",
                "policy": relation_normalization_audit.get("policy"),
            })

    normalized, redundant_floor_attaches = filter_redundant_floor_attaches_for_object_supported_targets(normalized)
    for item in redundant_floor_attaches:
        skipped.append({**item, "merge_source": "single_gravity_support_path_policy"})

    coverage_after = _target_coverage(object_ids, normalized)
    normalized.sort(key=lambda item: (sorted_object_id_key(item["parameters"]["object_target"]), item.get("relation"), item["parameters"]["object_anchor"]))
    return normalized, {
        "accepted_candidates": collection["accepted"],
        "skipped_candidates": skipped,
        "attach_contain_conflict_resolution": {
            "defensive_removed_relation_count": len(attach_contain_conflicts),
            "items": attach_contain_conflicts,
            "policy": "No relation type has fixed priority; any pair reaching this late guard with both attach and contain is removed as unresolved.",
        },
        "contain_relation_count": len([rel for rel in normalized if rel.get("relation") == "contain"]),
        "contain_conflict_resolution": {
            "schema_version": "contain_conflict_resolution_audit",
            "detected_count": len(contain_conflict_report.get("conflicts", [])),
            "conflicts": contain_conflict_report.get("conflicts", []),
            "pre_arbitration_removed_candidates": contain_conflict_candidate_removals,
            "schema_guard_resolutions": [
                contain_conflict_resolutions[conflict_id]
                for conflict_id in sorted(contain_conflict_resolutions)
                if any(item.get("conflict_id") == conflict_id for item in contain_conflict_report.get("conflicts", []))
            ],
            "applied": contain_conflict_applied,
            "unresolved": contain_conflict_unresolved,
            "policy": (
                "Independent accepted attach/contain verdicts, support-inside versus contain-reject disagreements, "
                "and accepted-attach versus high-recall-contain-reject disagreements are escalated to the existing "
                "DeliverySchemaGuard call. Neither relation type wins by default. Generic coverage fallback runs only "
                "after arbitration."
            ),
        },
        "contain_geometry_validation": contain_geometry_validation,
        "multi_level_support_contain_promotion": multi_level_contain_promotion,
        "gravity_attach_normalization": {
            "removed_floor_attach_count": len(redundant_floor_attaches),
            "items": redundant_floor_attaches,
            "policy": "Object support and floor support are mutually exclusive gravity paths; keep object support and remove generic floor fallback.",
        },
        "floor_wall_semantic_policy": {
            "trusted_contact_overrides": list(floor_wall_policy["trusted_contact"].values()),
            "proximity_only_vetoes": list(floor_wall_policy["proximity_only"].values()),
            "policy": (
                "The accepted-candidate preservation policy is unchanged: "
                "verified flat contact plus local possible-contact evidence, or its "
                "existing high-confidence tall-storage rear-contact exception, may "
                "survive noisy lower-mesh edges. New missing-pair completion is handled "
                "only by semantic_wall_contact_reconciliation."
            ),
            "additional_api_calls": 0,
            "additional_geometry_or_render_passes": 0,
            "accepted_candidate_preservation_policy_unchanged": True,
        },
        "semantic_wall_contact_reconciliation": semantic_wall_contact_reconciliation,
        "flat_contact_edge_wall_attach_filter": {
            "filtered_count": len(flat_edge_filtered_wall_attaches),
            "items": flat_edge_filtered_wall_attaches,
            "evaluations": flat_edge_evaluations,
            "visual_reconstruction_overrides": list(raw_visual_reconstruction_overrides.values()),
            "policy": "Ordinary floor-supported wall attach requires a finite wall-parallel flat contact edge from the real lower-mesh XZ hull. Consecutive collinear fragments may be merged; wrapper bbox/OBB edges and isolated point/corner contact are insufficient. An exact pair may be preserved only when the wall-attach Agent marks its decision final and supplies both visual and mask evidence for a reconstruction conflict.",
        },
        "surface_supported_wall_attach_filter": {
            "filtered_count": len(filtered_surface_wall_attaches),
            "items": filtered_surface_wall_attaches,
            "policy": "Object-supported targets keep wall attach only with explicit high-confidence wall-mounted/embedded/hung/rack/bracket evidence.",
        },
        "wall_mounted_floor_attach_filter": {
            "filtered_count": len(filtered_wall_mounted_floor_attaches),
            "items": filtered_wall_mounted_floor_attaches,
            "policy": "Explicitly wall-mounted objects retain their accepted wall attach and do not also receive floor attach solely for coverage.",
        },
        "single_wall_attach_per_target": {
            "removed_count": len(removed_multi_wall_attaches),
            "items": removed_multi_wall_attaches,
            "policy": "Each object target may have at most one wall attach; floor/object support relations remain independent.",
        },
        "gravity_support_completion": gravity_support_completion,
        "relation_normalization": relation_normalization_audit,
        "ceiling_confirmed": ceiling_confirmed,
        "blocked_wall_attach_anchors": blocked_wall_attach_anchors,
        "wall_id_remap_applied_to_relations": wall_id_remap,
        "coverage_repair": {
            "missing_before": coverage_before["missing_object_targets"],
            "missing_after": coverage_after["missing_object_targets"],
            "used_repairs": used_repairs,
            "skipped_candidates": coverage_skipped,
        },
    }


def _geometry_by_object_wall(wall_evidence: dict[str, Any] | None) -> dict[tuple[str, str], dict[str, Any]]:
    if not isinstance(wall_evidence, dict):
        return {}
    result = {}
    for item in wall_evidence.get("wall_attach_geometry", []) if isinstance(wall_evidence.get("wall_attach_geometry"), list) else []:
        if isinstance(item, dict) and item.get("object_id") and item.get("wall_id"):
            result[(item["object_id"], item["wall_id"])] = item
    return result


def _first_non_none(*values: Any) -> Any:
    for value in values:
        if value is not None:
            return value
    return None


def normalize_geometry_evidence(raw: Any) -> dict[str, Any]:
    """Normalize unstable agent geometry audit fields without dropping raw evidence."""
    normalized: dict[str, Any] = {
        "center_to_wall_distance": None,
        "footprint_min_distance_to_wall": None,
        "bbox_side_contact": None,
        "possible_against_wall_contact": None,
        "is_floor_supported": None,
        "geometry_classification": None,
        "geometry_reliability": None,
        "raw_geometry_evidence": raw,
        "geometry_evidence_items": [],
    }

    def merge_dict(item: dict[str, Any]) -> None:
        updates = {
            "center_to_wall_distance": item.get("center_to_wall_distance"),
            "footprint_min_distance_to_wall": item.get("footprint_min_distance_to_wall"),
            "bbox_side_contact": _first_non_none(item.get("bbox_side_contact"), item.get("bbox_side_near_wall")),
            "possible_against_wall_contact": item.get("possible_against_wall_contact"),
            "is_floor_supported": item.get("is_floor_supported"),
            "geometry_classification": item.get("geometry_classification"),
            "geometry_reliability": item.get("geometry_reliability"),
        }
        for key, value in updates.items():
            if normalized.get(key) is None and value is not None:
                normalized[key] = value

    if isinstance(raw, dict):
        merge_dict(raw)
    elif isinstance(raw, list):
        for item in raw:
            if isinstance(item, dict):
                merge_dict(item)
            elif isinstance(item, str):
                normalized["geometry_evidence_items"].append(item)
            elif item is not None:
                normalized["geometry_evidence_items"].append(str(item))
    elif isinstance(raw, str):
        normalized["geometry_evidence_items"].append(raw)
    elif raw is not None:
        normalized["geometry_evidence_items"].append(str(raw))

    return normalized


def _raw_geometry_from_candidate(item: dict[str, Any], fallback: Any) -> Any:
    for key in ["geometry_evidence", "geometry", "geometric_evidence", "geometry_evidence_items"]:
        if key in item and item.get(key) is not None:
            return item.get(key)
    return fallback


def collect_wall_attach_recall(task_results: list[dict[str, Any]], guard_result: dict[str, Any] | None = None, wall_evidence: dict[str, Any] | None = None) -> dict[str, Any]:
    roots: list[Any] = list(task_results)
    if guard_result:
        roots.append(guard_result)
    candidates: list[dict[str, Any]] = []
    geometry_lookup = _geometry_by_object_wall(wall_evidence)
    for root in roots:
        for node in _walk(root):
            if not isinstance(node, dict):
                continue
            for key in ["wall_attach_candidates", "wall_attach_recall", "wall_attach_decisions", "attach_decisions"]:
                value = node.get(key)
                if isinstance(value, list):
                    for item in value:
                        if isinstance(item, dict):
                            candidates.append(item)
                elif isinstance(value, dict) and isinstance(value.get("candidates"), list):
                    for item in value["candidates"]:
                        if isinstance(item, dict):
                            candidates.append(item)
    accepted = []
    rejected = []
    intermediate = []
    warnings = []
    accept_decisions = {"accept", "accepted", "true", "yes"}
    reject_decisions = {"reject", "rejected", "false", "no"}
    intermediate_decisions = {"needs_consistency_check", "needs_wall_resolution", "not_final_yet", "pending", "possible_wall_contact", "needs_more_evidence"}
    for item in candidates:
        decision = _candidate_decision(item)
        anchor = item.get("candidate_wall_id") or item.get("wall_id") or item.get("object_anchor")
        if not anchor and isinstance(item.get("parameters"), dict):
            anchor = item["parameters"].get("object_anchor")
        target = item.get("object_target") or item.get("object_id")
        if not target and isinstance(item.get("parameters"), dict):
            target = item["parameters"].get("object_target")
        raw_geometry = _raw_geometry_from_candidate(item, geometry_lookup.get((target, anchor)))
        geometry = normalize_geometry_evidence(raw_geometry)
        visual = item.get("visual_evidence") or item.get("evidence") or item.get("evidence_used")
        mask_evidence = item.get("mask_evidence") or item.get("mask_overlay_evidence")
        topdown_evidence = item.get("topdown_evidence") or item.get("top_down_evidence")
        geometry_reliability = item.get("geometry_reliability") or geometry.get("geometry_reliability") or "weak"
        footprint_distance = geometry.get("footprint_min_distance_to_wall")
        possible_contact = geometry.get("possible_against_wall_contact") is True or (isinstance(footprint_distance, (int, float)) and footprint_distance <= 0.03)
        visual_suggests_contact = isinstance(visual, str) and any(token in visual.lower() for token in ["against wall", "touch", "contact", "mounted", "embedded", "wall"])
        if decision is None and (possible_contact or visual_suggests_contact):
            decision = "needs_consistency_check"
        decision_is_final = bool(item.get("decision_is_final")) if "decision_is_final" in item else decision in (accept_decisions | reject_decisions)
        if geometry_reliability == "depends_on_unstable_wall" or decision in intermediate_decisions:
            decision_is_final = False
        normalized = {
            "decision": decision or None,
            "decision_is_final": decision_is_final,
            "object_target": target,
            "candidate_wall_id": anchor,
            "wall_contact_type": item.get("wall_contact_type") or geometry.get("geometry_classification"),
            "visual_evidence": visual,
            "mask_evidence": mask_evidence,
            "topdown_evidence": topdown_evidence,
            "geometry_evidence": {
                "center_to_wall_distance": geometry.get("center_to_wall_distance"),
                "footprint_min_distance_to_wall": geometry.get("footprint_min_distance_to_wall"),
                "bbox_side_contact": geometry.get("bbox_side_contact"),
                "possible_against_wall_contact": geometry.get("possible_against_wall_contact"),
                "is_floor_supported": geometry.get("is_floor_supported"),
                "geometry_reliability": geometry_reliability,
                "raw_geometry_evidence": geometry.get("raw_geometry_evidence"),
                "geometry_evidence_items": geometry.get("geometry_evidence_items", []),
            },
            "description_used_as": item.get("description_used_as") or "weak_support",
            "reason": item.get("reason") or item.get("rationale"),
            "selective_topdown_used": item.get("selective_topdown_used"),
            "raw": item,
        }
        if decision in accept_decisions:
            accepted.append(normalized)
        elif decision in intermediate_decisions:
            intermediate.append(normalized)
        else:
            reason_text = " ".join(
                str(value or "")
                for value in [
                    normalized.get("reason"),
                    normalized.get("description_used_as"),
                    normalized.get("wall_contact_type"),
                    normalized["geometry_evidence"].get("geometry_reliability"),
                ]
            ).lower()
            weak_reject = (
                decision in reject_decisions
                and (
                    normalized["geometry_evidence"].get("footprint_min_distance_to_wall") is None
                    or visual_suggests_contact
                    or possible_contact
                    or any(token in reason_text for token in ["center", "distance", "description", "category", "floor", "non-final", "unstable"])
                )
            )
            if weak_reject or (normalized["geometry_evidence"].get("footprint_min_distance_to_wall") is None and not visual):
                normalized["warning"] = "weak_or_unstable_rejection_reason"
                warnings.append({"object_target": target, "candidate_wall_id": anchor, "warning": "weak_or_unstable_rejection_reason"})
            rejected.append(normalized)
    return {"candidate_count": len(candidates), "accepted": accepted, "rejected": rejected, "intermediate": intermediate, "warnings": warnings}


def write_accepted_attach_merge_check(
    output_dir: Path,
    accepted_candidates: list[dict[str, Any]],
    final_relations: list[dict[str, Any]],
    final_outputs_blocked_reason: str | None = None,
    skipped_candidates: list[dict[str, Any]] | None = None,
    wall_id_remap: dict[str, str] | None = None,
) -> dict[str, Any]:
    wall_id_remap = wall_id_remap or {}

    def remapped_key(rel: dict[str, Any]) -> tuple[Any, Any, Any]:
        params = rel.get("parameters", {}) if isinstance(rel.get("parameters"), dict) else {}
        anchor = params.get("object_anchor")
        target = params.get("object_target")
        if anchor in wall_id_remap:
            anchor = wall_id_remap[anchor]
        if target in wall_id_remap:
            target = wall_id_remap[target]
        return (rel.get("relation"), anchor, target)

    # Final relations have already been canonicalized during collect_relations;
    # only accepted candidate keys need remapping for comparison.
    final_keys = {
        (rel.get("relation"), rel.get("parameters", {}).get("object_anchor"), rel.get("parameters", {}).get("object_target"))
        for rel in final_relations
        if isinstance(rel, dict)
    }
    policy_filtered_keys: dict[tuple[Any, Any, Any], dict[str, Any]] = {}
    for skipped in skipped_candidates or []:
        if not isinstance(skipped, dict):
            continue
        if skipped.get("reason") not in {
            "relation_type_not_allowed",
            "object_target_not_in_final_objects",
            "object_anchor_not_allowed",
            "contain_anchor_must_be_object",
            "contain_anchor_target_same_object",
            "weak_or_inferred_wall_anchor_not_allowed_for_attach",
            "surface_supported_object_wall_attach_filtered",
            "floor_supported_wall_attach_without_valid_flat_contact_edge",
            "schema_guard_attach_contain_arbitration",
            "unresolved_attach_contain_same_pair_requires_schema_guard",
            "contain_target_clearly_outside_anchor_obb",
            "duplicate_relation",
            "multiple_wall_attach_same_target_deduped",
            "target_already_covered",
            "floor_covering_anchor_normalized_to_floor",
            "duplicate_after_floor_covering_normalization_dropped",
        }:
            continue
        rel = skipped.get("relation") or skipped.get("relation_candidate")
        if not rel and isinstance(skipped.get("raw_candidate"), dict):
            rel = _relation_from_candidate(skipped["raw_candidate"])
        if not rel and isinstance(skipped.get("candidate"), dict):
            rel = _relation_from_candidate(skipped["candidate"])
        if isinstance(rel, dict):
            key = remapped_key(rel)
            policy_filtered_keys[key] = skipped
    items = []
    errors = []
    for candidate in accepted_candidates:
        rel = candidate.get("relation") or candidate.get("relation_candidate") or candidate.get("relation")
        if not rel and isinstance(candidate.get("raw_candidate"), dict):
            rel = _relation_from_candidate(candidate["raw_candidate"])
        if not rel and isinstance(candidate.get("relation"), dict):
            rel = candidate["relation"]
        if isinstance(rel, dict):
            source_path = str(candidate.get("source_path") or "")
            source_path_l = source_path.lower()
            # Schema/supervisor roots may provide a whole relations_json snapshot.
            # That snapshot is useful as a merge hint, but it is not an original
            # task-level accepted candidate. Treating it as a hard requirement can
            # falsely fail finalization when local canonical wall remapping or
            # policy filters choose the authoritative final relation set.
            if source_path_l.startswith("root_") and source_path_l.endswith(".relations_json"):
                items.append({
                    "accepted_candidate": rel,
                    "status": "schema_snapshot_not_hard_checked",
                    "source_path": source_path,
                    "severity": "informational",
                })
                continue
            key = remapped_key(rel)
            if key in final_keys:
                status = "present_in_final_relations"
            elif key in policy_filtered_keys:
                status = "intentionally_filtered_by_policy"
            elif final_outputs_blocked_reason:
                status = "blocked_by_missing_final_outputs"
            else:
                status = "missing_from_final_relations"
            item = {"accepted_candidate": rel, "status": status, "source_path": candidate.get("source_path")}
            if status == "intentionally_filtered_by_policy":
                item["policy_filter_reason"] = policy_filtered_keys[key].get("reason")
                item["blocked_wall_anchor_reason"] = policy_filtered_keys[key].get("blocked_wall_anchor_reason")
                item["policy"] = policy_filtered_keys[key].get("policy")
                item["severity"] = "policy_filtered"
                item["note"] = "Accepted candidate was intentionally skipped by local merge policy and is not required in final relations."
            elif final_outputs_blocked_reason and status == "blocked_by_missing_final_outputs":
                item["blocked_reason"] = final_outputs_blocked_reason
                item["severity"] = "blocked"
            elif status != "present_in_final_relations":
                item["severity"] = "error"
                errors.append(item)
            items.append(item)
    data = {
        "valid": not errors,
        "items": items,
        "errors": errors,
        "final_outputs_blocked_reason": final_outputs_blocked_reason,
        "wall_id_remap_used_for_check": wall_id_remap,
    }
    write_audit_pair(output_dir, "attach", "accepted_attach_merge_check", data, "Accepted Attach Merge Check")
    return data


def write_attach_audit(
    output_dir: Path,
    task_results: list[dict[str, Any]],
    guard_result: dict[str, Any] | None,
    relations: list[dict[str, Any]],
    merge_report: dict[str, Any],
    dry_run: bool,
    wall_evidence: dict[str, Any] | None = None,
    final_outputs_blocked_reason: str | None = None,
) -> dict[str, Any]:
    recall = collect_wall_attach_recall(task_results, guard_result, wall_evidence=wall_evidence)
    wall_relations = [rel for rel in relations if rel.get("relation") == "attach" and str(rel.get("parameters", {}).get("object_anchor", "")).startswith("wall_")]
    merge_check = write_accepted_attach_merge_check(
        output_dir,
        merge_report.get("accepted_candidates", []),
        relations,
        final_outputs_blocked_reason=final_outputs_blocked_reason,
        skipped_candidates=merge_report.get("skipped_candidates", []),
        wall_id_remap=merge_report.get("wall_id_remap_applied_to_relations", {}),
    )
    data = {
        "dry_run": dry_run,
        "status": "completed" if merge_check.get("valid") else "failed",
        "wall_attach_candidate_recall": recall,
        "accepted_wall_attach_relations": wall_relations,
        "wall_attach_relation_count": len(wall_relations),
        "accepted_attach_merge": merge_check,
        "skipped_candidates": merge_report.get("skipped_candidates", []),
    }
    write_audit_pair(output_dir, "attach", "attach_decision_audit", data, "Attach Decision Audit")
    return merge_check


def write_partial_output(output_dir: Path, name: str, payload: Any) -> Path:
    path = output_dir / "debug" / "partial_outputs" / name
    write_json(path, payload)
    return path


def apply_simulation_physics(
    objects: list[dict[str, Any]],
    output_dir: Path,
    dry_run: bool = False,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    sidecar_path = output_dir / "physics_sidecar" / "integrated_async" / "object_physics.json"
    agent_payload: dict[str, Any] = {}
    source = "local_fallback_sidecar_not_ready"
    read_error = None
    if sidecar_path.exists():
        try:
            loaded = json.loads(sidecar_path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                agent_payload = loaded
                source = "async_physics_sidecar"
            else:
                read_error = "object_physics.json top level is not an object"
        except (OSError, json.JSONDecodeError) as exc:
            read_error = str(exc)

    base_payload = {"walls": [], "objects": objects}
    sidecar, enriched, audit = resolve_object_physics(base_payload, agent_payload)
    expected_ids = [str(obj.get("id")) for obj in objects]
    validation_errors = validate_object_physics(sidecar, expected_ids)
    if validation_errors:
        sidecar, enriched, audit = resolve_object_physics(base_payload, {})
        fallback_errors = validate_object_physics(sidecar, expected_ids)
        source = "local_fallback_invalid_sidecar"
        validation_errors.extend(fallback_errors)

    # apply_upright_properties has already combined the Sidecar prior with
    # relation-aware wall, ceiling, and floor-covering overrides. Physics
    # enrichment may add numeric fields, but must not replace that final value.
    final_upright_by_id = {
        str(obj.get("id")): obj.get("upright")
        for obj in objects
        if isinstance(obj, dict) and isinstance(obj.get("upright"), bool)
    }
    for obj in enriched.get("objects", []):
        object_id = str(obj.get("id") or "")
        if object_id in final_upright_by_id:
            obj["upright"] = final_upright_by_id[object_id]

    audit.update({
        "integration_source": source,
        "sidecar_path": str(sidecar_path),
        "sidecar_available_at_final_assembly": sidecar_path.exists(),
        "relation_flow_waited_for_sidecar": False,
        "relation_aware_upright_preserved": True,
        "read_error": read_error,
        "validation_errors": validation_errors,
    })
    write_audit_pair(
        output_dir,
        "objects",
        "simulation_physics_audit",
        audit,
        "Simulation Physics Audit",
    )
    if not dry_run:
        write_json(output_dir / "debug" / "objects" / "resolved_object_physics.json", sidecar)
    return enriched["objects"], audit


def assemble_final_payloads(
    scene_state: dict[str, Any],
    task_results: list[dict[str, Any]],
    guard_result: dict[str, Any] | None,
    output_dir: Path,
    dry_run: bool = False,
    wall_evidence: dict[str, Any] | None = None,
) -> dict[str, Any]:
    base_objects = candidate_objects(scene_state)
    patches, patch_warnings = collect_object_revision_patches(task_results, guard_result)
    final_objects, applied_revisions, apply_warnings = apply_object_revision_patches(base_objects, patches)
    object_warnings = patch_warnings + apply_warnings
    write_object_revision_audit(output_dir, applied_revisions, object_warnings, dry_run=dry_run)

    walls, wall_warnings, wall_records = collect_walls(task_results, guard_result, wall_evidence=wall_evidence)
    room_materials_audit = collect_room_materials(task_results, guard_result)
    room_materials_audit["dry_run"] = dry_run
    write_audit_pair(output_dir, "objects", "room_materials_audit", room_materials_audit, "Room Materials Audit")

    wall_final_adjudication = summarize_wall_final_adjudication(task_results, guard_result, accepted_walls=walls)
    walls, wall_records, authoritative_wall_geometry_gate = _authoritative_obb_walls_from_evidence(walls, wall_records, wall_evidence)
    if authoritative_wall_geometry_gate.get("applied"):
        wall_warnings.append("Final wall coordinates were overridden by local obb_from_mesh_footprint_min_rect authoritative geometry gate.")
        wall_final_adjudication["accepted_final_walls"] = True
        wall_final_adjudication["coordinate_source_overridden_by"] = FORMAL_OBB_WALL_METHOD
    else:
        wall_warnings.append(f"Formal OBB wall geometry gate did not apply: {authoritative_wall_geometry_gate.get('reason')}.")
    walls, wall_records, wall_canonicalization_gate = canonicalize_walls_for_screen_clockwise(walls, wall_records, wall_evidence)
    if wall_canonicalization_gate.get("changed_wall_ids"):
        wall_warnings.append("Final wall ids were remapped by the local screen-clockwise canonicalization gate; relation anchors are remapped during merge.")
    wall_final_semantics = collect_wall_final_semantics(task_results, guard_result)
    wall_semantic_remap = (
        wall_canonicalization_gate.get("old_to_new_wall_id", {})
        if isinstance(wall_canonicalization_gate, dict)
        else {}
    )
    wall_records, wall_semantic_overlay_gate = apply_wall_final_semantics(
        wall_records,
        wall_final_semantics,
        wall_semantic_remap,
    )
    authoritative_wall_geometry_gate["semantic_metadata_overlay"] = wall_semantic_overlay_gate
    wall_geometry = write_wall_audit(
        output_dir,
        walls,
        wall_records,
        wall_warnings,
        dry_run=dry_run,
        wall_final_adjudication=wall_final_adjudication,
        wall_canonicalization_gate=wall_canonicalization_gate,
        authoritative_wall_geometry_gate=authoritative_wall_geometry_gate,
    )
    wall_blocked_reason = None
    if not walls or len(walls) != 4 or not wall_geometry.get("valid"):
        wall_blocked_reason = "missing_or_invalid_final_eligible_walls"
    object_ids = [obj["id"] for obj in final_objects]
    blocked_wall_attach_anchors = weak_or_inferred_wall_attach_blocks(wall_records)
    relation_wall_id_remap = build_wall_relation_id_remap(
        wall_records,
        wall_canonicalization_gate if isinstance(wall_canonicalization_gate, dict) else None,
    )
    object_physical_properties = collect_object_physical_properties(task_results, guard_result, final_objects, scene_state)
    write_object_physical_properties_audit(output_dir, object_physical_properties, dry_run=dry_run)
    relations, merge_report = collect_relations(
        guard_result,
        task_results,
        object_ids,
        blocked_wall_attach_anchors=blocked_wall_attach_anchors,
        wall_id_remap=relation_wall_id_remap,
        objects=final_objects,
        object_properties=object_physical_properties,
        wall_evidence=wall_evidence,
        scene_state=scene_state,
    )
    contain_geometry_audit = dict(merge_report.get("contain_geometry_validation", {}))
    contain_geometry_audit["dry_run"] = dry_run
    write_audit_pair(output_dir, "relations", "contain_geometry_audit", contain_geometry_audit, "Contain Geometry Audit")
    contain_conflict_audit = dict(merge_report.get("contain_conflict_resolution", {}))
    contain_conflict_audit["dry_run"] = dry_run
    write_audit_pair(output_dir, "relations", "contain_conflict_resolution_audit", contain_conflict_audit, "Contain Conflict Resolution Audit")
    relation_normalization_audit = write_relation_normalization_audit(output_dir, merge_report.get("relation_normalization", {}), dry_run=dry_run)
    flat_contact_edge_wall_attach_audit = write_flat_contact_edge_wall_attach_audit(
        output_dir,
        merge_report,
        dry_run=dry_run,
    )
    semantic_wall_contact_reconciliation_audit = (
        write_semantic_wall_contact_reconciliation_audit(
            output_dir,
            merge_report,
            dry_run=dry_run,
        )
    )
    sidecar_upright_priors = load_async_sidecar_upright_priors(output_dir, object_ids)
    final_objects, upright_audit = apply_upright_properties(
        final_objects,
        relations,
        object_physical_properties,
        scene_state,
        output_dir,
        dry_run=dry_run,
        upright_priors=sidecar_upright_priors,
    )
    final_objects, contain_anchor_audit = apply_contain_anchor_properties(
        final_objects,
        relations,
        task_results,
        guard_result,
        scene_state,
        output_dir,
        dry_run=dry_run,
    )
    final_objects, simulation_physics_audit = apply_simulation_physics(final_objects, output_dir, dry_run=dry_run)
    gravity_support_completion_audit = write_gravity_support_completion_audit(
        output_dir,
        merge_report.get("gravity_support_completion", {}),
        dry_run=dry_run,
    )
    coverage_repair_audit = write_coverage_repair_audit(output_dir, merge_report.get("coverage_repair", {}), dry_run=dry_run)
    coverage_missing = merge_report.get("coverage_repair", {}).get("missing_after", [])
    merge_check = write_attach_audit(
        output_dir,
        task_results,
        guard_result,
        relations,
        merge_report,
        dry_run=dry_run,
        wall_evidence=wall_evidence,
        final_outputs_blocked_reason=wall_blocked_reason,
    )

    # This is deliberately the final geometry postprocess: all Agent decisions,
    # relation normalization, coverage repair, and merge auditing are complete.
    walls, wall_records, virtual_wall_shift_audit = shift_walls_from_attach_policy(
        walls,
        wall_records,
        merge_report,
        relations=relations,
        offset_ratio=0.05,
    )
    if virtual_wall_shift_audit.get("status") == "applied":
        wall_warnings.append(
            "Only unattached weak/inferred/closure-only wall support lines were moved outward after final relation assembly."
        )
        wall_final_adjudication["coordinate_postprocess"] = "wall_attach_policy_outward_shift"
        wall_geometry = write_wall_audit(
            output_dir,
            walls,
            wall_records,
            wall_warnings,
            dry_run=dry_run,
            wall_final_adjudication=wall_final_adjudication,
            wall_canonicalization_gate=wall_canonicalization_gate,
            authoritative_wall_geometry_gate=authoritative_wall_geometry_gate,
        )
        if not wall_geometry.get("valid"):
            wall_blocked_reason = "invalid_walls_after_wall_attach_policy_shift"
    weak_wall_attach_audit = write_weak_wall_attach_audit(
        output_dir,
        merge_report,
        dry_run=dry_run,
        virtual_wall_shift_audit=virtual_wall_shift_audit,
    )
    partial_objects_payload = {
        "walls": walls,
        "room_materials": room_materials_audit.get("room_materials") or {},
        "objects": final_objects,
    }

    errors: list[str] = []
    repair_tasks: list[dict[str, Any]] = []
    if room_materials_audit.get("status") != "ready":
        errors.append("Final objects cannot be written because room_materials is missing or invalid.")
        repair_tasks.append(
            {
                "task_type": "room_material_classification",
                "reason": "objects.json requires image-grounded room_materials with non-empty walls, floor, and ceiling strings",
                "inputs": ["original_image_path"],
                "output_schema": {
                    "room_materials": {
                        "walls": "concise visible wall material/finish",
                        "floor": "concise visible floor material/finish",
                        "ceiling": "concise visible ceiling material/finish",
                    }
                },
            }
        )
    if wall_blocked_reason:
        errors.append("Final objects cannot be written because wall_001..wall_004 are missing, geometrically invalid, or lack final-eligible layout XZ source with topdown verification.")
        repair_tasks.append(
            {
                "task_type": "wall_final_adjudication",
                "reason": "objects.json requires final adjudication: wall_001..wall_004 must be verified layout XZ coordinates before finalization",
                "original_repair_task_type": "infer_four_clockwise_walls_from_topdown",
                "inputs": [
                    "wall_evidence.visual_wall_extraction",
                    "canonical_topdown.png",
                    "canonical_topdown_grid_mapping.json",
                    "wall_evidence.object_footprints",
                ],
                "output_schema": {
                    "walls": [
                        {
                            "id": "wall_001",
                            "first_point": ["x", "z"],
                            "second_point": ["x", "z"],
                            "source": "obb_from_mesh_footprint_min_rect",
                            "evidence": "layout XZ coordinate evidence verified by topdown/fixed views",
                            "topdown_verification": "passed",
                        }
                    ]
                },
            }
        )
    if coverage_missing:
        errors.append(f"Final relations are missing attach/contain target coverage for objects: {coverage_missing}.")
        repair_tasks.append(
            {
                "task_type": "repair_missing_relation_targets",
                "missing_object_targets": coverage_missing,
                "instruction": "Choose the most plausible physical anchor or container for each missing object target. Prefer local physical contact or interior containment over generic floor fallback.",
                "inputs": [
                    "missing_object_targets",
                    "scene_state.object_candidates",
                    "current_accepted_relations",
                    "candidate_anchor_objects",
                    "original_image_path",
                    "mask_overlay_path",
                    "canonical_topdown",
                    "fixed_layout_views_manifest",
                    "layout_bbox_or_footprint_evidence",
                    "selective_render_tool_capability",
                ],
                "output_schema": {
                    "coverage_repair_relations": [
                        {
                            "relation": "attach",
                            "parameters": {"object_anchor": "...", "object_target": "object_xxx"},
                            "repair_reason": "physical anchor reasoning",
                            "evidence_used": ["mask_overlay", "layout_bbox", "fixed_layout_views"],
                            "coverage_repair": True,
                        }
                    ]
                },
            }
        )
    if not merge_check.get("valid"):
        errors.append("Accepted attach candidates were missing from final relations after merge.")

    partial_path = None
    if errors:
        partial_path = write_partial_output(output_dir, "objects_or_relations_validation_failed.json", {"objects": partial_objects_payload, "relations": relations})

    if repair_tasks:
        write_json(output_dir / "debug" / "agents" / "schema_repair_tasks.json", {"tasks": repair_tasks, "errors": errors})
        write_md(
            output_dir / "debug" / "agents" / "schema_repair_tasks.md",
            "Schema Repair Tasks",
            ["```json", json.dumps({"tasks": repair_tasks, "errors": errors}, indent=2, ensure_ascii=False), "```"],
        )

    objects_payload = None if errors else partial_objects_payload
    objects_path = output_dir / "final_outputs" / "objects.json"
    relations_path = output_dir / "final_outputs" / "relations.json"
    final_objects_written = False
    final_relations_written = False
    if not errors:
        write_json(objects_path, partial_objects_payload)
        write_json(relations_path, relations)
        final_objects_written = True
        final_relations_written = True

    return {
        "objects_payload": objects_payload,
        "relations_payload": relations if not errors else None,
        "objects_path": str(objects_path) if final_objects_written else None,
        "relations_path": str(relations_path) if final_relations_written else None,
        "partial_objects_payload": partial_objects_payload,
        "simulation_physics_audit": simulation_physics_audit,
        "partial_objects_path": str(partial_path) if partial_path else None,
        "object_revision_warnings": object_warnings,
        "applied_object_revisions": applied_revisions,
        "wall_geometry": wall_geometry,
        "wall_source_records": wall_records,
        "wall_final_adjudication": wall_final_adjudication,
        "wall_errors": wall_geometry.get("errors", []),
        "attach_merge_check": merge_check,
        "ceiling_confirmed": merge_report.get("ceiling_confirmed"),
        "coverage_repair_audit": coverage_repair_audit,
        "semantic_wall_contact_reconciliation_audit": (
            semantic_wall_contact_reconciliation_audit
        ),
        "gravity_support_completion_audit": gravity_support_completion_audit,
        "relation_normalization_audit": relation_normalization_audit,
        "weak_wall_attach_audit": weak_wall_attach_audit,
        "virtual_wall_shift_audit": virtual_wall_shift_audit,
        "upright_audit": upright_audit,
        "contain_anchor_audit": contain_anchor_audit,
        "contain_geometry_audit": contain_geometry_audit,
        "contain_conflict_audit": contain_conflict_audit,
        "object_physical_properties": object_physical_properties,
        "room_materials_audit": room_materials_audit,
        "missing_object_targets": coverage_missing,
        "errors": errors,
        "repair_tasks": repair_tasks,
        "final_objects_written": final_objects_written,
        "final_relations_written": final_relations_written,
    }
