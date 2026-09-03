from __future__ import annotations

import copy
import re
from typing import Any


OBJECT_ANCHOR_MODES = {
    "ceiling_only",
    "wall_only",
    "floor_wall",
    "support_only",
    "contain_candidate",
    "uncertain",
}
OBJECT_ANCHOR_CONFIDENCES = {"high", "medium", "low"}


TASK_KIND_REGISTRY: dict[str, dict[str, Any]] = {
    "support_contact_extraction": {
        "description": "Extract gravity-direction floor/object support and direct contact relations.",
        "runner_action": "api_dynamic_task",
        "runtime_class": "required_reasoning",
        "requires_artifacts": ["object_list", "object_footprints", "original_image", "mask_overlay", "canonical_topdown"],
        "provides_capabilities": ["support_relations_ready"],
        "image_profile": ["original_image", "mask_overlay", "canonical_topdown"],
        "instructions": [
            "Gravity-direction support is the first relation pass. Return one explicit support_contact_candidates verdict for every object in the hydrated gravity_coverage_scope before resolving Manager-priority ambiguous cases.",
            "Extract physically supported attach relations in the gravity direction: floor-to-object and object-to-object support/contact. Judge actual contact from the supplied image, mask, and footprints; do not replace a plausible object support with floor merely for coverage.",
            "Wall contact is a separate horizontal relation. This task may report floor/object support without deciding wall attachment.",
            "The Manager priority scope identifies difficult cases; it never replaces gravity_coverage_scope. For every object you accept as a support anchor for another object, also decide that anchor's own gravity support unless it is exclusively wall-mounted, ceiling-mounted, or contained. Floor support and wall attach may coexist for the same freestanding object.",
            "Use concise reason_code values for obvious floor support. When an object is visibly inside a container or exclusively wall/ceiling mounted, return defer_to_contain or exclusive_wall_or_ceiling rather than inventing floor support. For a clearly visible inside-container case, include containment_hypothesis with decision=inside and the object anchor/target; otherwise use decision=uncertain. Do not use contain as a support_mode.",
            "An open-front shelf, bookcase bay, display cubby, or cabinet compartment is an interior containment volume. An object resting on an internal shelf board is contained by that furniture even though the board also provides gravity support. For ordinary single-level furniture, a target on a plain exterior top remains object-to-object attach.",
            "Multi-level furniture uses a simpler ontology: a target placed on or within any functional level of a bunk bed, bookcase/shelving unit, multi-tier service cart, or multi-level rack is contain, including the top level. Do not emit object-to-object attach for the same multi-level-furniture pair. Use defer_to_contain with containment_hypothesis.decision=inside and support_surface_role=multi_level_tier. Decide tier membership from the original image, not SAM3D bbox overlap alone.",
            "Keep support_mode structurally consistent with the anchor: floor requires object_anchor=floor; object requires object_anchor=object_###; exclusive_wall_or_ceiling is valid only for a genuinely wall- or ceiling-mounted target. Never label an object described as on another object's surface as exclusive_wall_or_ceiling.",
            "Treat Manager scene focus as a hypothesis, never as contact evidence. Judge gravity support independently from any contain task. Accept exterior object support only when the supplied images show direct surface contact; use defer_to_contain only when the same images positively show that the target is inside an interior volume.",
            "Before every support verdict, ground the object_id from its image bbox and mask region. Category and description are weak labels and location phrases such as on the bunk, on the desk, or inside the cabinet must never override the grounded mask. When a clear image/mask conflict would change the support anchor, follow the mask for the relation and emit one concise object_revisions patch. Do not spend output on cosmetic naming improvements that cannot change a relation.",
        ],
        "expected_output_schema": {
            "support_contact_candidates": [{
                "decision": "accept|reject|uncertain",
                "support_mode": "floor|object|defer_to_contain|exclusive_wall_or_ceiling|uncertain",
                "object_anchor": "floor|object_###|none",
                "object_target": "object_###",
                "reason_code": "short_reason_code",
                "support_surface_role": "internal_shelf|bounded_tier|multi_level_tier|outer_top|external_surface|not_applicable|uncertain",
                "containment_hypothesis": {
                    "decision": "inside|uncertain",
                    "object_anchor": "object_###",
                    "object_target": "object_###",
                },
            }],
            "accepted_relations": [{
                "relation": "attach",
                "parameters": {"object_anchor": "floor|object_###", "object_target": "object_###"},
            }],
            "object_revisions": [{
                "object_id": "object_###",
                "new_category": "optional corrected category",
                "new_description": "optional corrected description",
                "reason": "image/mask-grounded conflict that changes support",
                "confidence": "high|medium|low",
            }],
        },
    },
    "contain_relation_extraction": {
        "description": "Extract object-in-container relations from positive visual/spatial evidence.",
        "runner_action": "api_dynamic_task",
        "runtime_class": "optional_reasoning",
        "requires_artifacts": ["object_list", "object_footprints", "original_image", "mask_overlay", "contain_candidates"],
        "provides_capabilities": ["contain_relations_ready"],
        "image_profile": ["original_image", "mask_overlay"],
        "instructions": [
            "Extract contain only when the target lies inside an anchor object's cabinet, shelf, cubby, box, drawer, basket, tray, or other interior storage/display volume. Local candidates only select pairs to inspect and carry zero positive evidentiary weight. Decide from the original image and mask: reject false positives and add directly evident omissions.",
            "For ordinary single-level anchors, a usable interior, shelf/cubby, cabinet compartment, drawer, box, basket, tray, or other container volume is contain, while a target merely resting on a plain exterior top is attach. A door or fully enclosed cavity is not required.",
            "For multi-level furniture, every occupied functional level is contain, regardless of whether it is lower, middle, or top and regardless of rails or rims. This includes both bunks of a bunk bed and every tier of a multi-tier service cart, bookcase/shelving unit, or rack. Do not output attach for the same anchor-target pair. Decide whether the target belongs to a level from the original image and mask, not from local bbox overlap or candidate score.",
            "For the same anchor-target pair, contain and attach are mutually exclusive; emit contain when interior membership is the correct relation.",
            "Treat the Manager object scope as a priority rather than a closed set. Return an explicit accept, reject, or uncertain decision for every retained local recall pair supplied at runtime, even when its objects were outside the original Manager scope. Bbox overlap, a local score, or category compatibility never proves containment; when the image does not positively show interior membership, return uncertain or reject.",
            "For every accepted contain anchor, report render_direction as front for front-open cabinets/shelves/display volumes and front-accessed internal tiers of multi-level furniture such as bunk beds, or up for top-open boxes/baskets/trays/drawers. Final assembly validates and writes this property to objects.json.",
            "Before every contain verdict, ground the object_id from its image bbox and mask region. Category and description are weak labels and location phrases such as on the bunk, on the desk, or inside the cabinet must never override the grounded mask. When a clear image/mask conflict would change contain versus attach, follow the mask for the relation and emit one concise object_revisions patch. Do not spend output on cosmetic naming improvements that cannot change a relation.",
        ],
        "expected_output_schema": {
            "contain_candidates": [{
                "decision": "accept|reject|uncertain",
                "object_anchor": "object_###",
                "object_target": "object_###",
                "evidence": ["..."],
                "reason": "...",
            }],
            "accepted_contain_relations": [{
                "relation": "contain",
                "parameters": {"object_anchor": "object_###", "object_target": "object_###"},
            }],
            "contain_anchor_properties": [{
                "object_id": "object_###",
                "contain_anchor": True,
                "render_direction": "front|up",
                "reason": "...",
            }],
            "object_revisions": [{
                "object_id": "object_###",
                "new_category": "optional corrected category",
                "new_description": "optional corrected description",
                "reason": "image/mask-grounded conflict that changes contain versus attach",
                "confidence": "high|medium|low",
            }],
        },
    },
    "object_revision_check": {
        "description": "Propose only semantic object revisions that materially affect relation extraction.",
        "runner_action": "api_dynamic_task",
        "runtime_class": "optional_reasoning",
        "requires_artifacts": ["object_list", "original_image", "mask_overlay"],
        "provides_capabilities": ["object_revisions_ready"],
        "image_profile": ["original_image", "mask_overlay"],
        "instructions": [
            "Check only category or description errors that materially affect relation extraction. Preserve every SAM3D object id and return revision patches rather than a replacement object list.",
            "Treat descriptions as weak evidence and ground revisions in the original image and mask.",
        ],
        "expected_output_schema": {
            "object_revisions": [{
                "object_id": "object_###",
                "new_category": "optional corrected category",
                "new_description": "optional corrected description",
                "reason": "...",
                "confidence": "high|medium|low",
            }],
        },
    },
    "object_physical_property_reasoning": {
        "description": "Judge floor-covering semantics needed by deterministic relation normalization.",
        "runner_action": "api_dynamic_task",
        "runtime_class": "optional_reasoning",
        "requires_artifacts": ["object_list", "object_footprints", "original_image", "mask_overlay"],
        "provides_capabilities": ["object_physical_properties_ready"],
        "image_profile": ["original_image", "mask_overlay"],
        "instructions": [
            "Judge only whether an object is a floor covering whose apparent object-to-object support relations should normalize to floor support. Return uncertain judgements explicitly; final assembly has deterministic fallback hints.",
            "Do not output upright. The independent Physics Sidecar owns the general upright prior, and final assembly applies relation-aware wall/ceiling overrides.",
            "Do not infer relation changes beyond the floor-covering classification itself.",
        ],
        "expected_output_schema": {
            "object_physical_properties": [{
                "object_id": "object_###",
                "is_floor_covering": True,
                "floor_covering_confidence": "high|medium|low",
                "floor_covering_reason": "...",
            }],
        },
    },
    "room_material_classification": {
        "description": "Describe the dominant visible wall, floor, and ceiling finish materials for final room metadata.",
        "runner_action": "api_dynamic_task",
        "runtime_class": "required_reasoning",
        "requires_artifacts": ["original_image"],
        "provides_capabilities": ["room_materials_ready"],
        "image_profile": ["original_image"],
        "instructions": [
            "Inspect the original scene image and describe the dominant visible finish for walls, floor, and ceiling.",
            "Return exactly one room_materials object with the keys walls, floor, and ceiling. Each value must be a concise non-empty English material/finish phrase suitable for rendering, such as off-white painted wall, light oak wooden floor, or flat white plaster ceiling.",
            "Do not copy the example phrases unless they match the image. Do not infer materials from SAM3D object categories or wall geometry. When a surface is only partly visible, return the most specific description supported by the image and avoid unsupported brand, construction, or hidden-layer claims.",
            "This capability may be merged into another task that already receives the original image; do not return relations, wall coordinates, or object revisions solely for this capability.",
        ],
        "expected_output_schema": {
            "room_materials": {
                "walls": "concise visible wall material/finish",
                "floor": "concise visible floor material/finish",
                "ceiling": "concise visible ceiling material/finish",
            },
        },
    },
    "ceiling_judgement": {
        "description": "Decide whether ceiling is a valid attach anchor for concrete objects.",
        "runner_action": "api_dynamic_task",
        "runtime_class": "optional_reasoning",
        "requires_artifacts": ["object_list", "original_image", "mask_overlay"],
        "provides_capabilities": ["ceiling_judgement_ready"],
        "image_profile": ["original_image", "mask_overlay"],
        "instructions": [
            "Judge whether the scene provides positive evidence that ceiling is a valid anchor, and identify only concrete ceiling-mounted targets. Do not allow ceiling attachment from category text alone when visual evidence contradicts it.",
        ],
        "expected_output_schema": {
            "ceiling_decision": {
                "allow_ceiling_anchor": True,
                "ceiling_confirmed": True,
                "evidence": ["..."],
                "reason": "...",
            },
            "accepted_relations": [{
                "relation": "attach",
                "parameters": {"object_anchor": "ceiling", "object_target": "object_###"},
            }],
        },
    },
    "wall_alignment_object_review": {
        "description": "Identify real wall-following objects; local code scores finite OBB axes from their canonical top-down rotations.",
        "runner_action": "api_dynamic_task",
        "runtime_class": "required_reasoning",
        "requires_artifacts": ["object_list", "original_image", "mask_overlay", "wall_orientation_candidates_overlay"],
        "provides_capabilities": ["wall_alignment_objects_ready", "wall_object_hypotheses_ready"],
        "image_profile": ["original_image", "mask_overlay", "wall_orientation_candidates_overlay"],
        "formal_geometry_controlled": True,
        "instructions": [
            "Identify only objects whose real pose reliably reveals a room-wall axis: wall-mounted, wall-embedded, physically against a wall, or clearly parallel/aligned with a wall. Use the original image for semantics and the mask overlay for object ids.",
            "Treat object_scope as a priority, not a closed set. Include directly evident omissions, but exclude central furniture, round objects, small clutter, and objects whose heading does not reliably follow a wall.",
            "Do not choose the winning room-orientation candidate, final wall id, coordinate, angle, visibility label, or attach relation. Local code reads rotation_degrees_topdown for the selected object ids and scores the finite formal OBB candidates.",
            "For each likely wall object, use the isolated candidate panels to return candidate_wall_bindings keyed by exact candidate_id and provisional_wall_id. The binding is semantic evidence only and does not select a final candidate or accept attach.",
            "Keep the response compact. Use high or medium confidence only when the wall-following evidence is concrete; omit uncertain objects instead of returning low-confidence filler.",
        ],
        "expected_output_schema": {
            "likely_wall_objects": [{
                "object_id": "object_###",
                "role": "wall_mounted|wall_embedded|against_wall|wall_aligned",
                "confidence": "high|medium",
                "evidence_code": "short_visual_evidence_code",
            }],
            "candidate_wall_bindings": [{
                "object_id": "object_###",
                "candidate_id": "copy exactly from candidate panel",
                "provisional_wall_id": "wall_001|wall_002|wall_003|wall_004",
                "role": "wall_mounted|wall_embedded|against_wall|wall_aligned",
                "confidence": "high|medium",
                "evidence_code": "short_visual_binding_code",
            }],
        },
    },
    "wall_object_hypothesis": {
        "description": "Identify likely wall-following objects as semantic evidence for local formal-wall orientation scoring.",
        "runner_action": "api_dynamic_task",
        "runtime_class": "required_reasoning",
        "requires_artifacts": ["object_list", "original_image", "mask_overlay"],
        "provides_capabilities": ["wall_alignment_objects_ready", "wall_object_hypotheses_ready"],
        "image_profile": ["original_image", "mask_overlay"],
        "formal_geometry_controlled": True,
        "instructions": [
            "Identify only real wall-mounted, wall-embedded, physically against-wall, or clearly wall-aligned objects. These object ids are semantic evidence for local orientation scoring and later wall-contact review.",
            "Do not output candidate wall ids, coordinates, lines, angles, visibility labels, or relations before local wall orientation is resolved.",
            "Treat object_scope as a priority rather than a closed set, and omit uncertain central or round objects.",
        ],
        "expected_output_schema": {
            "likely_wall_objects": [{
                "object_id": "object_###",
                "role": "wall_mounted|wall_embedded|against_wall|wall_aligned",
                "confidence": "high|medium",
                "evidence_code": "short_visual_evidence_code",
            }],
        },
    },
    "provisional_wall_visibility_review": {
        "description": "Review original-image visibility semantics for labelled edges in finite pre-orientation OBB candidate panels.",
        "runner_action": "api_dynamic_task",
        "runtime_class": "required_reasoning",
        "requires_artifacts": ["original_image", "mask_overlay", "wall_orientation_candidates_overlay"],
        "provides_capabilities": ["provisional_wall_visibility_ready"],
        "image_profile": ["original_image", "mask_overlay", "wall_orientation_candidates_overlay"],
        "formal_geometry_controlled": True,
        "instructions": [
            "Use the isolated finite candidate panels only to judge which labelled candidate edges correspond to directly visible real walls versus inferred, weak, or closure-only room boundaries. Each panel is independently labelled wall_001 through wall_004.",
            "Return compact records keyed by candidate_id and provisional_wall_id. You may judge visibility semantics, but do not choose the winning orientation candidate, create coordinates, alter an edge, emit a final wall id, or emit any attach relation.",
            "Visible or partially_visible requires direct source-image evidence of a wall surface or a wall-floor/wall-ceiling junction. Do not treat a furniture contour, crop border, occlusion edge, or proximity to a mesh rectangle as a visible wall.",
            "When evidence is absent, use inferred/not_visible and can_anchor_wall_attach=false rather than guessing. These are provisional records; local code binds only the records belonging to the orientation selected later from wall-following objects.",
        ],
        "expected_output_schema": {
            "provisional_wall_visibility_review": [{
                "candidate_id": "copy exactly from the candidate panel header",
                "provisional_wall_id": "wall_001|wall_002|wall_003|wall_004",
                "original_image_visibility": "visible|partially_visible|not_visible|inferred",
                "weak_wall": False,
                "closure_only": False,
                "can_anchor_wall_attach": True,
                "reason_code": "short_reason_code",
            }],
        },
    },
    "wall_visibility_semantics_review": {
        "description": "Review visibility and weak/inferred semantics for the already selected, canonical formal OBB walls.",
        "runner_action": "api_dynamic_task",
        "runtime_class": "required_reasoning",
        "requires_artifacts": ["formal_obb_walls", "original_image", "mask_overlay"],
        "requires_capabilities": ["wall_alignment_objects_ready"],
        "provides_capabilities": ["wall_visibility_semantics_ready"],
        "image_profile": ["original_image", "wall_attach_prefilter_overlay"],
        "formal_geometry_controlled": True,
        "instructions": [
            "The local scorer has already selected and applied the formal OBB orientation. Judge original-image visibility, weak/inferred or closure-only status, and attach eligibility for the supplied final wall references.",
            "Copy every supplied wall_id and geometry_key exactly. Do not choose another candidate and do not rotate, edit, renumber, or replace wall geometry.",
            "Return one compact record for each of wall_001 through wall_004.",
        ],
        "expected_output_schema": {
            "wall_visibility_review": [{
                "wall_id": "wall_001",
                "geometry_key": "copy supplied wall_edge_... exactly",
                "original_image_visibility": "visible|partially_visible|not_visible|inferred",
                "weak_wall": False,
                "closure_only": False,
                "can_anchor_wall_attach": True,
                "reason_code": "short_reason_code",
            }],
        },
    },
    "wall_final_adjudication": {
        "description": "Perform detailed semantic adjudication of already selected canonical formal OBB walls.",
        "runner_action": "api_dynamic_task",
        "runtime_class": "required_reasoning",
        "requires_artifacts": ["formal_obb_walls", "wall_evidence", "wall_visual_evidence", "original_image", "mask_overlay"],
        "requires_capabilities": ["wall_alignment_objects_ready"],
        "provides_capabilities": ["wall_visibility_semantics_ready"],
        "image_profile": ["original_image", "mask_overlay"],
        "formal_geometry_controlled": True,
        "instructions": [
            "The formal OBB orientation and canonical wall ids have already been selected by local scoring from Agent-identified wall-following objects. Audit visibility, weak/inferred closure status, and attach eligibility only.",
            "Copy every supplied wall_id and geometry_key exactly. Never choose another candidate or invent/edit coordinates, angles, ids, or methods.",
        ],
        "expected_output_schema": {
            "decision": "verified_final_walls|missing_or_invalid_formal_obb_wall_evidence",
            "wall_visibility_review": [{
                "wall_id": "wall_001",
                "geometry_key": "copy supplied wall_edge_... exactly",
                "original_image_visibility": "visible|partially_visible|not_visible|inferred",
                "weak_wall": False,
                "closure_only": False,
                "can_anchor_wall_attach": True,
                "reason_code": "short_reason_code",
            }],
            "reason": "required only when final references are invalid",
        },
    },
    "wall_attach_candidate_recall": {
        "description": "Review final-wall visibility and judge compact locally prefiltered wall-attach candidates in one downstream call.",
        "runner_action": "api_dynamic_task",
        "runtime_class": "required_reasoning",
        "requires_artifacts": ["object_list", "object_footprints", "formal_obb_walls", "wall_evidence", "original_image", "mask_overlay"],
        "requires_capabilities": ["wall_alignment_objects_ready", "wall_attach_candidates_ready"],
        "provides_capabilities": ["wall_visibility_semantics_ready", "wall_attach_decisions_ready"],
        "image_profile": ["original_image", "mask_overlay", "wall_attach_prefilter_overlay"],
        "formal_geometry_controlled": True,
        "instructions": [
            "The local scorer has already selected and applied the formal OBB orientation. First return visibility/weak/inferred/closure-only semantics for all four supplied final wall references, copying wall_id and geometry_key exactly.",
            "Then review the compact local wall-attach candidate set. One object may have up to two plausible neighboring wall ids; compare every supplied pair on the numbered overlay, accept at most one wall for each object, and reject the other pair explicitly.",
            "The candidate wall id is a review option, not a pre-decided answer. Choose only among the supplied formal wall ids and never transfer visual evidence from one wall to its opposite or adjacent edge.",
            "Floor/object support may coexist with one wall attach. Do not confuse proximity, facing, an isolated corner, or orientation alignment with physical wall contact.",
            "When a target is already supported by another object, reject ordinary wall proximity. Preserve an additional wall attach only with explicit wall-mounted, embedded, hung, rack, or bracket evidence.",
            "A weak, hidden, inferred, closure-only, or can_anchor_wall_attach=false wall may close room geometry but cannot anchor a final wall attach.",
            "An image crop boundary, furniture side panel, occlusion contour, or room-closure edge is not evidence that a wall is visible. Mark visible or partially_visible only when the actual wall surface or a wall-floor/wall-ceiling junction is directly visible in the source image.",
            "Do not choose another orientation candidate and do not edit coordinates, angles, wall ids, or methods.",
        ],
        "expected_output_schema": {
            "wall_visibility_review": [{
                "wall_id": "wall_001",
                "geometry_key": "copy supplied wall_edge_... exactly",
                "original_image_visibility": "visible|partially_visible|not_visible|inferred",
                "weak_wall": False,
                "closure_only": False,
                "can_anchor_wall_attach": True,
                "reason_code": "short_reason_code",
            }],
            "wall_attach_candidates": [{
                "decision": "accept|reject",
                "object_target": "object_###",
                "candidate_wall_id": "wall_001|wall_002|wall_003|wall_004",
                "evidence_codes": ["short_code"],
                "reason_code": "short_reason_code",
            }],
            "accepted_attach_candidates": [{
                "relation": "attach",
                "parameters": {"object_anchor": "wall_001", "object_target": "object_###"},
            }],
        },
    },
    "local_validation": {
        "description": "Represent deterministic local validation without an API DynamicTaskAgent call.",
        "runner_action": "local_schema_validation",
        "runtime_class": "local_redundant_validation",
        "requires_artifacts": ["scene_state"],
        "provides_capabilities": ["local_validation_ready"],
        "image_profile": [],
        "instructions": ["Run deterministic local relation/object schema validation."],
        "expected_output_schema": {"valid": True, "errors": []},
    },
}


TASK_KIND_ALIASES = {
    "support_contact": "support_contact_extraction",
    "support_relation_extraction": "support_contact_extraction",
    "contain_relations": "contain_relation_extraction",
    "contain_relation_probe": "contain_relation_extraction",
    "object_revision": "object_revision_check",
    "object_revision_audit": "object_revision_check",
    "object_physical_properties": "object_physical_property_reasoning",
    "room_materials": "room_material_classification",
    "room_materials_classification": "room_material_classification",
    "ceiling_anchor_judgement": "ceiling_judgement",
    "ceiling_attach_judgement": "ceiling_judgement",
    "ceiling_anchor_review": "ceiling_judgement",
    "wall_alignment_review": "wall_alignment_object_review",
    "wall_orientation_object_review": "wall_alignment_object_review",
    "provisional_wall_semantic_review": "provisional_wall_visibility_review",
    "candidate_wall_visibility_review": "provisional_wall_visibility_review",
    "wall_attach_recall": "wall_attach_candidate_recall",
    "final_validation": "local_validation",
    "final_consistency_validation": "local_validation",
}


CAPABILITY_TO_KIND: dict[str, str] = {}
for _kind, _spec in TASK_KIND_REGISTRY.items():
    for _capability in _spec.get("provides_capabilities", []):
        CAPABILITY_TO_KIND.setdefault(str(_capability), _kind)


def normalize_task_kind(value: Any) -> str:
    kind = str(value or "").strip().lower().replace("-", "_").replace(" ", "_")
    return TASK_KIND_ALIASES.get(kind, kind)


def task_image_artifact_profiles() -> dict[str, tuple[str, ...]]:
    return {
        kind: tuple(str(item) for item in spec.get("image_profile", []))
        for kind, spec in TASK_KIND_REGISTRY.items()
    }


def compact_manager_capability_catalog() -> list[dict[str, Any]]:
    result = []
    for kind, spec in TASK_KIND_REGISTRY.items():
        result.append({
            "canonical_task_kind": kind,
            "description": spec["description"],
            "provides_capabilities": list(spec.get("provides_capabilities", [])),
            "requires_capabilities": list(spec.get("requires_capabilities", [])),
            "runner_controlled": bool(spec.get("runner_controlled", False)),
            "formal_geometry_controlled": bool(spec.get("formal_geometry_controlled", False)),
        })
    return result


def _ordered_union(*collections: Any) -> list[str]:
    result: list[str] = []
    for collection in collections:
        for item in collection or []:
            value = str(item)
            if value and value not in result:
                result.append(value)
    return result


def _safe_task_id(value: Any, index: int) -> str:
    raw = str(value or f"node_{index:03d}").strip().lower()
    normalized = re.sub(r"[^a-z0-9_]+", "_", raw).strip("_") or f"node_{index:03d}"
    return normalized if normalized.startswith("task_") else f"task_{normalized}"


def _merge_schema(target: dict[str, Any], source: Any) -> None:
    if not isinstance(source, dict):
        return
    for key, value in source.items():
        if key not in target:
            target[key] = copy.deepcopy(value)
        elif isinstance(target[key], dict) and isinstance(value, dict):
            _merge_schema(target[key], value)


def normalize_object_anchor_routing(value: Any) -> list[dict[str, str]]:
    """Normalize compact Manager routing without making scene decisions locally."""
    if not isinstance(value, list):
        return []
    confidence_rank = {"low": 1, "medium": 2, "high": 3}
    by_object: dict[str, dict[str, str]] = {}
    order: list[str] = []
    for item in value:
        if not isinstance(item, dict):
            continue
        object_id = str(item.get("object_id") or "").strip()
        anchor_mode = str(item.get("anchor_mode") or "").strip().lower()
        confidence = str(item.get("confidence") or "").strip().lower()
        if not object_id.startswith("object_") or anchor_mode not in OBJECT_ANCHOR_MODES:
            continue
        if confidence not in OBJECT_ANCHOR_CONFIDENCES:
            confidence = "low"
        record = {
            "object_id": object_id,
            "anchor_mode": anchor_mode,
            "confidence": confidence,
        }
        previous = by_object.get(object_id)
        if previous is None:
            order.append(object_id)
        if previous is None or confidence_rank[confidence] > confidence_rank[previous["confidence"]]:
            by_object[object_id] = record
    return [by_object[object_id] for object_id in order]


def compile_compact_manager_plan(raw_plan: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    raw_tasks = [item for item in raw_plan.get("tasks", []) if isinstance(item, dict)]
    compiled_tasks: list[dict[str, Any]] = []
    aliases: dict[str, str] = {}
    used_ids: set[str] = set()
    audits: list[dict[str, Any]] = []

    for index, node in enumerate(raw_tasks, start=1):
        base_id = _safe_task_id(node.get("node_id") or node.get("task_id"), index)
        task_id = base_id
        suffix = 2
        while task_id in used_ids:
            task_id = f"{base_id}_{suffix}"
            suffix += 1
        used_ids.add(task_id)
        for alias in (node.get("node_id"), node.get("task_id"), task_id):
            if alias:
                aliases[str(alias)] = task_id

        primary_kind = normalize_task_kind(node.get("canonical_task_kind") or node.get("task_type") or "custom_reasoning_task")
        requested_capabilities = _ordered_union(node.get("provides_capabilities"))
        template_kinds: list[str] = []
        if primary_kind in TASK_KIND_REGISTRY:
            template_kinds.append(primary_kind)
        for capability in requested_capabilities:
            template_kind = CAPABILITY_TO_KIND.get(capability)
            if template_kind and template_kind not in template_kinds:
                template_kinds.append(template_kind)

        specs = [TASK_KIND_REGISTRY[kind] for kind in template_kinds]
        manager_custom = not specs
        task_type = primary_kind
        agent_role = "DynamicTaskAgent"
        runner_action = "api_dynamic_task"
        runtime_class = "optional_reasoning" if manager_custom else (
            "required_reasoning" if any(spec.get("runtime_class") == "required_reasoning" for spec in specs)
            else str(specs[0].get("runtime_class", "optional_reasoning"))
        )
        if specs:
            runner_action = str(specs[0].get("runner_action", runner_action))
            if any(spec.get("runner_action") == "api_dynamic_task" for spec in specs):
                runner_action = "api_dynamic_task"

        required_artifacts = _ordered_union(
            *(spec.get("requires_artifacts", []) for spec in specs),
            node.get("requires_artifacts", []),
        )
        requires_capabilities = _ordered_union(
            *(spec.get("requires_capabilities", []) for spec in specs),
            node.get("requires_capabilities", []),
        )
        provides_capabilities = _ordered_union(
            *(spec.get("provides_capabilities", []) for spec in specs),
            requested_capabilities,
        )
        image_profile = _ordered_union(*(spec.get("image_profile", []) for spec in specs))
        instructions = _ordered_union(*(spec.get("instructions", []) for spec in specs))
        scene_focus = str(node.get("scene_focus") or "").strip()
        object_scope = [str(item) for item in node.get("object_scope", [])] if isinstance(node.get("object_scope"), list) else []
        reason = str(node.get("reason") or "").strip()
        custom_instructions = node.get("custom_instructions")
        if scene_focus:
            if "support_contact_extraction" in template_kinds:
                instructions.append(
                    "Manager-proposed scene focus (hypothesis only; verify every claimed surface or contact against supplied evidence): "
                    + scene_focus
                )
            elif "contain_relation_extraction" in template_kinds:
                instructions.append(
                    "Manager-proposed scene focus is a planning hypothesis only, not semantic evidence. "
                    "Verify every claimed inside, outside, or top-surface statement independently from the original image, mask, "
                    "and per-pair image bboxes; explicitly review every hydrated candidate pair even when the Manager omitted it: "
                    + scene_focus
                )
            else:
                instructions.append(f"Scene-specific focus from Manager: {scene_focus}")
        if object_scope:
            instructions.append("Manager-selected priority object scope: " + ", ".join(object_scope))
        if isinstance(custom_instructions, list):
            instructions.extend(str(item) for item in custom_instructions if str(item).strip())
        elif isinstance(custom_instructions, str) and custom_instructions.strip():
            instructions.append(custom_instructions.strip())
        if manager_custom and not instructions:
            instructions.append("Complete the custom reasoning task described by the Manager reason and scene focus using only supplied evidence.")

        expected_output_schema: dict[str, Any] = {}
        for spec in specs:
            _merge_schema(expected_output_schema, spec.get("expected_output_schema"))
        _merge_schema(expected_output_schema, node.get("expected_output_schema"))
        if not expected_output_schema:
            expected_output_schema = {
                "status": "completed|insufficient_evidence",
                "findings": [],
                "proposed_state_updates": {},
                "unresolved_items": [],
            }

        task = {
            "task_id": task_id,
            "task_type": task_type,
            "canonical_task_kind": primary_kind,
            "agent_role": agent_role,
            "runner_action": runner_action,
            "runtime_class": runtime_class,
            "depends_on": list(node.get("depends_on", [])) if isinstance(node.get("depends_on"), list) else [],
            "requires_artifacts": required_artifacts,
            "requires_capabilities": requires_capabilities,
            "provides_capabilities": provides_capabilities,
            "needs_tool": False,
            "use_vision_model": bool(image_profile),
            "image_artifact_profile": image_profile,
            "instructions": instructions,
            "expected_output_schema": expected_output_schema,
            "reason": reason,
            "object_scope": object_scope,
            "scene_focus": scene_focus,
            "compiled_from_compact_manager_plan": True,
        }
        compiled_tasks.append(task)
        audits.append({
            "task_id": task_id,
            "source_node_id": node.get("node_id") or node.get("task_id"),
            "canonical_task_kind": primary_kind,
            "template_kinds": template_kinds,
            "custom_task": manager_custom,
            "field_sources": {
                "task_id": "compiler",
                "task_type": "registry_default" if specs else "manager",
                "canonical_task_kind": "manager",
                "agent_role": "registry_default",
                "runner_action": "registry_default",
                "runtime_class": "registry_default",
                "depends_on": "manager_then_compiler_remap",
                "requires_artifacts": "registry_default+manager",
                "requires_capabilities": "registry_default+manager",
                "provides_capabilities": "registry_default+manager",
                "instructions": "registry_default+manager_scene_brief",
                "expected_output_schema": "registry_default+manager",
                "needs_tool": "registry_default",
            },
        })

    for task in compiled_tasks:
        remapped: list[str] = []
        for dependency in task.get("depends_on", []):
            mapped = aliases.get(str(dependency), str(dependency))
            if mapped != task["task_id"] and mapped not in remapped:
                remapped.append(mapped)
        task["depends_on"] = remapped

    object_anchor_routing = normalize_object_anchor_routing(raw_plan.get("object_anchor_routing"))
    compiled_plan = {
        "plan_id": raw_plan.get("plan_id", "manager_plan_000"),
        "objective": raw_plan.get("objective", ""),
        "object_anchor_routing": object_anchor_routing,
        "tasks": compiled_tasks,
        "stop_if": raw_plan.get("stop_if", []),
        "manager_plan_format": "compact_compiled",
        "compact_source_task_count": len(raw_tasks),
    }
    audit = {
        "schema_version": "compact_manager_plan_compilation",
        "status": "compiled",
        "raw_task_count": len(raw_tasks),
        "compiled_task_count": len(compiled_tasks),
        "raw_plan_bytes": len(str(raw_plan).encode("utf-8")),
        "object_anchor_routing_count": len(object_anchor_routing),
        "object_anchor_routing_source": "manager_semantic_routing_normalized_by_compiler",
        "tasks": audits,
        "policy": "Manager selects and groups capabilities; local registry supplies stable execution contracts without changing scene-specific decisions.",
    }
    return compiled_plan, audit

