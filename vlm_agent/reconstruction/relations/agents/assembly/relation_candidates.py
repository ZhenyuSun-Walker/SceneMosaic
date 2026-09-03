from __future__ import annotations

import re
from typing import Any

from relations.agents.assembly.object_properties import _walk_with_path

def _candidate_decision(candidate: dict[str, Any]) -> str | None:
    decision = (
        candidate.get("decision")
        or candidate.get("status")
        or candidate.get("result")
        or candidate.get("relation_recommendation")
        or candidate.get("recommendation")
    )
    return str(decision).lower() if decision is not None else None


def _relation_from_candidate(candidate: dict[str, Any], source_path: str | None = None) -> dict[str, Any] | None:
    relation_type = str(candidate.get("relation") or candidate.get("relation_type") or "").strip().lower()
    source_path_l = str(source_path or "").lower()
    contains_contain_fields = bool(
        candidate.get("container")
        or candidate.get("container_id")
        or candidate.get("container_anchor")
        or candidate.get("contained_object")
        or candidate.get("contained_object_id")
    )
    if not relation_type and ("contain" in source_path_l or contains_contain_fields):
        relation_type = "contain"
    if relation_type not in {"attach", "contain"}:
        relation_type = "attach"
    if isinstance(candidate.get("parameters"), dict):
        params = candidate["parameters"]
        anchor = (
            params.get("object_anchor")
            or params.get("anchor")
            or params.get("container")
            or params.get("container_anchor")
            or params.get("container_id")
            or candidate.get("candidate_wall_id")
        )
        target = (
            params.get("object_target")
            or params.get("target")
            or params.get("contained_object")
            or params.get("contained_object_id")
            or params.get("object_id")
            or candidate.get("object_target")
        )
    else:
        anchor = (
            candidate.get("object_anchor")
            or candidate.get("container")
            or candidate.get("container_anchor")
            or candidate.get("container_id")
            or candidate.get("support_object_id")
            or candidate.get("support_anchor_id")
            or candidate.get("candidate_wall_id")
            or candidate.get("wall_id")
            or candidate.get("anchor")
            or candidate.get("anchor_id")
            or candidate.get("subject_id")
        )
        target = (
            candidate.get("object_target")
            or candidate.get("object_target_id")
            or candidate.get("contained_object")
            or candidate.get("contained_object_id")
            or candidate.get("target_object_id")
            or candidate.get("object_id")
        )
    if not anchor or not target:
        return None
    return {"relation": relation_type, "parameters": {"object_anchor": anchor, "object_target": target}}


def _repair_inconsistent_support_candidate(candidate: dict[str, Any]) -> dict[str, Any] | None:
    """Repair a structural contradiction using only fields the Agent supplied."""
    if _candidate_decision(candidate) not in {"accept", "accepted", "true", "yes"}:
        return None
    if str(candidate.get("support_mode") or "").lower() != "exclusive_wall_or_ceiling":
        return None
    hypothesis = candidate.get("containment_hypothesis")
    if not isinstance(hypothesis, dict):
        return None
    anchor = hypothesis.get("object_anchor") or hypothesis.get("anchor")
    target = hypothesis.get("object_target") or hypothesis.get("target") or candidate.get("object_target")
    reason = str(candidate.get("reason_code") or candidate.get("reason") or "").strip().lower()
    surface_reason = bool(re.fullmatch(r"on_[a-z0-9_]+_surface", reason))
    if not (
        surface_reason
        and "wall" not in reason
        and "ceiling" not in reason
        and isinstance(anchor, str)
        and anchor.startswith("object_")
        and isinstance(target, str)
        and target.startswith("object_")
        and anchor != target
    ):
        return None
    return {
        "decision": "accept",
        "relation": "attach",
        "parameters": {"object_anchor": anchor, "object_target": target},
        "reason_code": reason,
        "support_candidate_consistency_repair": {
            "original_support_mode": "exclusive_wall_or_ceiling",
            "repaired_support_mode": "object",
            "source_anchor_field": "containment_hypothesis.object_anchor",
            "reason": "accepted object-surface reason contradicted exclusive wall/ceiling mode",
        },
    }



def _relation_from_accepted_support_candidate(candidate: dict[str, Any]) -> dict[str, Any] | None:
    """Compile one accepted gravity-support verdict into the final relation schema."""
    if _candidate_decision(candidate) not in {"accept", "accepted", "true", "yes"}:
        return None
    support_mode = str(candidate.get("support_mode") or "").strip().lower()
    target = candidate.get("object_target") or candidate.get("target") or candidate.get("object_id")
    anchor = candidate.get("object_anchor") or candidate.get("anchor")
    if not (isinstance(target, str) and target.startswith("object_")):
        return None
    if support_mode == "floor":
        anchor = "floor"
    elif support_mode == "object":
        if not (isinstance(anchor, str) and anchor.startswith("object_") and anchor != target):
            return None
    else:
        return None
    return {
        "decision": "accept",
        "relation": "attach",
        "parameters": {"object_anchor": anchor, "object_target": target},
        "reason_code": candidate.get("reason_code") or candidate.get("reason"),
        "compiled_from_support_contact_candidate": True,
    }

def collect_accepted_attach_candidates(task_results: list[dict[str, Any]], guard_result: dict[str, Any] | None = None) -> dict[str, Any]:
    roots: list[Any] = list(task_results)
    if guard_result:
        roots.append(guard_result)
    accepted: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    intermediate_decisions = {
        "needs_consistency_check",
        "needs_wall_resolution",
        "not_final_yet",
        "pending",
        "possible_wall_contact",
        "needs_more_evidence",
        "candidate_pending_wall_resolution",
    }

    def add_candidate(candidate: dict[str, Any], source_path: str, acceptance_mode: str) -> None:
        relation = _relation_from_candidate(candidate, source_path)
        decision = _candidate_decision(candidate)
        if decision in intermediate_decisions or (decision and ("pending" in decision or decision.startswith("needs_"))):
            skipped.append({"candidate": candidate, "source_path": source_path, "reason": "intermediate_wall_attach_candidate_not_final"})
            return
        if acceptance_mode == "requires_accept" and decision not in {"accept", "accepted", "true", "yes"}:
            skipped.append({"candidate": candidate, "source_path": source_path, "reason": "decision_not_accept"})
            return
        if acceptance_mode == "implicit_unless_reject" and decision in {"reject", "rejected", "false", "no"}:
            skipped.append({"candidate": candidate, "source_path": source_path, "reason": "decision_reject"})
            return
        if not relation:
            skipped.append({"candidate": candidate, "source_path": source_path, "reason": "could_not_normalize_relation"})
            return
        accepted.append({"relation": relation, "source_path": source_path, "raw_candidate": candidate, "decision": decision})

    def add_nested_candidates(node: dict[str, Any], source_path: str, acceptance_mode: str) -> bool:
        candidates = node.get("candidates")
        if not isinstance(candidates, list):
            return False
        parent_target = node.get("object_target") or node.get("object_id")
        any_child = False
        for idx, child in enumerate(candidates):
            if not isinstance(child, dict):
                continue
            candidate = dict(child)
            if parent_target and not candidate.get("object_target"):
                candidate["object_target"] = parent_target
            if candidate.get("anchor_id") and not candidate.get("object_anchor"):
                candidate["object_anchor"] = candidate.get("anchor_id")
            add_candidate(candidate, f"{source_path}.candidates[{idx}]", acceptance_mode)
            any_child = True
        return any_child

    def add_candidate_or_nested(candidate: dict[str, Any], source_path: str, acceptance_mode: str) -> None:
        if add_nested_candidates(candidate, source_path, acceptance_mode):
            return
        add_candidate(candidate, source_path, acceptance_mode)

    for root_idx, root in enumerate(roots):
        root_name = root.get("task_id", f"root_{root_idx}") if isinstance(root, dict) else f"root_{root_idx}"
        for path, node in _walk_with_path(root, root_name):
            if not isinstance(node, dict):
                continue
            support_values = node.get("support_contact_candidates")
            if isinstance(support_values, list):
                for idx, item in enumerate(support_values):
                    if not isinstance(item, dict):
                        continue
                    compiled = _relation_from_accepted_support_candidate(item)
                    if compiled:
                        add_candidate(
                            compiled,
                            f"{path}.support_contact_candidates[{idx}].compiled_support_verdict",
                            "requires_accept",
                        )
                    repaired = _repair_inconsistent_support_candidate(item)
                    if repaired:
                        add_candidate(
                            repaired,
                            f"{path}.support_contact_candidates[{idx}].consistency_repair",
                            "requires_accept",
                        )
            for key in ["accepted_relations", "accepted_attach_candidates", "accepted_contain_relations", "accepted_contain_candidates", "accepted_contain_findings", "verified_wall_attach_relations", "accepted_wall_attach_relations", "verified_wall_attach_candidates"]:
                value = node.get(key)
                if isinstance(value, list):
                    for item in value:
                        if isinstance(item, dict):
                            add_candidate_or_nested(item, f"{path}.{key}", "implicit_unless_reject")
            psu = node.get("proposed_state_updates")
            if isinstance(psu, dict):
                for key in ["wall_attach_candidates", "attach_candidates", "contain_candidates", "contain_relations", "contain_findings", "accepted_relations", "accepted_contain_relations", "accepted_contain_candidates", "accepted_contain_findings", "verified_wall_attach_relations", "accepted_wall_attach_relations", "verified_wall_attach_candidates", "final_relation_candidates", "final_attach_relation_candidates", "final_contain_relation_candidates", "proposed_attach_relations", "proposed_contain_relations"]:
                    value = psu.get(key)
                    if isinstance(value, list):
                        for item in value:
                            if isinstance(item, dict):
                                add_candidate_or_nested(item, f"{path}.proposed_state_updates.{key}", "implicit_unless_reject")
            findings = node.get("findings")
            if isinstance(findings, dict):
                for key in ["wall_attach_candidates", "attach_candidates", "contain_candidates", "contain_relations", "contain_findings", "relation_candidates", "relations"]:
                    value = findings.get(key)
                    if isinstance(value, list):
                        for item in value:
                            if isinstance(item, dict):
                                add_candidate_or_nested(item, f"{path}.findings.{key}", "requires_accept")
                for key in ["accepted_relations", "accepted_attach_candidates", "accepted_contain_relations", "accepted_contain_candidates", "accepted_contain_findings", "final_relation_candidates", "final_attach_relation_candidates", "final_contain_relation_candidates", "proposed_attach_relations", "proposed_contain_relations"]:
                    value = findings.get(key)
                    if isinstance(value, list):
                        for item in value:
                            if isinstance(item, dict):
                                add_candidate_or_nested(item, f"{path}.findings.{key}", "implicit_unless_reject")
            for key in ["relations", "relation_candidates", "wall_attach_candidates", "wall_attach_recall", "attach_candidates", "contain_candidates", "contain_relations", "contain_findings"]:
                value = node.get(key)
                if isinstance(value, list):
                    for item in value:
                        if isinstance(item, dict):
                            add_candidate_or_nested(item, f"{path}.{key}", "requires_accept")
            if isinstance(node.get("relations_json"), list):
                for item in node["relations_json"]:
                    if isinstance(item, dict):
                        add_candidate_or_nested(item, f"{path}.relations_json", "implicit_unless_reject")
    return {"accepted": accepted, "skipped": skipped}

def collect_coverage_repair_candidates(task_results: list[dict[str, Any]], guard_result: dict[str, Any] | None = None) -> dict[str, Any]:
    roots: list[Any] = list(task_results)
    if guard_result:
        roots.append(guard_result)
    accepted: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []

    def add_candidate(candidate: dict[str, Any], source_path: str, implicit: bool = False) -> None:
        if not isinstance(candidate, dict):
            return
        if not implicit and candidate.get("coverage_repair") is not True:
            skipped.append({"candidate": candidate, "source_path": source_path, "reason": "not_marked_coverage_repair"})
            return
        relation = _relation_from_candidate(candidate)
        if not relation:
            skipped.append({"candidate": candidate, "source_path": source_path, "reason": "could_not_normalize_relation"})
            return
        accepted.append(
            {
                "relation": relation,
                "source_path": source_path,
                "raw_candidate": candidate,
                "reason": candidate.get("repair_reason") or candidate.get("reason") or candidate.get("rationale"),
                "evidence_used": candidate.get("evidence_used") or candidate.get("evidence") or [],
                "coverage_repair": True,
            }
        )

    for root_idx, root in enumerate(roots):
        root_name = root.get("task_id", f"root_{root_idx}") if isinstance(root, dict) else f"root_{root_idx}"
        for path, node in _walk_with_path(root, root_name):
            if not isinstance(node, dict):
                continue
            psu = node.get("proposed_state_updates")
            if isinstance(psu, dict):
                for key in ["coverage_repair_relations", "coverage_repair_candidates"]:
                    value = psu.get(key)
                    if isinstance(value, list):
                        for item in value:
                            add_candidate(item, f"{path}.proposed_state_updates.{key}", implicit=True)
            findings = node.get("findings")
            if isinstance(findings, dict):
                for key in ["coverage_repair_relations", "coverage_repair_candidates"]:
                    value = findings.get(key)
                    if isinstance(value, list):
                        for item in value:
                            add_candidate(item, f"{path}.findings.{key}", implicit=True)
            for key in ["coverage_repair_relations", "coverage_repair_candidates"]:
                value = node.get(key)
                if isinstance(value, list):
                    for item in value:
                        add_candidate(item, f"{path}.{key}", implicit=True)
            for key in ["accepted_relations", "relation_candidates"]:
                value = node.get(key)
                if isinstance(value, list):
                    for item in value:
                        add_candidate(item, f"{path}.{key}", implicit=False)
    return {"accepted": accepted, "skipped": skipped}



WALL_MOUNTED_KEEP_KEYWORDS = {
    "attached to wall",
    "embedded",
    "mounted",
    "wall mounted",
    "wall-mounted",
    "wall-hung",
}

# Exact semantic categories that are wall-constrained even when the free-text
# description omits words such as "mounted" or "hanging".
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
