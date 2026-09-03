from __future__ import annotations

import copy
import math
import re
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from PIL import Image

def validate_b_actions(
    pass_a_objects: Sequence[Mapping[str, Any]],
    actions: Mapping[str, Any],
    *,
    expected_local_candidate_ids: Sequence[str] | None = None,
) -> None:
    known_ids = {str(item["object_id"]) for item in pass_a_objects}
    required = {"additions", "updates", "removals"}
    allowed = required | {
        "local_candidate_decisions",
        "local_candidate_rejections",
        "assigned_evidence_decisions",
        "assigned_evidence_semantic_gate",
    }
    if not required.issubset(actions) or not set(actions).issubset(allowed):
        raise ValueError(
            f"Pass B action keys must contain {sorted(required)} and only optional "
            "candidate or assigned-evidence audit fields"
        )
    update_ids = [str(item["pass_a_id"]) for item in actions["updates"]]
    removal_ids = [str(item) for item in actions["removals"]]
    if len(update_ids) != len(set(update_ids)):
        raise ValueError("Pass B updates one Pass-A ID more than once")
    if len(removal_ids) != len(set(removal_ids)):
        raise ValueError("Pass B removes one Pass-A ID more than once")
    unknown = (set(update_ids) | set(removal_ids)) - known_ids
    if unknown:
        raise ValueError(f"Pass B references unknown Pass-A IDs: {sorted(unknown)}")
    conflict = set(update_ids) & set(removal_ids)
    if conflict:
        raise ValueError(
            f"Pass B both updates and removes IDs: {sorted(conflict)}"
        )
    if (
        "local_candidate_decisions" in actions
        and "local_candidate_rejections" in actions
    ):
        raise ValueError("Pass B candidate decision protocols conflict")
    if "local_candidate_rejections" in actions:
        raw_rejections = actions["local_candidate_rejections"]
        if not isinstance(raw_rejections, list):
            raise ValueError("Pass B local candidate rejections must be a list")
        rejection_ids = [str(value) for value in raw_rejections]
        if not all(re.fullmatch(r"LC_\d{3}", value) for value in rejection_ids):
            raise ValueError("invalid Pass B local candidate rejection ID")
        if len(rejection_ids) != len(set(rejection_ids)):
            raise ValueError("duplicate Pass B local candidate rejection ID")

    if expected_local_candidate_ids is not None:
        if isinstance(expected_local_candidate_ids, (str, bytes)):
            raise ValueError("expected local candidate IDs must be a sequence")
        expected_ids = [str(value) for value in expected_local_candidate_ids]
        if len(expected_ids) != len(set(expected_ids)) or not all(
            re.fullmatch(r"LC_\d{3}", value) for value in expected_ids
        ):
            raise ValueError(
                "expected local candidate IDs must be unique LC_NNN values"
            )
        protocol_keys = {
            key
            for key in ("local_candidate_decisions", "local_candidate_rejections")
            if key in actions
        }
        if not expected_ids:
            if protocol_keys:
                raise ValueError(
                    "Pass B returned a local candidate protocol without a panel"
                )
        elif len(protocol_keys) != 1:
            raise ValueError(
                "Pass B must use exactly one local candidate protocol for a panel"
            )
        elif "local_candidate_rejections" in protocol_keys:
            unknown_rejections = sorted(
                set(rejection_ids) - set(expected_ids)
            )
            if unknown_rejections:
                raise ValueError(
                    "Pass B rejects unknown local candidate IDs: "
                    f"{unknown_rejections}"
                )
        else:
            decisions = actions["local_candidate_decisions"]
            if not isinstance(decisions, list):
                raise ValueError("Pass B local candidate decisions must be a list")
            decision_ids = [
                str(item.get("candidate_id") or "")
                for item in decisions
                if isinstance(item, Mapping)
            ]
            if len(decision_ids) != len(decisions):
                raise ValueError("invalid Pass B local candidate decision")
            if len(decision_ids) != len(set(decision_ids)):
                raise ValueError("duplicate Pass B local candidate decision ID")
            if set(decision_ids) != set(expected_ids):
                raise ValueError(
                    "Pass B local candidate decisions do not cover the panel"
                )

    evidence_keys = {
        key
        for key in (
            "assigned_evidence_decisions",
            "assigned_evidence_semantic_gate",
        )
        if key in actions
    }
    if evidence_keys and len(evidence_keys) != 2:
        raise ValueError(
            "assigned evidence decisions and semantic gate must travel together"
        )
    if evidence_keys:
        decisions = actions["assigned_evidence_decisions"]
        gate = actions["assigned_evidence_semantic_gate"]
        if not isinstance(decisions, list) or not isinstance(gate, Mapping):
            raise ValueError("invalid assigned evidence semantic gate")
        evidence_ids = []
        for decision in decisions:
            if not isinstance(decision, Mapping):
                raise ValueError("invalid assigned evidence decision")
            evidence_id = str(decision.get("evidence_id") or "")
            owner_id = str(decision.get("source_object_id") or "")
            if not re.fullmatch(r"E_\d{3}", evidence_id):
                raise ValueError("invalid assigned evidence decision ID")
            if owner_id not in known_ids:
                raise ValueError(
                    f"assigned evidence references unknown object: {owner_id}"
                )
            evidence_ids.append(evidence_id)
        if len(evidence_ids) != len(set(evidence_ids)):
            raise ValueError("duplicate assigned evidence decision ID")
        if int(gate.get("decision_count", -1)) != len(decisions):
            raise ValueError("assigned evidence gate count mismatch")
        if not bool(gate.get("complete_partition")):
            raise ValueError("assigned evidence gate partition is incomplete")





def apply_post_pass_b_policy_guard(
    pass_a_objects: Sequence[Mapping[str, Any]],
    actions: Mapping[str, Any],
    sam3_registry: Mapping[str, Any] | None = None,
    *,
    content_hosts: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    "Suppress bounded Pass-B semantic and geometry overcorrections."

    effective = copy.deepcopy(dict(actions))

    def tokens(item: Mapping[str, Any]) -> set[str]:
        text = " ".join(
            str(item.get(key) or "")
            for key in (
                "category",
                "sam3_prompt",
                "instance_name",
                "description",
            )
        ).lower()
        return set(re.findall(r"[a-z0-9]+", text))

    def box(item: Mapping[str, Any]) -> list[float] | None:
        try:
            value = [
                float(component)
                for component in item.get("bbox_2d_norm_xyxy")
            ]
        except (TypeError, ValueError):
            return None
        if (
            len(value) != 4
            or not all(math.isfinite(component) for component in value)
            or not (0.0 <= value[0] < value[2] <= 1.0)
            or not (0.0 <= value[1] < value[3] <= 1.0)
        ):
            return None
        return value

    def center_x(value: Sequence[float]) -> float:
        return (float(value[0]) + float(value[2])) / 2.0

    def is_bed(item: Mapping[str, Any]) -> bool:
        category_tokens = set(
            re.findall(
                r"[a-z0-9]+",
                str(item.get("category") or "").lower().replace("_", " "),
            )
        )
        return "bed" in category_tokens and not bool(
            category_tokens & {"bench", "bedside", "headboard"}
        )

    def is_complete_potted_plant(item: Mapping[str, Any]) -> bool:
        """Identify an inventory unit whose pot and foliage form one object."""

        return {"potted", "plant"} <= tokens(item)

    def is_container_only_botanical_downgrade(
        source: Mapping[str, Any], replacement: Mapping[str, Any]
    ) -> bool:
        """Reject a semantic split that turns one verified plant into its pot."""

        replacement_tokens = tokens(replacement)
        return (
            is_complete_potted_plant(source)
            and "plant" not in replacement_tokens
            and bool(replacement_tokens & {"pot", "planter", "vase"})
        )

    def center_y(value: Sequence[float]) -> float:
        return (float(value[1]) + float(value[3])) / 2.0

    def normalized_box(value: Any) -> list[float] | None:
        try:
            candidate = [float(component) for component in value]
        except (TypeError, ValueError):
            return None
        if (
            len(candidate) != 4
            or not all(math.isfinite(component) for component in candidate)
            or not (0.0 <= candidate[0] < candidate[2] <= 1.0)
            or not (0.0 <= candidate[1] < candidate[3] <= 1.0)
        ):
            return None
        return candidate

    def bbox_iou(left: Sequence[float], right: Sequence[float]) -> float:
        x1 = max(float(left[0]), float(right[0]))
        y1 = max(float(left[1]), float(right[1]))
        x2 = min(float(left[2]), float(right[2]))
        y2 = min(float(left[3]), float(right[3]))
        intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
        left_area = max(0.0, float(left[2]) - float(left[0])) * max(
            0.0, float(left[3]) - float(left[1])
        )
        right_area = max(0.0, float(right[2]) - float(right[0])) * max(
            0.0, float(right[3]) - float(right[1])
        )
        union = left_area + right_area - intersection
        return intersection / union if union else 0.0

    # H panels are attention context only. A returned update may legitimately
    # describe an A owner, but it must never replace that owner with the H crop
    # itself. This geometry check is independent of host category and remains
    # effective even if the VLM ignores the textual H contract.
    attention_crop_by_owner: dict[str, list[float]] = {}
    for raw_host in content_hosts or []:
        if not isinstance(raw_host, Mapping):
            continue
        owner_id = str(raw_host.get("source_object_id") or "")
        crop = normalized_box(raw_host.get("crop_norm_xyxy"))
        if owner_id and crop is not None:
            attention_crop_by_owner[owner_id] = crop

    live = [dict(item) for item in pass_a_objects]
    live_by_id = {str(item.get("object_id") or ""): item for item in live}
    registry_by_id = {
        str(item.get("object_id") or ""): item
        for item in (sam3_registry or {}).get("objects") or []
        if isinstance(item, Mapping)
    }

    def has_successful_sam3_mask(object_id: str) -> bool:
        registry_item = registry_by_id.get(object_id, {})
        return str((registry_item.get("sam3") or {}).get("status") or "") == "success"

    def preserves_failed_owner_semantics(
        source_item: Mapping[str, Any], replacement: Mapping[str, Any]
    ) -> bool:
        """Allow failed-mask updates only when their object class remains local."""
        ignored = {
            "unknown",
            "object",
            "item",
            "thing",
            "furniture",
            "appliance",
            "fixture",
            "device",
            "decor",
            "small",
        }
        source_terms = {
            token
            for token in re.findall(
                r"[a-z0-9]+", str(source_item.get("category") or "").lower()
            )
            if token not in ignored
        }
        replacement_terms = {
            token
            for token in re.findall(
                r"[a-z0-9]+", str(replacement.get("category") or "").lower()
            )
            if token not in ignored
        }
        # A generic/unknown label gives no reliable semantic signal. Preserve
        # the model action and let its geometry/evidence be audited elsewhere.
        return (
            not source_terms
            or not replacement_terms
            or bool(source_terms & replacement_terms)
        )

    sinks = [
        item
        for item in live
        if "sink" in tokens(item)
        and bool({"counter", "assembly", "vanity", "cabinet"} & tokens(item))
        and box(item) is not None
    ]
    counters = [
        item
        for item in live
        if "counter" in tokens(item)
        and "sink" not in tokens(item)
        and "island" not in tokens(item)
        and box(item) is not None
    ]
    separators = [
        item
        for item in live
        if bool({"range", "stove"} & tokens(item))
        and box(item) is not None
    ]
    partitions = []
    protected_roles: dict[str, str] = {}
    # A successful functional ROI refinement has already probed this complete
    # vanity (including fixed faucets) with the same local SAM3 session.  A
    # Pass-B wording-only update must not trigger a new prompt-group inference
    # and silently discard that validated fixture extent.
    for item in live:
        item_tokens = tokens(item)
        if (
            "vanity" in item_tokens
            and bool(item_tokens & {"sink", "sinks", "basin", "basins"})
            and bool(item_tokens & {"faucet", "faucets"})
        ):
            protected_roles[str(item.get("object_id") or "")] = (
                "complete_sink_vanity_assembly"
            )
    for sink in sinks:
        sink_box = box(sink)
        assert sink_box is not None
        for counter in counters:
            counter_box = box(counter)
            assert counter_box is not None
            for separator in separators:
                separator_box = box(separator)
                assert separator_box is not None
                sx = center_x(sink_box)
                cx = center_x(counter_box)
                rx = center_x(separator_box)
                if sx < rx < cx:
                    sink_side = "left"
                    sink_gap = max(0.0, separator_box[0] - sink_box[2])
                    counter_gap = max(0.0, counter_box[0] - separator_box[2])
                elif cx < rx < sx:
                    sink_side = "right"
                    sink_gap = max(0.0, sink_box[0] - separator_box[2])
                    counter_gap = max(0.0, separator_box[0] - counter_box[2])
                else:
                    continue
                if max(sink_gap, counter_gap) > 0.12:
                    continue
                if max(
                    abs(center_y(sink_box) - center_y(separator_box)),
                    abs(center_y(counter_box) - center_y(separator_box)),
                ) > 0.22:
                    continue
                sink_id = str(sink.get("object_id") or "")
                counter_id = str(counter.get("object_id") or "")
                separator_id = str(separator.get("object_id") or "")
                protected_roles[sink_id] = "sink_counter_assembly"
                protected_roles[counter_id] = "adjacent_counter_cabinet"
                partitions.append(
                    {
                        "sink_object_id": sink_id,
                        "counter_object_id": counter_id,
                        "separator_object_id": separator_id,
                        "sink_side": sink_side,
                    }
                )

    kept_updates = []
    suppressed_updates = []
    failed_mask_semantic_rejections = []
    for update in effective.get("updates") or []:
        source_id = str(update.get("pass_a_id") or "")
        source_item = live_by_id.get(source_id, {})
        source_tokens = tokens(source_item)
        replacement = update.get("replacement") or {}
        replacement_tokens = (
            tokens(replacement) if isinstance(replacement, Mapping) else set()
        )
        source_box = box(source_item)
        replacement_box = (
            box(replacement) if isinstance(replacement, Mapping) else None
        )
        replacement_is_sink = "sink" in replacement_tokens
        role = protected_roles.get(source_id)
        reason = None
        if (
            isinstance(replacement, Mapping)
            and not has_successful_sam3_mask(source_id)
            and not preserves_failed_owner_semantics(source_item, replacement)
        ):
            failed_mask_semantic_rejections.append(
                {
                    "pass_a_id": source_id,
                    "source_category": str(source_item.get("category") or ""),
                    "replacement_category": str(replacement.get("category") or ""),
                    "reason": "failed_mask_update_changes_object_semantics",
                }
            )
            continue
        attention_crop = attention_crop_by_owner.get(source_id)
        if (
            attention_crop is not None
            and replacement_box is not None
            and bbox_iou(attention_crop, replacement_box) >= 0.98
        ):
            reason = "content_attention_crop_cannot_replace_owner_bbox"
        elif (
            is_container_only_botanical_downgrade(source_item, replacement)
            and has_successful_sam3_mask(source_id)
        ):
            reason = (
                "verified_potted_plant_assembly_cannot_be_relabelled_as_container"
            )
        elif (
            is_bed(source_item)
            and isinstance(replacement, Mapping)
            and is_bed(replacement)
            and source_box is not None
            and replacement_box is not None
            and replacement_box[1] > source_box[1] + 0.015
        ):
            reason = "complete_bed_assembly_cannot_drop_headboard_or_bedding"
        elif role == "adjacent_counter_cabinet" and replacement_is_sink:
            reason = "counter_across_range_cannot_become_sink_assembly"
        elif (
            role == "sink_counter_assembly"
            and "counter" in replacement_tokens
            and not replacement_is_sink
        ):
            reason = "sink_assembly_cannot_become_plain_counter"
        elif (
            bool(source_tokens & {"pillow", "cushion"})
            and bool(replacement_tokens & {"bed", "bunk", "loft"})
        ):
            reason = "independent_pillow_cannot_become_bed"
        elif (
            role == "complete_sink_vanity_assembly"
            and "vanity" in replacement_tokens
            and bool(replacement_tokens & {"sink", "basin"})
        ):
            reason = "roi_refined_sink_vanity_preserves_fixed_fixture_mask"
        elif (
            role in {"sink_counter_assembly", "adjacent_counter_cabinet"}
            and str(replacement.get("category") or "")
            == str(live_by_id.get(source_id, {}).get("category") or "")
            and str(replacement.get("sam3_prompt") or "")
            == str(live_by_id.get(source_id, {}).get("sam3_prompt") or "")
        ):
            reason = "roi_refined_functional_role_preserves_assigned_mask"
        if reason is None:
            kept_updates.append(update)
        else:
            suppressed_updates.append(
                {
                    "pass_a_id": source_id,
                    "reason": reason,
                    "replacement_category": str(
                        replacement.get("category") or ""
                    ),
                    "attention_crop_norm_xyxy": attention_crop,
                }
            )
    effective["updates"] = kept_updates
    failed_mask_semantic_rejection_ids = {
        str(item["pass_a_id"]) for item in failed_mask_semantic_rejections
    }
    effective["removals"] = list(
        dict.fromkeys(
            [
                *[str(item) for item in effective.get("removals") or []],
                *sorted(failed_mask_semantic_rejection_ids),
            ]
        )
    )

    suppressed_update_ids = {
        str(item["pass_a_id"]) for item in suppressed_updates
    }
    rewritten_evidence_decisions = []
    for decision in effective.get("assigned_evidence_decisions") or []:
        source_id = str(decision.get("source_object_id") or "")
        if (
            source_id in failed_mask_semantic_rejection_ids
            and decision.get("decision") == "update_existing"
        ):
            previous_reason = str(decision.get("reason") or "")
            decision["decision"] = "remove_existing"
            decision["reason"] = (
                "failed SAM3 owner changed to a semantically different object"
            )
            rewritten_evidence_decisions.append(
                {
                    "source_object_id": source_id,
                    "previous_decision": "update_existing",
                    "effective_decision": "remove_existing",
                    "previous_reason": previous_reason,
                    "effective_reason": decision["reason"],
                }
            )
        elif (
            source_id in suppressed_update_ids
            and decision.get("decision") == "update_existing"
        ):
            previous_reason = str(decision.get("reason") or "")
            decision["decision"] = "pass"
            decision["reason"] = (
                "local object-unit policy retained the verified source role"
            )
            rewritten_evidence_decisions.append(
                {
                    "source_object_id": source_id,
                    "previous_decision": "update_existing",
                    "effective_decision": "pass",
                    "previous_reason": previous_reason,
                    "effective_reason": decision["reason"],
                }
            )

    raw_removal_ids = [str(value) for value in effective.get("removals") or []]
    raw_removal_id_set = set(raw_removal_ids)
    kept_removals = []
    suppressed_removals = []
    registry_by_id = {
        str(item.get("object_id") or ""): item
        for item in (sam3_registry or {}).get("objects") or []
        if isinstance(item, Mapping)
    }

    def category_key(item: Mapping[str, Any]) -> str:
        category = str(item.get("category") or "").lower()
        category_tokens = set(re.findall(r"[a-z0-9]+", category))
        if category_tokens & {"pillow", "cushion"}:
            return "pillow"
        return "_".join(sorted(category_tokens))

    def registry_mask(object_id: str):
        registry_item = registry_by_id.get(object_id, {})
        sam3 = registry_item.get("sam3") or {}
        if sam3.get("status") != "success":
            return None, "missing_successful_sam3_mask"
        path = Path(str(sam3.get("mask_path") or "")).expanduser().resolve()
        if not path.is_file():
            return None, "sam3_mask_path_missing"
        try:
            import numpy as np
            from PIL import Image

            mask = np.array(Image.open(path).convert("L")) > 0
        except Exception as exc:
            return None, f"sam3_mask_load_failed:{type(exc).__name__}"
        if mask.ndim != 2 or not mask.any():
            return None, "sam3_mask_invalid_or_empty"
        return mask, None

    decisions_by_source = {
        str(item.get("source_object_id") or ""): item
        for item in effective.get("assigned_evidence_decisions") or []
        if isinstance(item, Mapping)
    }
    has_complete_bed = any(is_bed(item) for item in live)
    for source_id in raw_removal_ids:
        decision = decisions_by_source.get(source_id)
        decision_reason = str((decision or {}).get("reason") or "").lower()
        is_duplicate_claim = bool(
            re.search(r"\bduplicate\b|same\s+visible\s+.+\s+region", decision_reason)
        )
        source_item = live_by_id.get(source_id, {})
        source_category = category_key(source_item)
        is_bed_assembly_exclusion_claim = bool(
            source_category == "pillow"
            and has_complete_bed
            and re.search(r"\bexclud(?:e|ed|ing)\b", decision_reason)
            and re.search(r"\b(?:bed|bedding|assembly)\b", decision_reason)
        )
        if is_bed_assembly_exclusion_claim and not is_duplicate_claim:
            source_mask, source_mask_error = registry_mask(source_id)
            if source_mask is not None:
                suppressed_removals.append(
                    {
                        "source_object_id": source_id,
                        "reason": (
                            "bed_mask_exclusion_preserves_independent_pillow"
                        ),
                        "source_mask_error": source_mask_error,
                        "retained_area_pixels": int(source_mask.sum()),
                    }
                )
                if (
                    decision is not None
                    and decision.get("decision") == "remove_existing"
                ):
                    previous_reason = str(decision.get("reason") or "")
                    decision["decision"] = "pass"
                    decision["reason"] = (
                        "local assembly policy keeps a pillow excluded from "
                        "the bed mask as an independent inventory instance"
                    )
                    rewritten_evidence_decisions.append(
                        {
                            "source_object_id": source_id,
                            "previous_decision": "remove_existing",
                            "effective_decision": "pass",
                            "previous_reason": previous_reason,
                            "effective_reason": decision["reason"],
                        }
                    )
                continue
        if not is_duplicate_claim:
            kept_removals.append(source_id)
            continue
        source_mask, source_mask_error = registry_mask(source_id)
        best_geometry = None
        if source_mask is not None and source_category:
            source_area = int(source_mask.sum())
            for candidate_id, candidate_item in live_by_id.items():
                if (
                    candidate_id == source_id
                    or candidate_id in raw_removal_id_set
                    or category_key(candidate_item) != source_category
                ):
                    continue
                candidate_mask, _ = registry_mask(candidate_id)
                if candidate_mask is None or candidate_mask.shape != source_mask.shape:
                    continue
                intersection = int((source_mask & candidate_mask).sum())
                if intersection < 30:
                    continue
                candidate_area = int(candidate_mask.sum())
                union = source_area + candidate_area - intersection
                geometry = {
                    "candidate_object_id": candidate_id,
                    "intersection_pixels": intersection,
                    "mask_iou": round(intersection / max(union, 1), 6),
                    "source_containment": round(
                        intersection / max(source_area, 1), 6
                    ),
                    "candidate_containment": round(
                        intersection / max(candidate_area, 1), 6
                    ),
                }
                if best_geometry is None or geometry["mask_iou"] > best_geometry["mask_iou"]:
                    best_geometry = geometry
        duplicate_proven = bool(
            best_geometry
            and (
                best_geometry["mask_iou"] >= 0.80
                or best_geometry["source_containment"] >= 0.95
                or best_geometry["candidate_containment"] >= 0.95
            )
        )
        if duplicate_proven:
            kept_removals.append(source_id)
            continue
        suppressed_removals.append(
            {
                "source_object_id": source_id,
                "reason": "duplicate_removal_requires_same_category_mask_geometry",
                "source_mask_error": source_mask_error,
                "best_same_category_geometry": best_geometry,
            }
        )
        if decision is not None and decision.get("decision") == "remove_existing":
            previous_reason = str(decision.get("reason") or "")
            decision["decision"] = "pass"
            decision["reason"] = (
                "local geometry policy retained a distinct same-category instance"
            )
            rewritten_evidence_decisions.append(
                {
                    "source_object_id": source_id,
                    "previous_decision": "remove_existing",
                    "effective_decision": "pass",
                    "previous_reason": previous_reason,
                    "effective_reason": decision["reason"],
                }
            )
    effective["removals"] = kept_removals

    hoods = [
        item
        for item in live
        if "hood" in tokens(item) and box(item) is not None
    ]
    shelves = [
        item
        for item in live
        if "shelf" in tokens(item) and box(item) is not None
    ]
    beds = [item for item in live if is_bed(item) and box(item) is not None]
    updated_ids = {
        str(item.get("pass_a_id") or "")
        for item in effective.get("updates") or []
        if isinstance(item, Mapping)
    }
    verified_potted_plant_assemblies = [
        item
        for item in live
        if (
            is_complete_potted_plant(item)
            and str(item.get("object_id") or "") not in updated_ids
            and str(item.get("object_id") or "") not in kept_removals
            and has_successful_sam3_mask(str(item.get("object_id") or ""))
            and box(item) is not None
        )
    ]
    kept_additions = []
    suppressed_additions = []
    addition_id_remap = {}
    for index, addition in enumerate(effective.get("additions") or [], start=1):
        addition_tokens = tokens(addition)
        addition_box = box(addition)
        matched_hood = None
        matched_shelf = None
        matched_bed = None
        matched_potted_plant = None
        if addition_box is not None and "headboard" in addition_tokens:
            addition_area = (
                (addition_box[2] - addition_box[0])
                * (addition_box[3] - addition_box[1])
            )
            for bed in beds:
                bed_box = box(bed)
                assert bed_box is not None
                intersection = (
                    max(
                        0.0,
                        min(addition_box[2], bed_box[2])
                        - max(addition_box[0], bed_box[0]),
                    )
                    * max(
                        0.0,
                        min(addition_box[3], bed_box[3])
                        - max(addition_box[1], bed_box[1]),
                    )
                )
                if intersection / max(addition_area, 1e-9) >= 0.85:
                    matched_bed = bed
                    break
        if (
            addition_box is not None
            and bool({"light", "lighting", "downlight"} & addition_tokens)
            and bool({"ceiling", "recessed", "downlight"} & addition_tokens)
        ):
            addition_center_x = center_x(addition_box)
            for hood in hoods:
                hood_box = box(hood)
                assert hood_box is not None
                if (
                    hood_box[0] - 0.01
                    <= addition_center_x
                    <= hood_box[2] + 0.01
                    and hood_box[1] <= addition_box[1] <= hood_box[3] + 0.04
                ):
                    matched_hood = hood
                    break
        if (
            matched_hood is None
            and addition_box is not None
            and bool({"light", "lighting"} & addition_tokens)
            and bool({"under", "undercabinet", "linear", "strip"} & addition_tokens)
        ):
            addition_width = addition_box[2] - addition_box[0]
            addition_height = addition_box[3] - addition_box[1]
            for shelf in shelves:
                shelf_box = box(shelf)
                assert shelf_box is not None
                shelf_width = shelf_box[2] - shelf_box[0]
                horizontal_overlap = max(
                    0.0,
                    min(addition_box[2], shelf_box[2])
                    - max(addition_box[0], shelf_box[0]),
                )
                overlap_ratio = horizontal_overlap / max(
                    min(addition_width, shelf_width), 1e-9
                )
                if (
                    addition_width >= 0.15
                    and addition_height <= 0.08
                    and addition_width / max(addition_height, 1e-9) >= 4.0
                    and overlap_ratio >= 0.90
                    and shelf_box[1] - 0.01 <= addition_box[1] <= shelf_box[3] + 0.02
                    and addition_box[3] <= shelf_box[3] + 0.08
                ):
                    matched_shelf = shelf
                    break
        if (
            matched_hood is None
            and matched_shelf is None
            and matched_bed is None
            and addition_box is not None
            and bool(
                addition_tokens
                & {
                    "plant",
                    "foliage",
                    "leaf",
                    "leaves",
                    "stem",
                    "stems",
                    "flower",
                    "flowers",
                    "branch",
                    "branches",
                }
            )
        ):
            addition_area = (
                (addition_box[2] - addition_box[0])
                * (addition_box[3] - addition_box[1])
            )
            for assembly in verified_potted_plant_assemblies:
                assembly_box = box(assembly)
                assert assembly_box is not None
                intersection = (
                    max(
                        0.0,
                        min(addition_box[2], assembly_box[2])
                        - max(addition_box[0], assembly_box[0]),
                    )
                    * max(
                        0.0,
                        min(addition_box[3], assembly_box[3])
                        - max(addition_box[1], assembly_box[1]),
                    )
                )
                if intersection / max(addition_area, 1e-9) >= 0.98:
                    matched_potted_plant = assembly
                    break
        if (
            matched_hood is None
            and matched_shelf is None
            and matched_bed is None
            and matched_potted_plant is None
        ):
            kept_additions.append(addition)
            addition_id_remap[f"B_{index:03d}"] = (
                f"B_{len(kept_additions):03d}"
            )
        else:
            suppression = {
                "raw_addition_index": index,
                "category": str(addition.get("category") or ""),
                "bbox_2d_norm_xyxy": list(
                    addition.get("bbox_2d_norm_xyxy") or []
                ),
            }
            if matched_potted_plant is not None:
                suppression.update(
                    {
                        "matched_potted_plant_object_id": str(
                            matched_potted_plant.get("object_id") or ""
                        ),
                        "reason": (
                            "foliage_inside_verified_potted_plant_is_not_a_second_object"
                        ),
                    }
                )
            elif matched_bed is not None:
                suppression.update(
                    {
                        "matched_bed_object_id": str(
                            matched_bed.get("object_id") or ""
                        ),
                        "reason": (
                            "physical_headboard_is_part_of_complete_bed_assembly"
                        ),
                    }
                )
            elif matched_shelf is not None:
                suppression.update(
                    {
                        "matched_shelf_object_id": str(
                            matched_shelf.get("object_id") or ""
                        ),
                        "reason": (
                            "thin_under_shelf_light_is_attached_host_assembly"
                        ),
                    }
                )
            else:
                suppression.update(
                    {
                        "matched_hood_object_id": str(
                            matched_hood.get("object_id") or ""
                        ),
                        "reason": "recessed_light_inside_range_hood_extent",
                    }
                )
            suppressed_additions.append(suppression)
    effective["additions"] = kept_additions
    suppressed_bed_addition_ids = {
        f"B_{int(item['raw_addition_index']):03d}"
        for item in suppressed_additions
        if item.get("matched_bed_object_id")
    }
    binding_errors = []
    for decision in effective.get("local_candidate_decisions") or []:
        if decision.get("decision") != "bind_to_addition":
            continue
        old_target = str(decision.get("target_object_id") or "")
        new_target = addition_id_remap.get(old_target)
        if new_target is None:
            binding_errors.append(
                {
                    "type": "candidate_bound_to_suppressed_addition",
                    "candidate_id": str(decision.get("candidate_id") or ""),
                    "target_object_id": old_target,
                }
            )
            decision["decision"] = "reject"
            decision["target_object_id"] = None
            decision["reason"] = "target addition suppressed by local policy"
        else:
            decision["target_object_id"] = new_target
    for decision in effective.get("assigned_evidence_decisions") or []:
        supporting = list(decision.get("supporting_addition_ids") or [])
        remapped = [
            addition_id_remap[value]
            for value in supporting
            if value in addition_id_remap
        ]
        missing_supporting = [
            value for value in supporting if value not in addition_id_remap
        ]
        source_id = str(decision.get("source_object_id") or "")
        bed_split_absorbed = bool(
            missing_supporting
            and decision.get("decision") == "split_instances"
            and is_bed(live_by_id.get(source_id, {}))
            and set(missing_supporting) <= suppressed_bed_addition_ids
        )
        if bed_split_absorbed:
            previous_reason = str(decision.get("reason") or "")
            effective_decision = (
                "update_existing"
                if any(
                    str(item.get("pass_a_id") or "") == source_id
                    for item in kept_updates
                )
                else "pass"
            )
            decision["decision"] = effective_decision
            decision["reason"] = (
                "local complete-bed policy retained the physical headboard"
            )
            rewritten_evidence_decisions.append(
                {
                    "source_object_id": source_id,
                    "previous_decision": "split_instances",
                    "effective_decision": effective_decision,
                    "previous_reason": previous_reason,
                    "effective_reason": decision["reason"],
                }
            )
        elif missing_supporting:
            binding_errors.append(
                {
                    "type": "assigned_evidence_references_suppressed_addition",
                    "evidence_id": str(decision.get("evidence_id") or ""),
                    "supporting_addition_ids": supporting,
                }
            )
        decision["supporting_addition_ids"] = remapped
    audit = {
        "schema_version": "scenemosaic_post_pass_b_policy_guard",
        "status": "pass" if not binding_errors else "blocked",
        "partition_count": len(partitions),
        "partitions": partitions,
        "raw_update_count": len(actions.get("updates") or []),
        "effective_update_count": len(kept_updates),
        "suppressed_updates": suppressed_updates,
        "failed_mask_semantic_rejections": failed_mask_semantic_rejections,
        "rewritten_evidence_decisions": rewritten_evidence_decisions,
        "raw_removal_count": len(actions.get("removals") or []),
        "effective_removal_count": len(kept_removals),
        "suppressed_removals": suppressed_removals,
        "raw_addition_count": len(actions.get("additions") or []),
        "effective_addition_count": len(kept_additions),
        "addition_id_remap": addition_id_remap,
        "suppressed_additions": suppressed_additions,
        "hard_errors": binding_errors,
        "formal_outputs_modified": False,
    }
    return {"actions": effective, "audit": audit}
