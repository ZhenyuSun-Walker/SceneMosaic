from __future__ import annotations

import copy
from collections import defaultdict
from typing import Any, Mapping, Sequence

from pipeline.core.pass_b_policy import validate_b_actions
from segmentation import runtime as SAM3_SESSION

def _prompt_groups(
    objects: Sequence[Mapping[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    """Mirror the production grouping without importing the GPU runtime."""

    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for raw_object in objects:
        obj = dict(raw_object)
        prompt = str(obj.get("sam3_prompt") or "").strip()
        if not prompt:
            prompt = str(obj.get("category") or "").strip().replace("_", " ")
        groups[prompt or "object"].append(obj)
    return dict(groups)


def _object_prompt_map(
    objects: Sequence[Mapping[str, Any]],
) -> dict[str, str]:
    result: dict[str, str] = {}
    for prompt, grouped_objects in _prompt_groups(objects).items():
        for obj in grouped_objects:
            result[str(obj["object_id"])] = prompt
    return result


def validate_prompt_group_parity(
    objects: Sequence[Mapping[str, Any]],
) -> None:
    """Ensure orchestration and the reusable SAM3 session group identically."""

    local = {
        prompt: [str(obj["object_id"]) for obj in grouped]
        for prompt, grouped in _prompt_groups(objects).items()
    }
    production = {
        prompt: [str(obj["object_id"]) for obj in grouped]
        for prompt, grouped in SAM3_SESSION.prompt_groups(objects).items()
    }
    if local != production:
        raise RuntimeError(
            f"SAM3 prompt grouping parity failure: local={local}, production={production}"
        )


def compute_affected_prompt_audit(
    pass_a_objects: Sequence[Mapping[str, Any]],
    pass_b_actions: Mapping[str, Any],
    fused_objects: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Determine complete old/new prompt groups invalidated by Pass B."""

    validate_b_actions(pass_a_objects, pass_b_actions)
    old_prompt_by_id = _object_prompt_map(pass_a_objects)
    new_prompt_by_id = _object_prompt_map(fused_objects)
    final_by_a: dict[str, Mapping[str, Any]] = {}
    additions: list[Mapping[str, Any]] = []
    for obj in fused_objects:
        sources = list(obj.get("proposal_sources") or [])
        for source in sources:
            if source.get("pass") == "A":
                final_by_a[str(source["source_id"])] = obj
            if (
                source.get("pass") == "B"
                and source.get("action") == "added"
            ):
                additions.append(obj)

    updated_ids = {
        str(item["pass_a_id"]) for item in pass_b_actions["updates"]
    }
    removed_ids = {str(item) for item in pass_b_actions["removals"]}
    old_prompts: set[str] = set()
    new_prompts: set[str] = set()
    reasons: list[dict[str, Any]] = []

    for old_obj in pass_a_objects:
        object_id = str(old_obj["object_id"])
        old_prompt = old_prompt_by_id[object_id]
        if object_id in removed_ids:
            old_prompts.add(old_prompt)
            reasons.append(
                {
                    "action": "removed",
                    "source_id": object_id,
                    "old_prompt": old_prompt,
                }
            )
            continue
        final_obj = final_by_a.get(object_id)
        if final_obj is None:
            raise ValueError(f"surviving A object lacks final identity: {object_id}")
        final_id = str(final_obj["object_id"])
        new_prompt = new_prompt_by_id[final_id]
        changed = object_id in updated_ids or old_prompt != new_prompt
        if changed:
            old_prompts.add(old_prompt)
            new_prompts.add(new_prompt)
            reasons.append(
                {
                    "action": (
                        "updated" if object_id in updated_ids else "regrouped"
                    ),
                    "source_id": object_id,
                    "proposal_id": final_id,
                    "old_prompt": old_prompt,
                    "new_prompt": new_prompt,
                }
            )

    for final_obj in additions:
        final_id = str(final_obj["object_id"])
        new_prompt = new_prompt_by_id[final_id]
        binding = final_obj.get("local_candidate_binding") or {}
        if binding.get("operation") == "replace":
            reasons.append(
                {
                    "action": "added_with_reviewed_local_mask",
                    "proposal_id": final_id,
                    "new_prompt": new_prompt,
                    "candidate_ids": [
                        item.get("candidate_id")
                        for item in binding.get("candidates") or []
                    ],
                }
            )
            continue
        new_prompts.add(new_prompt)
        sources = [
            source
            for source in final_obj.get("proposal_sources") or []
            if source.get("pass") == "B"
            and source.get("action") == "added"
        ]
        reasons.append(
            {
                "action": "added",
                "source_id": sources[0]["source_id"] if sources else None,
                "proposal_id": final_id,
                "new_prompt": new_prompt,
            }
        )

    # If an old group was invalidated and surviving members retain the same
    # prompt, that final prompt must also be recomputed after a removal.
    for object_id, final_obj in final_by_a.items():
        old_prompt = old_prompt_by_id[object_id]
        if old_prompt in old_prompts:
            new_prompts.add(new_prompt_by_id[str(final_obj["object_id"])])

    return {
        "old_prompts": sorted(old_prompts),
        "new_prompts": sorted(new_prompts),
        "reasons": reasons,
    }


def make_final_identity(
    fused_objects: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Assign continuous final IDs after any deterministic removal stage.

    Intermediate removals (for example a rejected carrier-boundary fragment)
    naturally create gaps. The final published inventory must nevertheless use
    continuous IDs, so retain the existing visual order and renumber only at
    this identity boundary. The manifest records every old-to-new mapping for
    auditability; duplicate source identities remain a hard error.
    """
    objects = [copy.deepcopy(dict(item)) for item in fused_objects]
    prior_ids = [str(item.get("object_id") or "") for item in objects]
    if len(prior_ids) != len(set(prior_ids)):
        raise ValueError(f"duplicate intermediate proposal IDs: {prior_ids}")
    renumbered = []
    for index, obj in enumerate(objects, start=1):
        previous_id = str(obj.get("object_id") or "")
        if not previous_id:
            raise ValueError("proposal is missing an intermediate object_id")
        proposal_id = f"proposal_{index:03d}"
        obj["object_id"] = proposal_id
        if previous_id != proposal_id:
            renumbered.append({
                "previous_proposal_id": previous_id,
                "proposal_id": proposal_id,
            })
    records = []
    source_to_proposal: dict[str, str] = {}
    for obj, previous_id in zip(objects, prior_ids):
        sources = list(obj.get("proposal_sources") or [])
        if not sources:
            raise ValueError(f"proposal has no source identity: {obj['object_id']}")
        for source in sources:
            source_key = f"{source.get('pass')}:{source.get('source_id')}"
            if source_key in source_to_proposal:
                raise ValueError(f"source identity reused: {source_key}")
            source_to_proposal[source_key] = obj["object_id"]
        record = {
            "proposal_id": obj["object_id"],
            "category": obj.get("category"),
            "sources": sources,
        }
        if previous_id != obj["object_id"]:
            record["previous_proposal_id"] = previous_id
        records.append(record)
    return objects, {
        "schema_version": "scenemosaic_attempt_identity",
        "object_id_format": "proposal_{index:03d}",
        "records": records,
        "source_to_proposal": source_to_proposal,
        "renumbered": renumbered,
    }
