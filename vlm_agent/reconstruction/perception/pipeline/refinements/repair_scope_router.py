#!/usr/bin/env python3
"""Conservative semantic routing for segmentation refinement actions."""

from __future__ import annotations

import re
from typing import Any, Mapping, Sequence


OPEN_STORAGE_ROLE = "open_storage_interior"
SUPPORT_SURFACE_ROLE = "support_surface"
ASSEMBLY_ROLE = "assembly_or_fixture"
ORDINARY_ROLE = "ordinary_object"
AMBIGUOUS_ROLE = "ambiguous"

OPEN_STORAGE_TOKENS = {
    "bookcase", "bookshelf", "cubby", "hutch", "rack", "shelf", "shelves",
    "shelving",
}
SUPPORT_SURFACE_TOKENS = {
    "console", "counter", "countertop", "desk", "dresser", "island",
    "nightstand", "sideboard", "table", "vanity", "workbench",
}
OPEN_QUALIFIERS = {"display", "open", "rack", "shelf", "shelves", "shelving", "tier", "tiers"}
CART_STORAGE_QUALIFIERS = OPEN_QUALIFIERS | {"storage"}
ASSEMBLY_TOKENS = {
    "bathtub", "bed", "faucet", "lamp", "light", "machine", "sink", "vanity",
}
DEDICATED_ASSEMBLY_TOKEN_SETS = (
    {"coat", "rack"},
    {"towel", "rack"},
    {"towel", "ladder"},
    {"drying", "rack"},
    {"fabric", "roll", "rack"},
    {"roll", "holder"},
    {"pegboard", "rack"},
)
NON_HOST_TYPES = {"appliance", "lighting", "opening", "plant", "small_object", "textile"}


def _tokens(value: Any) -> list[str]:
    return re.findall(r"[a-z0-9]+", str(value or "").lower().replace("_", " "))


def _fields(obj: Mapping[str, Any]) -> list[list[str]]:
    return [
        _tokens(obj.get(key))
        for key in ("category", "sam3_prompt", "instance_name")
    ]


def _field_vote(tokens: Sequence[str]) -> str | None:
    token_set = set(tokens)
    if token_set & SUPPORT_SURFACE_TOKENS:
        return SUPPORT_SURFACE_ROLE
    # A storage/tier/shelf cart is a small mobile open carrier. A plain
    # utility/serving cart remains a support surface.
    if "cart" in token_set:
        return (
            OPEN_STORAGE_ROLE
            if token_set & CART_STORAGE_QUALIFIERS
            else SUPPORT_SURFACE_ROLE
        )
    if token_set & OPEN_STORAGE_TOKENS:
        return OPEN_STORAGE_ROLE
    if "cabinet" in token_set:
        return (
            OPEN_STORAGE_ROLE
            if token_set & {"display", "open", "shelf", "shelves", "shelving"}
            else SUPPORT_SURFACE_ROLE
        )
    return None


def classify_object_role(obj: Mapping[str, Any]) -> dict[str, Any]:
    """Return a fail-closed role audit for one existing object."""

    fields = _fields(obj)
    votes = [_field_vote(tokens) for tokens in fields]
    object_type = "_".join(_tokens(obj.get("object_type"))) or "unknown"
    category_tokens = set(fields[0])
    all_tokens = set().union(*(set(tokens) for tokens in fields))
    dedicated_assembly = any(
        required.issubset(all_tokens)
        for required in DEDICATED_ASSEMBLY_TOKEN_SETS
    )
    open_votes = votes.count(OPEN_STORAGE_ROLE)
    support_votes = votes.count(SUPPORT_SURFACE_ROLE)

    if object_type in NON_HOST_TYPES:
        role = ORDINARY_ROLE
        confidence = 0.98
        reason = "semantic_type_is_not_a_furniture_host"
    elif dedicated_assembly:
        role = ASSEMBLY_ROLE
        confidence = 0.96
        reason = "dedicated_holder_and_attached_parts_form_one_assembly"
    elif "cart" in category_tokens and category_tokens & CART_STORAGE_QUALIFIERS:
        role = OPEN_STORAGE_ROLE
        confidence = 0.96
        reason = "explicit_storage_cart_category"
    elif category_tokens & SUPPORT_SURFACE_TOKENS:
        role = SUPPORT_SURFACE_ROLE
        confidence = 0.98 if support_votes >= 2 else 0.90
        reason = "support_surface_category_has_priority_over_modifier_text"
    elif category_tokens & OPEN_STORAGE_TOKENS:
        role = OPEN_STORAGE_ROLE
        confidence = 0.98 if open_votes >= 2 else 0.90
        reason = "explicit_open_storage_category"
    elif open_votes >= 2 and support_votes == 0:
        role = OPEN_STORAGE_ROLE
        confidence = 0.92
        reason = "two_semantic_fields_agree_on_open_storage"
    elif support_votes >= 2 and open_votes == 0:
        role = SUPPORT_SURFACE_ROLE
        confidence = 0.92
        reason = "two_semantic_fields_agree_on_support_surface"
    elif open_votes and support_votes:
        role = AMBIGUOUS_ROLE
        confidence = 0.50
        reason = "open_storage_and_support_surface_fields_conflict"
    elif open_votes == 1 or support_votes == 1:
        role = AMBIGUOUS_ROLE
        confidence = 0.45
        reason = "only_one_semantic_field_supports_a_host_role"
    elif category_tokens & ASSEMBLY_TOKENS:
        role = ASSEMBLY_ROLE
        confidence = 0.80
        reason = "category_requires_assembly_completeness_policy"
    else:
        role = ORDINARY_ROLE
        confidence = 0.80
        reason = "no_supported_host_semantics"

    host_mode = {
        OPEN_STORAGE_ROLE: "open_storage",
        SUPPORT_SURFACE_ROLE: "support_surface",
    }.get(role)
    return {
        "object_role": role,
        "host_mode": host_mode,
        "confidence": round(confidence, 6),
        "reason": reason,
        "field_votes": votes,
        "open_storage_votes": open_votes,
        "support_surface_votes": support_votes,
        "object_type": object_type,
        "host_minus_contents_allowed": role == OPEN_STORAGE_ROLE,
        "support_surface_review_allowed": role == SUPPORT_SURFACE_ROLE,
        "semantic_verification_required": role == AMBIGUOUS_ROLE,
    }
