from __future__ import annotations

import copy
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Iterable, Mapping


class RelationGraphError(ValueError):
    """Raised when published relations do not have deterministic ownership."""


def _is_wall(anchor_id: str) -> bool:
    return anchor_id.startswith("wall_")


@dataclass(frozen=True)
class RelationGraphView:

    full_relations: tuple[dict[str, Any], ...]
    primary_relation_by_target: Mapping[str, dict[str, Any]]
    primary_parent_by_target: Mapping[str, str]
    primary_children_by_anchor: Mapping[str, tuple[str, ...]]
    secondary_wall_constraints_by_target: Mapping[
        str, tuple[dict[str, Any], ...]
    ]

    @classmethod
    def from_relations(
        cls, relations: Iterable[dict[str, Any]]
    ) -> "RelationGraphView":
        original: list[dict[str, Any]] = []
        by_target: dict[str, list[dict[str, Any]]] = {}
        seen: set[tuple[str, str, str]] = set()

        for index, raw_relation in enumerate(relations):
            if not isinstance(raw_relation, dict):
                raise RelationGraphError(f"relation {index} must be an object")
            relation = str(raw_relation.get("relation") or "")
            if relation not in {"attach", "contain"}:
                raise RelationGraphError(
                    f"relation {index} has unsupported type: {relation or 'missing'}"
                )
            parameters = raw_relation.get("parameters")
            if not isinstance(parameters, dict):
                raise RelationGraphError(
                    f"relation {index}.parameters must be an object"
                )
            anchor = str(parameters.get("object_anchor") or "").strip()
            target = str(parameters.get("object_target") or "").strip()
            if not anchor or not target:
                raise RelationGraphError(
                    f"relation {index} must define object_anchor and object_target"
                )
            if anchor == target:
                raise RelationGraphError(
                    f"relation target cannot own itself: {target}"
                )
            if _is_wall(anchor) and relation != "attach":
                raise RelationGraphError(
                    f"wall constraint must use attach: {anchor} -> {target}"
                )
            triple = (relation, anchor, target)
            if triple in seen:
                raise RelationGraphError(f"duplicate relation: {triple}")
            seen.add(triple)

            # Keep the publication payload unchanged at the JSON-value level.
            # A normalized copy is used only for deterministic ownership.
            original_item = copy.deepcopy(raw_relation)
            original.append(original_item)
            normalized_item = copy.deepcopy(raw_relation)
            normalized_item["relation"] = relation
            normalized_parameters = copy.deepcopy(parameters)
            normalized_parameters["object_anchor"] = anchor
            normalized_parameters["object_target"] = target
            normalized_item["parameters"] = normalized_parameters
            by_target.setdefault(target, []).append(normalized_item)

        primary_relation_by_target: dict[str, dict[str, Any]] = {}
        primary_parent_by_target: dict[str, str] = {}
        primary_children: dict[str, list[str]] = {}
        secondary_walls: dict[str, tuple[dict[str, Any], ...]] = {}

        for target in sorted(by_target):
            target_relations = by_target[target]
            structural = [
                relation
                for relation in target_relations
                if not _is_wall(
                    str(relation["parameters"]["object_anchor"])
                )
            ]
            wall_attaches = [
                relation
                for relation in target_relations
                if _is_wall(str(relation["parameters"]["object_anchor"]))
            ]

            if len(structural) > 1:
                candidates = [
                    (
                        relation["relation"],
                        relation["parameters"]["object_anchor"],
                    )
                    for relation in structural
                ]
                raise RelationGraphError(
                    f"target {target} has competing non-wall owners: {candidates}"
                )

            if structural:
                primary = structural[0]
                if wall_attaches:
                    secondary_walls[target] = tuple(
                        copy.deepcopy(item)
                        for item in sorted(
                            wall_attaches,
                            key=lambda item: (
                                item["parameters"]["object_anchor"],
                                item["relation"],
                            ),
                        )
                    )
            elif wall_attaches:
                if len(wall_attaches) > 1:
                    candidates = [
                        relation["parameters"]["object_anchor"]
                        for relation in wall_attaches
                    ]
                    raise RelationGraphError(
                        f"target {target} has competing wall owners: {candidates}"
                    )
                primary = wall_attaches[0]
            else:
                raise RelationGraphError(
                    f"target {target} has no consumable ownership relation"
                )

            primary_copy = copy.deepcopy(primary)
            anchor = str(primary_copy["parameters"]["object_anchor"])
            primary_relation_by_target[target] = primary_copy
            primary_parent_by_target[target] = anchor
            primary_children.setdefault(anchor, []).append(target)

        return cls(
            full_relations=tuple(original),
            primary_relation_by_target=MappingProxyType(
                primary_relation_by_target
            ),
            primary_parent_by_target=MappingProxyType(
                primary_parent_by_target
            ),
            primary_children_by_anchor=MappingProxyType(
                {
                    anchor: tuple(sorted(targets))
                    for anchor, targets in sorted(primary_children.items())
                }
            ),
            secondary_wall_constraints_by_target=MappingProxyType(
                secondary_walls
            ),
        )

    def object_groups(self) -> dict[str, dict[str, Any]]:
        """Return primary-only groups accepted by the existing orchestrator."""

        groups: dict[str, dict[str, Any]] = {}
        for anchor, targets in self.primary_children_by_anchor.items():
            relation_types = {
                self.primary_relation_by_target[target]["relation"]
                for target in targets
            }
            relation_type = (
                next(iter(relation_types))
                if len(relation_types) == 1
                else "mixed"
            )
            groups[anchor] = {
                "relation_type": relation_type,
                "object_ids": list(targets),
            }
            secondary_constraints = [
                copy.deepcopy(relation)
                for target in targets
                for relation in self.secondary_wall_constraints_by_target.get(
                    target, ()
                )
            ]
            if secondary_constraints:
                groups[anchor]["secondary_wall_constraints"] = (
                    secondary_constraints
                )
        return groups

    def primary_parent(self, target: str) -> str | None:
        return self.primary_parent_by_target.get(target)

    def full_relations_list(self) -> list[dict[str, Any]]:
        return copy.deepcopy(list(self.full_relations))

    def filtered_to_object_ids(
        self,
        object_ids: Iterable[str],
    ) -> "RelationGraphView":
        """Restrict the graph to objects that survived physics filtering."""

        allowed = {str(object_id) for object_id in object_ids}
        filtered: list[dict[str, Any]] = []
        for relation in self.full_relations:
            parameters = relation.get("parameters", {})
            anchor = str(parameters.get("object_anchor") or "")
            target = str(parameters.get("object_target") or "")
            if target not in allowed:
                continue
            if anchor.startswith("object_") and anchor not in allowed:
                continue
            filtered.append(copy.deepcopy(relation))
        filtered_view = type(self).from_relations(filtered)
        missing = sorted(allowed - set(filtered_view.primary_parent_by_target))
        if missing:
            raise RelationGraphError(
                "surviving objects have no primary ownership after filtering: "
                f"{missing}"
            )
        return filtered_view

    def audit_summary(self) -> dict[str, Any]:
        return {
            "full_relation_count": len(self.full_relations),
            "primary_ownership_count": len(self.primary_parent_by_target),
            "primary_parent_by_target": dict(self.primary_parent_by_target),
            "primary_children_by_anchor": {
                anchor: list(targets)
                for anchor, targets in self.primary_children_by_anchor.items()
            },
            "secondary_wall_constraints_by_target": {
                target: copy.deepcopy(list(relations))
                for target, relations
                in self.secondary_wall_constraints_by_target.items()
            },
        }
