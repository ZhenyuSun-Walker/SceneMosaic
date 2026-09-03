import json
from pathlib import Path
import numpy as np

from .layout_combination import _layout_payload_to_matrix, _matrix_to_layout_payload


def build_anchor_dependency_graph(object_groups: dict) -> tuple[dict[str, set[str]], dict[str, set[str]], dict[str, int]]:
    parents_by_child: dict[str, set[str]] = {}
    children_by_parent: dict[str, set[str]] = {}
    indegree: dict[str, int] = {}

    for anchor_id, group_info in object_groups.items():
        if isinstance(group_info, dict):
            object_ids = group_info.get("object_ids", [])
        else:
            object_ids = group_info if isinstance(group_info, list) else []
        indegree.setdefault(anchor_id, 0)
        for object_id in object_ids:
            parents_by_child.setdefault(object_id, set()).add(anchor_id)
            children_by_parent.setdefault(anchor_id, set()).add(object_id)
            indegree.setdefault(object_id, 0)
            indegree[object_id] += 1

    return parents_by_child, children_by_parent, indegree

def collect_descendant_subtree(
    root_id: str,
    children_by_parent: dict[str, set[str]],
) -> list[str]:
    subtree_nodes: list[str] = []
    stack = [root_id]
    visited: set[str] = set()

    while stack:
        node_id = stack.pop()
        if node_id in visited:
            continue
        visited.add(node_id)
        subtree_nodes.append(node_id)
        for child_id in sorted(children_by_parent.get(node_id, []), reverse=True):
            if child_id not in visited:
                stack.append(child_id)

    return subtree_nodes

def iter_subtree_roots(
    anchor_id: str,
    children_by_parent: dict[str, set[str]],
) -> list[str]:
    subtree_roots: list[str] = []
    for child_id in sorted(children_by_parent.get(anchor_id, [])):
        subtree_nodes = collect_descendant_subtree(child_id, children_by_parent)
        if len(subtree_nodes) > 1:
            subtree_roots.append(child_id)
    return subtree_roots

def load_layout_snapshot(layout_dir: Path) -> dict[str, dict]:
    layouts: dict[str, dict] = {}
    for layout_file in layout_dir.glob("*.json"):
        try:
            with layout_file.open("r", encoding="utf-8") as f:
                payload = json.load(f)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            layouts[layout_file.stem] = payload
    return layouts

def _write_layout_payload(layout_dir: Path, object_id: str, payload: dict) -> None:
    layout_path = layout_dir / f"{object_id}.json"
    with layout_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

def propagate_subtree_transform(
    self,
    pending_layout_dir: Path,
    children_by_parent: dict[str, set[str]],
    anchor_id: str,
    subtree_root_id: str,
    before_layouts: dict[str, dict],
    after_layouts: dict[str, dict],
) -> None:
    subtree_nodes = collect_descendant_subtree(subtree_root_id, children_by_parent)
    if len(subtree_nodes) <= 1:
        return

    before_root_payload = before_layouts.get(subtree_root_id)
    after_root_payload = after_layouts.get(subtree_root_id)
    if before_root_payload is None or after_root_payload is None:
        return

    try:
        before_root_matrix = _layout_payload_to_matrix(before_root_payload)
        after_root_matrix = _layout_payload_to_matrix(after_root_payload)
        delta_matrix = after_root_matrix @ np.linalg.inv(before_root_matrix)
    except np.linalg.LinAlgError:
        return

    print(f"Propagating anchor '{anchor_id}' through subtree rooted at '{subtree_root_id}'")

    for node_id in subtree_nodes:
        if node_id == subtree_root_id:
            continue
        original_payload = before_layouts.get(node_id)
        if original_payload is None:
            continue
        composed_matrix = delta_matrix @ _layout_payload_to_matrix(original_payload)
        updated_payload = _matrix_to_layout_payload(composed_matrix, original_payload)
        after_layouts[node_id] = updated_payload
        _write_layout_payload(pending_layout_dir, node_id, updated_payload)
