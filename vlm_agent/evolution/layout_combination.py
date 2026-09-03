from __future__ import annotations

import copy
import itertools
import json
import math
import random
import re
import shutil
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .paths import ProjectPaths
from .scene_generator import SceneGenerator
from .tool_runner import ToolRunner
from .types import SceneArtifacts


def _sanitize_path_component(value: str) -> str:
    sanitized = re.sub(r"[^A-Za-z0-9_.=-]+", "_", value).strip("_")
    return sanitized or "item"


def _flatten_singleton_list(value: object) -> object:
    if isinstance(value, list) and len(value) == 1 and isinstance(value[0], list):
        return value[0]
    return value


def _parse_vector(value: object, *, expected_length: int) -> list[float]:
    flat_value = _flatten_singleton_list(value)
    array = np.asarray(flat_value, dtype=np.float32).reshape(-1)
    if array.size != expected_length:
        raise ValueError(f"expected {expected_length} values, got {array.size}")
    return [float(item) for item in array.tolist()]


def _parse_scale(value: object) -> list[float]:
    flat_value = _flatten_singleton_list(value)
    array = np.asarray(flat_value, dtype=np.float32).reshape(-1)
    if array.size == 1:
        return [float(array[0])] * 3
    if array.size == 3:
        return [float(item) for item in array.tolist()]
    raise ValueError(f"scale must have 1 or 3 values, got {array.size}")


def _quaternion_to_matrix_numpy(quaternion: object) -> np.ndarray:
    q = np.asarray(_parse_vector(quaternion, expected_length=4), dtype=np.float32)
    norm = float(np.linalg.norm(q))
    if norm <= 1e-8:
        raise ValueError("quaternion norm is zero")
    w, x, y, z = q / norm
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float32,
    )


def _matrix_to_quaternion(matrix: np.ndarray) -> list[float]:
    m = np.asarray(matrix, dtype=np.float32).reshape(3, 3)
    trace = float(m[0, 0] + m[1, 1] + m[2, 2])
    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        w = 0.25 * s
        x = (m[2, 1] - m[1, 2]) / s
        y = (m[0, 2] - m[2, 0]) / s
        z = (m[1, 0] - m[0, 1]) / s
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = math.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2.0
        w = (m[2, 1] - m[1, 2]) / s
        x = 0.25 * s
        y = (m[0, 1] + m[1, 0]) / s
        z = (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = math.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2.0
        w = (m[0, 2] - m[2, 0]) / s
        x = (m[0, 1] + m[1, 0]) / s
        y = 0.25 * s
        z = (m[1, 2] + m[2, 1]) / s
    else:
        s = math.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2.0
        w = (m[1, 0] - m[0, 1]) / s
        x = (m[0, 2] + m[2, 0]) / s
        y = (m[1, 2] + m[2, 1]) / s
        z = 0.25 * s
    return [float(w), float(x), float(y), float(z)]


def _layout_payload_to_matrix(layout: dict) -> np.ndarray:
    translation = np.asarray(
        _parse_vector(layout.get("translation", [0.0, 0.0, 0.0]), expected_length=3),
        dtype=np.float32,
    )
    rotation = _quaternion_to_matrix_numpy(layout.get("rotation", [1.0, 0.0, 0.0, 0.0]))
    scale = np.asarray(_parse_scale(layout.get("scale", [1.0, 1.0, 1.0])), dtype=np.float32)

    matrix = np.eye(4, dtype=np.float32)
    matrix[:3, :3] = rotation @ np.diag(scale)
    matrix[:3, 3] = translation
    return matrix


def _matrix_to_layout_payload(matrix: np.ndarray, template: dict | None = None) -> dict:
    affine = np.asarray(matrix, dtype=np.float32).reshape(4, 4)
    linear = affine[:3, :3]
    translation = affine[:3, 3]

    scale = np.linalg.norm(linear, axis=0)
    safe_scale = np.where(scale < 1e-8, 1.0, scale)
    rotation = linear @ np.diag(1.0 / safe_scale)

    u, _, vh = np.linalg.svd(rotation)
    rotation = u @ vh
    if np.linalg.det(rotation) < 0:
        u[:, -1] *= -1
        rotation = u @ vh

    payload = {
        "translation": [[float(translation[0]), float(translation[1]), float(translation[2])]],
        "rotation": [_matrix_to_quaternion(rotation)],
        "scale": [[float(scale[0]), float(scale[1]), float(scale[2])]],
    }
    if template:
        for key, value in template.items():
            if key not in payload:
                payload[key] = copy.deepcopy(value)
    return payload


@dataclass(frozen=True)
class LayoutVariantCandidate:
    anchor_id: str
    group_dir: Path
    variant_dir: Path
    layout_id: str
    layout_dir: Path
    mapping_path: Path

    @property
    def label(self) -> str:
        return f"{_sanitize_path_component(self.anchor_id)}={_sanitize_path_component(self.layout_id)}"


def discover_layout_variants(generated_root: Path) -> dict[str, list[LayoutVariantCandidate]]:
    grouped_variants: dict[str, list[LayoutVariantCandidate]] = {}
    if not generated_root.exists():
        return grouped_variants

    for group_dir in sorted(p for p in generated_root.glob("group_*") if p.is_dir()):
        anchor_id = group_dir.name.removeprefix("group_")
        candidates: list[LayoutVariantCandidate] = []

        for variant_dir in sorted(p for p in group_dir.glob("layout_*") if p.is_dir()):
            layout_id = variant_dir.name
            simulated_layout_dir = variant_dir / "layout_params_simulated"
            layout_params_dir = variant_dir / "layout_params"
            layout_dir = simulated_layout_dir if simulated_layout_dir.exists() else layout_params_dir
            if not layout_dir.exists():
                continue

            if layout_id.endswith("_org"):
                # For original layouts
                mapping_path = group_dir / f"grouped_scene_{anchor_id}.json"
                if not mapping_path.exists():
                    continue
            else:
                mapping_path = variant_dir / f"{layout_id}_render.json"
                if not mapping_path.exists():
                    continue

            candidates.append(
                LayoutVariantCandidate(
                    anchor_id=anchor_id,
                    group_dir=group_dir,
                    variant_dir=variant_dir,
                    layout_id=layout_id,
                    layout_dir=layout_dir,
                    mapping_path=mapping_path,
                )
            )

        if candidates:
            grouped_variants[anchor_id] = candidates

    return grouped_variants


def build_combo_name(variants: tuple[LayoutVariantCandidate, ...] | list[LayoutVariantCandidate]) -> str:
    parts: list[str] = []
    for variant in variants:
        anchor = _sanitize_path_component(variant.anchor_id)
        layout = _sanitize_path_component(variant.layout_id).removeprefix("layout_")
        anchor = anchor.replace("_", "")
        if layout.isdigit():
            layout = layout.zfill(3)
        parts.append(f"{anchor}@{layout}")
    return "__".join(parts)


class LayoutCombination:
    def __init__(self, paths: ProjectPaths, tool_runner: ToolRunner):
        self.paths = paths
        self.tool_runner = tool_runner

    @staticmethod
    def _load_json(path: Path) -> dict:
        with path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        return payload if isinstance(payload, dict) else {}

    def _load_relationships_and_objects(self) -> tuple[list[dict], list[dict], list[dict]]:
        relations_path = self.paths.outputs_dir / "relations.json"
        objects_path = self.paths.outputs_dir / "objects.json"

        relationships: list[dict] = []
        if relations_path.exists():
            with relations_path.open("r", encoding="utf-8") as f:
                payload = json.load(f)
            if isinstance(payload, list):
                relationships = [item for item in payload if isinstance(item, dict)]

        objects: list[dict] = []
        walls: list[dict] = []
        if objects_path.exists():
            with objects_path.open("r", encoding="utf-8") as f:
                payload = json.load(f)
            if isinstance(payload, dict):
                raw_objects = payload.get("objects", [])
                raw_walls = payload.get("walls", [])
                if isinstance(raw_objects, list):
                    objects = [item for item in raw_objects if isinstance(item, dict)]
                if isinstance(raw_walls, list):
                    walls = [item for item in raw_walls if isinstance(item, dict)]

        return relationships, objects, walls

    @staticmethod
    def _has_out_of_bounds(summary: dict | None) -> bool:
        if not isinstance(summary, dict):
            return False
        entries = summary.get("out_of_bounds_objects", [])
        return isinstance(entries, list) and bool(entries)

    @staticmethod
    def _is_root_anchor(anchor_id: str) -> bool:
        return anchor_id == "floor" or anchor_id == "ceiling" or anchor_id.startswith("wall")

    @staticmethod
    def _root_sort_key(anchor_id: str) -> tuple[int, str]:
        if anchor_id == "floor":
            return (0, "")
        if anchor_id == "ceiling":
            return (1, "")
        if anchor_id.startswith("wall"):
            return (2, anchor_id)
        return (3, anchor_id)

    def _local_variant_is_valid(
        self,
        candidate: LayoutVariantCandidate,
        artifacts: SceneArtifacts,
        generator: SceneGenerator,
        validation_root: Path,
    ) -> bool:
        is_org = candidate.layout_id.endswith("_org")
        if artifacts.collision_hulls_dir is None:
            return is_org

        validation_root.mkdir(parents=True, exist_ok=True)
        report_path = validation_root / f"{candidate.label}_local_collision_report.json"
        collision_result = self.tool_runner.check_object_collisions(
            layout_dir=candidate.layout_dir,
            hull_dir=artifacts.collision_hulls_dir,
            output=report_path,
        )
        if collision_result.returncode != 0 or not report_path.exists():
            msg = f"Collision check failed for {candidate.variant_dir}"
            if not is_org:
                print(f"Discarding local variant {candidate.label}: {msg}")
                return False

        collision_report = self._load_json(report_path)
        collisions = collision_report.get("collisions", [])
        if isinstance(collisions, list) and collisions:
            msg = f"collisions detected ({len(collisions)})"
            if not is_org:
                print(f"Discarding local variant {candidate.label}: {msg}")
                return False

        out_of_bounds_summary = generator.summarize_out_of_bounds_objects(
            candidate.mapping_path,
            anchor_id=candidate.anchor_id,
        )
        if self._has_out_of_bounds(out_of_bounds_summary):
            msg = "out-of-bounds objects detected"
            if not is_org:
                print(f"Discarding local variant {candidate.label}: {msg}")
                return False

        return True

    @staticmethod
    def _load_relationship_graph(relations: list[dict]) -> tuple[dict[str, set[str]], dict[str, set[str]]]:
        parents_by_child: dict[str, set[str]] = {}
        children_by_parent: dict[str, set[str]] = {}
        for relation in relations:
            if not isinstance(relation, dict) or relation.get("relation") not in ["attach", "contain"]:
                continue
            parameters = relation.get("parameters", {})
            if not isinstance(parameters, dict):
                continue
            parent_id = parameters.get("object_anchor")
            child_id = parameters.get("object_target")
            if not parent_id or not child_id:
                continue
            parent_id = str(parent_id)
            child_id = str(child_id)
            parents_by_child.setdefault(child_id, set()).add(parent_id)
            children_by_parent.setdefault(parent_id, set()).add(child_id)
        return parents_by_child, children_by_parent

    def _choose_parent_by_child(self, parents_by_child: dict[str, set[str]]) -> dict[str, str]:
        chosen_parent_by_child: dict[str, str] = {}
        for child_id, parents in parents_by_child.items():
            if not parents:
                continue
            if "floor" in parents:
                chosen_parent_by_child[child_id] = "floor"
                continue
            chosen_parent_by_child[child_id] = random.choice(sorted(parents))
        return chosen_parent_by_child

    def _collect_root_nodes(
        self,
        grouped_variants: dict[str, list[LayoutVariantCandidate]],
        chosen_parent_by_child: dict[str, str],
    ) -> list[str]:
        root_nodes = [
            anchor_id
            for anchor_id in grouped_variants
            if self._is_root_anchor(anchor_id) or anchor_id not in chosen_parent_by_child
        ]
        root_nodes.sort(key=self._root_sort_key)
        if not root_nodes:
            return []
        root_nodes = list(dict.fromkeys(root_nodes))
        root_nodes.sort(key=self._root_sort_key)
        return root_nodes

    @staticmethod
    def _merge_layout_maps(base_layouts: dict[str, dict], extra_layouts: dict[str, dict]) -> dict[str, dict]:
        merged = copy.deepcopy(base_layouts)
        for object_id, payload in extra_layouts.items():
            existing = merged.get(object_id)
            if existing is not None and existing != payload:
                raise ValueError(f"conflicting layout for object '{object_id}'")
            merged[object_id] = payload
        return merged

    @staticmethod
    def _write_layout_tree(layouts: dict[str, dict], target_layout_dir: Path) -> None:
        target_layout_dir.mkdir(parents=True, exist_ok=True)
        for object_id in sorted(layouts.keys()):
            output_path = target_layout_dir / f"{object_id}.json"
            with output_path.open("w", encoding="utf-8") as f:
                json.dump(layouts[object_id], f, ensure_ascii=False, indent=2)

    def _fill_missing_root_layouts_from_simulated(self, combo_layout_dir: Path) -> None:
        """Fill missing wall/floor/ceiling layouts from outputs/layout_params_simulated."""
        simulated_layout_dir = self.paths.outputs_dir / "layout_params_simulated"
        if not simulated_layout_dir.exists():
            return

        # Check for missing wall_*, floor, ceiling files
        for pattern in ["wall_*.json", "floor.json", "ceiling.json"]:
            for simulated_file in simulated_layout_dir.glob(pattern):
                target_file = combo_layout_dir / simulated_file.name
                if not target_file.exists():
                    shutil.copy2(simulated_file, target_file)

    def _compose_variant_into_scene(
        self,
        scene_layouts: dict[str, dict],
        candidate: LayoutVariantCandidate,
    ) -> bool:
        local_anchor_path = candidate.layout_dir / f"{candidate.anchor_id}.json"
        if not local_anchor_path.exists():
            print(f"Skipping {candidate.label}: missing local anchor layout {local_anchor_path}")
            return False

        if self._is_root_anchor(candidate.anchor_id):
            for layout_file in sorted(candidate.layout_dir.glob("*.json")):
                scene_layouts[layout_file.stem] = self._load_json(layout_file)
            return True

        parent_payload = scene_layouts.get(candidate.anchor_id)
        if parent_payload is None:
            print(
                f"Skipping {candidate.label}: parent anchor '{candidate.anchor_id}' has not been materialized yet"
            )
            return False

        try:
            parent_matrix = _layout_payload_to_matrix(parent_payload)
            local_anchor_payload = self._load_json(local_anchor_path)
            local_anchor_matrix = _layout_payload_to_matrix(local_anchor_payload)
            transform_matrix = parent_matrix @ np.linalg.inv(local_anchor_matrix)
        except np.linalg.LinAlgError:
            print(f"Skipping {candidate.label}: anchor transform is not invertible")
            return False

        for layout_file in sorted(candidate.layout_dir.glob("*.json")):
            object_id = layout_file.stem
            if object_id == candidate.anchor_id:
                continue
            local_payload = self._load_json(layout_file)
            composed_matrix = transform_matrix @ _layout_payload_to_matrix(local_payload)
            scene_layouts[object_id] = _matrix_to_layout_payload(composed_matrix, local_payload)

        return True

    def _expand_anchor(
        self,
        anchor_id: str,
        base_scene_layouts: dict[str, dict],
        grouped_variants: dict[str, list[LayoutVariantCandidate]],
        children_by_parent: dict[str, list[str]],
    ) -> list[tuple[dict[str, dict], list[LayoutVariantCandidate]]]:
        candidates = grouped_variants.get(anchor_id, [])
        if not candidates:
            return []

        results: list[tuple[dict[str, dict], list[LayoutVariantCandidate]]] = []
        child_ids = sorted(children_by_parent.get(anchor_id, []))

        for candidate in candidates:
            composed_scene_layouts = copy.deepcopy(base_scene_layouts)
            if not self._compose_variant_into_scene(composed_scene_layouts, candidate):
                continue

            branch_variants = [candidate]
            if not child_ids:
                results.append((composed_scene_layouts, branch_variants))
                continue

            child_result_groups: list[list[tuple[dict[str, dict], list[LayoutVariantCandidate]]]] = []
            for child_id in child_ids:
                child_results = self._expand_anchor(
                    child_id,
                    composed_scene_layouts,
                    grouped_variants,
                    children_by_parent,
                )
                if child_results:
                    child_result_groups.append(child_results)

            if not child_result_groups:
                results.append((composed_scene_layouts, branch_variants))
                continue

            for child_combo in itertools.product(*child_result_groups):
                merged_layouts = copy.deepcopy(composed_scene_layouts)
                merged_variants = list(branch_variants)
                valid_combo = True
                for child_layouts, child_variants in child_combo:
                    try:
                        merged_layouts = self._merge_layout_maps(merged_layouts, child_layouts)
                    except ValueError:
                        valid_combo = False
                        break
                    merged_variants.extend(child_variants)
                if valid_combo:
                    results.append((merged_layouts, merged_variants))

        return results

    def _write_combo_manifest(
        self,
        output_path: Path,
        *,
        combo_name: str,
        variants: tuple[LayoutVariantCandidate, ...],
        layout_dir: Path,
        collision_report_path: Path,
        novelty_metrics: dict | None = None,
    ) -> None:
        payload = {
            "combo_name": combo_name,
            "selected_variants": [
                {
                    "anchor_id": variant.anchor_id,
                    "layout_id": variant.layout_id,
                    "group_dir": str(variant.group_dir),
                    "variant_dir": str(variant.variant_dir),
                    "layout_dir": str(variant.layout_dir),
                    "mapping_path": str(variant.mapping_path),
                }
                for variant in variants
            ],
            "layout_dir": str(layout_dir),
            "collision_report_path": str(collision_report_path),
        }
        if novelty_metrics is not None:
            payload["novelty_metrics"] = novelty_metrics
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

    @staticmethod
    def _is_org_variant(variant: LayoutVariantCandidate) -> bool:
        """Check if variant is the org variant."""
        return variant.layout_id.endswith("_org")

    def _compose_selected_combo(
        self,
        combo_dir: Path,
        combo_name: str,
        combo_layout_dir: Path,
        unposed_objects_dir: Path,
    ) -> Path | None:
        """Generate GLB file for a selected combo. Returns glb_path on success, None on failure."""
        glb_path = combo_dir / f"{combo_name}.glb"
        compose_result = self.tool_runner.compose_scene(
            combo_layout_dir,
            unposed_objects_dir,
            glb_path,
        )
        if compose_result.returncode != 0:
            print(f"Failed to compose GLB for combo {combo_name}.")
            shutil.rmtree(combo_dir, ignore_errors=True)
            return None
        print(f"Created combined scene GLB for combo {combo_name} at {glb_path}")
        return glb_path

    def _load_layout_objects(self, layout_dir: Path) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """Load all object positions, rotations (as matrices), and quaternions from layout_params directory."""
        objects = []
        for obj_file in sorted(layout_dir.glob("*.json")):
            layout_data = self._load_json(obj_file)
            translation = np.asarray(
                _parse_vector(layout_data.get("translation", [0.0, 0.0, 0.0]), expected_length=3),
                dtype=np.float32,
            )
            # Parse quaternion for rotation matrix
            rotation = _quaternion_to_matrix_numpy(layout_data.get("rotation", [1.0, 0.0, 0.0, 0.0]))
            # Also keep raw quaternion for novelty calculation
            quaternion = np.asarray(
                _parse_vector(layout_data.get("rotation", [1.0, 0.0, 0.0, 0.0]), expected_length=4),
                dtype=np.float32,
            )
            # Normalize quaternion
            quaternion = quaternion / np.linalg.norm(quaternion)
            objects.append((translation, rotation, quaternion))
        return objects

    def _compute_novelty_metrics(
        self,
        combo_layout_dir: Path,
        org_layout_dir: Path,
        wp: float = 0.4,
        wu: float = 0.4,
        wr: float = 0.2,
    ) -> dict:
        """
        Compute enhanced novelty metrics between a combo layout and the pure org layout.

        Returns a dict with:
        - D_pos_raw: raw directed position difference
        - D_dist_raw: raw undirected distance difference
        - D_rot_raw: raw rotation difference
        - D_pos_normalized: normalized directed position difference
        - D_dist_normalized: normalized undirected distance difference
        - D_rot_normalized: normalized rotation difference
        - novelty: final novelty score
        """
        # Load objects from both layouts
        combo_objects = self._load_layout_objects(combo_layout_dir)
        org_objects = self._load_layout_objects(org_layout_dir)

        # Match objects by file name
        combo_files = {f.stem for f in combo_layout_dir.glob("*.json")}
        org_files = {f.stem for f in org_layout_dir.glob("*.json")}
        common_files = sorted(combo_files & org_files)

        if not common_files:
            return {
                "D_pos_raw": 0.0,
                "D_rot_raw": 0.0,
                "D_pos_normalized": 0.0,
                "D_rot_normalized": 0.0,
                "novelty": 0.0,
                "error": "no common objects",
            }

        # Build object dictionaries
        combo_dict = {f.stem: i for i, f in enumerate(sorted(combo_layout_dir.glob("*.json")))}
        org_dict = {f.stem: i for i, f in enumerate(sorted(org_layout_dir.glob("*.json")))}

        # Part 1: Position and distance differences
        N = len(common_files)
        if N < 2:
            D_pos_raw = 0.0
            D_dist_raw = 0.0
            s_pos = 1.0
            s_dist = 1.0
        else:
            # Compute pairwise differences over all ordered pairs i != j
            # (in the local-coordinate enhanced version, (i,j) and (j,i) are NOT symmetric
            # because each displacement is projected into object i's own local frame)
            total_pos_diff = 0.0
            total_pos_scale = 0.0
            total_dist_diff = 0.0
            total_dist_scale = 0.0
            count = 0
            for i, obj_i in enumerate(common_files):
                for j, obj_j in enumerate(common_files):
                    if i == j:
                        continue
                    # Get positions
                    combo_i = combo_objects[combo_dict[obj_i]][0]
                    combo_j = combo_objects[combo_dict[obj_j]][0]
                    org_i = org_objects[org_dict[obj_i]][0]
                    org_j = org_objects[org_dict[obj_j]][0]

                    # Global displacement vectors
                    v_ij_org = org_j - org_i
                    v_ij_combo = combo_j - combo_i

                    # Local coordinate system (enhanced version)
                    # Project to object i's local frame
                    R_i_org = org_objects[org_dict[obj_i]][1]
                    v_ij_org_local = R_i_org.T @ v_ij_org

                    R_i_combo = combo_objects[combo_dict[obj_i]][1]
                    v_ij_combo_local = R_i_combo.T @ v_ij_combo

                    # Difference in local coordinates
                    pos_diff = np.linalg.norm(v_ij_combo_local - v_ij_org_local)
                    total_pos_diff += pos_diff ** 2

                    # Undirected distance difference
                    dist_diff = abs(np.linalg.norm(v_ij_combo) - np.linalg.norm(v_ij_org))
                    total_dist_diff += dist_diff ** 2

                    # Scale factor (original layout distances)
                    total_pos_scale += np.linalg.norm(v_ij_org) ** 2
                    total_dist_scale += np.linalg.norm(v_ij_org) ** 2
                    count += 1

            D_pos_raw = math.sqrt(total_pos_diff / count) if count > 0 else 0.0
            D_dist_raw = math.sqrt(total_dist_diff / count) if count > 0 else 0.0
            s_pos = math.sqrt(total_pos_scale / count) if count > 0 else 1.0
            s_dist = math.sqrt(total_dist_scale / count) if count > 0 else 1.0

        # Normalize position differences
        D_pos_normalized = D_pos_raw / s_pos if s_pos > 1e-8 else 0.0
        D_dist_normalized = D_dist_raw / s_dist if s_dist > 1e-8 else 0.0

        # Part 2: Rotation difference (using quaternion dot product as per spec)
        theta_sum = 0.0
        for obj_id in common_files:
            combo_idx = combo_dict[obj_id]
            org_idx = org_dict[obj_id]

            # Extract quaternions (index 2 in the tuple)
            q_combo = combo_objects[combo_idx][2]
            q_org = org_objects[org_idx][2]

            # Compute quaternion dot product: θ = 2 * arccos(|q_combo · q_org|)
            dot_product = abs(np.dot(q_combo, q_org))
            dot_product = np.clip(dot_product, 0.0, 1.0)
            theta = 2.0 * math.acos(dot_product)  # in [0, pi]
            theta_sum += theta

        D_rot_raw = theta_sum / N if N > 0 else 0.0
        D_rot_normalized = D_rot_raw / math.pi

        # Final novelty score
        novelty = wp * D_pos_normalized + wu * D_dist_normalized + wr * D_rot_normalized

        return {
            "D_pos_raw": float(D_pos_raw),
            "D_dist_raw": float(D_dist_raw),
            "D_rot_raw": float(D_rot_raw),
            "D_pos_normalized": float(D_pos_normalized),
            "D_dist_normalized": float(D_dist_normalized),
            "D_rot_normalized": float(D_rot_normalized),
            "novelty": float(novelty),
            "common_objects_count": N,
            "weights": {"wp": wp, "wu": wu, "wr": wr},
        }

    def _select_representative_scenes(
        self,
        pending_manifests: list[tuple[Path, dict]],
        *,
        org_layout_dir: Path,
        k: int = 6,
        alpha: float = 0.1,
        wp: float = 0.4,
        wu: float = 0.4,
        wr: float = 0.2,
    ) -> list[dict]:
        """
        Select diverse representative scenes from candidate combos.

        Algorithm:
        0. Seed — the pure-org combo is directly placed into representatives (rank 1).
        1. Compute novelty-to-org for every non-org candidate.
        2. Iterative Max-Min greedy selection with **dynamic per-round filtering**:
           In each round, candidates whose min-novelty to *any* already-selected
           representative is below the current round's ``max_min_distance * alpha`` are
           removed, then the candidate maximising the minimum distance to all
           representatives is picked.

        Pairwise novelty scores are cached so each pair of layout dirs is computed
        at most once.

        Returns the list of selected manifest dicts (with an added ``representative_rank`` key).
        """
        # ---- Pairwise novelty cache (layout_dir_pair -> score) -----------------
        _pair_cache: dict[frozenset, float] = {}

        def _pairwise_novelty(layout_a: Path, layout_b: Path) -> float:
            key = frozenset((str(layout_a), str(layout_b)))
            cached = _pair_cache.get(key)
            if cached is not None:
                return cached
            score = self._compute_novelty_metrics(layout_a, layout_b, wp=wp, wu=wu, wr=wr)["novelty"]
            _pair_cache[key] = score
            return score

        def _min_novelty_to_reps(
            layout: Path, reps: list[tuple[dict, float]]
        ) -> float:
            """Minimum novelty from *layout* to any representative."""
            return min(
                _pairwise_novelty(layout, rep_data["layout_dir"])
                for rep_data, _ in reps
            )

        # ---- Step 0: Seed — place pure-org combo as rank 1 ---------------------
        org_manifest_data: dict | None = None
        for _manifest_path, manifest_data in pending_manifests:
            if self._is_pure_org_variant(manifest_data["variants"]):
                org_manifest_data = manifest_data
                break

        representatives: list[tuple[dict, float]] = []
        if org_manifest_data is not None:
            org_manifest_data["representative_rank"] = 1
            representatives.append((org_manifest_data, 0.0))
            print(f"  Seed (rank 1): {org_manifest_data['combo_name']} [pure org baseline]")
        else:
            print("Warning: no pure org combo found; proceeding without org seed.")

        # ---- Step 1: Compute novelty-to-org for non-org candidates -------------
        candidates: list[tuple[dict, float]] = []  # (manifest_data, novelty_to_org)
        for _manifest_path, manifest_data in pending_manifests:
            if self._is_pure_org_variant(manifest_data["variants"]):
                continue
            score = self._compute_novelty_metrics(
                manifest_data["layout_dir"],
                org_layout_dir,
                wp=wp,
                wu=wu,
                wr=wr,
            )["novelty"]
            candidates.append((manifest_data, score))

        if not candidates:
            print("No non-org candidates available for scene selection.")
            return [data for data, _ in representatives]

        # ---- Step 2: Iterative Max-Min greedy with per-round filtering ---------
        rank = len(representatives) + 1  # next rank number

        while len(representatives) < k and candidates:
            # Pick the candidate that maximises min-distance to all representatives.
            best_candidate: tuple[dict, float] | None = None
            best_idx = -1
            max_min_distance = -1.0

            for idx, (cand_data, _cand_score) in enumerate(candidates):
                min_dist = _min_novelty_to_reps(cand_data["layout_dir"], representatives)
                if min_dist > max_min_distance:
                    max_min_distance = min_dist
                    best_candidate = (cand_data, _cand_score)
                    best_idx = idx

            cand_data, _ = best_candidate  # type: ignore[misc]
            cand_data["representative_rank"] = rank
            representatives.append(best_candidate)  # type: ignore[arg-type]
            candidates.pop(best_idx)
            print(
                f"  Selected rank {rank}: {cand_data['combo_name']} "
                f"(min_dist_to_reps={max_min_distance:.4f})"
            )
            rank += 1

            if not candidates:
                break

            filter_threshold = max_min_distance * alpha
            before_count = len(candidates)
            candidates = [
                (cand_data, cand_score)
                for cand_data, cand_score in candidates
                if _min_novelty_to_reps(cand_data["layout_dir"], representatives) >= filter_threshold
            ]
            removed = before_count - len(candidates)
            if removed > 0:
                print(
                    f"  Round {rank - 1}: filtered out {removed} candidate(s) "
                    f"(min-novelty < {filter_threshold:.4f}, alpha={alpha})"
                )

        print(f"Scene selection complete: {len(representatives)} representative scenes chosen.")
        return [data for data, _ in representatives]


    @staticmethod
    def _is_pure_org_variant(variants: list[LayoutVariantCandidate]) -> bool:
        """Check if all variants are org variants."""
        return all(v.layout_id.endswith("_org") for v in variants)

    def _write_combo_index(self, output_path: Path, combinations: list[dict]) -> None:
        payload = {
            "combined_scene_count": len(combinations),
            "combined_scenes": combinations,
        }
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

    def run(self, artifacts: SceneArtifacts) -> list[Path]:
        run_t0 = time.time()
        if artifacts.unposed_objects_dir is None or artifacts.collision_hulls_dir is None:
            print("Skipping layout combination stage: missing unposed objects or collision hulls.")
            return []

        generated_root = self.paths.outputs_dir / "generated"
        t0 = time.time()
        grouped_variants = discover_layout_variants(generated_root)
        print(f"[Timing] Discover layout variants: {time.time() - t0:.2f}s ({sum(len(v) for v in grouped_variants.values())} variants in {len(grouped_variants)} groups)")
        if not grouped_variants:
            print("Skipping layout combination stage: no generated layout variants found.")
            return []

        generator = SceneGenerator(artifacts)
        validation_root = generated_root / "combined_scenes" / "_validation"
        valid_variants_by_group: dict[str, list[LayoutVariantCandidate]] = {}

        t0 = time.time()
        for anchor_id in sorted(grouped_variants.keys()):
            candidates = grouped_variants[anchor_id]
            valid_candidates: list[LayoutVariantCandidate] = []
            for candidate in candidates:
                if self._local_variant_is_valid(candidate, artifacts, generator, validation_root):
                    valid_candidates.append(candidate)
            if valid_candidates:
                valid_variants_by_group[anchor_id] = valid_candidates
            else:
                print(f"No valid local variants remain for anchor '{anchor_id}'.")
        print(f"[Timing] Local variant validation: {time.time() - t0:.2f}s ({sum(len(v) for v in valid_variants_by_group.values())} valid)")

        combined_root = generated_root / "combined_scenes"
        combined_root.mkdir(parents=True, exist_ok=True)

        created_glbs: list[Path] = []
        combo_index_entries: list[dict] = []
        # Deferred manifest records: (manifest_path, kwargs_for_manifest)
        pending_manifests: list[tuple[Path, dict]] = []

        t0 = time.time()
        relations, _, _ = self._load_relationships_and_objects()
        parents_by_child, children_by_parent = self._load_relationship_graph(relations)
        chosen_parent_by_child = self._choose_parent_by_child(parents_by_child)
        root_nodes = self._collect_root_nodes(valid_variants_by_group, chosen_parent_by_child)
        print(f"[Timing] Load relations & build graph: {time.time() - t0:.2f}s ({len(root_nodes)} root nodes)")

        if not root_nodes:
            print("Skipping layout combination stage: no root nodes found in relations.")
            return []

        root_result_groups: list[list[tuple[dict[str, dict], list[LayoutVariantCandidate]]]] = []
        t0 = time.time()
        for root_id in root_nodes:
            root_results = self._expand_anchor(root_id, {}, valid_variants_by_group, children_by_parent)
            if not root_results:
                print(f"No valid local variants remain for root '{root_id}'.")
                continue
            root_result_groups.append(root_results)
        total_combos = sum(len(g) for g in root_result_groups)
        print(f"[Timing] Expand anchor (combo generation): {time.time() - t0:.2f}s ({total_combos} combos)")

        if not root_result_groups:
            print("Skipping combination stage: no valid root branches remain.")
            return []

        t0 = time.time()
        for root_combo in itertools.product(*root_result_groups):
            combined_layouts: dict[str, dict] = {}
            selected_variants: list[LayoutVariantCandidate] = []
            valid_combo = True
            for branch_layouts, branch_variants in root_combo:
                try:
                    combined_layouts = self._merge_layout_maps(combined_layouts, branch_layouts)
                except ValueError as exc:
                    print(f"Skipping combo during merge: {exc}")
                    valid_combo = False
                    break
                selected_variants.extend(branch_variants)

            if not valid_combo or not selected_variants:
                continue

            combo_name = build_combo_name(tuple(selected_variants))
            combo_dir = combined_root / combo_name
            if combo_dir.exists():
                shutil.rmtree(combo_dir)
            combo_dir.mkdir(parents=True, exist_ok=True)

            combo_layout_dir = combo_dir / "layout_params"
            combo_layout_dir.mkdir(parents=True, exist_ok=True)

            self._write_layout_tree(combined_layouts, combo_layout_dir)

            # Fill missing wall/floor/ceiling layouts from simulated params if available.
            self._fill_missing_root_layouts_from_simulated(combo_layout_dir)

            # DEBUG
            collision_report_path = combo_dir / "collision_report.json"
            collision_result = self.tool_runner.check_object_collisions(
                layout_dir=combo_layout_dir,
                hull_dir=artifacts.collision_hulls_dir,
                output=collision_report_path,
                anchor_id=selected_variants[0].anchor_id if selected_variants else None,
            )
            if collision_result.returncode != 0 or not collision_report_path.exists():
                print(f"Skipping combo {combo_name}: collision check failed.")
                shutil.rmtree(combo_dir, ignore_errors=True)
                continue

            collision_report = self._load_json(collision_report_path)
            collisions = collision_report.get("collisions", [])
            if isinstance(collisions, list) and collisions:
                print(f"Skipping combo {combo_name}: final collisions detected ({len(collisions)}).")
                shutil.rmtree(combo_dir, ignore_errors=True)
                continue

            pending_manifests.append(
                (
                    combo_dir / "combo_manifest.json",
                    {
                        "combo_name": combo_name,
                        "variants": tuple(selected_variants),
                        "layout_dir": combo_layout_dir,
                        "combo_dir": combo_dir,
                        "collision_report_path": collision_report_path,
                    },
                )
            )
            combo_index_entries.append(
                {
                    "combo_name": combo_name,
                    "layout_dir": str(combo_layout_dir),
                    "selected_variants": [
                        {
                            "anchor_id": variant.anchor_id,
                            "layout_id": variant.layout_id,
                            "group_dir": str(variant.group_dir),
                            "variant_dir": str(variant.variant_dir),
                            "layout_dir": str(variant.layout_dir),
                            "mapping_path": str(variant.mapping_path),
                        }
                        for variant in selected_variants
                    ],
                }
            )

        print(f"[Timing] Combo enumeration & manifest building: {time.time() - t0:.2f}s ({len(pending_manifests)} combos)")

        # ========== Novelty Metrics Computation ==========
        # Find the pure org combo (all variants use _org layout)
        org_combo = None
        org_layout_dir = None
        for manifest_path, manifest_data in pending_manifests:
            if self._is_pure_org_variant(manifest_data["variants"]):
                org_combo = manifest_data
                org_layout_dir = manifest_data["layout_dir"]
                break

        if org_layout_dir is None:
            print("Warning: No pure org combo found. Skipping novelty metrics and scene selection.")
            # Still write manifests without novelty metrics
            for manifest_path, manifest_data in pending_manifests:
                self._write_combo_manifest(
                    manifest_path,
                    combo_name=manifest_data["combo_name"],
                    variants=manifest_data["variants"],
                    layout_dir=manifest_data["layout_dir"],
                    collision_report_path=manifest_data["collision_report_path"],
                )
        else:
            print(f"Found pure org combo. Computing novelty metrics for all combos...")
            t0 = time.time()
            # Compute novelty metrics for all combos
            for manifest_path, manifest_data in pending_manifests:
                combo_layout_dir = manifest_data["layout_dir"]
                novelty_metrics = self._compute_novelty_metrics(
                    combo_layout_dir,
                    org_layout_dir,
                    wp=0.4,
                    wu=0.4,
                    wr=0.2,
                )

                self._write_combo_manifest(
                    manifest_path,
                    combo_name=manifest_data["combo_name"],
                    variants=manifest_data["variants"],
                    layout_dir=manifest_data["layout_dir"],
                    collision_report_path=manifest_data["collision_report_path"],
                    novelty_metrics=novelty_metrics,
                )
                print(f"  {manifest_data['combo_name']}: novelty={novelty_metrics['novelty']:.4f}")
            print(f"[Timing] Novelty metrics computation: {time.time() - t0:.2f}s")

            # ========== Representative Scene Selection ==========
            print(f"\n{'='*60}")
            print("Starting representative scene selection...")
            print(f"{'='*60}")
            t0 = time.time()
            selected_reps = self._select_representative_scenes(
                pending_manifests,
                org_layout_dir=org_layout_dir,
                k=6,
                alpha=0.1,
                wp=0.8,
                wr=0.2,
            )
            print(f"[Timing] Representative scene selection: {time.time() - t0:.2f}s ({len(selected_reps)} selected)")

            # Write a selection summary file
            if selected_reps:
                selection_summary_path = combined_root / "selection_summary.json"
                summary_payload = {
                    "total_combos_evaluated": len(pending_manifests),
                    "selected_count": len(selected_reps),
                    "selected_scenes": [
                        {
                            "rank": rep.get("representative_rank"),
                            "combo_name": rep["combo_name"],
                            "layout_dir": str(rep["layout_dir"]),
                        }
                        for rep in selected_reps
                    ],
                }
                with selection_summary_path.open("w", encoding="utf-8") as f:
                    json.dump(summary_payload, f, ensure_ascii=False, indent=2)
                print(f"Selection summary written to {selection_summary_path}")

            # ========== Compose GLB for Selected Combos (including org seed) ==========
            print(f"\nComposing GLB files for {len(selected_reps)} selected representative scenes...")
            t0 = time.time()
            for rep in selected_reps:
                combo_name = rep["combo_name"]
                combo_dir = rep["combo_dir"]
                combo_layout_dir = rep["layout_dir"]
                glb_path = self._compose_selected_combo(
                    combo_dir, combo_name, combo_layout_dir, artifacts.unposed_objects_dir
                )
                if glb_path is not None:
                    created_glbs.append(glb_path)
            representative_scenes_dir = self.paths.representative_scenes_dir
            if representative_scenes_dir.exists():
                shutil.rmtree(representative_scenes_dir)
            representative_scenes_dir.mkdir(parents=True, exist_ok=True)
            for glb_path in created_glbs:
                link_path = representative_scenes_dir / glb_path.name
                if link_path.exists() or link_path.is_symlink():
                    link_path.unlink()
                link_path.symlink_to(glb_path.resolve())
            print(f"Linked representative GLBs to {representative_scenes_dir}")
            print(f"[Timing] GLB composition for selected scenes: {time.time() - t0:.2f}s ({len(created_glbs)} GLBs)")

        combo_index_path = combined_root / "combo_index.json"
        self._write_combo_index(combo_index_path, combo_index_entries)

        print(f"Layout combination stage completed with {len(created_glbs)} composed scenes.")
        print(f"[Timing] Total layout combination: {time.time() - run_t0:.2f}s")
        return created_glbs