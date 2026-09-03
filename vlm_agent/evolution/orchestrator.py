from __future__ import annotations

import copy
import json
import math
import numpy as np
import shutil
import torch
from dataclasses import dataclass, field, replace
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from pytorch3d.transforms import quaternion_to_matrix

from .scene_optimizer import build_anchor_dependency_graph, iter_subtree_roots, load_layout_snapshot, propagate_subtree_transform, collect_descendant_subtree
from .layout_combination import LayoutCombination
from .scene_generator import SceneGenerator
from .tool_runner import ToolRunner
from .types import PipelineReport, SceneArtifacts
from .vlm_client import VLMClient
from .paths import ProjectPaths


@dataclass
class PipelineOrchestrator:
    """Pipeline: optimize existing artifacts -> generate -> compose."""

    paths: ProjectPaths
    tool_runner: ToolRunner
    vlm_client: VLMClient
    max_rounds: int = 3
    collision_policy: str = "strict"
    history: list[PipelineReport] = field(default_factory=list)

    def _load_prompt(self, prompt_name: str) -> str:
        """Load a prompt from the prompts directory."""
        prompt_path = Path(__file__).resolve().parent / "prompts" / f"{prompt_name}.md"
        if prompt_path.exists():
            return prompt_path.read_text(encoding="utf-8")
        return f"[Missing prompt: {prompt_name}]"

    def _load_relationships_and_objects(self, suffix: str = "") -> tuple[list, dict, dict]:
        """Load relationships and objects from JSON files.

        Returns:
            (relationships, objects_data, walls_data)
        """
        relations_path = self.paths.outputs_dir / f"relations{suffix}.json"
        objects_path = self.paths.outputs_dir / f"objects{suffix}.json"

        relationships = []
        if relations_path.exists():
            with relations_path.open("r", encoding="utf-8") as f:
                relationships = json.load(f)

        objects_data = {"objects": [], "walls": []}
        if objects_path.exists():
            with objects_path.open("r", encoding="utf-8") as f:
                objects_data = json.load(f)

        return relationships, objects_data.get("objects", []), objects_data.get("walls", [])

    def _analyze_cross_anchor_constraints(self, object_groups: dict, objects: list) -> dict:
        """Generate and persist cross-anchor constraints with the VLM."""
        constraints_path = self.paths.outputs_dir / "cross_anchor_constraints.json"
        constraints_path.parent.mkdir(parents=True, exist_ok=True)

        prompt_template = self._load_prompt("analyze_cross_anchor_constraints")
        image_path = self.paths.outputs_dir / "image.png"
        object_categories = {
            obj.get("id"): obj.get("category")
            for obj in objects
            if obj.get("id") and obj.get("category")
        }

        def _format_object_ref(object_id: str) -> str:
            category = object_categories.get(object_id)
            if category:
                return f"{object_id} ({category})"
            return object_id

        def _annotate_object_groups(value: object) -> object:
            if isinstance(value, dict):
                annotated: dict[str, object] = {}
                for key, item in value.items():
                    if key == "object_ids" and isinstance(item, list):
                        annotated[key] = [
                            _format_object_ref(obj_id) if isinstance(obj_id, str) else obj_id
                            for obj_id in item
                        ]
                    elif key.startswith("object_") and isinstance(item, dict):
                        new_key = _format_object_ref(key)
                        annotated[new_key] = _annotate_object_groups(item)
                    else:
                        annotated[key] = _annotate_object_groups(item)
                return annotated
            if isinstance(value, list):
                return [_annotate_object_groups(item) for item in value]
            return value

        prompt = (
            f"{prompt_template}\n\n"
            "OBJECT_GROUPS (JSON):\n"
            f"{json.dumps(_annotate_object_groups(object_groups), ensure_ascii=False, indent=2)}\n"
        )
        images = [str(image_path)] if image_path.exists() else None
        vlm_response = self.vlm_client.plan(prompt, images=images)
        payload = self.vlm_client.extract_json_payload(vlm_response)

        with constraints_path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

        return payload

    def _load_cross_anchor_constraints(self, constraints_path: Path | None = None) -> dict[str, list[dict[str, list[str] | str]]]:
        """Load cross-anchor constraints from a JSON file.

        Returns:
            Dict mapping anchor_id -> list of constraints, where each constraint is a dict with keys:
            - "related_objects": list of object IDs
            - "layout_note": string describing the layout constraint
        """
        if constraints_path is None:
            constraints_path = self.paths.outputs_dir / "cross_anchor_constraints.json"

        if not constraints_path.exists():
            return {}

        with constraints_path.open("r", encoding="utf-8") as f:
            payload = json.load(f)

        grouped_constraints: dict[str, list[dict[str, list[str] | str]]] = {}
        if not isinstance(payload, list):
            return grouped_constraints

        for item in payload:
            if not isinstance(item, dict):
                continue
            for anchor_key in ("anchor_a", "anchor_b"):
                anchor_data = item.get(anchor_key)
                if not isinstance(anchor_data, dict):
                    continue
                anchor_id = anchor_data.get("anchor_id")
                if not anchor_id:
                    continue
                other_key = "anchor_b" if anchor_key == "anchor_a" else "anchor_a"
                other_data = item.get(other_key)
                if isinstance(other_data, dict) and other_data.get("anchor_id") == anchor_id:
                    continue
                entry = {
                    "related_objects": anchor_data.get("related_objects", []),
                    "layout_note": anchor_data.get("layout_note", ""),
                }
                grouped_constraints.setdefault(str(anchor_id), []).append(entry)

        return grouped_constraints

    def _parse_collision_report(self, stdout: str) -> dict | None:
        text = stdout.strip()
        if not text:
            return None
        start = text.find("{")
        if start == -1:
            return None
        payload_text = text[start:]
        end = payload_text.rfind("}")
        if end == -1:
            return None
        payload_text = payload_text[: end + 1]
        try:
            payload = json.loads(payload_text)
        except json.JSONDecodeError:
            return None
        return payload if isinstance(payload, dict) else None

    def _keep_y_axis_only(self, quaternion: list[float]) -> list[float]:
        w, _, y, _ = quaternion
        mag = math.sqrt(w**2 + y**2)
        if mag < 1e-6:
            return [1.0, 0.0, 0.0, 0.0]
        return [w / mag, 0.0, y / mag, 0.0]

    def _normalize_upright_layout_rotations(self, layout_dir: Path, objects: list[dict]) -> None:
        for obj in objects:
            if obj.get("upright", False) is not True:
                continue
            layout_path = layout_dir / f'{obj["id"]}.json'
            if not layout_path.exists():
                continue
            with layout_path.open("r", encoding="utf-8") as f:
                layout = json.load(f)
            rotation = layout.get("rotation")
            if isinstance(rotation, list) and len(rotation) == 1 and isinstance(rotation[0], list):
                rotation = rotation[0]
            if not isinstance(rotation, list) or len(rotation) != 4:
                continue
            layout["rotation"] = [self._keep_y_axis_only([float(v) for v in rotation])]
            with layout_path.open("w", encoding="utf-8") as f:
                json.dump(layout, f, indent=2)

    def _format_object_with_category(
        self, object_id: str | None, object_categories: dict[str, str] | None
    ) -> str | None:
        if not object_id:
            return object_id
        if not object_categories:
            return object_id
        category = object_categories.get(object_id)
        if not category:
            return object_id
        return f"{object_id} ({category})"

    def _get_deep_collision_pairs(
        self,
        report: dict | None,
        *,
        contact_threshold: int,
        object_categories: dict[str, str] | None = None,
    ) -> list[dict]:
        if not report:
            return []
        collisions = report.get("collisions", [])
        if not isinstance(collisions, list):
            return []
        results = []
        for entry in collisions:
            if not isinstance(entry, dict):
                continue
            contacts = entry.get("contacts")
            if isinstance(contacts, (int, float)) and contacts > contact_threshold:
                results.append(
                    {
                        "a": self._format_object_with_category(
                            entry.get("a"), object_categories
                        ),
                        "b": self._format_object_with_category(
                            entry.get("b"), object_categories
                        ),
                        "contacts": int(contacts),
                    }
                )
        return results

    def _parse_wall_normal(self, wall: dict) -> list:
        """Compute the normal vector for a wall in 3D space.
        
        Given two points on the wall (x, z coordinates at y=0), compute the normal vector
        pointing outward from the wall.

        Returns:
            Normal vector [x, y, z] in world coordinates
        """
        p1 = wall.get("first_point", [0, 0])
        p2 = wall.get("second_point", [0, 0])
        
        # Extract x, z coordinates
        x1, z1 = p1[0], p1[1]
        x2, z2 = p2[0], p2[1]
        
        # Direction along the wall
        wall_dir = np.array([x2 - x1, 0, z2 - z1], dtype=np.float32)
        wall_dir_norm = np.linalg.norm(wall_dir)
        
        if wall_dir_norm < 1e-6:
            print(f"Warning: Wall {wall.get('id')} has zero length, using default normal")
            return [1.0, 0.0, 0.0]
        
        wall_dir = wall_dir / wall_dir_norm
        
        # p1, p2 must be clockwise
        # Normal perpendicular to wall_dir in XZ plane, pointing outward
        # For a wall from p1 to p2, the outward normal is rotated 90 degrees
        # clockwise in XZ plane when viewed from above (+Y)
        normal = np.array([wall_dir[2], 0.0, -wall_dir[0]], dtype=np.float32)
        
        return normal.tolist()

    def _group_objects_by_anchor(
        self, relationships: list, objects: list, walls: list
    ) -> dict:
        """Group objects by their anchor point (floor, ceiling, or wall).

        Returns:
            Dict mapping anchor_id -> {"relation_type": str, "object_ids": list[str]}
        """
        groups: dict[str, dict[str, list[str] | str]] = {}

        for rel in relationships:
            relation_type = rel.get("relation")
            if relation_type != "attach" and relation_type != "contain":
                continue

            anchor_id = rel["parameters"]["object_anchor"]
            target_id = rel["parameters"]["object_target"]

            if anchor_id not in groups:
                groups[anchor_id] = {"relation_type": relation_type, "object_ids": []}
            elif groups[anchor_id].get("relation_type") != relation_type:
                groups[anchor_id]["relation_type"] = "mixed"

            object_ids = groups[anchor_id]["object_ids"]
            if target_id not in object_ids:
                object_ids.append(target_id)

        return groups

    def _compute_gravity_for_anchor(self, anchor_id: str, walls_by_id: dict) -> list:
        """Compute the gravity direction for a given anchor.
            
        Returns:
            Gravity vector [x, y, z]
        """
        if anchor_id == "floor":
            # Default gravity pointing downward (-Y)
            return [0.0, -2.0, 0.0]
        elif anchor_id == "ceiling":
            # Reverse gravity pointing upward (+Y)
            return [0.0, 2.0, 0.0]
        elif anchor_id.startswith("wall"):
            # Compute normal from wall geometry
            if anchor_id in walls_by_id:
                wall = walls_by_id[anchor_id]
                normal = self._parse_wall_normal(wall)
                # Scale to match gravity magnitude
                magnitude = 2.0
                gravity = [n * magnitude for n in normal]
                return gravity
            else:
                print(f"Warning: Wall {anchor_id} not found, using default gravity")
                return [0.0, -2.0, 0.0]
        else:
            # Unknown anchor, use default gravity
            return [0.0, -2.0, 0.0]

    def _get_wall_object_ids(self, layout_dir: Path) -> list[str]:
        """Get wall object IDs from the layout directory.
        
        Returns object IDs that match wall patterns (e.g., wall_001, ceiling, floor).
        """
        wall_object_ids = []
        if not layout_dir.exists():
            return wall_object_ids
        
        for layout_file in layout_dir.glob("*.json"):
            if layout_file.stem.startswith("object_"):
                continue
            obj_id = layout_file.stem
            # Check if it matches wall patterns
            if "wall" in obj_id.lower() or obj_id.lower() in ["ceiling", "floor"]:
                wall_object_ids.append(obj_id)
        
        return sorted(wall_object_ids)

    def _compute_camera_direction_for_anchor(self, anchor_id: str, walls_by_id: dict, objects: list,
                                             relation_type: str, artifacts: SceneArtifacts) -> tuple[list, str]:
        """Compute the camera direction for a given anchor.

        Returns:
            Camera direction vector [x, y, z] (normalized)
            Render anchor ID
        """
        if anchor_id == "floor":
            # Camera looking downward
            return [0.0, -1.0, 0.0], anchor_id
        elif anchor_id == "ceiling":
            # Camera looking upward
            return [0.0, 1.0, 0.0], anchor_id
        elif anchor_id.startswith("wall"):
            # Use normal from wall geometry (already normalized by _parse_wall_normal)
            if anchor_id in walls_by_id:
                wall = walls_by_id[anchor_id]
                normal = self._parse_wall_normal(wall)
                return normal, None
            else:
                print(f"Warning: Wall {anchor_id} not found, using default camera direction")
                return [0.0, -1.0, 0.0], None
        else:
            if relation_type == "contain" or relation_type == "mixed":
                if any(obj.get("id") == anchor_id and obj.get("render_direction") == "front" for obj in objects):
                    object_pose_path = artifacts.layout_params_dir / f"{anchor_id}.json"
                    if object_pose_path.exists():
                        with object_pose_path.open("r", encoding="utf-8") as f:
                            pose_data = json.load(f)
                        direction_vector = torch.tensor([0.0, 0.0, -1.0], dtype=torch.float32)
                        q_wxyz = pose_data["rotation"][0]
                        quat_tensor = torch.tensor(q_wxyz, dtype=torch.float32)
                        rot_matrix = quaternion_to_matrix(quat_tensor)
                        rotated_vector = torch.matmul(rot_matrix, direction_vector)
                        camera_direction = -rotated_vector
                        return camera_direction.tolist(), None
                    else:
                        return [0.0, 0.0, 1.0], anchor_id
            return [0.0, -1.0, 0.0], anchor_id

    def _stage_optimization(self, artifacts: SceneArtifacts) -> SceneArtifacts:
        """Optimize the existing scene using physics simulation."""
        if artifacts.unposed_objects_dir and artifacts.collision_hulls_dir:
            print("Running generate_room_mesh tool.")
            room_mesh_result = self.tool_runner.generate_room_mesh(
                artifacts.layout_params_dir,
                artifacts.unposed_objects_dir,
                self.paths.outputs_dir / "objects.json",
            )
            if room_mesh_result.returncode != 0:
                print(f"generate_room_mesh failed: {room_mesh_result.stderr}")
            else:
                print("Room mesh generated.")

            print("Running build_collision_hulls tool.")
            manifest = artifacts.collision_hulls_dir / "manifest.json"
            result = self.tool_runner.build_collision_hulls(
                artifacts.unposed_objects_dir,
                artifacts.collision_hulls_dir,
                manifest,
                split_mode="none",
            )
            if result.failed_object_ids:
                print(f"Collision hulls built with failed objects: {', '.join(result.failed_object_ids)}")
            else:
                print("Collision hulls built.")

        if artifacts.layout_params_dir and artifacts.collision_hulls_dir and artifacts.simulated_layout_dir:
            print("Running simulate_scene_physics tool with grouped simulation.")

            artifacts.simulated_layout_dir.mkdir(parents=True, exist_ok=True)
            for layout_file in artifacts.layout_params_dir.glob("*.json"):
                shutil.copy2(layout_file, artifacts.simulated_layout_dir / layout_file.name)

            # Load relationships and objects
            relationships, objects, walls = self._load_relationships_and_objects()
            walls_by_id = {w["id"]: w for w in walls}
            self._normalize_upright_layout_rotations(artifacts.simulated_layout_dir, objects)

            # Get wall object IDs to be used as ground objects
            wall_object_ids = self._get_wall_object_ids(artifacts.simulated_layout_dir)

            # Group objects by anchor point
            object_groups = self._group_objects_by_anchor(relationships, objects, walls)

            if not object_groups:
                print("No grouped relationships found, running standard simulation.")
                self.tool_runner.simulate_scene_physics(
                    artifacts.simulated_layout_dir,
                    artifacts.collision_hulls_dir,
                    artifacts.simulated_layout_dir,
                    cpu_only=True,
                    use_ground_object=True,
                    ground_object_ids=wall_object_ids,
                )
            else:
                print(f"Found {len(object_groups)} anchor groups")

                # Fix containment for each group (skip walls and ceiling)
                for anchor_id, group_info in object_groups.items():
                    object_ids = group_info["object_ids"] if isinstance(group_info, dict) else group_info
                    relation_type = group_info.get("relation_type") if isinstance(group_info, dict) else None
                    if anchor_id.startswith("wall") or anchor_id == "ceiling":
                        continue

                    print(
                        f"Fixing containment for {len(object_ids)} objects "
                        f"{'attached to' if relation_type == 'attach' else 'inside'} '{anchor_id}'"
                    )

                    combined_object_ids = sorted(list(set(object_ids) | set([anchor_id])))
                    self.tool_runner.fix_layout_containment(
                        artifacts.simulated_layout_dir,
                        artifacts.unposed_objects_dir,
                        artifacts.simulated_layout_dir,
                        object_ids=combined_object_ids,
                        anchor_id=anchor_id,
                        tolerated_out_of_bounds=0.0,
                    )

                parents_by_child, children_by_parent, indegree = build_anchor_dependency_graph(object_groups)
                ready_queue = [anchor_id for anchor_id, degree in indegree.items() if degree == 0]
                ready_queue.sort(key=lambda anchor: (0 if anchor.startswith("wall") else 1 if anchor == "ceiling" else 2 if anchor == "floor" else 3, anchor))
                processed_anchors: set[str] = set()
                anchors_with_simulation: set[str] = set()

                pending_layout_dir = artifacts.simulated_layout_dir.parent / f"{artifacts.simulated_layout_dir.name}_pending"
                if pending_layout_dir.exists():
                    shutil.rmtree(pending_layout_dir)
                if artifacts.simulated_layout_dir.exists():
                    shutil.copytree(artifacts.simulated_layout_dir, pending_layout_dir)
                else:
                    pending_layout_dir.mkdir(parents=True, exist_ok=True)
                
                illegal_objects = set()
                while ready_queue:
                    anchor_id = ready_queue.pop(0)
                    if anchor_id in processed_anchors:
                        continue
                    processed_anchors.add(anchor_id)

                    object_ids = object_groups.get(anchor_id, {}).get("object_ids", [])
                    relation_type = object_groups.get(anchor_id, {}).get("relation_type", None)
                    if not object_ids:
                        for child_id in sorted(children_by_parent.get(anchor_id, [])):
                            indegree[child_id] = max(indegree.get(child_id, 0) - 1, 0)
                            if indegree[child_id] == 0:
                                ready_queue.append(child_id)
                        ready_queue.sort(key=lambda anchor: (0 if anchor.startswith("wall") else 1 if anchor == "ceiling" else 2 if anchor == "floor" else 3, anchor))
                        continue

                    anchors_with_simulation.add(anchor_id)
                    gravity = self._compute_gravity_for_anchor(anchor_id, walls_by_id)
                    print(f"Simulating {len(object_ids)} objects attached to or inside '{anchor_id}' with gravity {gravity}")

                    pending_layout_dir.mkdir(parents=True, exist_ok=True)
                    ground_object_ids = list(set(wall_object_ids) | {anchor_id})
                    combined_object_ids = sorted(list(set(object_ids) | set(ground_object_ids)))
                    before_layouts = load_layout_snapshot(pending_layout_dir)
                    init_non_ground_y_offset = True if anchor_id not in wall_object_ids and relation_type == "attach" else False

                    sim_result = self.tool_runner.simulate_scene_physics(
                        pending_layout_dir,
                        artifacts.collision_hulls_dir,
                        pending_layout_dir,
                        gravity=gravity,
                        object_ids=combined_object_ids,
                        cpu_only=True,
                        use_ground_object=True,
                        ground_object_ids=ground_object_ids,
                        init_non_ground_y_offset=init_non_ground_y_offset,
                    )

                    if sim_result.returncode != 0:
                        print(f"Simulation failed for anchor '{anchor_id}': {sim_result.stderr}")
                        continue

                    if not (anchor_id.startswith("wall") or anchor_id == "ceiling"):
                        detect_only = (self.collision_policy == "strict")
                        print(f"{'Fixing' if not detect_only else 'Detecting'} containment for {len(object_ids)} objects attached to or inside '{anchor_id}'")

                        combined_object_ids = sorted(list(set(object_ids) | set([anchor_id])))
                        fix_result = self.tool_runner.fix_layout_containment(
                            pending_layout_dir,
                            artifacts.unposed_objects_dir,
                            pending_layout_dir,
                            object_ids=combined_object_ids,
                            anchor_id=anchor_id,
                            tolerated_out_of_bounds=0.5 if anchor_id != "floor" else 0.0,
                            detect_only=detect_only,
                        )

                        # If strict policy, collect all out-of-bounds objects and their descendants
                        if self.collision_policy == "strict" and fix_result.out_of_bounds_object_ids:
                            out_ids = set(fix_result.out_of_bounds_object_ids)
                            for obj_id in fix_result.out_of_bounds_object_ids:
                                subtree_ids = collect_descendant_subtree(obj_id, children_by_parent)
                                out_ids.update(subtree_ids)
                            illegal_objects.update(out_ids)
                            print(f"Anchor '{anchor_id}' has illegal objects: {', '.join(x for x in out_ids)}")

                    after_layouts = load_layout_snapshot(pending_layout_dir)
                    propagated_subtrees: list[str] = []
                    for subtree_root_id in iter_subtree_roots(anchor_id, children_by_parent):
                        propagate_subtree_transform(
                            self,
                            pending_layout_dir,
                            children_by_parent,
                            anchor_id,
                            subtree_root_id,
                            before_layouts,
                            after_layouts,
                        )
                        propagated_subtrees.append(subtree_root_id)

                    if propagated_subtrees:
                        print(
                            f"Anchor '{anchor_id}' propagated subtrees: {', '.join(propagated_subtrees)}"
                        )

                    for child_id in sorted(children_by_parent.get(anchor_id, [])):
                        indegree[child_id] = max(indegree.get(child_id, 0) - 1, 0)
                        if indegree[child_id] == 0:
                            ready_queue.append(child_id)
                    ready_queue.sort(key=lambda anchor: (0 if anchor.startswith("wall") else 1 if anchor == "ceiling" else 2 if anchor == "floor" else 3, anchor))

                if self.collision_policy == "strict":
                    # If strict policy, check for deep collisions
                    collision_output_path = self.paths.outputs_dir / "collision_orig_scene.json"
                    collision_result = self.tool_runner.check_object_collisions(
                        layout_dir=pending_layout_dir,
                        hull_dir=artifacts.collision_hulls_dir,
                        output=collision_output_path,
                    )
                    if collision_result.returncode != 0:
                        print("Collision check failed for anchor "
                              f"{anchor_id}: {collision_result.stderr}"
                        )
                    else:
                        collision_report = self._parse_collision_report(collision_result.stdout)
                        collision_pairs = collision_report.get("collisions", []) if collision_report else []
                        anchor_object_ids = set(object_groups.keys())

                        def _subtree_size(object_id: str) -> int:
                            return len(collect_descendant_subtree(object_id, children_by_parent))

                        def _object_volume(pair_entry: dict, object_id: str) -> float:
                            size_key = "size_a" if pair_entry.get("a") == object_id else "size_b"
                            sizes = pair_entry.get(size_key, [])
                            if not isinstance(sizes, list) or len(sizes) != 3:
                                return float("inf")
                            return float(sizes[0]) * float(sizes[1]) * float(sizes[2])

                        for pair_entry in collision_pairs:
                            if not isinstance(pair_entry, dict):
                                continue
                            obj0 = str(pair_entry.get("a", ""))
                            obj1 = str(pair_entry.get("b", ""))
                            if not obj0 or not obj1:
                                continue
                            if obj0 in illegal_objects or obj1 in illegal_objects:
                                continue

                            obj0_is_anchor = obj0 in anchor_object_ids
                            obj1_is_anchor = obj1 in anchor_object_ids

                            if not obj0_is_anchor and not obj1_is_anchor:
                                vol0 = _object_volume(pair_entry, obj0)
                                vol1 = _object_volume(pair_entry, obj1)
                                illegal_objects.add(obj0 if vol0 <= vol1 else obj1)
                                print(f"Deep collision detected between non-anchor objects '{obj0}' and '{obj1}', "
                                      f"marking smaller volume object '{obj0 if vol0 <= vol1 else obj1}' as illegal.")
                                continue

                            if obj0_is_anchor ^ obj1_is_anchor:
                                illegal_objects.add(obj0 if not obj0_is_anchor else obj1)
                                print(f"Collision detected between anchor object '{obj0 if obj0_is_anchor else obj1}' and non-anchor object "
                                      f"'{obj1 if obj0_is_anchor else obj0}', marking non-anchor object as illegal.")
                                continue

                            def _anchor_priority(object_id: str) -> tuple[int, int, float]:
                                if object_id == "wall" or object_id.startswith("wall"):
                                    category_priority = 0
                                elif object_id == "floor":
                                    category_priority = 0
                                elif object_id == "ceiling":
                                    category_priority = 0
                                else:
                                    category_priority = 1
                                subtree_priority = -_subtree_size(object_id)
                                volume_priority = -_object_volume(pair_entry, object_id)
                                return (category_priority, subtree_priority, volume_priority)

                            if _anchor_priority(obj0) <= _anchor_priority(obj1):
                                illegal_objects.update(collect_descendant_subtree(obj1, children_by_parent))
                                print(f"Collision detected between anchor objects '{obj0}' and '{obj1}', marking '{obj1}' and its descendants as illegal.")
                            else:
                                illegal_objects.update(collect_descendant_subtree(obj0, children_by_parent))
                                print(f"Collision detected between anchor objects '{obj0}' and '{obj1}', marking '{obj0}' and its descendants as illegal.")

                print(f"Illegal objects after simulation: {illegal_objects if illegal_objects else 'None'}")

                if illegal_objects:
                    filtered_relationships = []
                    for relation in relationships:
                        if not isinstance(relation, dict):
                            continue
                        parameters = relation.get("parameters", {})
                        if not isinstance(parameters, dict):
                            continue
                        if parameters.get("object_anchor") in illegal_objects:
                            continue
                        if parameters.get("object_target") in illegal_objects:
                            continue
                        filtered_relationships.append(relation)

                    filtered_objects = [
                        obj for obj in objects
                        if isinstance(obj, dict) and obj.get("id") not in illegal_objects
                    ]

                filtered_relationships = filtered_relationships if illegal_objects else relationships
                filtered_objects = filtered_objects if illegal_objects else objects
                with (self.paths.outputs_dir / "relations_filtered.json").open("w", encoding="utf-8") as f:
                    json.dump(filtered_relationships, f, ensure_ascii=False, indent=2)
                with (self.paths.outputs_dir / "objects_filtered.json").open("w", encoding="utf-8") as f:
                    json.dump({"walls": walls, "objects": filtered_objects}, f, ensure_ascii=False, indent=2)

                if pending_layout_dir.exists():
                    if artifacts.simulated_layout_dir.exists():
                        if illegal_objects:
                            for obj_id in illegal_objects:
                                obj_file = pending_layout_dir / f"{obj_id}.json"
                                if obj_file.exists():
                                    obj_file.unlink()
                        shutil.rmtree(artifacts.simulated_layout_dir)
                    shutil.move(str(pending_layout_dir), str(artifacts.simulated_layout_dir))

                if not anchors_with_simulation and object_groups:
                    print("No anchors were processed by queue-based grouped simulation.")

        return artifacts

    def _stage_generate_scene(
        self, artifacts: SceneArtifacts,
        deep_collision_contact_threshold: int = 8,
        max_refine_rounds: int = 3,
        enable_memory: bool = True,
        refine_movement_threshold: float = 0.03,
        refine_rotation_tolerance: float = 1e-2,
        draw_object_directions: bool = True,
        draw_object_bboxes: bool = True,
    ) -> None:
        """Generate new scene variants based on grouped renders."""
        generator = SceneGenerator(artifacts)
        prompt_template = self._load_prompt("generate_layout_description")
        pose_prompt_template = self._load_prompt("generate_2d_pose")
        pose_prompt_fix_rot_template = self._load_prompt("generate_2d_pose_fixed_rotation")
        evaluate_prompt_template = self._load_prompt("evaluate_layout_modification")
        revise_prompt_template = self._load_prompt("apply_layout_modification")

        relationships_raw, objects, walls = self._load_relationships_and_objects(suffix="_filtered")
        object_categories = {
            obj.get("id"): obj.get("category")
            for obj in objects
            if obj.get("id") and obj.get("category")
        }

        layout_dir = artifacts.simulated_layout_dir or artifacts.layout_params_dir
        unposed_dir = artifacts.unposed_objects_dir

        generation_output_dir = self.paths.outputs_dir / "generated"
        if generation_output_dir.exists():
            shutil.rmtree(generation_output_dir)
        generation_output_dir.mkdir(parents=True, exist_ok=True)

        walls_by_id = {w.get("id"): w for w in walls}
        object_groups = self._group_objects_by_anchor(relationships_raw, objects, walls)
        if not object_groups:
            all_ids = []
            for layout_file in layout_dir.glob("*.json"):
                all_ids.append(layout_file.stem)
            if all_ids:
                object_groups = {"floor": {"object_ids": sorted(all_ids), "relation_type": "attach"}}

        # Ensure each object has a unique anchor: remove objects from wall anchors if they are already anchored by non-wall anchors
        non_wall_anchors_object_ids = set()
        for anchor_id, group_info in object_groups.items():
            if not anchor_id.startswith("wall"):
                object_ids = group_info.get("object_ids", []) if isinstance(group_info, dict) else group_info
                non_wall_anchors_object_ids.update(object_ids)

        for anchor_id, group_info in list(object_groups.items()):
            if anchor_id.startswith("wall"):
                object_ids = group_info.get("object_ids", []) if isinstance(group_info, dict) else group_info
                filtered_object_ids = [obj_id for obj_id in object_ids if obj_id not in non_wall_anchors_object_ids]
                if not filtered_object_ids:
                    object_groups.pop(anchor_id, None)
                    continue
                if isinstance(group_info, dict):
                    group_info["object_ids"] = filtered_object_ids
                else:
                    object_groups[anchor_id] = filtered_object_ids

        if not (self.paths.outputs_dir / "cross_anchor_constraints.json").exists():
            self._analyze_cross_anchor_constraints(object_groups, objects)
        cross_anchor_constraints = self._load_cross_anchor_constraints()

        def _process_group(anchor_id: str, group_info: dict | list) -> dict | None:
            group_logs: list[str] = []

            def group_log(message: str) -> None:
                line = f"[{anchor_id}] {message}"
                print(line, flush=True)
                group_logs.append(line)

            object_ids = group_info.get("object_ids", []) if isinstance(group_info, dict) else group_info
            relation_type = group_info.get("relation_type") if isinstance(group_info, dict) else None

            group_dir = generation_output_dir / f"group_{anchor_id}"
            group_dir.mkdir(parents=True, exist_ok=True)

            camera_direction, render_anchor_id = self._compute_camera_direction_for_anchor(
                anchor_id,
                walls_by_id,
                objects,
                relation_type,
                artifacts,
            )
            output_path = group_dir / f"grouped_scene_{anchor_id}.png"
            mapping_path = output_path.with_suffix(".json")

            render_object_ids = list(object_ids)
            if anchor_id not in render_object_ids:
                render_object_ids.append(anchor_id)

            group_log(f"Rendering grouped scene with camera direction {camera_direction}.")
            render_result = self.tool_runner.render_scene(
                layout_dir=layout_dir,
                unposed_dir=unposed_dir,
                output=output_path,
                object_ids=render_object_ids,
                camera_direction=camera_direction,
                radius=0.0,
                image_size=1024,
                add_grid=True,
                draw_object_directions=draw_object_directions if relation_type == "attach" else False,
                draw_object_bboxes=draw_object_bboxes,
                anchor_id=render_anchor_id,
                hide_object_ids=[anchor_id],
                use_aabb_for_object_bboxes=True if relation_type != "attach" else False,
            )

            if render_result.returncode != 0:
                group_log(f"Failed to render group '{anchor_id}': {render_result.stderr}")
                return None

            cross_anchor_constraint = cross_anchor_constraints.get(anchor_id)
            cross_anchor_constraint_text = ""
            if cross_anchor_constraint:
                cross_anchor_constraint_text = (
                    "CROSS_ANCHOR_CONSTRAINTS:\n"
                    f"{json.dumps(cross_anchor_constraint, ensure_ascii=False, indent=2)}\n\n"
                )

            # group_log(f"Cross-anchor constraints: {cross_anchor_constraint_text}")

            # Generate similar layout description using VLM based on the rendered image and scene context
            summary, anchor_category = generator.generate_similar_layout_context(
                mapping_path=mapping_path,
                objects=objects,
                anchor_id=anchor_id,
            )
            description_prompt = (
                f"{prompt_template}\n\n"
                f"ANCHOR_CONSTRAINT: The anchor is {anchor_category or anchor_id} and must not change in the new layout.\n\n"
                f"{cross_anchor_constraint_text}"
                "CURRENT_LAYOUT_DATA (\"bbox_2d\": [min_u, min_v, max_u, max_v]):\n"
                f"{json.dumps(summary, ensure_ascii=False, indent=2)}\n"
            )

            group_log(f"Generating new layout description.")
            image_inputs = [str(output_path)] if output_path.exists() else None
            vlm_response = self.vlm_client.plan(description_prompt, images=image_inputs)
            response_payload = self.vlm_client.extract_json_payload(vlm_response)
            response_payload["usage"] = self.vlm_client.last_usage or {}

            output_json = group_dir / "generated_layout_description.json"
            with output_json.open("w", encoding="utf-8") as f:
                json.dump(response_payload, f, ensure_ascii=False, indent=2)

            # For each generated layout description, also generate a 2D pose estimation
            layout_entries = response_payload.get("alternative_layouts", [])

            # Insert the original layout as layout_id='org' at the beginning
            layout_entries.insert(0, {"layout_id": "org"})

            group_log(f"Processing {len(layout_entries)} layout entries for anchor '{anchor_id}'.")

            # Helper function to check collisions and render a layout round
            def _check_collisions_and_render_round(
                source_layout_dir: Path,
                current_round_dir: Path,
                current_round_index: int,
                layout_suffix: str,
            ) -> tuple[Path, Path, Path, list[dict]]:
                ground_object_ids = []
                anchor_layout_path = source_layout_dir / f"{anchor_id}.json"
                if anchor_layout_path.exists():
                    ground_object_ids = [anchor_id]

                if ground_object_ids:
                    group_log(f"Checking collisions for round {current_round_index} with ground objects {ground_object_ids}")
                else:
                    group_log(f"Checking collisions for round {current_round_index}")

                deep_collision_pairs: list[dict] = []
                collision_output_path = (
                    current_round_dir
                    / f"layout_{layout_suffix}_collisions_r{current_round_index}.json"
                )
                collision_result = self.tool_runner.check_object_collisions(
                    layout_dir=source_layout_dir,
                    hull_dir=artifacts.collision_hulls_dir,
                    output=collision_output_path,
                )
                if collision_result.returncode == 0:
                    collision_report = self._parse_collision_report(collision_result.stdout)
                    deep_collision_pairs = self._get_deep_collision_pairs(
                        collision_report,
                        contact_threshold=deep_collision_contact_threshold,
                        object_categories=object_categories,
                    )
                else:
                    group_log(
                        "Collision check failed for layout "
                        f"{layout_suffix} round {current_round_index}: {collision_result.stderr}"
                    )

                group_log(f"Rendering layout {layout_suffix} round {current_round_index}")
                render_output_path = (
                    current_round_dir / f"layout_{layout_suffix}_render_r{current_round_index}.png"
                )
                self.tool_runner.render_scene(
                    layout_dir=source_layout_dir,
                    unposed_dir=unposed_dir,
                    output=render_output_path,
                    object_ids=render_object_ids,
                    camera_direction=camera_direction,
                    radius=0.0,
                    image_size=1024,
                    add_grid=True,
                    draw_object_directions=draw_object_directions if relation_type == "attach" else False,
                    draw_object_bboxes=draw_object_bboxes,
                    anchor_id=render_anchor_id,
                    hide_object_ids=[anchor_id],
                    grid_mapping=mapping_path,
                    use_aabb_for_object_bboxes=True if relation_type != "attach" else False,
                )
                return (
                    source_layout_dir,
                    render_output_path,
                    render_output_path.with_suffix(".json"),
                    deep_collision_pairs,
                )

            # Function to process a single layout entry
            def _process_layout_entry(layout_entry: dict, anchor_id: str, output_path: Path, mapping_path: Path) -> dict | None:
                layout_logs: list[str] = []

                def layout_log(message: str) -> None:
                    line = f"[{anchor_id}][layout:{layout_entry.get('layout_id', '')}] {message}"
                    layout_logs.append(line)

                layout_id = layout_entry.get("layout_id")

                layout_suffix = str(layout_id) if layout_id is not None else "unknown"
                base_variant_dir = group_dir / f"layout_{layout_suffix}"
                base_variant_dir.mkdir(parents=True, exist_ok=True)

                memory = None
                if enable_memory:
                    memory_path = base_variant_dir / "memory.md"
                    memory = generator.create_layout_memory(
                        output_path=memory_path,
                        anchor_id=anchor_id,
                        layout_id=layout_suffix,
                    )

                # Get the initial layout summary and anchor category for the current layout entry
                summary, anchor_category = generator.generate_similar_layout_context(
                    mapping_path=mapping_path,
                    objects=objects,
                    anchor_id=anchor_id,
                )

                # For the original layout (org), skip VLM pose generation and use existing layout directly
                if layout_id == "org":
                    layout_log(f"Processing original layout as round 0 for group '{anchor_id}'")
                    round_records: list[dict] = []
                    round_index = 0
                    round_dir = base_variant_dir / f"round_{round_index}"
                    round_dir.mkdir(parents=True, exist_ok=True)

                    # Use the original layout as the starting point (no VLM generation needed)
                    source_layout_dir = artifacts.simulated_layout_dir or artifacts.layout_params_dir
                    current_layout_dir = round_dir / "layout_params"
                    current_layout_dir.mkdir(parents=True, exist_ok=True)

                    # Copy only the relevant objects for this anchor group
                    org_object_ids = set(object_ids) | {anchor_id}
                    for layout_file in source_layout_dir.glob("*.json"):
                        if layout_file.stem in org_object_ids:
                            shutil.copy2(layout_file, current_layout_dir / layout_file.name)

                    # Jump to the while True optimization loop
                    # The rest of the optimization loop will be executed below

                else:  # layout_id != "org"
                    # Generate 2D pose for the layout entry using VLM
                    pose_prompt = (
                        f"{pose_prompt_template if relation_type == 'attach' else pose_prompt_fix_rot_template}\n\n"
                        f"ANCHOR_CONSTRAINT: The anchor is {anchor_category or anchor_id} and must not change in the new layout.\n\n"
                        f"{cross_anchor_constraint_text}"
                        f"LAYOUT_DESCRIPTION:\n{json.dumps(layout_entry.get('layout_reasoning'), ensure_ascii=False, indent=2)}\n\n"
                        "CURRENT_LAYOUT_DATA (\"bbox_2d\": [min_u, min_v, max_u, max_v]):\n"
                        f"{json.dumps(summary, ensure_ascii=False, indent=2)}\n"
                    )

                    layout_log(f"Generating 2D pose for layout variant '{layout_entry.get('layout_id')}' of group '{anchor_id}'")
                    pose_response = self.vlm_client.plan(
                        pose_prompt,
                        images=[str(output_path)] if output_path.exists() else None,
                    )
                    pose_payload = self.vlm_client.extract_json_payload(pose_response)
                    pose_payload_output = {
                        "vlm_output": pose_payload,
                        "usage": self.vlm_client.last_usage or {},
                    }

                    if memory is not None:
                        initial_pose = generator.extract_2d_pose_with_rotation(pose_payload)
                        memory.record_initial_pose(initial_pose)

                    round_records = []
                    round_index = 0
                    round_dir = base_variant_dir / f"round_{round_index}"
                    pose_output_path = round_dir / f"layout_{layout_suffix}_2d_pose_r{round_index}.json"
                    pose_output_path.parent.mkdir(parents=True, exist_ok=True)

                    # Compute 3D pose from 2D pose and mapping data
                    vlm_output = generator.convert_2d_pose_to_3d(mapping_path, pose_payload_output.get("vlm_output", {}))
                    pose_payload_output["vlm_output"] = vlm_output
                    with pose_output_path.open("w", encoding="utf-8") as f:
                        json.dump(pose_payload_output, f, ensure_ascii=False, indent=2)

                    generator.generate_posed_layouts(
                        pose_output_path=pose_output_path,
                        pose_payload=vlm_output,
                        render_object_ids=render_object_ids,
                        camera_direction=camera_direction,
                        layout_dir=layout_dir,
                        anchor_id=anchor_id,
                        render_transform_path=mapping_path,
                    )

                    current_layout_dir = round_dir / "layout_params"
                    if not current_layout_dir.exists():
                        layout_log(f"Warning: Generated layout params not found for {layout_suffix} round {round_index}")
                        return {
                            "layout_suffix": None,
                            "logs": layout_logs,
                        }

                    source_layout_dir = current_layout_dir

                # Check collisions and render the layout for the first round
                layout_log(
                    f"Checking collisions and rendering layout variant {layout_entry.get('layout_id', '')} "
                    f"round {round_index} for group {anchor_id}"
                )
                (
                    current_layout_dir,
                    current_render_path,
                    current_mapping_path,
                    current_collision_pairs,
                ) = _check_collisions_and_render_round(
                    current_layout_dir,
                    round_dir,
                    round_index,
                    layout_suffix,
                )

                round_records.append(
                    {
                        "round_index": round_index,
                        "source_layout_dir": source_layout_dir,
                        "render_path": current_render_path,
                        "mapping_path": current_mapping_path,
                        "collision_pairs": current_collision_pairs,
                        "out_of_bounds_summary": generator.summarize_out_of_bounds_objects(
                            current_mapping_path,
                            anchor_id=anchor_id,
                        ),
                    }
                )

                # Initially detect out-of-bounds objects
                current_out_of_bounds_summary = round_records[-1]["out_of_bounds_summary"]
                current_out_of_bounds_notes = generator.format_out_of_bounds_modification_notes(
                    current_out_of_bounds_summary
                )

                if memory is not None:
                    layout_log(f"Recording initial pose and collisions for round {round_index}")
                    memory.record_round(
                        round_index,
                        collisions=current_collision_pairs,
                    )

                # Initialize memory context for VLM agents
                memory_context = ""
                if memory is not None and memory.output_path.exists():
                    memory_context = (
                        "LAYOUT_MEMORY:\n"
                        f"{memory.output_path.read_text(encoding='utf-8')}\n\n"
                    )

                while True:
                    if round_index + 1 >= max_refine_rounds:
                        layout_log(f"Reached max refinement rounds for layout {layout_suffix}")
                        break

                    # Evaluation step
                    eval_summary, _ = generator.generate_similar_layout_context(
                        mapping_path=current_mapping_path,
                        objects=objects,
                        anchor_id=anchor_id,
                    )
                    collision_context = ""
                    if current_collision_pairs:
                        collision_context = (
                            "POTENTIAL_DEEP_COLLISIONS:\n"
                            f"{json.dumps(current_collision_pairs, ensure_ascii=False, indent=2)}\n"
                            "These pairs may be in collision. Inspect the image and explicitly compute their 2D boundaries "
                            "to confirm deep collision (ignore planar objects like walls, rugs or paper sheets)."
                            "If confirmed, include them in modification_notes with separation guidance.\n\n"
                        )
                    eval_prompt = (
                        f"{evaluate_prompt_template}\n\n"
                        f"ANCHOR_CONSTRAINT: The anchor is {anchor_category or anchor_id} and must not change in the new layout.\n\n"
                        f"ROTATION_CONSTRAINT: Do not change the rotation of any object.\n\n" if relation_type != "attach" else ""
                        f"{cross_anchor_constraint_text}"
                        f"{collision_context}"
                        "CURRENT_LAYOUT_DATA (\"bbox_2d\": [min_u, min_v, max_u, max_v]):\n"
                        f"{json.dumps(eval_summary, ensure_ascii=False, indent=2)}\n\n"
                        f"{memory_context}"
                    )

                    layout_log(f"Evaluating layout variant {layout_entry.get('layout_id', '')} round {round_index}")
                    eval_response = self.vlm_client.plan(
                        eval_prompt,
                        images=[str(current_render_path)] if current_render_path.exists() else None,
                    )
                    eval_payload = self.vlm_client.extract_json_payload(eval_response)
                    eval_payload_output = {
                        "vlm_output": eval_payload,
                        "usage": self.vlm_client.last_usage or {},
                    }
                    eval_output_path = round_dir / f"layout_{layout_suffix}_evaluation_r{round_index}.json"
                    with eval_output_path.open("w", encoding="utf-8") as f:
                        json.dump(eval_payload_output, f, ensure_ascii=False, indent=2)

                    should_modify = eval_payload.get("should_modify")
                    if isinstance(should_modify, str):
                        should_modify = should_modify.strip().lower() in {"true", "yes", "y", "modify"}
                    if not isinstance(should_modify, bool):
                        decision = str(eval_payload.get("decision", "")).strip().lower()
                        should_modify = decision in {"modify", "yes", "true"}

                    modification_notes = eval_payload.get("modification_notes", [])
                    if isinstance(modification_notes, str):
                        modification_notes = [modification_notes]
                    if not isinstance(modification_notes, list):
                        modification_notes = []
                    modification_notes = [str(note) for note in modification_notes if str(note).strip()]
                    modification_notes = generator.merge_modification_notes(
                        modification_notes,
                        current_out_of_bounds_notes,
                    )

                    if current_out_of_bounds_notes:
                        should_modify = True

                    if memory is not None:
                        layout_log(f"Recording evaluation feedback for round {round_index}")
                        memory.record_round(round_index, modification_notes=modification_notes)

                    if not should_modify:
                        layout_log(f"No further modification requested for layout {layout_suffix}")
                        break

                    next_round_index = round_index + 1
                    next_round_dir = base_variant_dir / f"round_{next_round_index}"
                    next_round_dir.mkdir(parents=True, exist_ok=True)

                    revise_prompt = (
                        f"{revise_prompt_template}\n\n"
                        f"ANCHOR_CONSTRAINT: The anchor is {anchor_category or anchor_id} and must not change in the new layout.\n\n"
                        "CURRENT_LAYOUT_DATA (\"bbox_2d\": [min_u, min_v, max_u, max_v]):\n"
                        f"{json.dumps(eval_summary, ensure_ascii=False, indent=2)}\n\n"
                        "MODIFICATION_GUIDANCE:\n"
                        f"{json.dumps(modification_notes, ensure_ascii=False, indent=2)}\n\n"
                        f"{memory_context}"
                    )

                    layout_log(
                        f"Applying modification for layout variant {layout_entry.get('layout_id', '')} round {next_round_index}"
                    )
                    revise_response = self.vlm_client.plan(
                        revise_prompt,
                        images=[str(current_render_path)] if current_render_path.exists() else None,
                    )
                    revise_payload = self.vlm_client.extract_json_payload(revise_response)
                    revise_payload = generator.resolve_pose_expressions(
                        current_mapping_path,
                        revise_payload,
                        anchor_id=anchor_id,
                    )
                    revise_payload_output = {
                        "vlm_output": revise_payload,
                        "usage": self.vlm_client.last_usage or {},
                    }

                    if memory is not None:
                        layout_log(f"Recording revised pose for round {next_round_index}")
                        original_pose_expressions = copy.deepcopy(revise_payload.get("pose_expressions"))
                        revised_pose = generator.extract_2d_pose_with_rotation(revise_payload)
                        memory.record_round(
                            next_round_index,
                            revised_pose=revised_pose,
                            pose_expressions=original_pose_expressions,
                        )
                    revise_pose_path = (
                        next_round_dir / f"layout_{layout_suffix}_2d_pose_r{next_round_index}.json"
                    )

                    # Convert 2D pose into 3D using the latest grid mapping
                    revise_vlm_output = generator.convert_2d_pose_to_3d(
                        current_mapping_path, revise_payload_output.get("vlm_output", {})
                    )
                    revise_payload_output["vlm_output"] = revise_vlm_output
                    with revise_pose_path.open("w", encoding="utf-8") as f:
                        json.dump(revise_payload_output, f, ensure_ascii=False, indent=2)

                    generator.generate_posed_layouts(
                        pose_output_path=revise_pose_path,
                        pose_payload=revise_vlm_output,
                        render_object_ids=render_object_ids,
                        camera_direction=camera_direction,
                        layout_dir=current_layout_dir,
                        anchor_id=anchor_id,
                        render_transform_path=current_mapping_path,
                    )

                    current_layout_dir = next_round_dir / "layout_params"
                    if not current_layout_dir.exists():
                        layout_log(
                            f"Warning: Revised layout params not found for {layout_suffix} round {next_round_index}"
                        )
                        break

                    source_layout_dir = current_layout_dir
                    previous_mapping_path = current_mapping_path
                    round_index = next_round_index
                    round_dir = next_round_dir
                    (
                        current_layout_dir,
                        current_render_path,
                        current_mapping_path,
                        current_collision_pairs,
                    ) = _check_collisions_and_render_round(
                        current_layout_dir,
                        round_dir,
                        round_index,
                        layout_suffix,
                    )

                    if memory is not None:
                        layout_log(f"Recording simulated pose and collisions for round {round_index}")
                        memory.record_round(
                            round_index,
                            collisions=current_collision_pairs,
                        )

                    # Evaluate whether to early stop refinement based on collisions and out-of-bounds objects
                    current_out_of_bounds_summary = generator.summarize_out_of_bounds_objects(
                        current_mapping_path,
                        anchor_id=anchor_id,
                    )
                    current_out_of_bounds_notes = generator.format_out_of_bounds_modification_notes(
                        current_out_of_bounds_summary
                    )

                    round_records.append(
                        {
                            "round_index": round_index,
                            "source_layout_dir": source_layout_dir,
                            "render_path": current_render_path,
                            "mapping_path": current_mapping_path,
                            "collision_pairs": current_collision_pairs,
                            "out_of_bounds_summary": current_out_of_bounds_summary,
                        }
                    )

                    early_stop = generator.should_early_stop_refinement(
                        previous_mapping_path,
                        current_mapping_path,
                        anchor_id=anchor_id,
                        collision_pairs=current_collision_pairs,
                        out_of_bounds_summary=current_out_of_bounds_summary,
                        movement_threshold=refine_movement_threshold,
                        rotation_tolerance=refine_rotation_tolerance,
                    )
                    if early_stop.get("should_stop"):
                        layout_log(
                            "Early stop refinement: "
                            f"max_translation={early_stop.get('max_translation')}, "
                            f"max_rotation_delta={early_stop.get('max_rotation_delta')}, "
                            "no collisions and no out-of-bound objects."
                        )
                        break

                if not round_records:
                    layout_log(f"No valid round records generated for layout {layout_suffix}")
                    return {
                        "layout_suffix": None,
                        "logs": layout_logs,
                    }

                final_round_record = round_records[-1]
                if not generator._has_layout_issues(
                    final_round_record["collision_pairs"],
                    final_round_record["out_of_bounds_summary"],
                ):
                    selected_round_record = final_round_record
                else:
                    candidate_records = round_records[:-1]
                    if not candidate_records:
                        selected_round_record = final_round_record
                    else:
                        selected_round_record = max(
                            candidate_records,
                            key=lambda record: (
                                generator._score_layout_quality(
                                    record["collision_pairs"],
                                    record["out_of_bounds_summary"],
                                ),
                                record["round_index"],
                            ),
                        )

                selected_layout_dir = group_dir / f"layout_{layout_suffix}" / "layout_params"
                selected_layout_dir.mkdir(parents=True, exist_ok=True)
                shutil.copytree(
                    selected_round_record["source_layout_dir"],
                    selected_layout_dir,
                    dirs_exist_ok=True,
                )
                layout_log(f"selected_round_record: {selected_round_record['source_layout_dir']}")

                selected_simulated_layout_dir = selected_layout_dir.parent / "layout_params_simulated"
                selected_simulated_layout_dir.mkdir(parents=True, exist_ok=True)

                gravity = self._compute_gravity_for_anchor(anchor_id, walls_by_id)
                anchor_layout_path = selected_layout_dir / f"{anchor_id}.json"
                ground_object_ids = [anchor_id] if anchor_layout_path.exists() else []
                layout_log(
                    f"Simulating final selected layout {layout_suffix} for group {anchor_id} "
                    f"with gravity {gravity} and ground objects {ground_object_ids}"
                )
                self.tool_runner.simulate_scene_physics(
                    selected_layout_dir,
                    artifacts.collision_hulls_dir,
                    selected_simulated_layout_dir,
                    gravity=gravity,
                    object_ids=render_object_ids,
                    cpu_only=True,
                    use_ground_object=bool(ground_object_ids),
                    ground_object_ids=ground_object_ids or None,
                )

                selected_render_output_path = selected_layout_dir.parent / f"layout_{layout_suffix}_render.png"
                self.tool_runner.render_scene(
                    layout_dir=selected_simulated_layout_dir,
                    unposed_dir=unposed_dir,
                    output=selected_render_output_path,
                    object_ids=render_object_ids,
                    camera_direction=camera_direction,
                    radius=0.0,
                    image_size=1024,
                    add_grid=True,
                    draw_object_directions=draw_object_directions if relation_type == "attach" else False,
                    draw_object_bboxes=draw_object_bboxes,
                    anchor_id=render_anchor_id,
                    hide_object_ids=[anchor_id],
                    grid_mapping=mapping_path,
                    use_aabb_for_object_bboxes=True if relation_type != "attach" else False
                )

                return {
                    "layout_suffix": layout_suffix,
                    "logs": layout_logs,
                }

            valid_layout_entries = [layout_entry for layout_entry in layout_entries if isinstance(layout_entry, dict)]
            if valid_layout_entries:
                layout_workers = min(4, len(valid_layout_entries))
                layout_results: dict[int, dict | None] = {}
                with ThreadPoolExecutor(max_workers=layout_workers) as layout_executor:
                    layout_futures = {
                        layout_executor.submit(_process_layout_entry, layout_entry, anchor_id, output_path, mapping_path): index
                        for index, layout_entry in enumerate(valid_layout_entries)
                    }
                    for future in as_completed(layout_futures):
                        layout_index = layout_futures[future]
                        try:
                            layout_results[layout_index] = future.result()
                        except Exception as exc:
                            layout_entry = valid_layout_entries[layout_index]
                            layout_results[layout_index] = {
                                "layout_suffix": None,
                                "logs": [
                                    f"[{anchor_id}][layout:{layout_entry.get('layout_id', '')}] Failed to process layout for group '{anchor_id}': {exc}"
                                ],
                            }

                for layout_index, layout_entry in enumerate(valid_layout_entries):
                    layout_result = layout_results.get(layout_index) or {}
                    layout_logs = layout_result.get("logs", []) if isinstance(layout_result, dict) else []
                    for line in layout_logs:
                        group_logs.append(line)

                    layout_suffix = layout_result.get("layout_suffix") if isinstance(layout_result, dict) else None
                    if layout_suffix:
                        group_log(f"Completed processing layout {layout_suffix} for group '{anchor_id}'")
                    else:
                        group_log(f"Failed to process layout {layout_entry.get('layout_id', '')} for group '{anchor_id}'")

            return {
                "anchor_id": anchor_id,
                "num_layout_entries": len(layout_entries),
                "logs": group_logs,
            }

        group_outputs = []
        group_results: dict[str, dict | None] = {}
        with ThreadPoolExecutor(max_workers=min(8, max(1, len(object_groups)))) as executor:
            futures = {
                executor.submit(_process_group, anchor_id, group_info): anchor_id
                for anchor_id, group_info in object_groups.items()
            }
            for future in as_completed(futures):
                anchor_id = futures[future]
                group_results[anchor_id] = future.result()

        group_logs_path = generation_output_dir / "group_logs.txt"
        with group_logs_path.open("w", encoding="utf-8") as log_file:
            for anchor_id, _group_info in object_groups.items():
                result = group_results.get(anchor_id)
                if not result:
                    continue
                log_file.write(f"=== group {anchor_id} ===\n")
                for line in result.get("logs", []):
                    log_file.write(f"{line}\n")
                log_file.write("\n")
                if result.get("failed"):
                    continue
                group_outputs.append(
                    {
                        "anchor_id": anchor_id,
                        "num_layout_entries": result.get("num_layout_entries", 0),
                    }
                )

        print("Scene generation completed for all groups.")
        return group_outputs

    def run_full_pipeline(self) -> PipelineReport:
        """Execute optimization, generation, and composition for existing artifacts."""
        artifacts = SceneArtifacts(
            unposed_objects_dir=self.paths.unposed_objects_dir,
            layout_params_dir=self.paths.layout_params_dir,
            collision_hulls_dir=self.paths.collision_hulls_dir,
            composed_scene_mesh=self.paths.composed_scene_path,
            simulated_layout_dir=self.paths.simulated_layout_dir,
        )
        artifacts = self._stage_optimization(artifacts)
        _ = self._stage_generate_scene(artifacts, max_refine_rounds=self.max_rounds)

        combined_scene_paths = LayoutCombination(self.paths, self.tool_runner).run(artifacts)
        combined_scene_index = self.paths.outputs_dir / "generated" / "combined_scenes" / "combo_index.json"
        if combined_scene_index.exists():
            artifacts = replace(artifacts, composed_scene_mesh=combined_scene_index)
        report = PipelineReport(
            stage="composition",
            artifacts=artifacts,
            tool_outputs={
                "combined_scene_index": str(combined_scene_index) if combined_scene_index.exists() else "",
                "combined_scene_count": str(len(combined_scene_paths)),
            },
        )
        self.history.append(report)
        return report

    def save_report(self, report: PipelineReport, output_path: Path | None = None) -> Path:
        """Serialize the complete pipeline report to JSON."""
        if output_path is None:
            output_path = self.paths.outputs_dir / "pipeline_report.json"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "stage": report.stage,
            "artifacts": {
                "layout_params_dir": str(report.artifacts.layout_params_dir),
                "composed_scene_mesh": str(report.artifacts.composed_scene_mesh),
                "simulated_layout_dir": str(report.artifacts.simulated_layout_dir),
            },
        }
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        return output_path
