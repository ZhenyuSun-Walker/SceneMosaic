from __future__ import annotations

import ast
import json
import math
import random
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .types import SceneArtifacts


@dataclass
class LayoutMemory:
    output_path: Path
    anchor_id: str
    layout_id: str | None = None
    initial_pose: list[dict] = field(default_factory=list)
    rounds: list[dict] = field(default_factory=list)

    def record_initial_pose(self, pose: list[dict] | None) -> None:
        self.initial_pose = pose or []
        self.write()

    def record_round(
        self,
        round_index: int,
        simulated_pose: list[dict] | None = None,
        collisions: list[dict] | None = None,
        modification_notes: list[str] | None = None,
        revised_pose: list[dict] | None = None,
        pose_expressions: list | None = None,
    ) -> None:
        entry = None
        for candidate in self.rounds:
            if candidate.get("round_index") == round_index:
                entry = candidate
                break
        if entry is None:
            entry = {
                "round_index": round_index,
                "simulated_pose": None,
                "collisions": None,
                "modification_notes": None,
                "revised_pose": None,
                "pose_expressions": None,
            }
            self.rounds.append(entry)

        if simulated_pose is not None:
            entry["simulated_pose"] = simulated_pose
        if collisions is not None:
            entry["collisions"] = collisions
        if modification_notes is not None:
            entry["modification_notes"] = modification_notes
        if revised_pose is not None:
            entry["revised_pose"] = revised_pose
        if pose_expressions is not None:
            entry["pose_expressions"] = pose_expressions

        self.write()

    def _format_number(self, value: float | int | None) -> str:
        if value is None:
            return ""
        if isinstance(value, (int, float)):
            return str(round(float(value), 3))
        return str(value)

    def _format_pose_list(self, pose_list: list[dict] | None, include_rotation: bool) -> list[str]:
        if not pose_list:
            return ["- (none)"]
        lines = []
        for pose in pose_list:
            if not isinstance(pose, dict):
                continue
            object_id = pose.get("object_id")
            center = pose.get("center_2d")
            if not isinstance(center, list) or len(center) < 2:
                continue
            u_val, v_val = center[0], center[1]
            line = f"- {object_id}: u={self._format_number(u_val)}, v={self._format_number(v_val)}"
            if include_rotation:
                rotation = pose.get("rotation_degrees")
                if rotation is not None:
                    line += f", rotation_degrees={self._format_number(rotation)}"
            lines.append(line)
        return lines or ["- (none)"]

    def _format_collisions(self, collisions: list[dict] | None) -> list[str]:
        if not collisions:
            return ["- (none)"]
        lines = []
        for item in collisions:
            if not isinstance(item, dict):
                continue
            a_id = item.get("a")
            b_id = item.get("b")
            contacts = item.get("contacts")
            line = f"- {a_id} vs {b_id} (contacts={self._format_number(contacts)})"
            lines.append(line)
        return lines or ["- (none)"]

    def _format_notes(self, notes: list[str] | None) -> list[str]:
        if not notes:
            return ["- (none)"]
        return [f"- {note}" for note in notes]

    def _format_pose_expressions(self, expressions: list | None) -> list[str]:
        if not expressions:
            return ["- (none)"]
        lines = []
        for expression in expressions:
            if isinstance(expression, str):
                lines.append(f"- {expression}")
            else:
                lines.append(f"- {json.dumps(expression, ensure_ascii=False)}")
        return lines or ["- (none)"]

    def write(self) -> None:
        lines: list[str] = ["# Layout Memory", ""]
        lines.append(f"- anchor_id: {self.anchor_id}")
        if self.layout_id is not None:
            lines.append(f"- layout_id: {self.layout_id}")
        lines.append("")
        lines.append("## Initial 2D Pose")
        lines.extend(self._format_pose_list(self.initial_pose, include_rotation=True))

        sorted_rounds = sorted(self.rounds, key=lambda item: item.get("round_index", 0))
        for entry in sorted_rounds:
            round_index = entry.get("round_index")
            lines.append("")
            lines.append(f"## Round {round_index}")

            if entry.get("pose_expressions") is not None:
                lines.append("- Original Pose Expressions")
                lines.extend(self._format_pose_expressions(entry.get("pose_expressions")))

            if entry.get("revised_pose") is not None:
                lines.append("- Revised 2D Pose")
                lines.extend(self._format_pose_list(entry.get("revised_pose"), include_rotation=True))

            if entry.get("simulated_pose") is not None:
                lines.append("- 2D Pose after Simulation ")
                lines.extend(self._format_pose_list(entry.get("simulated_pose"), include_rotation=True))

            if entry.get("collisions") is not None:
                lines.append("- Possible Deep Collisions")
                lines.extend(self._format_collisions(entry.get("collisions")))

            if entry.get("modification_notes") is not None:
                lines.append("- Modification Notes for Next Round")
                lines.extend(self._format_notes(entry.get("modification_notes")))

        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.output_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


class SceneGenerator:
    """Generate new scenes based on learned relationships and original artifacts."""

    def __init__(self, original_artifacts: SceneArtifacts):
        self.original_artifacts = original_artifacts

    def _score_layout_quality(
        self,
        collision_pairs: list[dict] | None,
        out_of_bounds_summary: dict | None,
    ) -> int:
        collision_count = len(collision_pairs or [])
        out_of_bounds_count = 0
        if isinstance(out_of_bounds_summary, dict):
            entries = out_of_bounds_summary.get("out_of_bounds_objects", [])
            if isinstance(entries, list):
                out_of_bounds_count = len(entries)
        return max(0, 100 - 20 * (collision_count + out_of_bounds_count))

    def _has_layout_issues(
        self,
        collision_pairs: list[dict] | None,
        out_of_bounds_summary: dict | None,
    ) -> bool:
        if collision_pairs:
            return True
        if not isinstance(out_of_bounds_summary, dict):
            return False
        entries = out_of_bounds_summary.get("out_of_bounds_objects", [])
        return isinstance(entries, list) and bool(entries)

    def generate_similar_layout_context(
        self,
        mapping_path: Path | None = None,
        objects: list[dict] | None = None,
        anchor_id: str | None = None,
    ) -> dict:
        """Create the layout context for generate_layout_description prompt."""
        objects_by_id = {}
        if objects:
            for obj in objects:
                if not isinstance(obj, dict):
                    continue
                object_id = str(obj.get("id", ""))
                objects_by_id[object_id] = {
                    "category": obj.get("category", ""),
                    "description": obj.get("description", ""),
                }

        objects_context = {
            "objects": [],
        }

        def _round_numbers(value):
            if isinstance(value, float | int):
                return round(value, 2)
            if isinstance(value, list):
                return [_round_numbers(item) for item in value]
            if isinstance(value, dict):
                return {k: _round_numbers(v) for k, v in value.items()}
            return value

        anchor_category = None
        if mapping_path and mapping_path.exists():
            with mapping_path.open("r", encoding="utf-8") as f:
                mapping_payload = json.load(f)

            mapping_objects = mapping_payload.get("objects", [])
            if isinstance(mapping_objects, list):
                for obj_info in mapping_objects:
                    if not isinstance(obj_info, dict):
                        continue
                    object_id = str(obj_info.get("object_id", ""))
                    meta = objects_by_id.get(object_id)
                    if meta is None and object_id.startswith("object_"):
                        alt_id = f"obj_{object_id.split('_', 1)[1]}"
                        meta = objects_by_id.get(alt_id)
                    category = (meta or {}).get("category", object_id)
                    description = (meta or {}).get("description", object_id)

                    if anchor_id and object_id == anchor_id:
                        anchor_category = category
                        description += " (This object is the anchor and should not change in the new layout)"

                    obj_info = {k: v for k, v in obj_info.items() if k != "signed_distance_to_plane"}
                    obj_info["category"] = category
                    obj_info["description"] = description
                    objects_context["objects"].append(_round_numbers(obj_info))

        return {"objects_context": objects_context}, anchor_category

    def create_layout_memory(
        self,
        output_path: Path,
        anchor_id: str,
        layout_id: str | None = None,
    ) -> LayoutMemory:
        return LayoutMemory(output_path=output_path, anchor_id=anchor_id, layout_id=layout_id)

    def extract_2d_pose_centers(self, pose_payload: dict) -> list[dict]:
        return self._extract_2d_pose_entries(pose_payload, include_rotation=False)

    def extract_2d_pose_with_rotation(self, pose_payload: dict) -> list[dict]:
        return self._extract_2d_pose_entries(pose_payload, include_rotation=True)

    def extract_2d_pose_from_mapping(self, mapping_path: Path | None) -> list[dict]:
        if not mapping_path or not mapping_path.exists():
            return []
        with mapping_path.open("r", encoding="utf-8") as f:
            mapping_payload = json.load(f)

        objects = mapping_payload.get("objects", [])
        if not isinstance(objects, list):
            return []
        results = []
        for obj in objects:
            if not isinstance(obj, dict):
                continue
            center = obj.get("center_2d")
            if not isinstance(center, list) or len(center) < 2:
                continue
            object_id = obj.get("object_id")
            entry = {
                "object_id": object_id,
                "center_2d": [center[0], center[1]],
            }
            rotation = obj.get("rotation_degrees")
            if rotation is not None:
                entry["rotation_degrees"] = rotation
            results.append(entry)
        return results

    def _parse_bbox_2d(self, obj: dict) -> list[float] | None:
        bbox = obj.get("bbox_2d")
        if not isinstance(bbox, list) or len(bbox) < 4:
            return None
        try:
            return [float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])]
        except (TypeError, ValueError):
            return None

    def summarize_out_of_bounds_objects(
        self,
        mapping_path: Path | None,
        anchor_id: str | None,
    ) -> dict:
        if not mapping_path or not mapping_path.exists():
            return {
                "anchor_id": anchor_id,
                "anchor_bbox_2d": None,
                "out_of_bounds_objects": [],
            }

        with mapping_path.open("r", encoding="utf-8") as f:
            mapping_payload = json.load(f)

        objects = mapping_payload.get("objects", [])
        if not isinstance(objects, list):
            return {
                "anchor_id": anchor_id,
                "anchor_bbox_2d": None,
                "out_of_bounds_objects": [],
            }

        anchor_bbox = None
        for obj in objects:
            if not isinstance(obj, dict):
                continue
            if str(obj.get("object_id")) != str(anchor_id):
                continue
            anchor_bbox = self._parse_bbox_2d(obj)
            break

        summary = {
            "anchor_id": anchor_id,
            "anchor_bbox_2d": anchor_bbox,
            "out_of_bounds_objects": [],
        }
        if anchor_bbox is None:
            return summary

        anchor_min_u, anchor_min_v, anchor_max_u, anchor_max_v = anchor_bbox
        for obj in objects:
            if not isinstance(obj, dict):
                continue
            object_id = obj.get("object_id")
            if object_id is None or str(object_id) == str(anchor_id):
                continue

            bbox = self._parse_bbox_2d(obj)
            if bbox is None:
                continue

            min_u, min_v, max_u, max_v = bbox
            u_buffer = (max_u - min_u) / 2.0 if anchor_id.startswith("object_") else 0.0
            v_buffer = (max_v - min_v) / 2.0 if anchor_id.startswith("object_") else 0.0
            exceeded_sides = []
            if min_u < anchor_min_u - u_buffer - 1e-2:
                exceeded_sides.append("left")
            if min_v < anchor_min_v - v_buffer - 1e-2:
                exceeded_sides.append("bottom")
            if max_u > anchor_max_u + u_buffer + 1e-2:
                exceeded_sides.append("right")
            if max_v > anchor_max_v + v_buffer + 1e-2:
                exceeded_sides.append("top")

            if exceeded_sides:
                summary["out_of_bounds_objects"].append(
                    {
                        "object_id": object_id,
                        "bbox_2d": [min_u, min_v, max_u, max_v],
                        "exceeded_sides": exceeded_sides,
                    }
                )

        return summary

    def format_out_of_bounds_modification_notes(self, out_of_bounds_summary: dict | None) -> list[str]:
        if not isinstance(out_of_bounds_summary, dict):
            return []
        entries = out_of_bounds_summary.get("out_of_bounds_objects", [])
        if not isinstance(entries, list):
            return []

        notes: list[str] = []
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            object_id = entry.get("object_id")
            if not object_id:
                continue
            exceeded_sides = entry.get("exceeded_sides", [])
            if isinstance(exceeded_sides, list) and exceeded_sides:
                side_text = ", ".join(str(side) for side in exceeded_sides)
                notes.append(
                    f"Move ({object_id}) inward so its bbox stays inside the anchor boundary (exceeded: {side_text})."
                )
            else:
                notes.append(f"Move ({object_id}) inward so it stays inside the anchor boundary.")
        return notes

    def merge_modification_notes(
        self,
        base_notes: list[str] | None,
        out_of_bounds_notes: list[str] | None,
    ) -> list[str]:
        merged: list[str] = []
        for note_list in (base_notes or [], out_of_bounds_notes or []):
            for note in note_list:
                text = str(note).strip()
                if not text:
                    continue
                if text not in merged:
                    merged.append(text)
        return merged

    def _extract_mapping_pose_by_id(
        self,
        mapping_path: Path | None,
        anchor_id: str | None = None,
    ) -> dict[str, dict[str, float | None]]:
        if not mapping_path or not mapping_path.exists():
            return {}
        with mapping_path.open("r", encoding="utf-8") as f:
            mapping_payload = json.load(f)

        objects = mapping_payload.get("objects", [])
        if not isinstance(objects, list):
            return {}

        pose_by_id: dict[str, dict[str, float | None]] = {}
        for obj in objects:
            if not isinstance(obj, dict):
                continue
            object_id = obj.get("object_id")
            if object_id is None:
                continue
            object_id = str(object_id)
            if anchor_id is not None and object_id == str(anchor_id):
                continue

            center = obj.get("center_2d")
            if not isinstance(center, list) or len(center) < 2:
                continue
            try:
                u_val = float(center[0])
                v_val = float(center[1])
            except (TypeError, ValueError):
                continue

            rotation = obj.get("rotation_degrees")
            rotation_value = None
            if isinstance(rotation, (int, float)):
                rotation_value = float(rotation)

            pose_by_id[object_id] = {
                "u": u_val,
                "v": v_val,
                "rotation_degrees": rotation_value,
            }

        return pose_by_id

    def summarize_pose_change(
        self,
        previous_mapping_path: Path | None,
        current_mapping_path: Path | None,
        *,
        anchor_id: str | None = None,
    ) -> dict:
        prev_pose = self._extract_mapping_pose_by_id(previous_mapping_path, anchor_id=anchor_id)
        curr_pose = self._extract_mapping_pose_by_id(current_mapping_path, anchor_id=anchor_id)
        shared_ids = sorted(set(prev_pose.keys()) & set(curr_pose.keys()))

        deltas: list[dict] = []
        max_translation = 0.0
        max_rotation_delta = 0.0
        for object_id in shared_ids:
            prev_entry = prev_pose[object_id]
            curr_entry = curr_pose[object_id]

            dx = float(curr_entry["u"]) - float(prev_entry["u"])
            dy = float(curr_entry["v"]) - float(prev_entry["v"])
            translation = math.sqrt(dx * dx + dy * dy)
            max_translation = max(max_translation, translation)

            prev_rot = prev_entry.get("rotation_degrees")
            curr_rot = curr_entry.get("rotation_degrees")
            rotation_delta = 0.0
            if isinstance(prev_rot, (int, float)) and isinstance(curr_rot, (int, float)):
                rotation_delta = abs(float(curr_rot) - float(prev_rot))
            max_rotation_delta = max(max_rotation_delta, rotation_delta)

            deltas.append(
                {
                    "object_id": object_id,
                    "translation": translation,
                    "rotation_delta": rotation_delta,
                }
            )

        return {
            "object_count": len(shared_ids),
            "max_translation": max_translation,
            "max_rotation_delta": max_rotation_delta,
            "object_deltas": deltas,
        }

    def should_early_stop_refinement(
        self,
        previous_mapping_path: Path | None,
        current_mapping_path: Path | None,
        *,
        anchor_id: str | None,
        collision_pairs: list[dict] | None,
        out_of_bounds_summary: dict | None,
        movement_threshold: float,
        rotation_tolerance: float,
    ) -> dict:
        change_summary = self.summarize_pose_change(
            previous_mapping_path,
            current_mapping_path,
            anchor_id=anchor_id,
        )
        object_count = int(change_summary.get("object_count", 0))
        max_translation = float(change_summary.get("max_translation", 0.0))
        max_rotation_delta = float(change_summary.get("max_rotation_delta", 0.0))

        has_collision = bool(collision_pairs)
        out_of_bounds_objects = []
        if isinstance(out_of_bounds_summary, dict):
            raw = out_of_bounds_summary.get("out_of_bounds_objects", [])
            if isinstance(raw, list):
                out_of_bounds_objects = raw
        has_out_of_bounds = bool(out_of_bounds_objects)

        should_stop = (
            object_count > 0
            and max_rotation_delta <= float(rotation_tolerance)
            and max_translation < float(movement_threshold)
            and not has_collision
            and not has_out_of_bounds
        )

        return {
            "should_stop": should_stop,
            "object_count": object_count,
            "max_translation": max_translation,
            "max_rotation_delta": max_rotation_delta,
            "has_collision": has_collision,
            "has_out_of_bounds": has_out_of_bounds,
        }

    def _extract_2d_pose_entries(self, pose_payload: dict, include_rotation: bool) -> list[dict]:
        if not isinstance(pose_payload, dict):
            return []
        poses = pose_payload.get("poses", [])
        if not isinstance(poses, list):
            return []
        results = []
        for pose in poses:
            if not isinstance(pose, dict):
                continue
            coords = pose.get("center_coordinates", {})
            if not isinstance(coords, dict):
                continue
            u_val = coords.get("u")
            v_val = coords.get("v")
            if u_val is None or v_val is None:
                continue
            object_id = pose.get("object_id") or pose.get("id")
            entry = {
                "object_id": object_id,
                "center_2d": [u_val, v_val],
            }
            if include_rotation:
                rotation = pose.get("rotation_degrees")
                if rotation is not None:
                    entry["rotation_degrees"] = rotation
            results.append(entry)
        return results

    def _load_grid_mapping(self, mapping_path: Path) -> tuple[int | None, dict, dict]:
        grid_size = None
        grid_map = {}
        with mapping_path.open("r", encoding="utf-8") as f:
            mapping_payload = json.load(f)

        raw_grid_size = mapping_payload.get("grid_size")
        if isinstance(raw_grid_size, int):
            grid_size = raw_grid_size

        grid_points = mapping_payload.get("grid_points", [])
        if isinstance(grid_points, list):
            for point in grid_points:
                if not isinstance(point, dict):
                    continue
                i = point.get("grid_i")
                j = point.get("grid_j")
                p3d = point.get("point_3d_original")
                if isinstance(i, int) and isinstance(j, int) and isinstance(p3d, list) and len(p3d) == 3:
                    grid_map[(i, j)] = p3d

        return grid_size, grid_map, mapping_payload

    def _nearest_grid_point(self, grid_map: dict, u: float, v: float) -> list[float] | None:
        best = None
        best_dist = None
        for (i, j), point in grid_map.items():
            dist = (i - u) ** 2 + (j - v) ** 2
            if best_dist is None or dist < best_dist:
                best_dist = dist
                best = point
        return best

    def _interpolate_grid_point(
        self,
        grid_map: dict,
        u: float,
        v: float,
        max_index: int,
    ) -> list[float] | None:
        if max_index < 0:
            return None
        u_clamped = max(0.0, min(float(max_index), u))
        v_clamped = max(0.0, min(float(max_index), v))

        i0 = int(math.floor(u_clamped))
        j0 = int(math.floor(v_clamped))
        i1 = min(i0 + 1, max_index)
        j1 = min(j0 + 1, max_index)

        p00 = grid_map.get((i0, j0))
        p10 = grid_map.get((i1, j0))
        p01 = grid_map.get((i0, j1))
        p11 = grid_map.get((i1, j1))

        if i0 == i1 and j0 == j1:
            if p00:
                return p00
            return self._nearest_grid_point(grid_map, u_clamped, v_clamped)

        if j0 == j1 and p00 and p10:
            t = u_clamped - i0
            return [
                (1 - t) * p00[0] + t * p10[0],
                (1 - t) * p00[1] + t * p10[1],
                (1 - t) * p00[2] + t * p10[2],
            ]

        if i0 == i1 and p00 and p01:
            s = v_clamped - j0
            return [
                (1 - s) * p00[0] + s * p01[0],
                (1 - s) * p00[1] + s * p01[1],
                (1 - s) * p00[2] + s * p01[2],
            ]

        if not (p00 and p10 and p01 and p11):
            return self._nearest_grid_point(grid_map, u_clamped, v_clamped)

        t = u_clamped - i0
        s = v_clamped - j0

        return [
            (1 - t) * (1 - s) * p00[0] + t * (1 - s) * p10[0] + (1 - t) * s * p01[0] + t * s * p11[0],
            (1 - t) * (1 - s) * p00[1] + t * (1 - s) * p10[1] + (1 - t) * s * p01[1] + t * s * p11[1],
            (1 - t) * (1 - s) * p00[2] + t * (1 - s) * p10[2] + (1 - t) * s * p01[2] + t * s * p11[2],
        ]

    def _quaternion_from_axis_angle(self, axis: list, angle_degrees: float) -> list:
        angle_rad = math.radians(angle_degrees)
        half_angle = angle_rad / 2.0
        sin_half = math.sin(half_angle)
        cos_half = math.cos(half_angle)
        
        w = cos_half
        x = axis[0] * sin_half
        y = axis[1] * sin_half
        z = axis[2] * sin_half
        
        return [w, x, y, z]

    def _camera_axes(self, camera_direction: list) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        forward = np.asarray(camera_direction, dtype=np.float32).reshape(3)
        norm = np.linalg.norm(forward)
        if norm > 1e-6:
            forward = forward / norm
        up_hint = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        right = np.cross(up_hint, forward)
        if np.linalg.norm(right) <= 1e-8:
            up_hint = np.array([0.0, 0.0, 1.0], dtype=np.float32)
            right = np.cross(up_hint, forward)
        right = right / np.linalg.norm(right)
        up = np.cross(forward, right)
        up = up / np.linalg.norm(up)
        return right, up, forward

    def _quaternion_to_matrix(self, quat: list | np.ndarray) -> np.ndarray:
        q = np.asarray(quat, dtype=np.float32).reshape(4)
        norm = np.linalg.norm(q)
        if norm > 1e-8:
            q = q / norm
        w, x, y, z = q
        return np.array(
            [
                [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
            ],
            dtype=np.float32,
        )

    def _matrix_to_quaternion(self, matrix: np.ndarray) -> list:
        m = np.asarray(matrix, dtype=np.float32)
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

    def _euler_from_matrix_xyz(self, matrix: np.ndarray) -> tuple[float, float, float]:
        r00, r10, r20 = matrix[0, 0], matrix[1, 0], matrix[2, 0]
        r21, r22 = matrix[2, 1], matrix[2, 2]
        if r20 < -0.995:
            x = 0.0
            y = math.pi / 2
            z = math.atan2(matrix[0, 1], matrix[1, 1])
        elif r20 > 0.995:
            x = 0.0
            y = -math.pi / 2
            z = -math.atan2(matrix[0, 1], matrix[1, 1])
        else:
            x = math.atan2(r21, r22)
            y = math.asin(-r20)
            z = math.atan2(r10, r00)
        return x, y, z

    def _matrix_from_euler_xyz(self, x: float, y: float, z: float) -> np.ndarray:
        cx, sx = math.cos(x), math.sin(x)
        cy, sy = math.cos(y), math.sin(y)
        cz, sz = math.cos(z), math.sin(z)

        return np.array(
            [
                [cz * cy, cz * sy * sx - sz * cx, cz * sy * cx + sz * sx],
                [sz * cy, sz * sy * sx + cz * cx, sz * sy * cx - cz * sx],
                [-sy, cy * sx, cy * cx],
            ],
            dtype=np.float32,
        )

    def update_rotation_about_camera_direction(
        self,
        rotation_degrees: float,
        original_quaternion: list,
        camera_direction: list,
    ) -> list:
        """Update the Euler angle around camera_direction and return a new quaternion."""
        quat = np.asarray(original_quaternion, dtype=np.float32).reshape(-1)
        if quat.size != 4:
            raise ValueError("Original rotation must have 4 elements.")

        right, up, forward = self._camera_axes(camera_direction)
        basis = np.stack([right, up, forward], axis=1)
        rotation_world = self._quaternion_to_matrix(quat)
        rotation_local = basis.T @ rotation_world @ basis
        euler_x, euler_y, _ = self._euler_from_matrix_xyz(rotation_local)
        euler_z = math.radians(float(rotation_degrees))
        rotation_local = self._matrix_from_euler_xyz(euler_x, euler_y, euler_z)
        rotation_world = basis @ rotation_local @ basis.T
        return self._matrix_to_quaternion(rotation_world)

    def generate_posed_layouts(
        self,
        pose_output_path: Path,
        pose_payload: dict,
        render_object_ids: list,
        camera_direction: list,
        layout_dir: Path,
        anchor_id: str,
        render_transform_path: Path | None = None,
    ) -> None:
        """Generate layout JSON files from 2D pose estimates.
        
        Args:
            pose_output_path: Path to the 2D pose output JSON file
            pose_payload: Parsed pose payload dict
            render_object_ids: List of object IDs that were rendered
            camera_direction: Camera direction vector for this anchor
            layout_dir: Source layout directory (layout_params_simulated)
            anchor_id: ID of the anchor object
        """
        layout_output_dir = pose_output_path.parent / "layout_params"
        layout_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Extract pose list from payload
        poses = pose_payload.get("poses", [])
        if not isinstance(poses, list):
            return
        
        # Build a mapping of pose object_id to pose data
        pose_map = {}
        for pose in poses:
            if isinstance(pose, dict):
                obj_id = pose.get("object_id")
                if obj_id:
                    pose_map[str(obj_id)] = pose
        
        # Normalize camera direction
        camera_dir_array = np.array(camera_direction, dtype=np.float32)
        camera_dir_norm = np.linalg.norm(camera_dir_array)
        if camera_dir_norm > 1e-6:
            camera_direction_normalized = camera_dir_array / camera_dir_norm
        else:
            camera_direction_normalized = camera_dir_array
        
        # Align object rotation with the normalized scene coordinate system in the rendering script
        rot_fix = 0.0
        if abs(camera_direction_normalized[0]) < 1e-6 and abs(camera_direction_normalized[2]) < 1e-6:
            if render_transform_path and render_transform_path.exists():
                with render_transform_path.open("r", encoding="utf-8") as f:
                    render_transform = json.load(f)

                transform_info = render_transform.get("transform_info", {})
                rotation_matrix = transform_info.get("rotation_matrix_col_major")
                if isinstance(rotation_matrix, list) and len(rotation_matrix) == 3:
                    rot_mat = np.asarray(rotation_matrix, dtype=np.float32).reshape(3, 3)
                    forward_vec = rot_mat @ np.array([0.0, 0.0, 1.0], dtype=np.float32)
                    rot_fix = math.degrees(math.atan2(float(forward_vec[0]), float(forward_vec[2])))

        # Process each object in render_object_ids
        for obj_id in render_object_ids:
            obj_id_str = str(obj_id)
            
            # Load source layout from layout_dir
            source_layout_path = layout_dir / f"{obj_id_str}.json"
            if not source_layout_path.exists():
                print(f"Warning: Source layout not found for {obj_id_str}")
                continue
            
            with source_layout_path.open("r", encoding="utf-8") as f:
                source_layout = json.load(f)
            
            # If object not in pose_map, it means it wasn't moved (anchor case)
            if obj_id_str not in pose_map or obj_id_str == anchor_id:
                # Copy anchor layout as-is
                output_layout_path = layout_output_dir / f"{obj_id_str}.json"
                with output_layout_path.open("w", encoding="utf-8") as f:
                    json.dump(source_layout, f, ensure_ascii=False, indent=2)
                continue
            
            pose_data = pose_map[obj_id_str]
            
            # Extract 3D coordinates from pose
            coords_3d = pose_data.get("center_coordinates_3d", {})
            if not isinstance(coords_3d, dict):
                print(f"Warning: Invalid 3D coordinates for {obj_id_str}")
                continue
            
            x = coords_3d.get("x")
            y = coords_3d.get("y")
            z = coords_3d.get("z")
            
            if x is None or y is None or z is None:
                print(f"Warning: Incomplete 3D coordinates for {obj_id_str}")
                continue
            
            # Extract rotation in degrees
            rotation_degrees = pose_data.get("rotation_degrees", None)
            
            # Create new layout with source rotation/scale
            original_rotation = source_layout.get("rotation", [[1, 0, 0, 0]])
            new_layout = {
                "translation": [[x, y, z]],
                "rotation": original_rotation,
                "scale": source_layout.get("scale", [[1, 1, 1]]),
            }
            
            # Apply rotation
            if rotation_degrees is not None:
                # Get original rotation quaternion
                if isinstance(original_rotation, list) and len(original_rotation) > 0:                    
                    if len(original_rotation) == 4 and all(
                        isinstance(val, (int, float)) for val in original_rotation
                    ):
                        orig_quat = original_rotation
                    else:
                        orig_quat = original_rotation[0]
                else:
                    orig_quat = [1, 0, 0, 0]
                
                try:
                    new_quat = self.update_rotation_about_camera_direction(
                        rotation_degrees + rot_fix,
                        orig_quat,
                        camera_direction_normalized.tolist(),
                    )
                except ValueError:
                    print(f"Warning: Invalid original quaternion for {obj_id_str}, skipping rotation")
                    new_quat = orig_quat

                new_layout["rotation"] = [new_quat]
            else:
                new_layout["rotation"] = original_rotation
            
            # Save new layout
            output_layout_path = layout_output_dir / f"{obj_id_str}.json"
            with output_layout_path.open("w", encoding="utf-8") as f:
                json.dump(new_layout, f, ensure_ascii=False, indent=2)

    def convert_2d_pose_to_3d(self, mapping_path: Path | None, pose_payload: dict) -> dict:
        if not mapping_path or not mapping_path.exists():
            return pose_payload
        if not isinstance(pose_payload, dict):
            return pose_payload

        poses = pose_payload.get("poses")
        if not isinstance(poses, list) or not poses:
            return pose_payload

        grid_size, grid_map, mapping_payload = self._load_grid_mapping(mapping_path)
        if not grid_map:
            return pose_payload

        if isinstance(grid_size, int) and grid_size > 1:
            max_index = grid_size - 1
        else:
            max_i = max(i for i, _ in grid_map.keys())
            max_j = max(j for _, j in grid_map.keys())
            max_index = max(max_i, max_j)

        signed_distance_by_id = {}
        mapping_objects = mapping_payload.get("objects", [])
        if isinstance(mapping_objects, list):
            for obj in mapping_objects:
                if not isinstance(obj, dict):
                    continue
                object_id = obj.get("object_id")
                signed_distance = obj.get("signed_distance_to_plane")
                if object_id is not None and isinstance(signed_distance, (int, float)):
                    signed_distance_by_id[str(object_id)] = float(signed_distance)

        plane_normal = None
        plane_info = mapping_payload.get("plane", {})
        if isinstance(plane_info, dict):
            normal = plane_info.get("normal")
            if isinstance(normal, list) and len(normal) == 3:
                plane_normal = np.asarray(normal, dtype=np.float32)
                norm = np.linalg.norm(plane_normal)
                if norm > 1e-6:
                    plane_normal = plane_normal / norm
                else:
                    plane_normal = None

        for pose in poses:
            if not isinstance(pose, dict):
                continue
            coords = pose.get("center_coordinates", {})
            if not isinstance(coords, dict):
                continue
            u = coords.get("u")
            v = coords.get("v")
            try:
                u_val = float(u)
                v_val = float(v)
            except (TypeError, ValueError):
                continue

            point_3d = self._interpolate_grid_point(grid_map, u_val, v_val, max_index)
            if not point_3d:
                continue

            object_id = pose.get("object_id") or pose.get("id")
            signed_distance = None
            if object_id is not None:
                signed_distance = signed_distance_by_id.get(str(object_id))

            if signed_distance is not None and plane_normal is not None:
                point_3d = [
                    point_3d[0] + plane_normal[0] * signed_distance,
                    point_3d[1] + plane_normal[1] * signed_distance,
                    point_3d[2] + plane_normal[2] * signed_distance,
                ]

            pose["center_coordinates_3d"] = {
                "x": point_3d[0],
                "y": point_3d[1],
                "z": point_3d[2],
            }
        
        return pose_payload

    _POSE_REF_PATTERN = re.compile(r"[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)+")

    def _build_pose_context(self, mapping_payload: dict) -> dict:
        context = {}
        objects = mapping_payload.get("objects", [])
        if not isinstance(objects, list):
            return context
        for obj in objects:
            if not isinstance(obj, dict):
                continue
            object_id = obj.get("object_id")
            if object_id is None:
                continue
            object_id = str(object_id)
            entry = {}

            center = obj.get("center_2d")
            if isinstance(center, list) and len(center) >= 2:
                entry["center"] = {
                    "u": float(center[0]),
                    "v": float(center[1]),
                }

            size = obj.get("size_2d")
            if isinstance(size, list) and len(size) >= 2:
                entry["size"] = {
                    "u": float(size[0]),
                    "v": float(size[1]),
                }

            bbox = obj.get("bbox_2d")
            if isinstance(bbox, list) and len(bbox) >= 4:
                entry["bbox"] = {
                    "min_u": float(bbox[0]),
                    "min_v": float(bbox[1]),
                    "max_u": float(bbox[2]),
                    "max_v": float(bbox[3]),
                }

            rotation_degrees = obj.get("rotation_degrees")
            if isinstance(rotation_degrees, (int, float)):
                entry["rotation_degrees"] = float(rotation_degrees)

            if entry:
                context[object_id] = entry

        return context

    def _normalize_pose_section(self, section: str) -> str:
        section = section.strip().lower()
        if section in {"center", "center_2d", "center2d"}:
            return "center"
        if section in {"size", "size_2d", "size2d"}:
            return "size"
        if section in {"bbox", "bbox_2d", "bbox2d"}:
            return "bbox"
        if section in {"rotation", "rotation_degrees"}:
            return "rotation_degrees"
        return section

    def _lookup_pose_reference(self, reference: str, pose_context: dict) -> float | None:
        parts = [part.strip() for part in reference.split(".") if part.strip()]
        if len(parts) < 2:
            return None
        object_id = parts[0]
        obj = pose_context.get(object_id)
        if not isinstance(obj, dict):
            return None

        if len(parts) == 2:
            key = self._normalize_pose_section(parts[1])
            if key == "rotation_degrees":
                value = obj.get("rotation_degrees")
                return float(value) if isinstance(value, (int, float)) else None
            return None

        section = self._normalize_pose_section(parts[1])
        key = parts[2]
        if section == "center":
            value = (obj.get("center") or {}).get(key)
            return float(value) if isinstance(value, (int, float)) else None
        if section == "size":
            value = (obj.get("size") or {}).get(key)
            return float(value) if isinstance(value, (int, float)) else None
        if section == "bbox":
            value = (obj.get("bbox") or {}).get(key)
            return float(value) if isinstance(value, (int, float)) else None
        if section == "rotation_degrees":
            value = obj.get("rotation_degrees")
            return float(value) if isinstance(value, (int, float)) else None
        return None

    def _parse_pose_target(self, target: str) -> tuple[str, str, str | None] | None:
        if not target:
            return None
        parts = [part.strip() for part in target.split(".") if part.strip()]
        if len(parts) < 2:
            return None
        object_id = parts[0]

        if len(parts) == 2:
            section = self._normalize_pose_section(parts[1])
            if section == "rotation_degrees":
                return object_id, "rotation_degrees", None
            return None

        section = self._normalize_pose_section(parts[1])
        if section == "center":
            axis = parts[2]
            if axis in {"u", "v"}:
                return object_id, "center", axis
        if section == "rotation_degrees":
            return object_id, "rotation_degrees", None
        return None

    def _parse_pose_assignment(self, statement: object) -> tuple[str, str] | None:
        target = None
        expression = None

        if isinstance(statement, dict):
            target = statement.get("target") or statement.get("lhs") or statement.get("left")
            expression = statement.get("expression") or statement.get("rhs") or statement.get("value")
        elif isinstance(statement, str):
            raw = statement.strip().rstrip(";")
            if "+=" in raw:
                left, right = raw.split("+=", 1)
                target = left.strip()
                expression = f"{target} + ({right.strip()})"
            elif "-=" in raw:
                left, right = raw.split("-=", 1)
                target = left.strip()
                expression = f"{target} - ({right.strip()})"
            elif "=" in raw:
                left, right = raw.split("=", 1)
                target = left.strip()
                expression = right.strip()

        if not target or not expression:
            return None
        return str(target), str(expression)

    def _rewrite_pose_expression(self, expression: str, pose_context: dict) -> tuple[str | None, dict | None]:
        variables: dict[str, float] = {}
        missing_reference = False

        def _replace(match: re.Match) -> str:
            nonlocal missing_reference
            reference = match.group(0)
            value = self._lookup_pose_reference(reference, pose_context)
            if value is None:
                missing_reference = True
                return "0"
            key = f"ref_{len(variables)}"
            variables[key] = float(value)
            return key

        rewritten = self._POSE_REF_PATTERN.sub(_replace, expression)
        if missing_reference:
            return None, None
        return rewritten, variables

    def _safe_eval_expression(self, expression: str, variables: dict[str, float]) -> float | None:
        try:
            tree = ast.parse(expression, mode="eval")
        except SyntaxError:
            return None

        def _eval(node: ast.AST) -> float:
            if isinstance(node, ast.Expression):
                return _eval(node.body)
            if isinstance(node, ast.Constant):
                if isinstance(node.value, (int, float)):
                    return float(node.value)
                raise ValueError("Unsupported constant")
            if isinstance(node, ast.Name):
                if node.id in variables:
                    return float(variables[node.id])
                raise ValueError("Unknown variable")
            if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
                operand = _eval(node.operand)
                return operand if isinstance(node.op, ast.UAdd) else -operand
            if isinstance(node, ast.BinOp) and isinstance(
                node.op, (ast.Add, ast.Sub, ast.Mult, ast.Div)
            ):
                left = _eval(node.left)
                right = _eval(node.right)
                if isinstance(node.op, ast.Add):
                    return left + right
                if isinstance(node.op, ast.Sub):
                    return left - right
                if isinstance(node.op, ast.Mult):
                    return left * right
                if isinstance(node.op, ast.Div):
                    return left / right
            raise ValueError("Unsupported expression")

        try:
            return float(_eval(tree))
        except Exception:
            return None

    def _update_center(self, obj_entry: dict, axis: str, value: float) -> None:
        center = obj_entry.get("center") or {}
        previous_u = center.get("u")
        previous_v = center.get("v")
        if axis == "u":
            center["u"] = float(value)
        elif axis == "v":
            center["v"] = float(value)
        obj_entry["center"] = center

        if previous_u is None or previous_v is None:
            return
        new_u = center.get("u")
        new_v = center.get("v")
        if not isinstance(new_u, (int, float)) or not isinstance(new_v, (int, float)):
            return

        dx = float(new_u) - float(previous_u)
        dy = float(new_v) - float(previous_v)
        bbox = obj_entry.get("bbox")
        if isinstance(bbox, dict):
            for key, delta in (("min_u", dx), ("max_u", dx), ("min_v", dy), ("max_v", dy)):
                if key in bbox and isinstance(bbox[key], (int, float)):
                    bbox[key] = float(bbox[key]) + delta
            obj_entry["bbox"] = bbox
        elif isinstance(obj_entry.get("size"), dict):
            size = obj_entry["size"]
            half_u = float(size.get("u", 0.0)) / 2.0
            half_v = float(size.get("v", 0.0)) / 2.0
            obj_entry["bbox"] = {
                "min_u": float(new_u) - half_u,
                "min_v": float(new_v) - half_v,
                "max_u": float(new_u) + half_u,
                "max_v": float(new_v) + half_v,
            }

    def resolve_pose_expressions(
        self,
        mapping_path: Path | None,
        pose_payload: dict,
        anchor_id: str | None = None,
    ) -> dict:
        if not mapping_path or not mapping_path.exists():
            return pose_payload
        if not isinstance(pose_payload, dict):
            return pose_payload

        pose_expressions = pose_payload.get("pose_expressions")
        if not pose_expressions:
            if "poses" in pose_payload:
                pose_payload["poses"] = []
            return pose_payload

        with mapping_path.open("r", encoding="utf-8") as f:
            mapping_payload = json.load(f)

        pose_context = self._build_pose_context(mapping_payload)
        original_rotation_by_object = {
            object_id: obj_entry.get("rotation_degrees")
            for object_id, obj_entry in pose_context.items()
            if isinstance(obj_entry, dict)
        }
        updated_objects: set[str] = set()
        rotation_updates: set[str] = set()

        if isinstance(pose_expressions, list):
            for statement in pose_expressions:
                parsed = self._parse_pose_assignment(statement)
                if not parsed:
                    continue
                target, expression = parsed
                target_info = self._parse_pose_target(target)
                if not target_info:
                    continue
                object_id, field, axis = target_info
                if anchor_id and object_id == anchor_id:
                    continue
                if object_id not in pose_context:
                    continue

                rewritten, variables = self._rewrite_pose_expression(expression, pose_context)
                if rewritten is None or variables is None:
                    continue
                value = self._safe_eval_expression(rewritten, variables)
                if value is None:
                    continue

                obj_entry = pose_context.get(object_id)
                if not isinstance(obj_entry, dict):
                    continue
                if field == "center" and axis in {"u", "v"}:
                    self._update_center(obj_entry, axis, float(value))
                    updated_objects.add(object_id)
                elif field == "rotation_degrees":
                    obj_entry["rotation_degrees"] = float(value)
                    updated_objects.add(object_id)
                    rotation_updates.add(object_id)

        if not updated_objects:
            return pose_payload

        resolved_poses = []
        for object_id in sorted(updated_objects):
            obj_entry = pose_context.get(object_id) or {}
            center = obj_entry.get("center") or {}
            if not isinstance(center.get("u"), (int, float)) or not isinstance(center.get("v"), (int, float)):
                continue
            entry = {
                "object_id": object_id,
                "center_coordinates": {
                    "u": float(center["u"]),
                    "v": float(center["v"]),
                },
            }
            if object_id in rotation_updates:
                rotation = obj_entry.get("rotation_degrees")
                if isinstance(rotation, (int, float)):
                    entry["rotation_degrees"] = float(rotation)
            elif isinstance(original_rotation_by_object.get(object_id), (int, float)):
                entry["rotation_degrees"] = float(original_rotation_by_object[object_id])
            resolved_poses.append(entry)

        pose_payload["poses"] = resolved_poses
        return pose_payload
