from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ToolRunResult:
    command: list[str]
    returncode: int
    stdout: str
    stderr: str
    failed_object_ids: tuple[str, ...] = ()
    out_of_bounds_object_ids: tuple[str, ...] = ()


class ToolRunner:
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.tools_dir = Path(__file__).resolve().parent / "tools"
        self._build_collision_hulls_summary_prefix = "BUILD_COLLISION_HULLS_SUMMARY:"
        self._fix_layout_containment_summary_prefix = "FIX_LAYOUT_CONTAINMENT_SUMMARY:"
        self._retrieve_room_materials_summary_prefix = "ROOM_MATERIAL_RETRIEVAL_SUMMARY:"

    @staticmethod
    def _load_object_ids(unposed_dir: Path) -> list[str]:
        return sorted(
            p.stem.split("_")[-1]
            for p in unposed_dir.glob("object_*.glb")
            if p.stem.startswith("object_")
        )

    @staticmethod
    def _load_collision_object_ids(output_dir: Path) -> list[str]:
        return sorted(
            p.stem.split("_")[-1]
            for p in output_dir.glob("object_*.obj")
            if p.stem.startswith("object_")
        )

    def _missing_collision_object_ids(self, unposed_dir: Path, output_dir: Path) -> list[str]:
        expected_ids = self._load_object_ids(unposed_dir)
        existing_ids = set(self._load_collision_object_ids(output_dir))
        return [object_id for object_id in expected_ids if object_id not in existing_ids]

    def _run_script(self, script_name: str, args: list[str]) -> ToolRunResult:
        script_path = self.tools_dir / script_name
        command = [sys.executable, str(script_path), *args]
        try:
            completed = subprocess.run(command, capture_output=True, text=True, check=False)
        except Exception as exc:
            stderr = f"Failed to run {script_name}: {exc}"
            print(stderr, file=sys.stderr)
            return ToolRunResult(command=command, returncode=1, stdout="", stderr=stderr)

        result = ToolRunResult(
            command=command,
            returncode=completed.returncode,
            stdout=completed.stdout,
            stderr=completed.stderr,
        )
        if result.returncode != 0:
            print(f"[{script_name}] failed with exit code {result.returncode}", file=sys.stderr)
            if result.stdout:
                print(result.stdout, file=sys.stderr, end="" if result.stdout.endswith("\n") else "\n")
            if result.stderr:
                print(result.stderr, file=sys.stderr, end="" if result.stderr.endswith("\n") else "\n")
        return result

    def _extract_build_collision_hulls_failed_object_ids(self, stdout: str) -> tuple[str, ...]:
        for line in reversed(stdout.splitlines()):
            if not line.startswith(self._build_collision_hulls_summary_prefix):
                continue
            payload_text = line[len(self._build_collision_hulls_summary_prefix) :].strip()
            if not payload_text:
                return ()
            try:
                payload = json.loads(payload_text)
            except json.JSONDecodeError:
                return ()
            failed_object_ids = payload.get("failed_object_ids", []) if isinstance(payload, dict) else []
            if not isinstance(failed_object_ids, list):
                return ()
            return tuple(str(object_id) for object_id in failed_object_ids)
        return ()

    def _extract_fix_layout_containment_out_of_bounds_object_ids(self, stdout: str) -> tuple[str, ...]:
        for line in reversed(stdout.splitlines()):
            if not line.startswith(self._fix_layout_containment_summary_prefix):
                continue
            payload_text = line[len(self._fix_layout_containment_summary_prefix) :].strip()
            if not payload_text:
                return ()
            try:
                payload = json.loads(payload_text)
            except json.JSONDecodeError:
                return ()
            out_of_bounds_object_ids = payload.get("out_of_bounds_object_ids", []) if isinstance(payload, dict) else []
            if not isinstance(out_of_bounds_object_ids, list):
                return ()
            return tuple(str(object_id) for object_id in out_of_bounds_object_ids)
        return ()

    def _run_build_collision_hulls_once(
        self,
        unposed_dir: Path,
        output_dir: Path,
        manifest: Path,
        *,
        object_ids: list[str] | None = None,
        workers: int | None = None,
        target_faces_per_part: int | None = None,
        split_mode: str | None = "none",
    ) -> ToolRunResult:
        args = ["--unposed_dir", str(unposed_dir), "--output_dir", str(output_dir), "--manifest", str(manifest)]
        if workers is not None:
            args.extend(["--workers", str(workers)])
        if target_faces_per_part is not None:
            args.extend(["--target_faces_per_part", str(target_faces_per_part)])
        if split_mode is not None:
            args.extend(["--split_mode", str(split_mode)])
        if object_ids:
            args.append("--object_ids")
            args.extend(str(object_id) for object_id in object_ids)
        result = self._run_script("build_collision_hulls.py", args)
        return ToolRunResult(
            command=result.command,
            returncode=result.returncode,
            stdout=result.stdout,
            stderr=result.stderr,
            failed_object_ids=self._extract_build_collision_hulls_failed_object_ids(result.stdout),
        )

    def compose_scene(self, layout_dir: Path, unposed_dir: Path, output: Path) -> ToolRunResult:
        return self._run_script(
            "compose_scene_from_layout.py",
            ["--layout_dir", str(layout_dir), "--unposed_dir", str(unposed_dir), "--output", str(output)],
        )

    def _extract_room_material_dirs(self, stdout: str) -> tuple[Path, Path, Path] | None:
        for line in reversed(stdout.splitlines()):
            if not line.startswith(self._retrieve_room_materials_summary_prefix):
                continue
            payload_text = line[len(self._retrieve_room_materials_summary_prefix) :].strip()
            try:
                payload = json.loads(payload_text)
            except json.JSONDecodeError:
                return None
            if not isinstance(payload, dict) or payload.get("status") != "resolved":
                return None
            paths = (
                payload.get("wall_material_dir"),
                payload.get("floor_material_dir"),
                payload.get("ceiling_material_dir"),
            )
            if not all(isinstance(path, str) and path for path in paths):
                return None
            return tuple(Path(path) for path in paths)
        return None

    def retrieve_room_materials(self, objects_json: Path) -> tuple[Path, Path, Path] | None:
        result = self._run_script(
            "retrieve_room_materials.py",
            [
                "--objects_json",
                str(objects_json),
                "--materials_dir",
                str(self.project_root / "data" / "materials"),
                "--embeddings_dir",
                str(self.project_root / "data" / "embeddings"),
            ],
        )
        material_dirs = self._extract_room_material_dirs(result.stdout)
        if material_dirs is not None:
            print("Resolved room materials for walls, floor, and ceiling.")
            return material_dirs
        if result.returncode == 0:
            print("No room_materials found; using legacy room material.")
        else:
            print("Room material retrieval failed; using legacy room material.", file=sys.stderr)
        return None

    def generate_room_mesh(
        self,
        layout_dir: Path,
        unposed_dir: Path,
        objects_json: Path,
        *,
        material_dir: Path | None = None,
        thickness: float | None = None,
        wall_padding: float | None = None,
    ) -> ToolRunResult:
        args = [
            "--layout_dir",
            str(layout_dir),
            "--unposed_dir",
            str(unposed_dir),
            "--objects_json",
            str(objects_json),
        ]
        material_dirs = self.retrieve_room_materials(objects_json)
        if material_dirs is not None:
            wall_material_dir, floor_material_dir, ceiling_material_dir = material_dirs
            args.extend(["--wall_material_dir", str(wall_material_dir)])
            args.extend(["--floor_material_dir", str(floor_material_dir)])
            args.extend(["--ceiling_material_dir", str(ceiling_material_dir)])
        if material_dir is not None:
            args.extend(["--material_dir", str(material_dir)])
        if thickness is not None:
            args.extend(["--thickness", str(thickness)])
        if wall_padding is not None:
            args.extend(["--wall_padding", str(wall_padding)])
        return self._run_script("generate_room_mesh.py", args)

    def render_scene(
        self,
        layout_dir: Path,
        unposed_dir: Path,
        output: Path,
        *,
        object_ids: list[str] | None = None,
        camera_direction=None,
        radius: float | None = None,
        image_size: int | None = None,
        add_grid: bool = False,
        grid_size: int | None = None,
        draw_grid_lines: bool = False,
        draw_object_directions: bool = False,
        draw_object_bboxes: bool = False,
        anchor_id: str | None = None,
        grid_mapping: str | None = None,
        hide_object_ids: list[str] | None = None,
        use_aabb_for_object_bboxes: bool = False,
    ) -> ToolRunResult:
        args = ["--layout_dir", str(layout_dir), "--unposed_dir", str(unposed_dir), "--output", str(output)]
        if object_ids:
            args.append("--object_ids")
            args.extend(str(object_id) for object_id in object_ids)
        if camera_direction is not None:
            args.append("--camera_direction")
            if isinstance(camera_direction, (list, tuple)):
                args.extend(str(value) for value in camera_direction)
            else:
                args.append(str(camera_direction))
        if radius is not None:
            args.extend(["--radius", str(radius)])
        if image_size is not None:
            args.extend(["--image_size", str(image_size)])
        if add_grid:
            args.append("--add_grid")
        if grid_size is not None:
            args.extend(["--grid_size", str(grid_size)])
        if draw_grid_lines:
            args.append("--draw_grid_lines")
        if draw_object_directions:
            args.append("--draw_object_directions")
        if draw_object_bboxes:
            args.append("--draw_object_bboxes")
        if anchor_id is not None:
            args.append("--anchor_id")
            args.append(str(anchor_id))
        if grid_mapping is not None:
            args.append("--grid_mapping")
            args.append(str(grid_mapping))
        if hide_object_ids:
            args.append("--hide_object_ids")
            args.extend(str(object_id) for object_id in hide_object_ids)
        if use_aabb_for_object_bboxes:
            args.append("--use_aabb_for_object_bboxes")
        return self._run_script("render_scene_from_layout.py", args)

    def build_collision_hulls(
        self,
        unposed_dir: Path,
        output_dir: Path,
        manifest: Path,
        *,
        workers: int | None = None,
        target_faces_per_part: int | None = None,
        max_retries: int = 4,
        split_mode: str | None = "none",
    ) -> ToolRunResult:
        if max_retries < 1:
            raise ValueError("max_retries must be >= 1")

        attempt_faces_per_part: list[int | None] = [target_faces_per_part, 5000, 10000, 20000]
        result = self._run_build_collision_hulls_once(
            unposed_dir,
            output_dir,
            manifest,
            workers=workers,
            target_faces_per_part=attempt_faces_per_part[0],
            split_mode=split_mode,
        )
        if result.returncode != 0 or not result.failed_object_ids:
            return result

        retry_index = 1
        while retry_index < min(max_retries, len(attempt_faces_per_part)):
            current_faces_per_part = attempt_faces_per_part[retry_index]
            retry_object_ids = list(result.failed_object_ids)
            print(
                "Retrying build_collision_hulls "
                f"(attempt {retry_index + 1}/{max_retries}) with failed_object_ids={list(result.failed_object_ids)}, "
                f"workers={workers}, target_faces_per_part={current_faces_per_part}..."
            )
            retry_result = self._run_build_collision_hulls_once(
                unposed_dir,
                output_dir,
                manifest,
                object_ids=retry_object_ids,
                workers=workers,
                target_faces_per_part=current_faces_per_part,
                split_mode=split_mode,
            )
            result = retry_result
            if retry_result.returncode != 0 or not retry_result.failed_object_ids:
                return retry_result
            retry_index += 1

        return result

    def simulate_scene_physics(
        self,
        layout_dir: Path,
        hull_dir: Path,
        output_layout_dir: Path,
        *,
        gravity=None,
        object_ids: list[str] | None = None,
        verify_only: bool = False,
        cpu_only: bool = True,
        use_ground_object: bool = False,
        ground_object_ids: list[str] | None = None,
        lock_rotation_all: bool = False,
        init_non_ground_y_offset: bool = False,
    ) -> ToolRunResult:
        args = [
            "--layout_dir",
            str(layout_dir),
            "--hull_dir",
            str(hull_dir),
            "--output_layout_dir",
            str(output_layout_dir),
        ]
        if gravity is not None:
            args.append("--gravity")
            if isinstance(gravity, (list, tuple)):
                args.extend(str(value) for value in gravity)
            else:
                args.append(str(gravity))
        if object_ids:
            args.append("--object_ids")
            args.extend(str(object_id) for object_id in object_ids)
        if verify_only:
            args.append("--verify_only")
        if cpu_only:
            args.append("--cpu_only")
        if lock_rotation_all:
            args.append("--lock_rotation_all")
        if use_ground_object:
            args.append("--use_ground_object")
        if ground_object_ids:
            args.append("--ground_object_id")
            args.extend(str(object_id) for object_id in ground_object_ids)
        if init_non_ground_y_offset:
            args.append("--init_non_ground_y_offset")
        return self._run_script("simulate_scene_physics.py", args)

    def fix_layout_containment(
        self,
        layout_dir: Path,
        unposed_dir: Path,
        output_layout_dir: Path,
        *,
        object_ids: list[str] | None = None,
        anchor_id: str,
        tolerated_out_of_bounds: float = 0.0,
        detect_only: bool = False,
    ) -> ToolRunResult:
        args = [
            "--layout_dir",
            str(layout_dir),
            "--unposed_dir",
            str(unposed_dir),
            "--output_layout_dir",
            str(output_layout_dir),
            "--anchor_id",
            str(anchor_id),
            "--tolerated_out_of_bounds",
            str(tolerated_out_of_bounds),
        ]
        if object_ids:
            args.append("--object_ids")
            args.extend(str(object_id) for object_id in object_ids)
        if detect_only:
            args.append("--detect_only")
        result = self._run_script("fix_layout_containment.py", args)
        return ToolRunResult(
            command=result.command,
            returncode=result.returncode,
            stdout=result.stdout,
            stderr=result.stderr,
            out_of_bounds_object_ids=self._extract_fix_layout_containment_out_of_bounds_object_ids(result.stdout),
        )

    def check_object_collisions(
        self,
        layout_dir: Path,
        hull_dir: Path,
        *,
        output: Path | None = None,
        object_ids: list[str] | None = None,
        steps: int | None = None,
        mass: float | None = None,
        anchor_id: str | None = None,
    ) -> ToolRunResult:
        args = ["--layout_dir", str(layout_dir), "--hull_dir", str(hull_dir)]
        if output is not None:
            args.extend(["--output", str(output)])
        if object_ids:
            args.append("--object_ids")
            args.extend(str(object_id) for object_id in object_ids)
        if steps is not None:
            args.extend(["--steps", str(steps)])
        if mass is not None:
            args.extend(["--mass", str(mass)])
        if anchor_id is not None:
            args.extend(["--anchor_id", str(anchor_id)])
        return self._run_script("check_object_collisions.py", args)
