from __future__ import annotations

import ast
import hashlib
import json
import os
import re
import shutil
import subprocess
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from .graph import RelationGraphView
from ...artifacts import RunContract
from ...config import verify_relation_runtime


READY_STATUSES = frozenset({"sam3d_ready", "sam3d_ready_degraded"})
OBJECT_ID_RE = re.compile(r"^object_\d{3}$")
SAFE_RUN_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
ENV_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
SECRET_NAME_RE = re.compile(
    r"(?:KEY|TOKEN|SECRET|PASSWORD|CREDENTIAL|AUTHORIZATION|BASE_URL)", re.I
)

RENDERER_FUNCTIONS: Mapping[str, tuple[str, ...]] = {
    "compose_scene": ("layout_dir", "unposed_dir"),
    "denormalize_point": ("point_normalized", "transform_info"),
    "generate_2d_grid_on_plane": (
        "posed_meshes",
        "camera_direction",
        "grid_size",
    ),
    "render_scene_with_grid": (
        "meshes",
        "camera_direction",
        "radius",
        "image_size",
    ),
    "save_grid_mapping": ("mapping", "output_path", "transform_info"),
    "resolve_orthographic_camera_frame": (
        "meshes",
        "camera_direction",
        "radius",
    ),
    "build_camera_pose": ("camera_direction", "radius"),
    "get_camera_matrix": ("yfov_deg", "image_size"),
    "project_3d_to_2d": (
        "point_3d",
        "camera_pose",
        "camera_matrix",
        "image_size",
    ),
}


class RelationStageError(RuntimeError):
    """Raised when relation input, publication or execution gates fail."""


@dataclass(frozen=True)
class RelationArtifacts:
    objects_path: Path
    relations_path: Path
    schema_validation_path: Path
    publication_manifest_path: Path
    source_run_dir: Path
    release_id: str
    publication_id: str
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class CandidateRunResult:
    candidate_dir: Path
    command: tuple[str, ...]
    returncode: int
    stdout: str
    stderr: str


@dataclass(frozen=True)
class CandidateValidationResult:
    candidate_dir: Path
    summary_path: Path
    objects_path: Path
    relations_path: Path
    schema_validation_path: Path
    validation_path: Path
    object_ids: tuple[str, ...]
    source_hashes: Mapping[str, str]
    graph_audit: Mapping[str, Any]
    runtime_audit: Mapping[str, Any]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RelationStageError(f"cannot parse {label}: {path}") from exc
    if not isinstance(value, dict):
        raise RelationStageError(f"{label} must contain a JSON object: {path}")
    return value


def _read_relations(path: Path) -> list[dict[str, Any]]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RelationStageError(f"cannot parse relations: {path}") from exc
    if not isinstance(value, list) or not all(
        isinstance(item, dict) for item in value
    ):
        raise RelationStageError("relations.json must contain an array of objects")
    return value


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}.{uuid.uuid4().hex}.tmp"
    _write_json(temporary, value)
    os.replace(temporary, path)


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def _resolve_confined(
    raw_path: Any,
    *,
    base_dir: Path,
    root: Path,
    label: str,
) -> Path:
    if not isinstance(raw_path, str) or not raw_path.strip():
        raise RelationStageError(f"{label} must be a non-empty path")
    candidate = Path(raw_path).expanduser()
    if not candidate.is_absolute():
        candidate = base_dir / candidate
    resolved = candidate.resolve()
    if not _is_within(resolved, root):
        raise RelationStageError(f"{label} escapes its trusted root: {resolved}")
    return resolved


def _object_ids(items: Any, label: str) -> tuple[str, ...]:
    if not isinstance(items, list):
        raise RelationStageError(f"{label} must be an array")
    values: list[str] = []
    for index, item in enumerate(items):
        if not isinstance(item, dict):
            raise RelationStageError(f"{label}[{index}] must be an object")
        object_id = str(item.get("object_id") or item.get("id") or "")
        if not OBJECT_ID_RE.fullmatch(object_id):
            raise RelationStageError(
                f"{label}[{index}] has invalid object ID: {object_id}"
            )
        values.append(object_id)
    if len(values) != len(set(values)):
        raise RelationStageError(f"{label} contains duplicate object IDs")
    return tuple(values)


def _relation_runtime_audit(path: Path) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink() or path.stat().st_size == 0:
        raise RelationStageError(
            f"relation execution summary is missing or unsafe: {path}"
        )
    summary = _read_json(path, "relation execution summary")
    api_calls = summary.get("api_calls")
    if (
        summary.get("schema_version") != "relation_execution_summary"
        or not isinstance(api_calls, int)
        or isinstance(api_calls, bool)
        or api_calls <= 0
        or not isinstance(summary.get("stages"), dict)
    ):
        raise RelationStageError(
            "relation execution summary has invalid execution closure"
        )
    return {
        "runtime_summary_path": str(path),
        "runtime_summary_sha256": _sha256(path),
        "api_calls": api_calls,
    }


class RelationStageAdapter:
    """SceneMosaic-facing gate for the verified relation runtime."""

    def __init__(
        self,
        *,
        scene_config_path: Path,
        relation_package_root: Path,
        renderer_script: Path,
    ) -> None:
        self.contract = RunContract.from_scene_config(scene_config_path)
        self.package_root = Path(relation_package_root).expanduser().resolve()
        self.renderer_script = Path(renderer_script).expanduser().resolve()

    @property
    def relations_root(self) -> Path:
        return self.contract.relations_dir

    def preflight(self) -> dict[str, Any]:
        release = self.verify_runtime()
        renderer = self.verify_renderer_contract()
        inputs = self.validate_inputs()
        return {
            "release": release,
            "renderer": renderer,
            "inputs": inputs,
            "api_calls": 0,
        }

    def verify_runtime(self) -> dict[str, Any]:
        report = verify_relation_runtime(self.package_root)
        if not report.get("valid"):
            raise RelationStageError(
                f"relation runtime verification failed: {report}"
            )
        release_id = report.get("release_id")
        if (
            not isinstance(release_id, str)
            or not SAFE_RUN_ID_RE.fullmatch(release_id)
        ):
            raise RelationStageError("relation runtime release_id is unsafe")
        return report

    def verify_renderer_contract(self) -> dict[str, Any]:
        if not self.renderer_script.is_file():
            raise RelationStageError(
                f"renderer script is missing: {self.renderer_script}"
            )
        try:
            tree = ast.parse(
                self.renderer_script.read_text(encoding="utf-8"),
                filename=str(self.renderer_script),
            )
        except (OSError, SyntaxError) as exc:
            raise RelationStageError(
                f"cannot parse renderer script: {self.renderer_script}"
            ) from exc
        functions = {
            node.name: tuple(
                argument.arg
                for argument in [*node.args.posonlyargs, *node.args.args]
            )
            for node in tree.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        }
        failures: list[str] = []
        for name, required in RENDERER_FUNCTIONS.items():
            actual = functions.get(name)
            if actual is None:
                failures.append(f"missing:{name}")
                continue
            if actual[: len(required)] != required:
                failures.append(
                    f"signature:{name}:expected_prefix={required}:actual={actual}"
                )
        if failures:
            raise RelationStageError(
                f"renderer contract verification failed: {failures}"
            )
        return {
            "renderer_path": str(self.renderer_script),
            "function_count": len(RENDERER_FUNCTIONS),
            "functions": {
                name: list(functions[name]) for name in RENDERER_FUNCTIONS
            },
            "coordinate_system": "OpenGL_glTF_global",
            "ground_plane": "XZ",
            "valid": True,
        }

    def validate_inputs(self) -> dict[str, Any]:
        run_root = self.contract.run_root
        reviewed_path = self.contract.stage03_dir / "reviewed_objects.json"
        quality_path = self.contract.stage03_dir / "quality_report.json"
        adoption_path = self.contract.stage03_dir / "perception_adoption.json"
        stage04_manifest_path = (
            self.contract.stage04_dir / "sam3d_input_manifest.json"
        )
        stage05 = self.contract.stage05_dir
        index_path = stage05 / "object_index_map.json"
        run_info_path = stage05 / "sam3d_run_info.json"
        required = (
            reviewed_path,
            quality_path,
            adoption_path,
            stage04_manifest_path,
            index_path,
            run_info_path,
        )
        missing = [str(path) for path in required if not path.is_file()]
        if missing:
            raise RelationStageError(
                f"relation input gate is missing required files: {missing}"
            )

        adoption = _read_json(adoption_path, "perception adoption")
        if (
            adoption.get("delivery_status") not in READY_STATUSES
            or adoption.get("sam3d_execution_allowed") is not True
            or adoption.get("human_review_required") is not False
        ):
            raise RelationStageError(
                "protected perception adoption is not accepted for SAM3D"
            )
        formal_artifacts = adoption.get("formal_artifacts")
        if not isinstance(formal_artifacts, list):
            raise RelationStageError(
                "perception adoption has no formal artifact hashes"
            )
        for record in formal_artifacts:
            if not isinstance(record, dict):
                raise RelationStageError(
                    "perception adoption artifact record is invalid"
                )
            artifact = Path(str(record.get("path") or "")).expanduser().resolve()
            if not _is_within(artifact, run_root):
                raise RelationStageError(
                    f"perception artifact escapes run root: {artifact}"
                )
            if (
                not artifact.is_file()
                or _sha256(artifact) != record.get("sha256")
            ):
                raise RelationStageError(
                    f"perception artifact hash mismatch: {artifact}"
                )

        reviewed = _read_json(reviewed_path, "reviewed objects")
        quality = _read_json(quality_path, "quality report")
        stage04_manifest = _read_json(
            stage04_manifest_path, "Stage 04 manifest"
        )
        index = _read_json(index_path, "Stage 05 object index")
        run_info = _read_json(run_info_path, "Stage 05 run info")
        if run_info.get("status") != "success":
            raise RelationStageError(
                f"Stage 05 is not successful: {run_info.get('status')}"
            )
        if (
            stage04_manifest.get("delivery_status") not in READY_STATUSES
            or stage04_manifest.get("sam3d_execution_allowed") is not True
        ):
            raise RelationStageError("Stage 04 delivery is not executable")

        reviewed_ids = _object_ids(reviewed.get("objects"), "reviewed objects")
        stage04_ids = _object_ids(
            stage04_manifest.get("objects"), "Stage 04 objects"
        )
        stage05_ids = _object_ids(index.get("objects"), "Stage 05 objects")
        expected_ids = tuple(
            f"object_{index:03d}"
            for index in range(1, len(stage04_ids) + 1)
        )
        if stage04_ids != expected_ids:
            raise RelationStageError(
                "Stage 04 IDs must be continuous object_001 through object_N"
            )

        layout_dir = stage05 / "layout_params"
        unposed_dir = stage05 / "unposed_objects"
        layout_ids = tuple(
            sorted(path.stem for path in layout_dir.glob("object_*.json"))
        )
        unposed_ids = tuple(
            sorted(path.stem for path in unposed_dir.glob("object_*.glb"))
        )
        if not (
            reviewed_ids
            == stage04_ids
            == stage05_ids
            == layout_ids
            == unposed_ids
        ):
            raise RelationStageError(
                "reviewed, Stage 04, Stage 05, layout and unposed object IDs "
                "must be identical"
            )

        layout_hashes: dict[str, str] = {}
        mesh_hashes: dict[str, str] = {}
        for object_id in expected_ids:
            layout_path = layout_dir / f"{object_id}.json"
            mesh_path = unposed_dir / f"{object_id}.glb"
            layout = _read_json(layout_path, f"layout {object_id}")
            if (
                layout.get("object_id") != object_id
                or layout.get("sam3d_internal_name") != object_id
            ):
                raise RelationStageError(
                    f"layout identity mismatch for {object_id}"
                )
            for key in ("translation", "rotation", "scale"):
                if not isinstance(layout.get(key), list) or not layout[key]:
                    raise RelationStageError(
                        f"layout {object_id} lacks valid {key}"
                    )
            if (
                not mesh_path.is_file()
                or mesh_path.is_symlink()
                or mesh_path.stat().st_size == 0
            ):
                raise RelationStageError(
                    f"unposed GLB must be a real non-empty file: {mesh_path}"
                )
            layout_hashes[object_id] = _sha256(layout_path)
            mesh_hashes[object_id] = _sha256(mesh_path)

        closure = adoption.get("object_id_closure")
        if (
            not isinstance(closure, dict)
            or closure.get("closed") is not True
            or tuple(closure.get("manifest", [])) != expected_ids
        ):
            raise RelationStageError(
                "perception adoption object ID closure is invalid"
            )
        return {
            "scene_name": self.contract.scene_name,
            "run_root": str(run_root),
            "object_ids": list(expected_ids),
            "object_count": len(expected_ids),
            "reviewed_objects_path": str(reviewed_path),
            "reviewed_objects_sha256": _sha256(reviewed_path),
            "quality_report_path": str(quality_path),
            "quality_report_sha256": _sha256(quality_path),
            "quality_report_keys": sorted(quality),
            "stage04_manifest_path": str(stage04_manifest_path),
            "stage04_manifest_sha256": _sha256(stage04_manifest_path),
            "stage05_index_path": str(index_path),
            "stage05_index_sha256": _sha256(index_path),
            "layout_sha256": layout_hashes,
            "unposed_glb_sha256": mesh_hashes,
            "segmentation_accepted": True,
            "sam3d_accepted": True,
            "id_closure": True,
        }

    def reuse(self) -> RelationArtifacts:
        preflight = self.preflight()
        current_candidates = (
            self.relations_root / "current.json",
            self.relations_root / "published" / "current.json",
        )
        existing = [path for path in current_candidates if path.is_file()]
        if not existing:
            raise RelationStageError(
                "an active Relation publication is not available"
            )
        if len(existing) > 1 and _sha256(existing[0]) != _sha256(existing[1]):
            raise RelationStageError(
                "relation current pointers disagree; refusing ambiguous reuse"
            )
        pointer_path = existing[0]
        pointer = _read_json(pointer_path, "relation current pointer")
        if pointer.get("active") is not True:
            raise RelationStageError("relation current pointer is not active")

        publication_manifest_path = _resolve_confined(
            pointer.get("publication_manifest"),
            base_dir=self.relations_root,
            root=self.relations_root,
            label="publication manifest",
        )
        artifacts = self._validate_publication(
            publication_manifest_path,
            tuple(preflight["inputs"]["object_ids"]),
            {
                key: preflight["inputs"][key]
                for key in (
                    "reviewed_objects_sha256",
                    "quality_report_sha256",
                    "stage04_manifest_sha256",
                    "stage05_index_sha256",
                )
            },
        )
        metadata = dict(artifacts.metadata)
        metadata.update(
            {
                "reused": True,
                "api_calls": 0,
                "preflight": preflight,
                "current_pointer_path": str(pointer_path),
            }
        )
        return RelationArtifacts(
            objects_path=artifacts.objects_path,
            relations_path=artifacts.relations_path,
            schema_validation_path=artifacts.schema_validation_path,
            publication_manifest_path=artifacts.publication_manifest_path,
            source_run_dir=artifacts.source_run_dir,
            release_id=artifacts.release_id,
            publication_id=artifacts.publication_id,
            metadata=metadata,
        )

    def _validate_publication(
        self,
        manifest_path: Path,
        expected_ids: tuple[str, ...],
        expected_input_hashes: Mapping[str, str],
    ) -> RelationArtifacts:
        publications_root = (self.relations_root / "publications").resolve()
        if not _is_within(manifest_path, publications_root):
            raise RelationStageError(
                f"publication is outside immutable publication root: {manifest_path}"
            )
        manifest = _read_json(manifest_path, "publication manifest")
        publication_id = str(manifest.get("publication_id") or "")
        if not SAFE_RUN_ID_RE.fullmatch(publication_id):
            raise RelationStageError("publication_id is missing or unsafe")
        release_id = str(manifest.get("release_id") or "")
        if not SAFE_RUN_ID_RE.fullmatch(release_id):
            raise RelationStageError(
                "publication release_id is missing or unsafe"
            )
        if manifest.get("input_object_ids") != list(expected_ids):
            raise RelationStageError(
                "publication input object IDs no longer match this run"
            )
        input_hashes = manifest.get("input_hashes")
        if not isinstance(input_hashes, dict):
            raise RelationStageError("publication lacks input hash bindings")
        mismatched_inputs = {
            key: (input_hashes.get(key), expected)
            for key, expected in expected_input_hashes.items()
            if input_hashes.get(key) != expected
        }
        if mismatched_inputs:
            raise RelationStageError(
                f"publication input hash binding mismatch: {mismatched_inputs}"
            )

        runtime_audit = manifest.get("runtime_audit")
        benchmark_evidence = manifest.get("benchmark_evidence")
        audit_sources = manifest.get("audit_sources")
        if (
            not isinstance(runtime_audit, dict)
            or not isinstance(runtime_audit.get("api_calls"), int)
            or runtime_audit.get("api_calls") <= 0
            or not isinstance(
                runtime_audit.get("runtime_summary_sha256"), str
            )
            or len(runtime_audit["runtime_summary_sha256"]) != 64
            or not isinstance(benchmark_evidence, dict)
            or benchmark_evidence.get("agent_api_calls")
            != runtime_audit.get("api_calls")
            or not isinstance(audit_sources, dict)
        ):
            raise RelationStageError(
                "publication Relation execution evidence is missing or inconsistent"
            )
        runtime_summary_path = str(
            runtime_audit.get("runtime_summary_path") or ""
        )
        runtime_source_record = audit_sources.get(
            runtime_summary_path
        )
        if (
            not isinstance(runtime_source_record, dict)
            or runtime_source_record.get("sha256")
            != runtime_audit["runtime_summary_sha256"]
        ):
            raise RelationStageError(
                "publication Relation execution summary hash is not source-bound"
            )

        publication_dir = manifest_path.parent
        required_names = (
            "objects.json",
            "relations.json",
            "schema_validation.json",
            "candidate_validation.json",
        )
        file_records = manifest.get("files")
        if not isinstance(file_records, dict):
            raise RelationStageError(
                "publication manifest lacks file hash records"
            )
        paths: dict[str, Path] = {}
        for name in required_names:
            path = publication_dir / name
            record = file_records.get(name)
            if (
                not path.is_file()
                or not isinstance(record, dict)
                or record.get("sha256") != _sha256(path)
                or record.get("bytes") != path.stat().st_size
            ):
                raise RelationStageError(
                    f"publication file verification failed: {path}"
                )
            paths[name] = path

        objects = _read_json(paths["objects.json"], "published objects")
        relations = _read_relations(paths["relations.json"])
        schema = _read_json(
            paths["schema_validation.json"], "published schema validation"
        )
        if schema.get("valid") is not True:
            raise RelationStageError("published relation Schema is invalid")
        graph = self._validate_relation_payloads(
            objects, relations, expected_ids
        )

        source_run_dir = Path(
            str(manifest.get("source_run_dir") or publication_dir)
        ).expanduser()
        return RelationArtifacts(
            objects_path=paths["objects.json"],
            relations_path=paths["relations.json"],
            schema_validation_path=paths["schema_validation.json"],
            publication_manifest_path=manifest_path,
            source_run_dir=source_run_dir,
            release_id=release_id,
            publication_id=publication_id,
            metadata={
                "publication_manifest_sha256": _sha256(manifest_path),
                "candidate_validated": True,
                "schema_valid": True,
                "relation_graph": graph.audit_summary(),
                "historical_api": runtime_audit,
                "historical_api_calls": runtime_audit["api_calls"],
                "api_calls_this_invocation": 0,
                "per_object_physics_consumed": False,
            },
        )

    @staticmethod
    def _validate_relation_payloads(
        objects: dict[str, Any],
        relations: list[dict[str, Any]],
        expected_ids: tuple[str, ...],
    ) -> RelationGraphView:
        walls = objects.get("walls")
        room_materials = objects.get("room_materials")
        object_items = objects.get("objects")
        if not isinstance(walls, list) or not walls:
            raise RelationStageError("objects.json must preserve walls")
        if not isinstance(room_materials, dict) or any(
            not isinstance(room_materials.get(key), str)
            or not room_materials[key].strip()
            for key in ("walls", "floor", "ceiling")
        ):
            raise RelationStageError(
                "room_materials walls/floor/ceiling must be non-empty"
            )
        output_ids = _object_ids(object_items, "published objects")
        if output_ids != expected_ids:
            raise RelationStageError(
                "published objects do not match reviewed and Stage 05 IDs"
            )
        physics_keys = {
            "mass",
            "linear_damping",
            "angular_damping",
            "static_friction",
            "dynamic_friction",
        }
        for item in object_items:
            physics = item.get("physics")
            if not isinstance(physics, dict) or not physics_keys.issubset(
                physics
            ):
                raise RelationStageError(
                    f"published object physics is incomplete: {item.get('id')}"
                )

        wall_ids = {
            str(item.get("id"))
            for item in walls
            if isinstance(item, dict) and item.get("id")
        }
        allowed_anchors = set(expected_ids) | wall_ids | {
            "floor",
            "ceiling",
        }
        targets: set[str] = set()
        for relation in relations:
            parameters = relation.get("parameters", {})
            anchor = str(parameters.get("object_anchor") or "")
            target = str(parameters.get("object_target") or "")
            if anchor not in allowed_anchors:
                raise RelationStageError(
                    f"relation anchor is not published: {anchor}"
                )
            if target not in expected_ids:
                raise RelationStageError(
                    f"relation target is not published: {target}"
                )
            targets.add(target)
        if targets != set(expected_ids):
            raise RelationStageError(
                "every published object must be covered as a relation target"
            )
        return RelationGraphView.from_relations(relations)

    def _candidate_path(self, candidate_dir: Path) -> Path:
        candidate = Path(candidate_dir).expanduser().resolve()
        benchmarks_root = self.contract.relation_candidates_dir.resolve()
        if (
            candidate.parent != benchmarks_root
            or not SAFE_RUN_ID_RE.fullmatch(candidate.name)
        ):
            raise RelationStageError(
                f"candidate is outside the direct benchmark root: {candidate}"
            )
        return candidate

    def validate_candidate(
        self,
        candidate_dir: Path,
    ) -> CandidateValidationResult:
        """Validate a production-wrapper result before any human decision."""

        preflight = self.preflight()
        candidate = self._candidate_path(candidate_dir)
        summary_path = candidate / "benchmark_summary.json"
        final_dir = candidate / "async_agent_run" / "final_outputs"
        objects_path = final_dir / "objects.json"
        relations_path = final_dir / "relations.json"
        schema_path = (
            candidate
            / "async_agent_run"
            / "debug"
            / "final_validation"
            / "schema_validation.json"
        )
        runtime_summary_path = (
            candidate
            / "async_agent_run"
            / "debug"
            / "agents"
            / "runtime_summary.json"
        )
        source_paths = {
            "benchmark_summary.json": summary_path,
            "objects.json": objects_path,
            "relations.json": relations_path,
            "schema_validation.json": schema_path,
            "runtime_summary.json": runtime_summary_path,
        }
        missing = [
            str(path)
            for path in source_paths.values()
            if not path.is_file()
        ]
        if missing:
            raise RelationStageError(
                f"candidate validation is missing files: {missing}"
            )
        unsafe = [
            str(path)
            for path in source_paths.values()
            if path.is_symlink() or path.stat().st_size == 0
        ]
        if unsafe:
            raise RelationStageError(
                f"candidate files must be real and non-empty: {unsafe}"
            )

        summary = _read_json(summary_path, "benchmark summary")
        if summary.get("schema_version") != "relation_async_benchmark":
            raise RelationStageError("candidate benchmark schema is invalid")
        if (
            summary.get("scene_name") != self.contract.scene_name
            or summary.get("benchmark_id") != candidate.name
        ):
            raise RelationStageError(
                "candidate benchmark identity does not match this scene"
            )
        for key, expected in (
            ("run_root", self.contract.run_root),
            ("benchmark_root", candidate),
        ):
            raw_path = summary.get(key)
            if (
                not isinstance(raw_path, str)
                or Path(raw_path).expanduser().resolve() != expected
            ):
                raise RelationStageError(
                    f"candidate benchmark {key} does not match: {raw_path}"
                )
        if summary.get("dry_run") is not False:
            raise RelationStageError(
                "dry-run relation benchmark cannot become a candidate"
            )
        if summary.get("benchmark_success") is not True:
            raise RelationStageError(
                "benchmark_summary does not report benchmark_success=true"
            )

        async_summary = summary.get("async")
        if not isinstance(async_summary, dict):
            raise RelationStageError("candidate lacks an async benchmark result")
        commands = async_summary.get("commands")
        required_commands = {"expected_only", "precompute", "runner"}
        if (
            not isinstance(commands, dict)
            or not required_commands.issubset(commands)
            or any(
                not isinstance(command, dict)
                or command.get("returncode") != 0
                for command in commands.values()
            )
        ):
            raise RelationStageError(
                "benchmark summary does not show all successful async commands"
            )
        metrics = async_summary.get("metrics")
        if (
            not isinstance(metrics, dict)
            or metrics.get("final_objects_exists") is not True
            or metrics.get("final_relations_exists") is not True
        ):
            raise RelationStageError(
                "async metrics do not confirm both final JSON outputs"
            )
        for key, expected in (
            ("final_objects_path", objects_path),
            ("final_relations_path", relations_path),
        ):
            raw_path = metrics.get(key)
            if (
                not isinstance(raw_path, str)
                or Path(raw_path).expanduser().resolve() != expected
            ):
                raise RelationStageError(
                    f"async metric {key} does not match candidate outputs"
                )

        runtime_audit = _relation_runtime_audit(
            runtime_summary_path
        )
        metric_api_calls = metrics.get("agent_api_calls")
        if (
            metric_api_calls is not None
            and metric_api_calls != runtime_audit["api_calls"]
        ):
            raise RelationStageError(
                "benchmark metrics and runtime summary api_calls disagree"
            )

        benchmark_evidence = {
            "agent_api_calls": runtime_audit["api_calls"],
            "command_returncodes": {
                name: int(commands[name]["returncode"])
                for name in sorted(required_commands)
            },
            "final_objects_path": str(objects_path),
            "final_relations_path": str(relations_path),
        }

        objects = _read_json(objects_path, "candidate objects")
        relations = _read_relations(relations_path)
        schema = _read_json(schema_path, "candidate schema validation")
        if schema.get("valid") is not True:
            raise RelationStageError(
                "schema-invalid candidate cannot be published"
            )
        object_ids = tuple(preflight["inputs"]["object_ids"])
        graph = self._validate_relation_payloads(
            objects,
            relations,
            object_ids,
        )
        source_hashes = {
            name: _sha256(path) for name, path in source_paths.items()
        }
        validation_path = candidate / "scenemosaic_candidate_validation.json"
        stable_payload = {
            "schema_version": "scenemosaic_relation_candidate_validation",
            "status": "candidate_valid",
            "candidate_dir": str(candidate),
            "release_id": preflight["release"]["release_id"],
            "object_ids": list(object_ids),
            "source_hashes": source_hashes,
            "relation_graph": graph.audit_summary(),
            "benchmark_evidence": benchmark_evidence,
            "runtime_audit": runtime_audit,
            "per_object_physics_consumed": False,
        }
        existing: dict[str, Any] | None = None
        if validation_path.is_file():
            existing = _read_json(
                validation_path, "candidate validation record"
            )
        if not isinstance(existing, dict) or any(
            existing.get(key) != value
            for key, value in stable_payload.items()
        ):
            payload = dict(stable_payload)
            payload["validated_at_utc"] = _utc_now()
            _atomic_write_json(validation_path, payload)

        return CandidateValidationResult(
            candidate_dir=candidate,
            summary_path=summary_path,
            objects_path=objects_path,
            relations_path=relations_path,
            schema_validation_path=schema_path,
            validation_path=validation_path,
            object_ids=object_ids,
            source_hashes=source_hashes,
            graph_audit=graph.audit_summary(),
            runtime_audit=runtime_audit,
        )

    def publish_candidate(
        self,
        *,
        candidate_dir: Path,
        publication_id: str,
    ) -> RelationArtifacts:
        preflight = self.preflight()
        if not SAFE_RUN_ID_RE.fullmatch(publication_id):
            raise RelationStageError("publication_id is unsafe")
        validation = self.validate_candidate(candidate_dir)
        candidate = validation.candidate_dir
        summary_path = validation.summary_path
        objects_path = validation.objects_path
        relations_path = validation.relations_path
        schema_path = validation.schema_validation_path
        required = (
            summary_path,
            objects_path,
            relations_path,
            schema_path,
            validation.validation_path,
        )
        missing = [str(path) for path in required if not path.is_file()]
        if missing:
            raise RelationStageError(
                f"candidate publication is missing files: {missing}"
            )

        summary = _read_json(summary_path, "benchmark summary")
        async_summary = summary.get("async")
        commands = (
            async_summary.get("commands")
            if isinstance(async_summary, dict)
            else None
        )
        if (
            not isinstance(commands, dict)
            or not commands
            or any(
                not isinstance(command, dict)
                or command.get("returncode") != 0
                for command in commands.values()
            )
        ):
            raise RelationStageError(
                "benchmark summary does not show successful async commands"
            )

        objects = _read_json(objects_path, "candidate objects")
        relations = _read_relations(relations_path)
        schema = _read_json(schema_path, "candidate schema validation")
        if schema.get("valid") is not True:
            raise RelationStageError(
                "schema-invalid candidate cannot be published"
            )
        self._validate_relation_payloads(
            objects,
            relations,
            tuple(preflight["inputs"]["object_ids"]),
        )

        publications_root = self.relations_root / "publications"
        publications_root.mkdir(parents=True, exist_ok=True)
        publication_dir = publications_root / publication_id
        if publication_dir.exists():
            raise RelationStageError(
                f"publication already exists: {publication_dir}"
            )
        temporary = (
            publications_root
            / f".{publication_id}.{uuid.uuid4().hex}.tmp"
        )
        temporary.mkdir(parents=False, exist_ok=False)
        try:
            source_files = {
                "objects.json": objects_path,
                "relations.json": relations_path,
                "schema_validation.json": schema_path,
                "candidate_validation.json": validation.validation_path,
            }
            for name, source in source_files.items():
                shutil.copy2(source, temporary / name)
            file_records = {
                name: {
                    "sha256": _sha256(temporary / name),
                    "bytes": (temporary / name).stat().st_size,
                }
                for name in source_files
            }
            for name in (
                "objects.json",
                "relations.json",
                "schema_validation.json",
            ):
                if (
                    file_records[name]["sha256"]
                    != validation.source_hashes[name]
                ):
                    raise RelationStageError(
                        f"candidate changed while publishing: {name}"
                    )
            audit_sources: dict[str, dict[str, Any]] = {}
            audit_candidates = (
                summary_path,
                candidate
                / "async_agent_run"
                / "debug"
                / "agents"
                / "runtime_summary.json",
                candidate
                / "async_agent_run"
                / "debug"
                / "agents"
                / "final_process_report.json",
            )
            for source in audit_candidates:
                if source.is_file():
                    audit_sources[str(source)] = {
                        "sha256": _sha256(source),
                        "bytes": source.stat().st_size,
                    }
            benchmark_evidence = {
                "agent_api_calls": validation.runtime_audit["api_calls"],
                "command_returncodes": {
                    name: int(command["returncode"])
                    for name, command in sorted(commands.items())
                },
                "final_objects_path": str(validation.objects_path),
                "final_relations_path": str(validation.relations_path),
            }
            manifest = {
                "schema_version": "scenemosaic_relation_publication",
                "publication_id": publication_id,
                "release_id": preflight["release"]["release_id"],
                "published_at_utc": _utc_now(),
                "source_run_dir": str(candidate),
                "files": file_records,
                "input_object_ids": preflight["inputs"]["object_ids"],
                "input_hashes": {
                    "reviewed_objects_sha256": preflight["inputs"][
                        "reviewed_objects_sha256"
                    ],
                    "quality_report_sha256": preflight["inputs"][
                        "quality_report_sha256"
                    ],
                    "stage04_manifest_sha256": preflight["inputs"][
                        "stage04_manifest_sha256"
                    ],
                    "stage05_index_sha256": preflight["inputs"][
                        "stage05_index_sha256"
                    ],
                },
                "audit_sources": audit_sources,
                "benchmark_evidence": benchmark_evidence,
                "runtime_audit": dict(validation.runtime_audit),
                "candidate_validated": True,
                "schema_valid": True,
                "per_object_physics_consumed": False,
            }
            _write_json(temporary / "manifest.json", manifest)
            os.replace(temporary, publication_dir)
        except Exception:
            if temporary.exists():
                shutil.rmtree(temporary)
            raise

        pointer = {
            "schema_version": "scenemosaic_relation_current",
            "active": True,
            "publication_id": publication_id,
            "publication_manifest": str(
                Path("publications") / publication_id / "manifest.json"
            ),
            "updated_at_utc": _utc_now(),
        }
        _atomic_write_json(self.relations_root / "current.json", pointer)
        return self.reuse()

    def run_candidate(
        self,
        *,
        benchmark_id: str,
        api_key_env: str,
        precompute_python: Path,
        runner_python: Path,
        model: str = "gpt-5.5",
        api_base: str | None = None,
        environ: Mapping[str, str] | None = None,
    ) -> CandidateRunResult:
        self.preflight()
        if not SAFE_RUN_ID_RE.fullmatch(benchmark_id):
            raise RelationStageError("benchmark_id is unsafe")
        if not ENV_NAME_RE.fullmatch(api_key_env):
            raise RelationStageError("API key environment variable name is invalid")
        if not isinstance(model, str) or model.strip().lower() not in {
            "gpt-5.5", "gpt-6.1-sol"
        }:
            raise RelationStageError("SceneMosaic Relation requires gpt-5.5 or gpt-6.1-sol")
        model = model.strip().lower()
        source_env = os.environ if environ is None else environ
        secret = source_env.get(api_key_env)
        if not secret:
            raise RelationStageError(
                f"API key environment variable is not set: {api_key_env}"
            )
        precompute = Path(precompute_python).expanduser().resolve()
        runner = Path(runner_python).expanduser().resolve()
        if not precompute.is_file() or not runner.is_file():
            raise RelationStageError(
                "precompute_python and runner_python must both exist"
            )
        candidate_dir = self.contract.relation_candidates_dir / benchmark_id
        if candidate_dir.exists():
            raise RelationStageError(
                f"benchmark directory already exists: {candidate_dir}"
            )
        wrapper = (
            self.package_root
            / "tools"
            / "benchmark_relation_async_planning.py"
        )
        command = [
            str(runner),
            str(wrapper),
            "--scene_name",
            self.contract.scene_name,
            "--run_root",
            str(self.contract.run_root),
            "--benchmark_id",
            benchmark_id,
            "--render_script",
            str(self.renderer_script),
            "--precompute_python",
            str(precompute),
            "--runner_python",
            str(runner),
            "--api_key_env",
            api_key_env,
            "--model",
            model,
            "--vision_model",
            model,
            "--wall_specialist_model",
            model,
            "--wall_specialist_vision_model",
            model,
        ]
        if "--force" in command or "--force_regenerate" in command:
            raise RelationStageError("destructive wrapper flags are forbidden")

        child_env = {
            name: value
            for name, value in source_env.items()
            if not SECRET_NAME_RE.search(name)
        }
        child_env[api_key_env] = secret
        child_env["PYTHONDONTWRITEBYTECODE"] = "1"
        child_env["PYTHONNOUSERSITE"] = "1"
        resolved_base = (
            api_base.strip()
            if api_base and api_base.strip() and api_base.strip() != "None"
            else source_env.get("OPEN_BASE_URL")
            or source_env.get("OPENAI_BASE_URL")
            or source_env.get("OPENAI_API_BASE")
        )
        if resolved_base:
            child_env["OPEN_BASE_URL"] = resolved_base
        secret_values = {
            value
            for name, value in source_env.items()
            if value and SECRET_NAME_RE.search(name)
        }
        secret_values.update(value for value in (secret, resolved_base) if value)

        def redact(value: str) -> str:
            result = value
            for secret_value in secret_values:
                result = result.replace(secret_value, "<redacted>")
            return result[: 2 * 1024 * 1024]

        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=False,
            env=child_env,
        )
        redacted_stdout = redact(completed.stdout)
        redacted_stderr = redact(completed.stderr)
        if completed.returncode != 0:
            raise RelationStageError(
                "relation wrapper failed with return code "
                f"{completed.returncode}; stdout={redacted_stdout[-2000:]}; "
                f"stderr={redacted_stderr[-2000:]}"
            )
        return CandidateRunResult(
            candidate_dir=candidate_dir,
            command=tuple(command),
            returncode=completed.returncode,
            stdout=redacted_stdout,
            stderr=redacted_stderr,
        )
