from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import uuid
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import trimesh

from .layout_combination import (
    LayoutCombination,
    LayoutVariantCandidate,
    _layout_payload_to_matrix,
)
from .orchestrator import PipelineOrchestrator
from ..reconstruction.relations.graph import RelationGraphError, RelationGraphView
from ..reconstruction.relations.stage import RelationArtifacts
from ..artifacts import RunContract
from .tool_runner import ToolRunner, ToolRunResult
from .types import PipelineReport, SceneArtifacts

SAFE_STAGE_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


class SceneMosaicArtifactError(RuntimeError):
    """Raised when an immutable Stage 08 attempt cannot be prepared."""


@dataclass(frozen=True)
class SceneMosaicStageArtifacts:
    stage_id: str
    workspace_dir: Path
    integration_manifest_path: Path
    relation_artifacts: RelationArtifacts


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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


class SceneMosaicStageAdapter:
    """Prepare a writable Stage 08 copy from immutable upstream facts."""

    def __init__(
        self,
        *,
        scene_config_path: Path,
        relation_artifacts: RelationArtifacts,
    ) -> None:
        self.contract = RunContract.from_scene_config(scene_config_path)
        self.relation_artifacts = relation_artifacts

    @property
    def stage08_root(self) -> Path:
        return self.contract.evolution_dir

    @property
    def attempts_dir(self) -> Path:
        return self.stage08_root / ".stage08_attempts"

    @property
    def publications_dir(self) -> Path:
        return self.stage08_root / ".stage08_publications"

    def attempt_workspace(self, stage_id: str) -> Path:
        if not SAFE_STAGE_ID_RE.fullmatch(stage_id):
            raise SceneMosaicArtifactError("Stage 08 stage_id is unsafe")
        return self.attempts_dir / stage_id

    def prepare(self, stage_id: str) -> SceneMosaicStageArtifacts:
        if not SAFE_STAGE_ID_RE.fullmatch(stage_id):
            raise SceneMosaicArtifactError("Stage 08 stage_id is unsafe")
        target = self.attempt_workspace(stage_id)
        if target.exists():
            raise SceneMosaicArtifactError(
                f"Stage 08 attempt already exists: {target}"
            )
        temporary = (
            target.parent / f".{stage_id}.{uuid.uuid4().hex}.tmp"
        )
        temporary.mkdir(parents=True, exist_ok=False)
        try:
            source_records, formal_records = self._materialize(
                temporary, target
            )
            manifest = {
                "schema_version": "scenemosaic_stage08_integration",
                "stage_id": stage_id,
                "created_at_utc": _utc_now(),
                "scene_name": self.contract.scene_name,
                "coordinate_system": "OpenGL_glTF_global",
                "ground_plane": "XZ",
                "relation_release_id": self.relation_artifacts.release_id,
                "relation_publication_id": (
                    self.relation_artifacts.publication_id
                ),
                "relation_publication_manifest": str(
                    self.relation_artifacts.publication_manifest_path
                ),
                "relation_publication_manifest_sha256": _sha256(
                    self.relation_artifacts.publication_manifest_path
                ),
                "source_artifacts": source_records,
                "formal_working_copies": formal_records,
                "relation_metadata": dict(
                    self.relation_artifacts.metadata
                ),
                "wall_padding": 0.0,
                "full_relations_preserved": True,
                "per_object_physics_consumed": False,
                "upstream_stage05_read_only": True,
                "relation_publication_read_only": True,
                "api_calls": 0,
            }
            _write_json(temporary / "integration_manifest.json", manifest)
            target.parent.mkdir(parents=True, exist_ok=True)
            os.replace(temporary, target)
        except Exception:
            if temporary.exists():
                shutil.rmtree(temporary)
            raise

        return SceneMosaicStageArtifacts(
            stage_id=stage_id,
            workspace_dir=target,
            integration_manifest_path=target / "integration_manifest.json",
            relation_artifacts=self.relation_artifacts,
        )

    def _materialize(
        self,
        temporary: Path,
        final_target: Path,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        stage05 = self.contract.stage05_dir
        layout_source = stage05 / "layout_params"
        unposed_source = stage05 / "unposed_objects"
        if not layout_source.is_dir() or not unposed_source.is_dir():
            raise SceneMosaicArtifactError(
                "Stage 05 layout_params and unposed_objects are required"
            )

        layout_target = temporary / "layout_params"
        unposed_target = temporary / "unposed_objects"
        collision_target = temporary / "collision_hulls"
        layout_target.mkdir()
        unposed_target.mkdir()
        collision_target.mkdir()

        copies: list[tuple[str, Path, Path, Path]] = [
            (
                "input_image",
                self.contract.input_image,
                temporary / "image.png",
                final_target / "image.png",
            ),
            (
                "relation_objects",
                self.relation_artifacts.objects_path,
                temporary / "objects.json",
                final_target / "objects.json",
            ),
            (
                "relation_relations",
                self.relation_artifacts.relations_path,
                temporary / "relations.json",
                final_target / "relations.json",
            ),
        ]
        for source in sorted(layout_source.glob("object_*.json")):
            copies.append(
                (
                    "layout_params",
                    source,
                    layout_target / source.name,
                    final_target / "layout_params" / source.name,
                )
            )
        for source in sorted(unposed_source.glob("object_*.glb")):
            copies.append(
                (
                    "unposed_object",
                    source,
                    unposed_target / source.name,
                    final_target / "unposed_objects" / source.name,
                )
            )

        source_records: list[dict[str, Any]] = []
        formal_records: list[dict[str, Any]] = []
        for role, source, temporary_path, final_path in copies:
            source = source.expanduser().resolve()
            if (
                not source.is_file()
                or source.is_symlink()
                or source.stat().st_size == 0
            ):
                raise SceneMosaicArtifactError(
                    f"source must be a real non-empty file: {source}"
                )
            source_sha = _sha256(source)
            shutil.copy2(source, temporary_path)
            formal_sha = _sha256(temporary_path)
            if source_sha != formal_sha:
                raise SceneMosaicArtifactError(
                    f"Stage 08 copy hash mismatch: {source}"
                )
            source_records.append(
                {
                    "role": role,
                    "path": str(source),
                    "sha256": source_sha,
                    "bytes": source.stat().st_size,
                }
            )
            formal_records.append(
                {
                    "role": role,
                    "path": str(final_path),
                    "sha256": formal_sha,
                    "bytes": temporary_path.stat().st_size,
                    "identical_to_source": True,
                }
            )

        layout_ids = tuple(
            record["path"].rsplit("/", 1)[-1].removesuffix(".json")
            for record in formal_records
            if record["role"] == "layout_params"
        )
        unposed_ids = tuple(
            record["path"].rsplit("/", 1)[-1].removesuffix(".glb")
            for record in formal_records
            if record["role"] == "unposed_object"
        )
        try:
            objects = json.loads(
                (temporary / "objects.json").read_text(encoding="utf-8")
            )
            relations = json.loads(
                (temporary / "relations.json").read_text(encoding="utf-8")
            )
        except (OSError, json.JSONDecodeError) as exc:
            raise SceneMosaicArtifactError(
                "copied relation artifacts are not valid JSON"
            ) from exc
        object_items = (
            objects.get("objects", []) if isinstance(objects, dict) else []
        )
        object_ids = tuple(
            str(item.get("id") or item.get("object_id") or "")
            for item in object_items
            if isinstance(item, dict)
        )
        if not (
            object_ids
            and object_ids == layout_ids == unposed_ids
            and isinstance(relations, list)
        ):
            raise SceneMosaicArtifactError(
                "Stage 08 object, layout and asset IDs are not closed"
            )
        graph = RelationGraphView.from_relations(relations)
        if set(graph.primary_parent_by_target) != set(object_ids):
            raise SceneMosaicArtifactError(
                "Stage 08 relations do not cover the object ID closure"
            )
        _write_json(
            temporary / "relation_graph_audit.json",
            graph.audit_summary(),
        )
        return source_records, formal_records

    @staticmethod
    def verify_upstream_unchanged(
        integration_manifest_path: Path,
    ) -> dict[str, Any]:
        try:
            manifest = json.loads(
                Path(integration_manifest_path).read_text(encoding="utf-8")
            )
        except (OSError, json.JSONDecodeError) as exc:
            raise SceneMosaicArtifactError(
                "cannot parse Stage 08 integration manifest"
            ) from exc
        records = manifest.get("source_artifacts")
        if not isinstance(records, list):
            raise SceneMosaicArtifactError(
                "integration manifest lacks source hashes"
            )
        failures: list[dict[str, Any]] = []
        for record in records:
            path = Path(str(record.get("path") or "")).expanduser().resolve()
            if not path.is_file():
                failures.append(
                    {"path": str(path), "reason": "missing"}
                )
                continue
            actual = _sha256(path)
            if actual != record.get("sha256"):
                failures.append(
                    {
                        "path": str(path),
                        "reason": "sha256_mismatch",
                        "actual": actual,
                        "expected": record.get("sha256"),
                    }
                )
        return {
            "valid": not failures,
            "checked": len(records),
            "failures": failures,
        }

    @staticmethod
    def verify_prepared_working_copies(
        integration_manifest_path: Path,
    ) -> dict[str, Any]:
        manifest_path = Path(
            integration_manifest_path
        ).expanduser().resolve()
        try:
            manifest = json.loads(
                manifest_path.read_text(encoding="utf-8")
            )
        except (OSError, json.JSONDecodeError) as exc:
            raise SceneMosaicArtifactError(
                "cannot parse Stage 08 integration manifest"
            ) from exc
        workspace = manifest_path.parent.resolve()
        records = manifest.get("formal_working_copies")
        if (
            manifest.get("schema_version")
            != "scenemosaic_stage08_integration"
            or manifest.get("wall_padding") != 0.0
            or manifest.get("full_relations_preserved") is not True
            or manifest.get("per_object_physics_consumed") is not False
            or not isinstance(records, list)
            or not records
        ):
            raise SceneMosaicArtifactError(
                "prepared Stage 08 integration facts are invalid"
            )
        failures: list[dict[str, Any]] = []
        seen: set[Path] = set()
        for record in records:
            if not isinstance(record, dict):
                failures.append({"reason": "invalid_record"})
                continue
            path = Path(
                str(record.get("path") or "")
            ).expanduser().resolve()
            if (
                path in seen
                or not path.is_relative_to(workspace)
                or not path.is_file()
                or path.is_symlink()
                or record.get("identical_to_source") is not True
            ):
                failures.append(
                    {"path": str(path), "reason": "missing_or_unsafe"}
                )
                continue
            seen.add(path)
            actual = _sha256(path)
            if (
                actual != record.get("sha256")
                or path.stat().st_size != record.get("bytes")
            ):
                failures.append(
                    {
                        "path": str(path),
                        "reason": "hash_or_size_mismatch",
                        "actual": actual,
                        "expected": record.get("sha256"),
                    }
                )
        return {
            "valid": not failures,
            "checked": len(records),
            "failures": failures,
        }

    def _validate_filtered_relation_view(
        self,
        workspace: Path,
    ) -> dict[str, Any]:
        relations_path = workspace / "relations_filtered.json"
        objects_path = workspace / "objects_filtered.json"
        if not relations_path.is_file() or not objects_path.is_file():
            raise SceneMosaicArtifactError(
                "final Stage 08 output requires filtered objects and relations"
            )
        try:
            filtered_relations = json.loads(
                relations_path.read_text(encoding="utf-8")
            )
            filtered_objects = json.loads(
                objects_path.read_text(encoding="utf-8")
            )
            full_relations = json.loads(
                (workspace / "relations.json").read_text(encoding="utf-8")
            )
        except (OSError, json.JSONDecodeError) as exc:
            raise SceneMosaicArtifactError(
                "Stage 08 filtered relation artifacts are not valid JSON"
            ) from exc
        if (
            not isinstance(filtered_relations, list)
            or not isinstance(full_relations, list)
            or not isinstance(filtered_objects, dict)
            or not isinstance(filtered_objects.get("objects"), list)
        ):
            raise SceneMosaicArtifactError(
                "Stage 08 filtered relation artifacts have invalid shapes"
            )
        object_ids = {
            str(item.get("id") or item.get("object_id") or "")
            for item in filtered_objects["objects"]
            if isinstance(item, dict)
        }
        if "" in object_ids:
            raise SceneMosaicArtifactError(
                "objects_filtered.json contains an empty object ID"
            )
        try:
            expected = RelationGraphView.from_relations(
                full_relations
            ).filtered_to_object_ids(object_ids)
            actual = RelationGraphView.from_relations(filtered_relations)
        except RelationGraphError as exc:
            raise SceneMosaicArtifactError(
                f"filtered relation graph is invalid: {exc}"
            ) from exc
        if actual.full_relations_list() != expected.full_relations_list():
            raise SceneMosaicArtifactError(
                "relations_filtered.json is not the exact surviving full graph"
            )
        return {
            "surviving_object_ids": sorted(object_ids),
            "graph": actual.audit_summary(),
            "secondary_wall_constraints_preserved": True,
            "primary_ownership_only": True,
        }

    def _validate_final_outputs(
        self,
        artifacts: SceneMosaicStageArtifacts,
        report_path: Path,
    ) -> dict[str, Any]:
        workspace = artifacts.workspace_dir.resolve()
        expected_workspace = self.attempt_workspace(artifacts.stage_id).resolve()
        if (
            workspace != expected_workspace
            or not workspace.is_dir()
            or artifacts.relation_artifacts.publication_id
            != self.relation_artifacts.publication_id
        ):
            raise SceneMosaicArtifactError(
                "Stage 08 artifacts do not belong to this adapter"
            )
        if (
            not report_path.is_file()
            or report_path.is_symlink()
            or report_path.stat().st_size == 0
            or not report_path.is_relative_to(workspace)
        ):
            raise SceneMosaicArtifactError(
                "pipeline report must be a real file inside Stage 08"
            )
        try:
            report = json.loads(report_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise SceneMosaicArtifactError(
                "pipeline report is not valid JSON"
            ) from exc
        if not isinstance(report, dict) or report.get("stage") != "composition":
            raise SceneMosaicArtifactError(
                "pipeline report must describe the composition stage"
            )

        integration_path = workspace / "integration_manifest.json"
        try:
            integration = json.loads(
                integration_path.read_text(encoding="utf-8")
            )
        except (OSError, json.JSONDecodeError) as exc:
            raise SceneMosaicArtifactError(
                "Stage 08 integration manifest is invalid"
            ) from exc
        if (
            integration.get("stage_id") != artifacts.stage_id
            or integration.get("relation_publication_id")
            != self.relation_artifacts.publication_id
            or integration.get("wall_padding") != 0.0
            or integration.get("full_relations_preserved") is not True
            or integration.get("per_object_physics_consumed") is not False
        ):
            raise SceneMosaicArtifactError(
                "Stage 08 integration invariants are not satisfied"
            )

        combined_root = workspace / "generated" / "combined_scenes"
        combo_index_path = combined_root / "combo_index.json"
        try:
            combo_index = json.loads(
                combo_index_path.read_text(encoding="utf-8")
            )
        except (OSError, json.JSONDecodeError) as exc:
            raise SceneMosaicArtifactError(
                "final Stage 08 combo_index.json is missing or invalid"
            ) from exc
        entries = (
            combo_index.get("combined_scenes")
            if isinstance(combo_index, dict)
            else None
        )
        count = (
            combo_index.get("combined_scene_count")
            if isinstance(combo_index, dict)
            else None
        )
        if (
            not isinstance(entries, list)
            or not isinstance(count, int)
            or count <= 0
            or count != len(entries)
        ):
            raise SceneMosaicArtifactError(
                "final combo index must contain at least one scene"
            )

        glb_paths: list[Path] = []
        for path in sorted(combined_root.rglob("*")):
            if path.is_symlink():
                raise SceneMosaicArtifactError(
                    f"final combined output cannot be a symlink: {path}"
                )
            if path.is_file() and path.suffix.lower() == ".glb":
                if path.stat().st_size == 0:
                    raise SceneMosaicArtifactError(
                        f"final combined GLB is empty: {path}"
                    )
                glb_paths.append(path)
        if not glb_paths:
            raise SceneMosaicArtifactError(
                "final Stage 08 output has no non-empty GLB"
            )

        return {
            "workspace": workspace,
            "report": report_path,
            "integration": integration_path,
            "combined_root": combined_root,
            "combo_index": combo_index_path,
            "combo_index_payload": combo_index,
            "glb_paths": glb_paths,
            "filtered_relations": self._validate_filtered_relation_view(
                workspace
            ),
        }

    @staticmethod
    def _copy_publication_file(
        *,
        source: Path,
        workspace: Path,
        temporary: Path,
        publication_dir: Path,
    ) -> dict[str, Any]:
        if (
            not source.is_file()
            or source.is_symlink()
            or source.stat().st_size == 0
            or not source.resolve().is_relative_to(workspace)
        ):
            raise SceneMosaicArtifactError(
                f"publication source must be a real workspace file: {source}"
            )
        relative = source.relative_to(workspace)
        destination = temporary / "outputs" / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        before = _sha256(source)
        shutil.copy2(source, destination)
        copied = _sha256(destination)
        after = _sha256(source)
        if before != copied or before != after:
            raise SceneMosaicArtifactError(
                f"Stage 08 output changed while publishing: {source}"
            )
        return {
            "source_path": str(source),
            "path": str(publication_dir / "outputs" / relative),
            "relative_path": str(Path("outputs") / relative),
            "sha256": copied,
            "bytes": destination.stat().st_size,
        }

    def publish(
        self,
        artifacts: SceneMosaicStageArtifacts,
        *,
        pipeline_report_path: Path,
    ) -> Path:
        report_path = Path(pipeline_report_path).expanduser().resolve()
        final = self._validate_final_outputs(artifacts, report_path)
        upstream = self.verify_upstream_unchanged(
            artifacts.integration_manifest_path
        )
        if upstream["valid"] is not True:
            raise SceneMosaicArtifactError(
                "upstream facts changed during Stage 08"
            )

        publication_dir = self.publications_dir / artifacts.stage_id
        if publication_dir.exists():
            raise SceneMosaicArtifactError(
                f"Stage 08 publication already exists: {publication_dir}"
            )
        temporary = (
            publication_dir.parent
            / f".{artifacts.stage_id}.{uuid.uuid4().hex}.tmp"
        )
        temporary.mkdir(parents=True, exist_ok=False)
        try:
            workspace = final["workspace"]
            selected: set[Path] = {
                final["report"],
                final["integration"],
                workspace / "relation_graph_audit.json",
                workspace / "objects.json",
                workspace / "relations.json",
                workspace / "objects_filtered.json",
                workspace / "relations_filtered.json",
            }
            optional = workspace / "cross_anchor_constraints.json"
            if optional.is_file():
                selected.add(optional)
            for path in final["combined_root"].rglob("*"):
                if path.is_symlink():
                    raise SceneMosaicArtifactError(
                        f"final combined output cannot be a symlink: {path}"
                    )
                if path.is_file():
                    selected.add(path)

            output_records = [
                self._copy_publication_file(
                    source=path,
                    workspace=workspace,
                    temporary=temporary,
                    publication_dir=publication_dir,
                )
                for path in sorted(selected)
            ]
            glb_relative_paths = {
                str(path.relative_to(workspace))
                for path in final["glb_paths"]
            }
            published_glbs = [
                {
                    "relative_path": record["relative_path"],
                    "sha256": record["sha256"],
                    "bytes": record["bytes"],
                }
                for record in output_records
                if record["relative_path"].removeprefix("outputs/")
                in glb_relative_paths
            ]
            if not published_glbs:
                raise SceneMosaicArtifactError(
                    "no final GLB was copied into the publication"
                )
            representative_records = self._materialize_representative_scenes(
                temporary=temporary,
                publication_dir=publication_dir,
                combined_root=final["combined_root"],
                published_glbs=published_glbs,
            )
            output_records.extend(representative_records)
            publication_index = {
                "schema_version": "scenemosaic_stage08_publication_index",
                "source_combo_index": str(
                    Path("outputs")
                    / final["combo_index"].relative_to(workspace)
                ),
                "combined_scene_count": final["combo_index_payload"][
                    "combined_scene_count"
                ],
                "published_glbs": published_glbs,
            }
            _write_json(
                temporary / "publication_combo_index.json",
                publication_index,
            )
            manifest = {
                "schema_version": "scenemosaic_stage08_publication",
                "stage_id": artifacts.stage_id,
                "published_at_utc": _utc_now(),
                "immutable_copy": True,
                "source_workspace_dir": str(workspace),
                "pipeline_report": str(
                    Path("outputs") / report_path.relative_to(workspace)
                ),
                "relation_publication_id": (
                    artifacts.relation_artifacts.publication_id
                ),
                "relation_publication_manifest_sha256": _sha256(
                    artifacts.relation_artifacts.publication_manifest_path
                ),
                "combo_index": publication_index["source_combo_index"],
                "publication_combo_index": "publication_combo_index.json",
                "glb_count": len(published_glbs),
                "outputs": output_records,
                "filtered_relations": final["filtered_relations"],
                "upstream_unchanged": upstream,
            }
            _write_json(
                temporary / "publication_manifest.json", manifest
            )
            publication_dir.parent.mkdir(parents=True, exist_ok=True)
            os.replace(temporary, publication_dir)
        except Exception:
            if temporary.exists():
                shutil.rmtree(temporary)
            raise

        current = {
            "schema_version": "scenemosaic_stage08_current",
            "active": True,
            "stage_id": artifacts.stage_id,
            "publication_manifest": str(
                Path(".stage08_publications")
                / artifacts.stage_id
                / "publication_manifest.json"
            ),
            "updated_at_utc": _utc_now(),
        }
        _atomic_write_json(
            self.publications_dir / "current.json",
            current,
        )
        self._publish_native_outputs(publication_dir)
        return publication_dir / "publication_manifest.json"

    @staticmethod
    def _materialize_representative_scenes(
        *,
        temporary: Path,
        publication_dir: Path,
        combined_root: Path,
        published_glbs: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Expose ranked final GLBs in the native evolution output tree."""

        summary_path = combined_root / "selection_summary.json"
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise SceneMosaicArtifactError(
                "final Stage 08 selection summary is missing or invalid"
            ) from exc
        selected = summary.get("selected_scenes")
        if not isinstance(selected, list) or len(selected) != len(published_glbs):
            raise SceneMosaicArtifactError(
                "final Stage 08 selected scenes do not match published GLBs"
            )
        ranks: dict[str, int] = {}
        for item in selected:
            if not isinstance(item, dict):
                raise SceneMosaicArtifactError("final Stage 08 selected scene is invalid")
            name = str(item.get("combo_name") or "")
            rank = item.get("rank")
            if not name or not isinstance(rank, int) or rank < 1 or name in ranks:
                raise SceneMosaicArtifactError("final Stage 08 selected scene rank is invalid")
            ranks[name] = rank
        if set(ranks.values()) != set(range(1, len(published_glbs) + 1)):
            raise SceneMosaicArtifactError("final Stage 08 selected scene ranks are incomplete")

        result_dir = temporary / "outputs" / "representative_scenes"
        result_dir.mkdir(parents=True, exist_ok=False)
        records: list[dict[str, Any]] = []
        for glb in published_glbs:
            relative = Path(str(glb["relative_path"]))
            source = temporary / relative
            combo_name = source.parent.name
            rank = ranks.get(combo_name)
            if rank is None or not source.is_file() or source.is_symlink():
                raise SceneMosaicArtifactError(
                    "final Stage 08 representative GLB is invalid"
                )
            destination = result_dir / f"scene_{rank:02d}.glb"
            os.link(source, destination)
            copied = _sha256(destination)
            if copied != glb["sha256"] or destination.stat().st_size != glb["bytes"]:
                raise SceneMosaicArtifactError(
                    "final Stage 08 representative GLB hash binding failed"
                )
            representative_relative = (
                Path("outputs") / "representative_scenes" / destination.name
            )
            records.append(
                {
                    "source_path": str(publication_dir / relative),
                    "path": str(publication_dir / representative_relative),
                    "relative_path": str(representative_relative),
                    "sha256": copied,
                    "bytes": destination.stat().st_size,
                }
            )
        return records

    def _publish_native_outputs(self, publication_dir: Path) -> None:
        """Atomically materialize the original evolution ``outputs/`` layout.

        Every regular file is hard-linked to the hash-bound immutable
        publication.  The public view is therefore a normal directory that
        costs no additional GLB storage and cannot diverge silently.
        """

        target = self.stage08_root / "outputs"
        if target.exists() or target.is_symlink():
            raise SceneMosaicArtifactError(
                "native evolution outputs already exist and will not be overwritten: "
                f"{target}"
            )
        source = publication_dir / "outputs"
        if not source.is_dir() or source.is_symlink():
            raise SceneMosaicArtifactError("Stage 08 publication outputs are missing")
        temporary = self.stage08_root / f".outputs.{uuid.uuid4().hex}.tmp"
        try:
            temporary.mkdir(parents=False, exist_ok=False)
            for source_path in sorted(source.rglob("*")):
                if source_path.is_symlink():
                    raise SceneMosaicArtifactError(
                        f"Stage 08 publication output is a symlink: {source_path}"
                    )
                destination = temporary / source_path.relative_to(source)
                if source_path.is_dir():
                    destination.mkdir()
                elif source_path.is_file():
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    os.link(source_path, destination)
                else:
                    raise SceneMosaicArtifactError(
                        f"Stage 08 publication output has unsupported type: {source_path}"
                    )
            os.replace(temporary, target)
        except Exception:
            if temporary.exists() or temporary.is_symlink():
                if temporary.is_dir() and not temporary.is_symlink():
                    shutil.rmtree(temporary)
                else:
                    temporary.unlink()
            raise

    def validate_publication(
        self,
        publication_manifest_path: Path | None = None,
    ) -> dict[str, Any]:
        """Revalidate an immutable Stage 08 publication and active pointer."""

        current_path = self.publications_dir / "current.json"
        try:
            current = json.loads(
                current_path.read_text(encoding="utf-8")
            )
        except (OSError, json.JSONDecodeError) as exc:
            raise SceneMosaicArtifactError(
                "Stage 08 active publication pointer is missing or invalid"
            ) from exc
        stage_id = str(current.get("stage_id") or "")
        if (
            current_path.is_symlink()
            or current.get("schema_version")
            != "scenemosaic_stage08_current"
            or current.get("active") is not True
            or not SAFE_STAGE_ID_RE.fullmatch(stage_id)
        ):
            raise SceneMosaicArtifactError(
                "Stage 08 active publication pointer is unsafe"
            )
        declared = Path(
            str(current.get("publication_manifest") or "")
        )
        if declared.is_absolute():
            raise SceneMosaicArtifactError(
                "Stage 08 current pointer must use a relative manifest path"
            )
        manifest_path = (
            self.stage08_root / declared
        ).resolve()
        expected_manifest = (
            self.publications_dir / stage_id / "publication_manifest.json"
        ).resolve()
        if (
            manifest_path != expected_manifest
            or (
                publication_manifest_path is not None
                and Path(
                    publication_manifest_path
                ).expanduser().resolve()
                != manifest_path
            )
            or not manifest_path.is_file()
            or manifest_path.is_symlink()
            or manifest_path.stat().st_size == 0
        ):
            raise SceneMosaicArtifactError(
                "Stage 08 current pointer and publication manifest disagree"
            )
        try:
            manifest = json.loads(
                manifest_path.read_text(encoding="utf-8")
            )
        except (OSError, json.JSONDecodeError) as exc:
            raise SceneMosaicArtifactError(
                "Stage 08 publication manifest is invalid"
            ) from exc
        if (
            manifest.get("schema_version")
            != "scenemosaic_stage08_publication"
            or manifest.get("stage_id") != stage_id
            or manifest.get("immutable_copy") is not True
            or manifest.get("relation_publication_id")
            != self.relation_artifacts.publication_id
            or manifest.get("relation_publication_manifest_sha256")
            != _sha256(
                self.relation_artifacts.publication_manifest_path
            )
            or not isinstance(manifest.get("upstream_unchanged"), dict)
            or manifest["upstream_unchanged"].get("valid") is not True
        ):
            raise SceneMosaicArtifactError(
                "Stage 08 publication invariants are invalid"
            )

        publication_dir = manifest_path.parent.resolve()
        outputs_root = publication_dir / "outputs"
        records = manifest.get("outputs")
        if not isinstance(records, list) or not records:
            raise SceneMosaicArtifactError(
                "Stage 08 publication has no output hash closure"
            )
        record_by_relative: dict[str, dict[str, Any]] = {}
        expected_output_paths: set[Path] = set()
        for record in records:
            if not isinstance(record, dict):
                raise SceneMosaicArtifactError(
                    "Stage 08 publication output record is invalid"
                )
            relative = str(record.get("relative_path") or "")
            relative_path = Path(relative)
            path = Path(
                str(record.get("path") or "")
            ).expanduser().resolve()
            expected = (
                publication_dir / relative_path
            ).resolve()
            if (
                not relative
                or relative in record_by_relative
                or relative_path.is_absolute()
                or not relative_path.is_relative_to(Path("outputs"))
                or path != expected
                or not path.is_relative_to(outputs_root)
                or not path.is_file()
                or path.is_symlink()
                or path.stat().st_size == 0
                or path.stat().st_size != record.get("bytes")
                or _sha256(path) != record.get("sha256")
            ):
                raise SceneMosaicArtifactError(
                    f"Stage 08 publication output verification failed: {relative}"
                )
            record_by_relative[relative] = record
            expected_output_paths.add(path)

        actual_output_paths: set[Path] = set()
        if not outputs_root.is_dir() or outputs_root.is_symlink():
            raise SceneMosaicArtifactError(
                "Stage 08 publication outputs directory is unsafe"
            )
        for path in outputs_root.rglob("*"):
            if path.is_symlink():
                raise SceneMosaicArtifactError(
                    f"Stage 08 publication contains a symlink: {path}"
                )
            if path.is_file():
                actual_output_paths.add(path.resolve())
        if actual_output_paths != expected_output_paths:
            raise SceneMosaicArtifactError(
                "Stage 08 publication output closure has missing or extra files"
            )

        combo_index_name = str(
            manifest.get("publication_combo_index") or ""
        )
        if combo_index_name != "publication_combo_index.json":
            raise SceneMosaicArtifactError(
                "Stage 08 publication combo index path is invalid"
            )
        publication_combo_path = publication_dir / combo_index_name
        try:
            publication_combo = json.loads(
                publication_combo_path.read_text(encoding="utf-8")
            )
        except (OSError, json.JSONDecodeError) as exc:
            raise SceneMosaicArtifactError(
                "Stage 08 publication combo index is invalid"
            ) from exc
        published_glbs = publication_combo.get("published_glbs")
        combined_count = publication_combo.get("combined_scene_count")
        source_combo = str(
            publication_combo.get("source_combo_index") or ""
        )
        if (
            publication_combo_path.is_symlink()
            or publication_combo.get("schema_version")
            != "scenemosaic_stage08_publication_index"
            or source_combo != manifest.get("combo_index")
            or source_combo not in record_by_relative
            or not isinstance(combined_count, int)
            or combined_count <= 0
            or not isinstance(published_glbs, list)
            or not published_glbs
            or len(published_glbs) != manifest.get("glb_count")
        ):
            raise SceneMosaicArtifactError(
                "Stage 08 publication combo facts are inconsistent"
            )
        for glb in published_glbs:
            if not isinstance(glb, dict):
                raise SceneMosaicArtifactError(
                    "Stage 08 published GLB record is invalid"
                )
            relative = str(glb.get("relative_path") or "")
            output = record_by_relative.get(relative)
            if (
                output is None
                or Path(relative).suffix.lower() != ".glb"
                or glb.get("sha256") != output.get("sha256")
                or glb.get("bytes") != output.get("bytes")
            ):
                raise SceneMosaicArtifactError(
                    f"Stage 08 published GLB hash binding failed: {relative}"
                )

        report_relative = str(manifest.get("pipeline_report") or "")
        report_record = record_by_relative.get(report_relative)
        if report_record is None:
            raise SceneMosaicArtifactError(
                "Stage 08 publication pipeline report is not hash-bound"
            )
        try:
            report = json.loads(
                Path(str(report_record["path"])).read_text(
                    encoding="utf-8"
                )
            )
        except (OSError, json.JSONDecodeError) as exc:
            raise SceneMosaicArtifactError(
                "Stage 08 published pipeline report is invalid"
            ) from exc
        filtered = manifest.get("filtered_relations")
        if (
            not isinstance(report, dict)
            or report.get("stage") != "composition"
            or not isinstance(filtered, dict)
            or filtered.get(
                "secondary_wall_constraints_preserved"
            )
            is not True
            or filtered.get("primary_ownership_only") is not True
        ):
            raise SceneMosaicArtifactError(
                "Stage 08 published relation-consumption facts are invalid"
            )

        allowed_top_level = {
            manifest_path.resolve(),
            publication_combo_path.resolve(),
        }
        actual_top_level = {
            path.resolve()
            for path in publication_dir.iterdir()
            if path.is_file()
        }
        if actual_top_level != allowed_top_level:
            raise SceneMosaicArtifactError(
                "Stage 08 publication has unexpected top-level files"
            )
        return {
            "valid": True,
            "stage_id": stage_id,
            "publication_manifest": str(manifest_path),
            "publication_manifest_sha256": _sha256(manifest_path),
            "current_pointer": str(current_path),
            "output_count": len(records),
            "glb_count": len(published_glbs),
            "combined_scene_count": combined_count,
            "relation_publication_id": (
                self.relation_artifacts.publication_id
            ),
        }


class IntegratedToolRunner(ToolRunner):
    """Existing SceneMosaic tools with integration invariants enforced."""

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
        if wall_padding is not None and float(wall_padding) != 0.0:
            raise ValueError(
                "published relation walls require wall_padding=0.0"
            )
        return super().generate_room_mesh(
            layout_dir,
            unposed_dir,
            objects_json,
            material_dir=material_dir,
            thickness=thickness,
            wall_padding=0.0,
        )


class RelationAwareLayoutCombination(LayoutCombination):
    """Combination driven by the same filtered primary ownership graph."""

    def _load_relationships_and_objects(
        self,
    ) -> tuple[list[dict], list[dict], list[dict]]:
        filtered_relations = self.paths.outputs_dir / "relations_filtered.json"
        filtered_objects = self.paths.outputs_dir / "objects_filtered.json"
        if filtered_relations.exists() != filtered_objects.exists():
            raise RelationGraphError(
                "filtered objects and relations must be published together"
            )
        suffix = "_filtered" if filtered_relations.is_file() else ""
        relations_path = self.paths.outputs_dir / f"relations{suffix}.json"
        objects_path = self.paths.outputs_dir / f"objects{suffix}.json"
        try:
            relations = json.loads(
                relations_path.read_text(encoding="utf-8")
            )
            objects_payload = json.loads(
                objects_path.read_text(encoding="utf-8")
            )
        except (OSError, json.JSONDecodeError) as exc:
            raise RelationGraphError(
                "combination relation inputs are invalid"
            ) from exc
        if not isinstance(relations, list) or not isinstance(
            objects_payload, dict
        ):
            raise RelationGraphError(
                "combination relation inputs have invalid shapes"
            )
        objects = objects_payload.get("objects")
        walls = objects_payload.get("walls")
        if not isinstance(objects, list) or not isinstance(walls, list):
            raise RelationGraphError(
                "combination objects must preserve objects and walls"
            )
        view = RelationGraphView.from_relations(relations)
        object_ids = {
            str(item.get("id") or item.get("object_id") or "")
            for item in objects
            if isinstance(item, dict)
        }
        if set(view.primary_parent_by_target) != object_ids:
            raise RelationGraphError(
                "combination relation targets do not match surviving objects"
            )
        if suffix:
            full_path = self.paths.outputs_dir / "relations.json"
            full = json.loads(full_path.read_text(encoding="utf-8"))
            expected = RelationGraphView.from_relations(
                full
            ).filtered_to_object_ids(object_ids)
            if view.full_relations_list() != expected.full_relations_list():
                raise RelationGraphError(
                    "combination must consume the exact filtered relation graph"
                )
        return relations, objects, walls

    @staticmethod
    def _load_relationship_graph(
        relations: list[dict],
    ) -> tuple[dict[str, set[str]], dict[str, set[str]]]:
        view = RelationGraphView.from_relations(relations)
        parents_by_child = {
            target: {parent}
            for target, parent in view.primary_parent_by_target.items()
        }
        children_by_parent = {
            anchor: set(targets)
            for anchor, targets in view.primary_children_by_anchor.items()
        }
        return parents_by_child, children_by_parent

    def _choose_parent_by_child(
        self, parents_by_child: dict[str, set[str]]
    ) -> dict[str, str]:
        selected: dict[str, str] = {}
        for target, parents in sorted(parents_by_child.items()):
            if len(parents) != 1:
                raise RelationGraphError(
                    f"target {target} must have one deterministic primary owner"
                )
            selected[target] = next(iter(parents))
        return selected

    @staticmethod
    def _mesh_wall_distance(
        *,
        object_id: str,
        layout_dir: Path,
        unposed_dir: Path,
        wall: dict[str, Any],
    ) -> float:
        layout_path = layout_dir / f"{object_id}.json"
        mesh_path = unposed_dir / f"{object_id}.glb"
        try:
            layout = json.loads(layout_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise RelationGraphError(
                f"cannot validate wall constraint layout for {object_id}"
            ) from exc
        if not isinstance(layout, dict) or not mesh_path.is_file():
            raise RelationGraphError(
                f"wall constraint artifacts are missing for {object_id}"
            )
        try:
            mesh = trimesh.load(mesh_path, force="mesh", process=False)
            vertices = np.asarray(mesh.vertices, dtype=np.float64)
            transform = np.asarray(
                _layout_payload_to_matrix(layout), dtype=np.float64
            )
        except Exception as exc:
            raise RelationGraphError(
                f"cannot load wall constraint geometry for {object_id}"
            ) from exc
        if (
            vertices.ndim != 2
            or vertices.shape[1] != 3
            or vertices.shape[0] == 0
            or not np.isfinite(vertices).all()
            or transform.shape != (4, 4)
            or not np.isfinite(transform).all()
        ):
            raise RelationGraphError(
                f"wall constraint geometry is invalid for {object_id}"
            )
        world = (
            np.concatenate(
                [vertices, np.ones((vertices.shape[0], 1), dtype=np.float64)],
                axis=1,
            )
            @ transform.T
        )[:, :3]
        first = np.asarray(wall.get("first_point"), dtype=np.float64).reshape(-1)
        second = np.asarray(
            wall.get("second_point"), dtype=np.float64
        ).reshape(-1)
        if (
            first.size != 2
            or second.size != 2
            or not np.isfinite(first).all()
            or not np.isfinite(second).all()
        ):
            raise RelationGraphError("wall endpoints must contain two finite values")
        segment = second - first
        length_squared = float(segment @ segment)
        if length_squared <= 1e-12:
            raise RelationGraphError("wall segment must have non-zero length")
        points = world[:, [0, 2]]
        projection = np.clip(
            ((points - first) @ segment) / length_squared,
            0.0,
            1.0,
        )
        closest = first + projection[:, None] * segment
        distances = np.linalg.norm(points - closest, axis=1)
        return float(np.min(distances))

    def _secondary_wall_candidate_audit(
        self, candidate: LayoutVariantCandidate
    ) -> dict[str, Any]:
        relations, _, walls = self._load_relationships_and_objects()
        view = RelationGraphView.from_relations(relations)
        walls_by_id = {
            str(wall.get("id")): wall
            for wall in walls
            if isinstance(wall, dict) and wall.get("id")
        }
        checks: list[dict[str, Any]] = []
        for target in sorted(view.secondary_wall_constraints_by_target):
            if view.primary_parent_by_target.get(target) != candidate.anchor_id:
                continue
            for relation in view.secondary_wall_constraints_by_target[target]:
                parameters = relation.get("parameters", {})
                wall_id = str(parameters.get("object_anchor") or "")
                wall = walls_by_id.get(wall_id)
                if wall is None:
                    raise RelationGraphError(
                        f"secondary wall constraint references missing {wall_id}"
                    )
                baseline_distance = self._mesh_wall_distance(
                    object_id=target,
                    layout_dir=self.paths.layout_params_dir,
                    unposed_dir=self.paths.unposed_objects_dir,
                    wall=wall,
                )
                candidate_distance = self._mesh_wall_distance(
                    object_id=target,
                    layout_dir=candidate.layout_dir,
                    unposed_dir=self.paths.unposed_objects_dir,
                    wall=wall,
                )
                allowed_distance = max(0.05, baseline_distance + 0.02)
                checks.append(
                    {
                        "constraint_type": "secondary_wall_attach",
                        "primary_owner": candidate.anchor_id,
                        "secondary_wall": wall_id,
                        "object_target": target,
                        "baseline_distance": baseline_distance,
                        "candidate_distance": candidate_distance,
                        "allowed_distance": allowed_distance,
                        "valid": candidate_distance <= allowed_distance,
                    }
                )
        return {
            "schema_version": "scenemosaic_secondary_wall_candidate_audit",
            "candidate": candidate.label,
            "valid": all(item["valid"] for item in checks),
            "checks": checks,
            "secondary_constraints_create_ownership": False,
        }

    def _local_variant_is_valid(
        self,
        candidate: LayoutVariantCandidate,
        artifacts: SceneArtifacts,
        generator: Any,
        validation_root: Path,
    ) -> bool:
        if not super()._local_variant_is_valid(
            candidate, artifacts, generator, validation_root
        ):
            return False
        audit = self._secondary_wall_candidate_audit(candidate)
        _atomic_write_json(
            validation_root
            / f"{candidate.label}_secondary_wall_constraints.json",
            audit,
        )
        if not audit["valid"]:
            print(
                "Discarding local variant "
                f"{candidate.label}: secondary wall constraint failed"
            )
            return False
        return True


class IntegratedPipelineOrchestrator(PipelineOrchestrator):
    """SceneMosaic pipeline using one relation interpretation end to end."""

    def _group_objects_by_anchor(
        self,
        relationships: list,
        objects: list,
        walls: list,
    ) -> dict[str, dict[str, Any]]:
        del objects, walls
        view = RelationGraphView.from_relations(relationships)
        self._latest_relation_graph_view = view
        return view.object_groups()

    @staticmethod
    def _secondary_wall_constraint_records(
        object_groups: dict[str, dict[str, Any]],
    ) -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = []
        for primary_owner, group in sorted(object_groups.items()):
            if not isinstance(group, dict):
                continue
            relations = group.get("secondary_wall_constraints", [])
            if not isinstance(relations, list):
                raise RelationGraphError(
                    "secondary_wall_constraints must be a relation list"
                )
            for relation in relations:
                if not isinstance(relation, dict):
                    raise RelationGraphError(
                        "secondary wall relation must be a JSON object"
                    )
                parameters = relation.get("parameters")
                if not isinstance(parameters, dict):
                    raise RelationGraphError(
                        "secondary wall relation parameters are invalid"
                    )
                wall_id = str(parameters.get("object_anchor") or "")
                target = str(parameters.get("object_target") or "")
                if (
                    not wall_id.startswith("wall")
                    or not target
                    or wall_id == primary_owner
                ):
                    raise RelationGraphError(
                        "secondary wall relation identity is invalid"
                    )
                records.append(
                    {
                        "constraint_id": (
                            f"secondary_wall_attach:{primary_owner}:"
                            f"{wall_id}:{target}"
                        ),
                        "constraint_type": "secondary_wall_attach",
                        "source": "relation_graph_view",
                        "description": (
                            f"{target} keeps primary ownership under "
                            f"{primary_owner} while remaining attached to "
                            f"{wall_id}."
                        ),
                        "primary_owner": primary_owner,
                        "secondary_wall": wall_id,
                        "object_target": target,
                        "relation": relation,
                        "anchor_a": {
                            "anchor_id": primary_owner,
                            "related_objects": [target],
                            "layout_note": (
                                f"Keep {target} flush with the room boundary "
                                f"defined by {wall_id}; do not create another "
                                "copy or change its primary owner."
                            ),
                        },
                        "anchor_b": {
                            "anchor_id": wall_id,
                            "related_objects": [target],
                            "layout_note": (
                                f"Preserve the existing {target} contact with "
                                f"{wall_id} as a secondary constraint."
                            ),
                        },
                    }
                )
        return sorted(
            records,
            key=lambda item: str(item["constraint_id"]),
        )

    def _analyze_cross_anchor_constraints(
        self, object_groups: dict, objects: list
    ) -> list[dict[str, Any]]:
        payload = super()._analyze_cross_anchor_constraints(
            object_groups, objects
        )
        if not isinstance(payload, list) or not all(
            isinstance(item, dict) for item in payload
        ):
            raise RelationGraphError(
                "cross-anchor analysis must return a JSON array of objects"
            )
        if any(
            item.get("constraint_type") == "secondary_wall_attach"
            or item.get("source") == "relation_graph_view"
            for item in payload
        ):
            raise RelationGraphError(
                "VLM cross-anchor output used reserved relation constraint fields"
            )
        combined = [
            *payload,
            *self._secondary_wall_constraint_records(object_groups),
        ]
        _atomic_write_json(
            self.paths.outputs_dir / "cross_anchor_constraints.json",
            combined,
        )
        return combined

    def _validate_secondary_wall_constraint_file(
        self,
        object_groups: dict[str, dict[str, Any]],
    ) -> dict[str, Any]:
        path = self.paths.outputs_dir / "cross_anchor_constraints.json"
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise RelationGraphError(
                "cross-anchor constraint file is missing or invalid"
            ) from exc
        if not isinstance(payload, list) or not all(
            isinstance(item, dict) for item in payload
        ):
            raise RelationGraphError(
                "cross-anchor constraint file must contain a JSON array"
            )
        expected = self._secondary_wall_constraint_records(object_groups)
        actual = [
            item
            for item in payload
            if item.get("constraint_type") == "secondary_wall_attach"
            or item.get("source") == "relation_graph_view"
        ]
        if actual != expected:
            raise RelationGraphError(
                "secondary wall generation constraints are missing or stale"
            )
        identifiers = [
            str(item["constraint_id"]) for item in expected
        ]
        if len(identifiers) != len(set(identifiers)):
            raise RelationGraphError(
                "secondary wall generation constraint IDs are not unique"
            )
        return {
            "schema_version": "scenemosaic_secondary_wall_generation_audit",
            "valid": True,
            "path": str(path),
            "sha256": _sha256(path),
            "constraint_count": len(expected),
            "constraint_ids": identifiers,
            "consumed_by_generation": True,
            "candidate_validation_required": True,
            "secondary_constraints_create_ownership": False,
        }

    def _stage_generate_scene(
        self,
        artifacts: SceneArtifacts,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        relationships, objects, walls = (
            self._load_relationships_and_objects(suffix="_filtered")
        )
        object_groups = self._group_objects_by_anchor(
            relationships, objects, walls
        )
        constraints_path = (
            self.paths.outputs_dir / "cross_anchor_constraints.json"
        )
        if constraints_path.is_file():
            self._validate_secondary_wall_constraint_file(object_groups)
        else:
            self._analyze_cross_anchor_constraints(
                object_groups, objects
            )
        super()._stage_generate_scene(artifacts, *args, **kwargs)
        audit = self._validate_secondary_wall_constraint_file(object_groups)
        relation_audit_path = (
            self.paths.outputs_dir / "relation_graph_audit.json"
        )
        relation_audit = json.loads(
            relation_audit_path.read_text(encoding="utf-8")
        )
        if not isinstance(relation_audit, dict):
            raise RelationGraphError("relation graph audit is invalid")
        relation_audit["secondary_wall_generation"] = audit
        _atomic_write_json(relation_audit_path, relation_audit)

    def _stage_optimization(
        self, artifacts: SceneArtifacts
    ) -> SceneArtifacts:
        artifacts = super()._stage_optimization(artifacts)
        self._preserve_filtered_room_materials()
        self._write_relation_graph_audit()
        return artifacts

    def _preserve_filtered_room_materials(self) -> None:
        source_path = self.paths.outputs_dir / "objects.json"
        filtered_path = self.paths.outputs_dir / "objects_filtered.json"
        if not source_path.is_file() or not filtered_path.is_file():
            return
        source = json.loads(source_path.read_text(encoding="utf-8"))
        filtered = json.loads(filtered_path.read_text(encoding="utf-8"))
        room_materials = (
            source.get("room_materials")
            if isinstance(source, dict)
            else None
        )
        if not isinstance(filtered, dict) or not isinstance(
            room_materials, dict
        ):
            raise RelationGraphError(
                "room_materials cannot be preserved in filtered objects"
            )
        filtered["room_materials"] = room_materials
        _write_json(filtered_path, filtered)

    def _write_relation_graph_audit(self) -> None:
        relations_path = self.paths.outputs_dir / "relations.json"
        filtered_path = self.paths.outputs_dir / "relations_filtered.json"
        objects_filtered_path = self.paths.outputs_dir / "objects_filtered.json"
        if not relations_path.is_file():
            raise RelationGraphError("full relations.json is missing")
        full = json.loads(relations_path.read_text(encoding="utf-8"))
        full_view = RelationGraphView.from_relations(full)
        payload: dict[str, Any] = {
            "full_relations": full_view.audit_summary(),
            "full_relations_preserved": True,
            "secondary_wall_constraints_create_ownership": False,
        }
        if filtered_path.is_file() or objects_filtered_path.is_file():
            if not filtered_path.is_file() or not objects_filtered_path.is_file():
                raise RelationGraphError(
                    "filtered objects and relations must be written together"
                )
            filtered = json.loads(
                filtered_path.read_text(encoding="utf-8")
            )
            objects_payload = json.loads(
                objects_filtered_path.read_text(encoding="utf-8")
            )
            objects = (
                objects_payload.get("objects")
                if isinstance(objects_payload, dict)
                else None
            )
            if not isinstance(filtered, list) or not isinstance(objects, list):
                raise RelationGraphError(
                    "filtered relation audit inputs are invalid"
                )
            object_ids = {
                str(item.get("id") or item.get("object_id") or "")
                for item in objects
                if isinstance(item, dict)
            }
            expected = full_view.filtered_to_object_ids(object_ids)
            actual = RelationGraphView.from_relations(filtered)
            if actual.full_relations_list() != expected.full_relations_list():
                raise RelationGraphError(
                    "filtered relation graph differs from surviving objects"
                )
            payload["filtered_relations"] = actual.audit_summary()
            payload["combination_uses_filtered_relations"] = True
        _write_json(
            self.paths.outputs_dir / "relation_graph_audit.json",
            payload,
        )

    def run_full_pipeline(self) -> PipelineReport:
        artifacts = SceneArtifacts(
            unposed_objects_dir=self.paths.unposed_objects_dir,
            layout_params_dir=self.paths.layout_params_dir,
            collision_hulls_dir=self.paths.collision_hulls_dir,
            composed_scene_mesh=self.paths.composed_scene_path,
            simulated_layout_dir=self.paths.simulated_layout_dir,
        )
        artifacts = self._stage_optimization(artifacts)
        self._stage_generate_scene(
            artifacts, max_refine_rounds=self.max_rounds
        )
        combined_scene_paths = RelationAwareLayoutCombination(
            self.paths, self.tool_runner
        ).run(artifacts)
        combined_scene_index = (
            self.paths.outputs_dir
            / "generated"
            / "combined_scenes"
            / "combo_index.json"
        )
        if not combined_scene_paths:
            raise RelationGraphError(
                "Stage 08 produced no composed GLB scenes"
            )
        if (
            not combined_scene_index.is_file()
            or any(
                not path.is_file() or path.stat().st_size == 0
                for path in combined_scene_paths
            )
        ):
            raise RelationGraphError(
                "Stage 08 final combination outputs are incomplete"
            )
        artifacts = replace(
            artifacts, composed_scene_mesh=combined_scene_index
        )
        report = PipelineReport(
            stage="composition",
            artifacts=artifacts,
            tool_outputs={
                "combined_scene_index": str(combined_scene_index),
                "combined_scene_count": str(len(combined_scene_paths)),
                "wall_padding": "0.0",
                "per_object_physics_consumed": "false",
                "combination_uses_filtered_relations": "true",
            },
        )
        self.history.append(report)
        return report

    def save_report(
        self,
        report: PipelineReport,
        output_path: Path | None = None,
    ) -> Path:
        path = super().save_report(report, output_path)
        payload = json.loads(path.read_text(encoding="utf-8"))
        audit_path = self.paths.outputs_dir / "relation_graph_audit.json"
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
        payload["integration"] = {
            "relation_graph": audit,
            "full_relations_preserved": True,
            "wall_padding": 0.0,
            "per_object_physics_consumed": False,
            "combination_uses_filtered_relations": True,
        }
        _write_json(path, payload)
        return path

