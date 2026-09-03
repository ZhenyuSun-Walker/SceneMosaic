from __future__ import annotations

import hashlib
import json
import os
import uuid
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from .evolution.vlm_client import VLMClient
from .reconstruction.perception.stage import (
    PerceptionStage,
    PerceptionStageError,
)
from .evolution.evolution_stage import (
    IntegratedPipelineOrchestrator,
    IntegratedToolRunner,
    SceneMosaicArtifactError,
    SceneMosaicStageAdapter,
    SceneMosaicStageArtifacts,
)
from .evolution.paths import ProjectPaths
from .reconstruction.relations.stage import (
    RelationArtifacts,
    RelationStageAdapter,
    RelationStageError,
)
from .state import RunState, RunStateError
from .config import RuntimeConfig
from .reconstruction.sam3d_stage import (
    Sam3DStage,
    Sam3DStageError,
    validate_stage05,
)
from .artifacts import SceneRun, initialize_scene_run, load_scene_run


class FullPipelineError(RuntimeError):
    exit_code = 2


class FullPipelineChildError(FullPipelineError):
    exit_code = 3


class FullPipelineIntegrityError(FullPipelineError):
    exit_code = 4


@dataclass(frozen=True)
class FullRunOptions:
    input_image: Path | None = None
    scene_name: str | None = None
    scene_root: Path | None = None
    run_root: Path | None = None
    perception_reuse_local_pre_pass_b_cache_root: Path | None = None
    perception_pass_b_reasoning_effort: str = "none"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _optional_url(value: str | None) -> str | None:
    if value is None:
        return None
    normalized = value.strip()
    return normalized if normalized and normalized != "None" else None


class FullSceneOrchestrator:

    def __init__(self, runtime: RuntimeConfig) -> None:
        self.runtime = runtime

    def verify(self, *, require_gpu: bool = False) -> dict[str, Any]:
        return self.runtime.verify(require_gpu=require_gpu)

    def _scene(self, options: FullRunOptions) -> SceneRun:
        starting = options.input_image is not None or options.scene_name is not None
        resuming = options.scene_root is not None
        if starting == resuming:
            raise FullPipelineError(
                "run requires either --input-image with --scene-name, or --scene-root"
            )
        if starting:
            if options.input_image is None or not options.scene_name:
                raise FullPipelineError(
                    "new run requires both input_image and scene_name"
                )
            return initialize_scene_run(
                self.runtime,
                scene_name=options.scene_name,
                input_image=options.input_image,
                run_root=options.run_root,
            )
        assert options.scene_root is not None
        if options.run_root is not None:
            raise FullPipelineError("run_root is valid only for a new scene")
        return load_scene_run(options.scene_root)

    def _relation_adapter(self, scene: SceneRun) -> RelationStageAdapter:
        return RelationStageAdapter(
            scene_config_path=scene.contract.scene_config_path,
            relation_package_root=self.runtime.relation_package_root,
            renderer_script=self.runtime.renderer_path,
        )

    def _aggregate_audit(self, scene: SceneRun) -> Path:
        state = scene.state.public_status()
        stages = state.get("stages", {})
        payload: dict[str, Any] = {
            "schema_version": "scenemosaic_full_pipeline_audit",
            "generated_at_utc": _utc_now(),
            "scene_name": scene.contract.scene_name,
            "scene_root": str(scene.contract.run_root),
            "vlm_model": self.runtime.vlm_model,
            "current_state": state.get("current_state"),
            "last_successful_state": state.get("last_successful_state"),
            "next_action": state.get("next_action"),
            "failure": state.get("failure"),
            "releases": {
                "perception": "scenemosaic-perception",
                "relation": "scenemosaic-relations",
                "sam3d_source": "SceneMosaic/sam-3d-objects",
            },
            "stages": stages,
            "stage_attempts": state.get("stage_attempts", {}),
            "invocations": state.get("invocations", []),
        }

        path = scene.contract.run_root / "audit" / "pipeline_audit.json"
        _atomic_json(path, payload)
        return path

    def _prepared_stage08(
        self,
        scene: SceneRun,
        relation: RelationArtifacts,
        adapter: SceneMosaicStageAdapter,
    ) -> SceneMosaicStageArtifacts | None:
        record = scene.state.data.get("stages", {}).get("stage08")
        if not isinstance(record, dict) or record.get("status") != "prepared":
            return None
        facts = record.get("facts")
        if not isinstance(facts, dict):
            raise FullPipelineIntegrityError("Stage 08 state facts are missing")
        stage_id = str(facts.get("stage_id") or "")
        workspace = Path(str(facts.get("workspace_dir") or "")).resolve()
        manifest = Path(str(facts.get("integration_manifest") or "")).resolve()
        expected = adapter.attempt_workspace(stage_id)
        if workspace != expected or manifest != workspace / "integration_manifest.json":
            raise FullPipelineIntegrityError("Stage 08 state paths are inconsistent")
        if not workspace.is_dir() or not manifest.is_file():
            raise FullPipelineIntegrityError("prepared Stage 08 workspace is missing")
        manifest_payload = json.loads(manifest.read_text(encoding="utf-8"))
        if manifest_payload.get("relation_publication_id") != relation.publication_id:
            raise FullPipelineIntegrityError(
                "prepared Stage 08 uses a different relation publication"
            )
        verification = adapter.verify_upstream_unchanged(manifest)
        if verification.get("valid") is not True:
            raise FullPipelineIntegrityError(
                "prepared Stage 08 upstream hashes changed"
            )
        working_copies = adapter.verify_prepared_working_copies(manifest)
        if working_copies.get("valid") is not True:
            raise FullPipelineIntegrityError(
                "prepared Stage 08 working-copy hashes changed"
            )
        if (workspace / "pipeline_report.json").exists():
            raise FullPipelineIntegrityError(
                "prepared Stage 08 already started and cannot be resumed in place"
            )
        return SceneMosaicStageArtifacts(
            stage_id=stage_id,
            workspace_dir=workspace,
            integration_manifest_path=manifest,
            relation_artifacts=relation,
        )

    def _validate_completed_scene(
        self,
        scene: SceneRun,
    ) -> dict[str, Any]:
        runtime_report = self.verify(require_gpu=False)
        if runtime_report.get("valid") is not True:
            raise FullPipelineIntegrityError(
                "completed run no longer passes runtime verification"
            )
        perception = PerceptionStage(
            self.runtime, scene.contract.scene_config_path
        ).reuse_formal()
        if perception is None:
            raise FullPipelineIntegrityError(
                "completed run has no valid formal perception publication"
            )
        sam3d = validate_stage05(
            scene.contract.scene_config_path,
            project_root=self.runtime.project_root,
        )
        relation = self._relation_adapter(scene).reuse()
        adapter = SceneMosaicStageAdapter(
            scene_config_path=scene.contract.scene_config_path,
            relation_artifacts=relation,
        )
        record = scene.state.data.get("stages", {}).get("stage08")
        if (
            not isinstance(record, dict)
            or record.get("status") != "published"
            or not isinstance(record.get("facts"), dict)
        ):
            raise FullPipelineIntegrityError(
                "completed run has no published Stage 08 state facts"
            )
        facts = record["facts"]
        stage_id = str(facts.get("stage_id") or "")
        if not stage_id:
            raise FullPipelineIntegrityError("completed Stage 08 state has no stage_id")
        workspace = Path(str(facts.get("workspace_dir") or "")).expanduser().resolve()
        integration = (
            Path(str(facts.get("integration_manifest") or "")).expanduser().resolve()
        )
        publication = (
            Path(str(facts.get("publication_manifest") or "")).expanduser().resolve()
        )
        expected_workspace = adapter.attempt_workspace(stage_id).resolve()
        if (
            workspace != expected_workspace
            or integration != workspace / "integration_manifest.json"
            or not integration.is_file()
            or integration.is_symlink()
            or facts.get("relation_publication_id") != relation.publication_id
            or facts.get("wall_padding") != 0.0
        ):
            raise FullPipelineIntegrityError(
                "completed Stage 08 state paths or invariants are inconsistent"
            )
        upstream = adapter.verify_upstream_unchanged(integration)
        if upstream.get("valid") is not True:
            raise FullPipelineIntegrityError(
                "completed Stage 08 upstream facts changed"
            )
        publication_facts = adapter.validate_publication(publication)
        expected_sha = facts.get("publication_manifest_sha256")
        if (
            publication_facts.get("stage_id") != stage_id
            or not isinstance(expected_sha, str)
            or publication_facts.get("publication_manifest_sha256") != expected_sha
        ):
            raise FullPipelineIntegrityError(
                "completed Stage 08 publication no longer matches pipeline state"
            )
        return {
            "runtime": runtime_report,
            "perception": {
                "delivery_status": (perception.artifacts.delivery_status),
                "object_ids": list(perception.artifacts.object_ids),
                "run_manifest_sha256": (perception.run_manifest_sha256),
            },
            "sam3d": {
                "object_ids": list(sam3d.object_ids),
                "source_manifest_sha256": (sam3d.source_manifest_sha256),
            },
            "relation": {
                "publication_id": relation.publication_id,
                "publication_manifest_sha256": _sha256(
                    relation.publication_manifest_path
                ),
            },
            "stage08": publication_facts,
        }

    def run(
        self,
        options: FullRunOptions,
        *,
        sanitized_argv: Sequence[str] = (),
        environ: Mapping[str, str] | None = None,
    ) -> dict[str, Any]:
        scene = self._scene(options)
        state = scene.state
        try:
            scene_lock = state.acquire_lock(exclusive=True, blocking=False)
        except RunStateError as exc:
            raise FullPipelineIntegrityError(str(exc)) from exc
        state.close_stale_invocations()
        invocation_id = state.begin_invocation(
            command="run", sanitized_argv=list(sanitized_argv)
        )
        try:
            if state.current_state == "scenemosaic_published":
                verification = self._validate_completed_scene(scene)
                publication = verification["stage08"]["publication_manifest"]
                state.finish_invocation(invocation_id, result="complete_reuse")
                return {
                    "complete": True,
                    "status": "scenemosaic_published",
                    "scene_root": str(scene.contract.run_root),
                    "publication_manifest": publication,
                    "reused": True,
                    "fact_verification": verification,
                    "pipeline_audit": str(self._aggregate_audit(scene)),
                }
            runtime_report = self.verify(require_gpu=False)
            if not runtime_report.get("valid"):
                raise FullPipelineIntegrityError(
                    "runtime preflight failed; run python -m vlm_agent.cli verify for details"
                )

            perception = PerceptionStage(self.runtime, scene.contract.scene_config_path)

            perception_result = perception.run(
                approve_gpu=True,
                approve_api=True,
                approve_third_party_upload=True,
                reuse_local_pre_pass_b_cache_root=(
                    options.perception_reuse_local_pre_pass_b_cache_root
                ),
                pass_b_reasoning_effort=(
                    options.perception_pass_b_reasoning_effort
                ),
                environ=environ,
            )
            state.record_stage(
                "perception",
                status="published",
                successful_state="stage04_published",
                facts={
                    "delivery_status": perception_result.artifacts.delivery_status,
                    "object_ids": list(perception_result.artifacts.object_ids),
                    "run_manifest": str(perception_result.run_manifest_path),
                    "run_manifest_sha256": perception_result.run_manifest_sha256,
                    "stage03": str(perception_result.artifacts.stage03_dir),
                    "stage04": str(perception_result.artifacts.stage04_dir),
                    "subprocess_started": perception_result.subprocess_started,
                    "audit": str(perception_result.audit_path)
                    if perception_result.audit_path
                    else None,
                },
            )

            sam3d = Sam3DStage(
                scene.contract.scene_config_path,
                runtime_config=self.runtime,
            )
            try:
                sam3d_artifacts = validate_stage05(
                    scene.contract.scene_config_path,
                    project_root=self.runtime.project_root,
                )
                sam3d_artifacts = replace(sam3d_artifacts, reused=True)
            except Sam3DStageError:
                dry_run = sam3d.dry_run()
                state.record_stage("sam3d_dry_run", status="passed", facts=dry_run)
                sam3d_artifacts = sam3d.run(approve_gpu=True)
            state.record_stage(
                "sam3d",
                status="published",
                successful_state="sam3d_published",
                facts={
                    "stage05_dir": str(sam3d_artifacts.stage05_dir),
                    "object_ids": list(sam3d_artifacts.object_ids),
                    "source_manifest_sha256": sam3d_artifacts.source_manifest_sha256,
                    "artifact_hashes": sam3d_artifacts.artifact_hashes,
                    "attempt_id": sam3d_artifacts.attempt_id,
                    "reused": sam3d_artifacts.reused,
                },
            )

            relation_adapter = self._relation_adapter(scene)
            relation: RelationArtifacts | None = None
            try:
                relation = relation_adapter.reuse()
            except RelationStageError:
                relation_record = state.data.get("stages", {}).get("relation", {})
                relation_facts = (
                    relation_record.get("facts", {})
                    if isinstance(relation_record, dict)
                    else {}
                )
                relation_status = (
                    relation_record.get("status")
                    if isinstance(relation_record, dict)
                    else None
                )
                candidate_raw = (
                    relation_facts.get("candidate_dir")
                    if relation_status
                    in {
                        "candidate_running",
                        "candidate_valid",
                        "candidate_interrupted",
                    }
                    else None
                )
                recovery_candidate = (
                    Path(candidate_raw).expanduser().resolve()
                    if isinstance(candidate_raw, str) and candidate_raw.strip()
                    else None
                )
                if recovery_candidate is not None and recovery_candidate.is_dir():
                    try:
                        recovered = relation_adapter.validate_candidate(
                            recovery_candidate
                        )
                        relation = relation_adapter.publish_candidate(
                            candidate_dir=recovered.candidate_dir,
                            publication_id=(
                                f"publication_{recovered.candidate_dir.name}"
                            ),
                        )
                    except RelationStageError as exc:
                        state.record_attempt(
                            "relation",
                            attempt_id=recovery_candidate.name,
                            status="recovery_failed",
                            facts={
                                "candidate_dir": str(recovery_candidate),
                                "error": str(exc),
                            },
                        )
                    else:
                        state.record_attempt(
                            "relation",
                            attempt_id=recovery_candidate.name,
                            status="recovered_and_published",
                            facts={
                                "candidate_dir": str(recovery_candidate),
                                "validation": str(recovered.validation_path),
                                "publication_id": relation.publication_id,
                            },
                        )

                if relation is None:
                    if not self.runtime.has_api_credentials(
                        "relation", environ=environ
                    ):
                        raise FullPipelineError(
                            f"environment {self.runtime.relation_api_key_env} "
                            "is required for Relation execution"
                        )
                    timestamp = datetime.now(timezone.utc).strftime(
                        "%Y%m%dT%H%M%SZ"
                    )
                    benchmark_id = (
                        f"relation_{scene.contract.scene_name}_"
                        f"{timestamp}_{uuid.uuid4().hex[:8]}"
                    )
                    candidate_dir = (
                        scene.contract.relation_candidates_dir / benchmark_id
                    )
                    state.record_stage(
                        "relation",
                        status="candidate_running",
                        facts={
                            "candidate_id": benchmark_id,
                            "candidate_dir": str(candidate_dir),
                            "invocation_id": invocation_id,
                        },
                    )
                    state.record_attempt(
                        "relation",
                        attempt_id=benchmark_id,
                        status="started",
                        facts={
                            "candidate_dir": str(candidate_dir),
                            "invocation_id": invocation_id,
                        },
                    )
                    source_env = os.environ if environ is None else environ
                    api_base = _optional_url(
                        source_env.get(self.runtime.relation_api_base_env)
                    )
                    candidate = relation_adapter.run_candidate(
                        benchmark_id=benchmark_id,
                        api_key_env=self.runtime.relation_api_key_env,
                        model=self.runtime.vlm_model,
                        precompute_python=(
                            self.runtime.relation_precompute_python
                        ),
                        runner_python=self.runtime.relation_runner_python,
                        api_base=api_base,
                        environ=self.runtime.child_environment(
                            "relation", environ=environ
                        ),
                    )
                    validated = relation_adapter.validate_candidate(
                        candidate.candidate_dir
                    )
                    state.record_attempt(
                        "relation",
                        attempt_id=benchmark_id,
                        status="candidate_valid",
                        facts={
                            "candidate_dir": str(candidate.candidate_dir),
                            "validation": str(validated.validation_path),
                        },
                    )
                    relation = relation_adapter.publish_candidate(
                        candidate_dir=validated.candidate_dir,
                        publication_id=f"publication_{benchmark_id}",
                    )
                    state.record_attempt(
                        "relation",
                        attempt_id=benchmark_id,
                        status="published",
                        facts={
                            "candidate_dir": str(candidate.candidate_dir),
                            "validation": str(validated.validation_path),
                            "publication_id": relation.publication_id,
                        },
                    )

            assert relation is not None
            state.record_stage(
                "relation",
                status="published",
                successful_state="relation_published",
                facts={
                    "publication_id": relation.publication_id,
                    "publication_manifest": str(
                        relation.publication_manifest_path
                    ),
                    "publication_manifest_sha256": _sha256(
                        relation.publication_manifest_path
                    ),
                    "objects": str(relation.objects_path),
                    "relations": str(relation.relations_path),
                    "candidate_validated": True,
                },
            )

            stage08_adapter = SceneMosaicStageAdapter(
                scene_config_path=scene.contract.scene_config_path,
                relation_artifacts=relation,
            )
            stage08 = self._prepared_stage08(scene, relation, stage08_adapter)
            if stage08 is None:
                stage_id = (
                    "stage08_"
                    + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ_")
                    + uuid.uuid4().hex[:8]
                )
                stage08 = stage08_adapter.prepare(stage_id)
                state.record_stage(
                    "stage08",
                    status="prepared",
                    successful_state="stage08_prepared",
                    facts={
                        "stage_id": stage08.stage_id,
                        "workspace_dir": str(stage08.workspace_dir),
                        "integration_manifest": str(stage08.integration_manifest_path),
                        "relation_publication_id": relation.publication_id,
                        "wall_padding": 0.0,
                        "per_object_physics_consumed": False,
                    },
                )
                state.record_attempt(
                    "stage08",
                    attempt_id=stage08.stage_id,
                    status="prepared",
                    facts={
                        "workspace_dir": str(stage08.workspace_dir),
                        "integration_manifest": str(stage08.integration_manifest_path),
                    },
                )

            source_env = os.environ if environ is None else environ
            secret = source_env.get(self.runtime.scenemosaic_api_key_env)
            if not secret:
                raise FullPipelineError(
                    f"environment {self.runtime.scenemosaic_api_key_env} "
                    "is required for scene evolution"
                )
            state.record_attempt(
                "stage08",
                attempt_id=stage08.stage_id,
                status="started",
                facts={
                    "workspace_dir": str(stage08.workspace_dir),
                    "invocation_id": invocation_id,
                },
            )
            state.record_stage(
                "stage08",
                status="running",
                facts={
                    "stage_id": stage08.stage_id,
                    "workspace_dir": str(stage08.workspace_dir),
                    "integration_manifest": str(stage08.integration_manifest_path),
                },
            )
            api_base = _optional_url(
                source_env.get(self.runtime.scenemosaic_api_base_env)
            )
            client = VLMClient(
                url=api_base,
                model=self.runtime.vlm_model,
                api_key=secret,
            )
            paths = ProjectPaths(
                self.runtime.project_root, output_dir=stage08.workspace_dir
            )
            downstream = IntegratedPipelineOrchestrator(
                paths=paths,
                tool_runner=IntegratedToolRunner(self.runtime.project_root),
                vlm_client=client,
                max_rounds=3,
                collision_policy="strict",
            )
            try:
                report = downstream.run_full_pipeline()
                report_path = downstream.save_report(
                    report, stage08.workspace_dir / "pipeline_report.json"
                )
            except Exception as exc:
                state.record_attempt(
                    "stage08",
                    attempt_id=stage08.stage_id,
                    status="failed",
                    facts={
                        "workspace_dir": str(stage08.workspace_dir),
                        "error_type": type(exc).__name__,
                    },
                )
                state.fail(
                    stage="stage08",
                    error_type=type(exc).__name__,
                    message="SceneMosaic downstream failed; inspect the retained workspace",
                    resumable=False,
                    next_action=(
                        "inspect the retained Stage 08 attempt; the next run will create a new stage08_id"
                    ),
                )
                state.finish_invocation(invocation_id, result="failed")
                self._aggregate_audit(scene)
                raise FullPipelineChildError("SceneMosaic downstream failed") from None
            publication = stage08_adapter.publish(
                stage08, pipeline_report_path=report_path
            )
            state.record_attempt(
                "stage08",
                attempt_id=stage08.stage_id,
                status="published",
                facts={
                    "workspace_dir": str(stage08.workspace_dir),
                    "publication_manifest": str(publication),
                    "publication_manifest_sha256": _sha256(publication),
                },
            )
            state.record_stage(
                "stage08",
                status="published",
                successful_state="scenemosaic_published",
                facts={
                    "stage_id": stage08.stage_id,
                    "workspace_dir": str(stage08.workspace_dir),
                    "integration_manifest": str(stage08.integration_manifest_path),
                    "pipeline_report": str(report_path),
                    "publication_manifest": str(publication),
                    "publication_manifest_sha256": _sha256(publication),
                    "relation_publication_id": relation.publication_id,
                    "per_object_physics_consumed": False,
                    "wall_padding": 0.0,
                },
            )
            state.complete()
            state.finish_invocation(invocation_id, result="complete")
            audit_path = self._aggregate_audit(scene)
            return {
                "complete": True,
                "status": "scenemosaic_published",
                "scene_root": str(scene.contract.run_root),
                "stage08_id": stage08.stage_id,
                "publication_manifest": str(publication),
                "pipeline_audit": str(audit_path),
            }
        except FullPipelineError as exc:
            if state.current_state != "failed":
                state.fail(
                    stage="integrity",
                    error_type=type(exc).__name__,
                    message=str(exc),
                    resumable=False,
                    next_action="run python -m vlm_agent.cli verify and inspect immutable facts",
                )
            if any(
                item.get("invocation_id") == invocation_id
                and item.get("result") == "running"
                for item in state.data.get("invocations", [])
            ):
                state.finish_invocation(invocation_id, result="failed")
            self._aggregate_audit(scene)
            raise
        except (
            PerceptionStageError,
            Sam3DStageError,
            RelationStageError,
            SceneMosaicArtifactError,
        ) as exc:
            stage = (
                "perception"
                if isinstance(exc, PerceptionStageError)
                else "sam3d"
                if isinstance(exc, Sam3DStageError)
                else "relation"
                if isinstance(exc, RelationStageError)
                else "stage08"
            )
            state.fail(
                stage=stage,
                error_type=type(exc).__name__,
                message=str(exc),
                resumable=stage != "stage08",
                next_action="inspect the retained audit and resume after correcting the failure",
            )
            state.finish_invocation(invocation_id, result="failed")
            self._aggregate_audit(scene)
            child = "failed with exit code" in str(exc) or "runner failed" in str(exc)
            error_type = FullPipelineChildError if child else FullPipelineIntegrityError
            raise error_type(str(exc)) from None
        except Exception as exc:
            state.fail(
                stage="orchestrator",
                error_type=type(exc).__name__,
                message=str(exc),
                resumable=False,
                next_action="inspect pipeline state and facts before retrying",
            )
            state.finish_invocation(invocation_id, result="failed")
            self._aggregate_audit(scene)
            raise FullPipelineIntegrityError(str(exc)) from None
        finally:
            scene_lock.release()

    def status(self, scene_root: Path) -> dict[str, Any]:
        scene = load_scene_run(scene_root)
        try:
            scene_lock = scene.state.acquire_lock(exclusive=False, blocking=False)
        except RunStateError as exc:
            raise FullPipelineIntegrityError(str(exc)) from exc
        try:
            scene = load_scene_run(scene_root)
            state = scene.state.public_status()
            stages = state.get("stages", {})
            checks: list[dict[str, Any]] = []

            def check(name: str, function: Any) -> None:
                try:
                    value = function()
                except Exception as exc:
                    checks.append(
                        {
                            "name": name,
                            "valid": False,
                            "error": str(exc),
                        }
                    )
                else:
                    checks.append(
                        {
                            "name": name,
                            "valid": True,
                            "facts": value,
                        }
                    )

            check(
                "runtime",
                lambda: self.verify(require_gpu=False),
            )

            perception_record = stages.get("perception")
            perception_expected = (
                isinstance(perception_record, dict)
                or (scene.contract.stage03_dir / "perception_adoption.json").exists()
            )
            if perception_expected:

                def perception_facts() -> dict[str, Any]:
                    result = PerceptionStage(
                        self.runtime,
                        scene.contract.scene_config_path,
                    ).reuse_formal()
                    if result is None:
                        raise FullPipelineIntegrityError(
                            "formal perception publication is missing"
                        )
                    recorded = (
                        perception_record.get("facts", {})
                        if isinstance(perception_record, dict)
                        else {}
                    )
                    object_ids = list(result.artifacts.object_ids)
                    if recorded and (
                        recorded.get("object_ids") != object_ids
                        or recorded.get("run_manifest_sha256")
                        != result.run_manifest_sha256
                    ):
                        raise FullPipelineIntegrityError(
                            "perception facts disagree with pipeline state"
                        )
                    return {
                        "delivery_status": (result.artifacts.delivery_status),
                        "object_ids": object_ids,
                        "run_manifest_sha256": (result.run_manifest_sha256),
                    }

                check("perception", perception_facts)

            sam3d_record = stages.get("sam3d")
            sam3d_expected = (
                isinstance(sam3d_record, dict)
                or (scene.contract.stage05_dir / "object_index_map.json").exists()
            )
            if sam3d_expected:

                def sam3d_facts() -> dict[str, Any]:
                    result = validate_stage05(
                        scene.contract.scene_config_path,
                        project_root=self.runtime.project_root,
                    )
                    recorded = (
                        sam3d_record.get("facts", {})
                        if isinstance(sam3d_record, dict)
                        else {}
                    )
                    object_ids = list(result.object_ids)
                    if recorded and (
                        recorded.get("object_ids") != object_ids
                        or recorded.get("source_manifest_sha256")
                        != result.source_manifest_sha256
                    ):
                        raise FullPipelineIntegrityError(
                            "SAM3D facts disagree with pipeline state"
                        )
                    return {
                        "object_ids": object_ids,
                        "source_manifest_sha256": (result.source_manifest_sha256),
                        "artifact_hashes": result.artifact_hashes,
                    }

                check("sam3d", sam3d_facts)

            relation_record = stages.get("relation")
            relation_current = scene.contract.relations_dir / "current.json"
            relation: RelationArtifacts | None = None
            if relation_current.is_file() or (
                isinstance(relation_record, dict)
                and relation_record.get("status") == "published"
            ):

                def relation_facts() -> dict[str, Any]:
                    nonlocal relation
                    relation = self._relation_adapter(scene).reuse()
                    recorded = (
                        relation_record.get("facts", {})
                        if isinstance(relation_record, dict)
                        else {}
                    )
                    manifest_sha = _sha256(relation.publication_manifest_path)
                    if recorded and (
                        recorded.get("publication_id") != relation.publication_id
                        or recorded.get("publication_manifest_sha256") != manifest_sha
                    ):
                        raise FullPipelineIntegrityError(
                            "Relation facts disagree with pipeline state"
                        )
                    return {
                        "publication_id": relation.publication_id,
                        "publication_manifest": str(relation.publication_manifest_path),
                        "publication_manifest_sha256": manifest_sha,
                        "candidate_validated": True,
                    }

                check("relation", relation_facts)
            elif isinstance(relation_record, dict) and relation_record.get(
                "status"
            ) in {
                "candidate_running",
                "candidate_valid",
                "candidate_interrupted",
            }:

                def relation_candidate_facts() -> dict[str, Any]:
                    facts = relation_record.get("facts", {})
                    candidate_raw = (
                        facts.get("candidate_dir") if isinstance(facts, dict) else None
                    )
                    if not isinstance(candidate_raw, str) or not candidate_raw.strip():
                        raise FullPipelineIntegrityError(
                            "Relation candidate path is missing"
                        )
                    candidate = Path(candidate_raw).expanduser().resolve()
                    expected_root = scene.contract.relation_candidates_dir.resolve()
                    if (
                        candidate.parent != expected_root
                        or not candidate.is_dir()
                        or candidate.is_symlink()
                    ):
                        raise FullPipelineIntegrityError(
                            "Relation candidate path is unsafe"
                        )
                    validation = candidate / ("scenemosaic_candidate_validation.json")
                    if relation_record.get("status") == "candidate_valid" and (
                        not validation.is_file() or validation.is_symlink()
                    ):
                        raise FullPipelineIntegrityError(
                            "validated Relation candidate record is missing"
                        )
                    return {
                        "status": relation_record.get("status"),
                        "candidate_id": candidate.name,
                        "candidate_dir": str(candidate),
                        "validation": (
                            str(validation) if validation.is_file() else None
                        ),
                    }

                check(
                    "relation_candidate",
                    relation_candidate_facts,
                )

            stage08_record = stages.get("stage08")
            if isinstance(stage08_record, dict):
                stage08_status = stage08_record.get("status")
                if stage08_status == "published":
                    check(
                        "stage08",
                        lambda: self._validate_completed_scene(scene)["stage08"],
                    )
                elif stage08_status == "prepared":

                    def prepared_stage08_facts() -> dict[str, Any]:
                        nonlocal relation
                        if relation is None:
                            relation = self._relation_adapter(scene).reuse()
                        adapter = SceneMosaicStageAdapter(
                            scene_config_path=(scene.contract.scene_config_path),
                            relation_artifacts=relation,
                        )
                        prepared = self._prepared_stage08(scene, relation, adapter)
                        if prepared is None:
                            raise FullPipelineIntegrityError(
                                "prepared Stage 08 state is missing"
                            )
                        return {
                            "stage_id": prepared.stage_id,
                            "workspace_dir": str(prepared.workspace_dir),
                            "integration_manifest": str(
                                prepared.integration_manifest_path
                            ),
                            "upstream": (
                                adapter.verify_upstream_unchanged(
                                    prepared.integration_manifest_path
                                )
                            ),
                            "working_copies": (
                                adapter.verify_prepared_working_copies(
                                    prepared.integration_manifest_path
                                )
                            ),
                        }

                    check("stage08_prepared", prepared_stage08_facts)
                elif stage08_status == "running":

                    def running_stage08_facts() -> dict[str, Any]:
                        facts = stage08_record.get("facts", {})
                        workspace = (
                            Path(str(facts.get("workspace_dir") or ""))
                            .expanduser()
                            .resolve()
                        )
                        integration = (
                            Path(str(facts.get("integration_manifest") or ""))
                            .expanduser()
                            .resolve()
                        )
                        stage_id = str(facts.get("stage_id") or "")
                        expected = (
                            scene.contract.evolution_dir
                            / ".stage08_attempts"
                            / stage_id
                        ).resolve()
                        if (
                            workspace != expected
                            or integration != workspace / "integration_manifest.json"
                            or not integration.is_file()
                            or integration.is_symlink()
                        ):
                            raise FullPipelineIntegrityError(
                                "running Stage 08 attempt paths are unsafe"
                            )
                        return {
                            "stage_id": stage_id,
                            "workspace_dir": str(workspace),
                            "upstream": (
                                SceneMosaicStageAdapter.verify_upstream_unchanged(
                                    integration
                                )
                            ),
                            "resumable_in_place": False,
                        }

                    check("stage08_running", running_stage08_facts)

            complete = state.get("current_state") == "scenemosaic_published"
            if complete and not any(item["name"] == "stage08" for item in checks):
                checks.append(
                    {
                        "name": "stage08",
                        "valid": False,
                        "error": ("completed state has no published Stage 08 facts"),
                    }
                )
            runtime_checks = next(
                (
                    item.get("facts", {}).get("checks", [])
                    for item in checks
                    if item["name"] == "runtime" and item["valid"]
                ),
                [],
            )
            if runtime_checks and not all(
                item.get("valid") is True
                for item in runtime_checks
                if item.get("required", True)
            ):
                for item in checks:
                    if item["name"] == "runtime":
                        item["valid"] = False
                        item["error"] = "one or more runtime checks failed"
                        break
            return {
                "complete": complete,
                "scene_root": str(scene.contract.run_root),
                "current_state": state.get("current_state"),
                "last_successful_state": state.get("last_successful_state"),
                "next_action": state.get("next_action"),
                "failure": state.get("failure"),
                "fact_checks": checks,
                "facts_valid": all(item["valid"] for item in checks),
                "pipeline_state": str(scene.state.path),
                "pipeline_audit": str(
                    scene.contract.run_root / "audit" / "pipeline_audit.json"
                ),
                "gpu_used": False,
            }
        finally:
            scene_lock.release()
