from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping

from .adoption import (
    PerceptionAdoptionAdapter,
    PerceptionAdoptionArtifacts,
    PerceptionAdoptionError,
)
from ...artifacts import RunContract
from ...config import RuntimeConfig, verify_perception_runtime


SECRET_NAME_RE = re.compile(
    r"(?:KEY|TOKEN|SECRET|PASSWORD|CREDENTIAL|AUTHORIZATION|BASE_URL)", re.I
)


class PerceptionStageError(RuntimeError):
    """Raised when the perception pipeline cannot run or be adopted safely."""


class PerceptionApprovalRequired(PerceptionStageError):
    """Raised before any paid/upload/GPU operation when approval is absent."""


@dataclass(frozen=True)
class PerceptionRunResult:
    artifacts: PerceptionAdoptionArtifacts
    run_manifest_path: Path
    attempt_output_dir: Path
    subprocess_started: bool
    audit_path: Path | None
    run_manifest_sha256: str



def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


class PerceptionStage:
    """Run the perception pipeline and adopt only its delivery."""

    def __init__(self, runtime: RuntimeConfig, scene_config_path: Path) -> None:
        self.runtime = runtime
        self.contract = RunContract.from_scene_config(scene_config_path)

    def verify_runtime(self) -> dict[str, Any]:
        report = verify_perception_runtime(self.runtime.perception_root)
        if not report.get("valid"):
            raise PerceptionStageError(
                f"perception runtime verification failed: {report}"
            )
        return report

    def adopt(self, run_manifest_path: Path) -> PerceptionRunResult:
        self.verify_runtime()
        manifest = Path(run_manifest_path).expanduser().resolve()
        try:
            artifacts = PerceptionAdoptionAdapter(
                run_manifest_path=manifest,
                scene_config_path=self.contract.scene_config_path,
                perception_attempts_root=self.runtime.perception_attempts_root,
            ).adopt()
        except PerceptionAdoptionError as exc:
            raise PerceptionStageError(str(exc)) from exc
        return PerceptionRunResult(
            artifacts=artifacts,
            run_manifest_path=manifest,
            attempt_output_dir=manifest.parent,
            subprocess_started=False,
            audit_path=None,
            run_manifest_sha256=_sha256(manifest),
        )

    def reuse_formal(self) -> PerceptionRunResult | None:
        adoption_path = self.contract.stage03_dir / "perception_adoption.json"
        if not adoption_path.is_file():
            return None
        self.verify_runtime()
        try:
            adoption = json.loads(adoption_path.read_text(encoding="utf-8"))
            source = adoption["source"]
            manifest = Path(str(source["run_manifest_path"]))
            manifest_sha256 = str(source["run_manifest_sha256"])
        except (OSError, json.JSONDecodeError, KeyError, TypeError) as exc:
            raise PerceptionStageError(
                "existing perception adoption is invalid"
            ) from exc
        try:
            artifacts = PerceptionAdoptionAdapter.validate_formal(
                self.contract.scene_config_path
            )
        except PerceptionAdoptionError as exc:
            raise PerceptionStageError(str(exc)) from exc
        return PerceptionRunResult(
            artifacts=artifacts,
            run_manifest_path=manifest,
            attempt_output_dir=manifest.parent,
            subprocess_started=False,
            audit_path=None,
            run_manifest_sha256=manifest_sha256,
        )

    def run(
        self,
        *,
        approve_gpu: bool,
        approve_api: bool,
        approve_third_party_upload: bool,
        reuse_local_pre_pass_b_cache_root: Path | None = None,
        pass_b_reasoning_effort: str = "none",
        environ: Mapping[str, str] | None = None,
        on_subprocess_start: Callable[[], None] | None = None,
    ) -> PerceptionRunResult:
        existing = self.reuse_formal()
        if existing is not None:
            return existing
        self.verify_runtime()
        missing: list[str] = []
        if not approve_gpu:
            missing.append("GPU approval")
        if not approve_api:
            missing.append("perception API approval")
        if not approve_third_party_upload:
            missing.append("third-party image upload approval")
        if missing:
            raise PerceptionApprovalRequired(
                "perception was not started; missing " + ", ".join(missing)
            )
        recovery_root: Path | None = None
        if reuse_local_pre_pass_b_cache_root is not None:
            trusted_root = self.runtime.perception_attempts_root.resolve(strict=True)
            candidate = Path(reuse_local_pre_pass_b_cache_root).expanduser()
            if candidate.is_symlink():
                raise PerceptionStageError(
                    "perception recovery root must not be a symlink"
                )
            try:
                recovery_root = candidate.resolve(strict=True)
            except OSError as exc:
                raise PerceptionStageError(
                    "perception recovery root does not exist"
                ) from exc
            try:
                recovery_root.relative_to(trusted_root)
            except ValueError as exc:
                raise PerceptionStageError(
                    "perception recovery root is outside perception attempts root"
                ) from exc
            if not recovery_root.is_dir():
                raise PerceptionStageError(
                    "perception recovery root is not a directory"
                )
        if pass_b_reasoning_effort not in {
            None,
            "none",
            "low",
            "medium",
            "high",
            "xhigh",
        }:
            raise PerceptionStageError("invalid Pass-B reasoning effort")
        if not self.runtime.has_api_credentials("perception", environ=environ):
            raise PerceptionApprovalRequired(
                f"perception was not started; environment {self.runtime.perception_api_key_env} is unset"
            )
        gpu_preflight = self.runtime.verify_gpu()
        if not gpu_preflight.get("valid"):
            raise PerceptionApprovalRequired(
                "perception was not started; GPU preflight failed"
            )

        attempt_id = f"{self.contract.scene_name}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}_{uuid.uuid4().hex[:8]}"
        output_dir = self.runtime.perception_attempts_root / attempt_id
        if output_dir.exists():
            raise PerceptionStageError(
                f"isolated perception attempt already exists: {output_dir}"
            )
        output_dir.parent.mkdir(parents=True, exist_ok=True)
        command = [
            str(self.runtime.perception_python),
            str(self.runtime.perception_entrypoint),
            "--scene-config",
            str(self.contract.scene_config_path),
            "--output-dir",
            str(output_dir),
            "--model-dir",
            str(self.runtime.perception_model_dir),
            "--pass-a-model",
            self.runtime.vlm_model,
            "--pass-b-model",
            self.runtime.vlm_model,
            "--allow-third-party-upload",
        ]
        if recovery_root is not None:
            command.extend(
                ["--reuse-local-pre-pass-b-cache-root", str(recovery_root)]
            )
        if pass_b_reasoning_effort is not None:
            command.extend(
                ["--pass-b-reasoning-effort", pass_b_reasoning_effort]
            )
        source_environment = os.environ if environ is None else environ
        secret_values = tuple(
            value
            for name, value in source_environment.items()
            if value and SECRET_NAME_RE.search(name)
        )

        def redact(value: str) -> str:
            result = value
            for secret_value in secret_values:
                result = result.replace(secret_value, "<redacted>")
            if len(result) > 2 * 1024 * 1024:
                result = result[: 2 * 1024 * 1024] + "\n<truncated>\n"
            return result

        if on_subprocess_start is not None:
            on_subprocess_start()
        completed = subprocess.run(
            command,
            cwd=self.runtime.perception_root,
            env=self.runtime.child_environment("perception", environ=environ),
            capture_output=True,
            text=True,
            check=False,
        )
        audit_path = self.contract.run_root / "audit" / f"perception_{attempt_id}.json"
        audit = {
            "schema_version": "scenemosaic_perception_subprocess_audit",
            "attempt_id": attempt_id,
            "command": command,
            "cwd": str(self.runtime.perception_root),
            "returncode": completed.returncode,
            "stdout": redact(completed.stdout),
            "stderr": redact(completed.stderr),
            "release_verification": self.verify_runtime(),
            "gpu_preflight": gpu_preflight,
            "approvals": {
                "gpu": True,
                "api": True,
                "third_party_image_upload": True,
            },
            "recovery": {
                "reuse_local_pre_pass_b_cache_root": (
                    str(recovery_root) if recovery_root is not None else None
                ),
                "pass_b_reasoning_effort": pass_b_reasoning_effort,
                "compact_pass_b_protocol": True,
            },
        }
        _write_json(audit_path, audit)
        if completed.returncode != 0:
            raise PerceptionStageError(
                f"perception failed with return code {completed.returncode}; audit: {audit_path}"
            )
        manifest = output_dir / "run_manifest.json"
        if not manifest.is_file():
            raise PerceptionStageError(
                f"perception returned success without run_manifest.json: {output_dir}"
            )
        try:
            adopted = self.adopt(manifest)
        except PerceptionStageError as exc:
            raise PerceptionStageError(
                f"perception output failed adoption; audit: {audit_path}; {exc}"
            ) from exc
        return PerceptionRunResult(
            artifacts=adopted.artifacts,
            run_manifest_path=manifest,
            attempt_output_dir=output_dir,
            subprocess_started=True,
            audit_path=audit_path,
            run_manifest_sha256=adopted.run_manifest_sha256,
        )
