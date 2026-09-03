from __future__ import annotations

import copy
import fcntl
import json
import os
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, IO, Iterable


STATE_SCHEMA = "scenemosaic_full_pipeline_state"
SECRET_KEYS = frozenset(
    {"api_key", "apikey", "secret", "authorization", "access_token", "refresh_token"}
)


class RunStateError(RuntimeError):
    """Raised when the durable pipeline state is invalid or unsafe."""


@dataclass
class SceneRunLock:
    handle: IO[str]
    exclusive: bool

    def release(self) -> None:
        if self.handle.closed:
            return
        try:
            fcntl.flock(self.handle.fileno(), fcntl.LOCK_UN)
        finally:
            self.handle.close()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sanitize(value: Any, secret_values: Iterable[str] = ()) -> Any:
    secrets = tuple(item for item in secret_values if item)
    if isinstance(value, dict):
        cleaned: dict[str, Any] = {}
        for key, item in value.items():
            normalized = str(key).lower().replace("-", "_")
            if normalized in SECRET_KEYS or normalized.endswith("_api_key"):
                cleaned[str(key)] = "<redacted>"
            else:
                cleaned[str(key)] = _sanitize(item, secrets)
        return cleaned
    if isinstance(value, list):
        return [_sanitize(item, secrets) for item in value]
    if isinstance(value, tuple):
        return [_sanitize(item, secrets) for item in value]
    if isinstance(value, str):
        result = value
        for secret in secrets:
            result = result.replace(secret, "<redacted>")
        return result
    return value


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


@dataclass
class RunState:
    """Atomic index for a fact-validated, resumable full pipeline."""

    path: Path
    data: dict[str, Any]

    @classmethod
    def create(cls, scene_root: Path, *, scene_name: str) -> "RunState":
        root = Path(scene_root).expanduser().resolve()
        path = root / "pipeline_state.json"
        if path.exists():
            raise RunStateError(f"pipeline state already exists: {path}")
        now = utc_now()
        state = cls(
            path=path,
            data={
                "schema_version": STATE_SCHEMA,
                "scene_name": scene_name,
                "scene_root": str(root),
                "current_state": "initialized",
                "last_successful_state": "initialized",
                "next_action": None,
                "failure": None,
                "stages": {},
                "stage_attempts": {},
                "invocations": [],
                "created_at_utc": now,
                "updated_at_utc": now,
            },
        )
        state.save()
        return state

    @classmethod
    def load(cls, scene_root_or_path: Path) -> "RunState":
        supplied = Path(scene_root_or_path).expanduser().resolve()
        path = supplied if supplied.name == "pipeline_state.json" else supplied / "pipeline_state.json"
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise RunStateError(f"cannot parse pipeline state: {path}") from exc
        if not isinstance(payload, dict) or payload.get("schema_version") != STATE_SCHEMA:
            raise RunStateError(f"unsupported pipeline state: {path}")
        expected_root = path.parent.resolve()
        if Path(str(payload.get("scene_root"))).expanduser().resolve() != expected_root:
            raise RunStateError("pipeline state scene_root does not match its location")
        return cls(path=path, data=payload)

    @property
    def current_state(self) -> str:
        return str(self.data.get("current_state") or "")

    @property
    def scene_root(self) -> Path:
        return self.path.parent

    def save(self, *, secret_values: Iterable[str] = ()) -> None:
        self.data["updated_at_utc"] = utc_now()
        self.data = _sanitize(self.data, secret_values)
        _atomic_json(self.path, self.data)

    def acquire_lock(
        self,
        *,
        exclusive: bool = True,
        blocking: bool = False,
    ) -> SceneRunLock:
        lock_path = self.scene_root / ".pipeline.lock"
        if not lock_path.is_file() or lock_path.is_symlink():
            raise RunStateError(f"pipeline lock file is missing or unsafe: {lock_path}")
        handle = lock_path.open("r+", encoding="utf-8")
        operation = fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH
        if not blocking:
            operation |= fcntl.LOCK_NB
        try:
            fcntl.flock(handle.fileno(), operation)
        except BlockingIOError as exc:
            handle.close()
            raise RunStateError(
                "another pipeline invocation currently owns this scene"
            ) from exc
        return SceneRunLock(handle=handle, exclusive=exclusive)

    def close_stale_invocations(self) -> int:
        count = 0
        for record in self.data.get("invocations", []):
            if record.get("result") == "running":
                record["ended_at_utc"] = utc_now()
                record["result"] = "interrupted"
                count += 1
        if count:
            self.save()
        return count

    def begin_invocation(self, *, command: str, sanitized_argv: list[str]) -> str:
        invocation_id = f"invocation_{uuid.uuid4().hex}"
        self.data.setdefault("invocations", []).append(
            {
                "invocation_id": invocation_id,
                "command": command,
                "argv": _sanitize(list(sanitized_argv)),
                "started_at_utc": utc_now(),
                "ended_at_utc": None,
                "result": "running",
            }
        )
        self.save()
        return invocation_id

    def finish_invocation(self, invocation_id: str, *, result: str) -> None:
        for record in reversed(self.data.get("invocations", [])):
            if record.get("invocation_id") == invocation_id:
                record["ended_at_utc"] = utc_now()
                record["result"] = result
                self.save()
                return
        raise RunStateError(f"unknown invocation: {invocation_id}")

    def record_attempt(
        self,
        stage: str,
        *,
        attempt_id: str,
        status: str,
        facts: dict[str, Any] | None = None,
    ) -> None:
        if not str(attempt_id).strip():
            raise RunStateError("attempt_id must be non-empty")
        self.data.setdefault("stage_attempts", {}).setdefault(
            stage, []
        ).append(
            {
                "attempt_id": str(attempt_id),
                "status": str(status),
                "facts": _sanitize(
                    copy.deepcopy(facts or {})
                ),
                "recorded_at_utc": utc_now(),
            }
        )
        self.save()

    def record_stage(
        self,
        stage: str,
        *,
        status: str,
        facts: dict[str, Any] | None = None,
        successful_state: str | None = None,
    ) -> None:
        self.data.setdefault("stages", {})[stage] = {
            "status": status,
            "facts": _sanitize(copy.deepcopy(facts or {})),
            "updated_at_utc": utc_now(),
        }
        if successful_state:
            self.data["current_state"] = successful_state
            self.data["last_successful_state"] = successful_state
            self.data["failure"] = None
            self.data["next_action"] = None
        self.save()

    def fail(
        self,
        *,
        stage: str,
        error_type: str,
        message: str,
        resumable: bool,
        next_action: str | None,
    ) -> None:
        self.data["current_state"] = "failed"
        self.data["failure"] = {
            "stage": stage,
            "error_type": error_type,
            "message": _sanitize(message),
            "resumable": bool(resumable),
            "failed_at_utc": utc_now(),
        }
        self.data["next_action"] = next_action
        self.save()

    def complete(self) -> None:
        self.data["current_state"] = "scenemosaic_published"
        self.data["last_successful_state"] = "scenemosaic_published"
        self.data["next_action"] = None
        self.data["failure"] = None
        self.save()

    def public_status(self) -> dict[str, Any]:
        return _sanitize(copy.deepcopy(self.data))
