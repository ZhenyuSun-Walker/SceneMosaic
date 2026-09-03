from __future__ import annotations

import copy
import hashlib
import json
import os
import re
import shutil
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from PIL import Image

from ...artifacts import RunContract


READY_STATUSES = frozenset({"sam3d_ready", "sam3d_ready_degraded"})
OBJECT_ID_RE = re.compile(r"^object_(\d{3})$")


class PerceptionAdoptionError(RuntimeError):
    """Raised when a protected perception delivery cannot be safely adopted."""


@dataclass(frozen=True)
class PerceptionAdoptionArtifacts:
    stage03_dir: Path
    stage04_dir: Path
    reviewed_objects_path: Path
    sam3d_manifest_path: Path
    adoption_report_path: Path
    delivery_status: str
    object_ids: tuple[str, ...]
    reused: bool = False


@dataclass(frozen=True)
class _ValidatedDelivery:
    run_manifest_path: Path
    run_manifest: dict[str, Any]
    run_manifest_sha256: str
    output_dir: Path
    source_manifest_path: Path
    source_manifest: dict[str, Any]
    delivery_report_path: Path
    delivery_report: dict[str, Any]
    run_info_path: Path | None
    quality_report_path: Path
    image_path: Path
    image_size: tuple[int, int]
    mask_paths: tuple[Path, ...]
    object_ids: tuple[str, ...]
    status: str


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
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PerceptionAdoptionError(f"cannot parse {label}: {path}") from exc
    if not isinstance(payload, dict):
        raise PerceptionAdoptionError(f"{label} must contain a JSON object: {path}")
    return payload


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


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
    trusted_root: Path,
    label: str,
) -> Path:
    if not isinstance(raw_path, str) or not raw_path.strip():
        raise PerceptionAdoptionError(f"{label} must be a non-empty path")
    candidate = Path(raw_path).expanduser()
    if not candidate.is_absolute():
        candidate = base_dir / candidate
    resolved = candidate.resolve()
    if not _is_within(resolved, trusted_root):
        raise PerceptionAdoptionError(
            f"{label} escapes perception attempts root: {resolved}"
        )
    return resolved


def _directory_has_files(path: Path) -> bool:
    if not path.exists():
        return False
    if not path.is_dir():
        return True
    return any(item.is_file() or item.is_symlink() for item in path.rglob("*"))


def _empty_tree_snapshot(path: Path) -> tuple[bool, tuple[Path, ...]]:
    if not path.exists():
        return False, ()
    if not path.is_dir():
        raise PerceptionAdoptionError(f"stage target is not a directory: {path}")
    entries = tuple(path.rglob("*"))
    if any(item.is_file() or item.is_symlink() for item in entries):
        raise PerceptionAdoptionError(f"stage target already contains files: {path}")
    directories = tuple(
        sorted(
            (item.relative_to(path) for item in entries if item.is_dir()),
            key=lambda value: (len(value.parts), str(value)),
        )
    )
    return True, directories


def _remove_empty_tree(path: Path) -> None:
    if not path.exists():
        return
    directories = sorted(
        (item for item in path.rglob("*") if item.is_dir()),
        key=lambda item: len(item.parts),
        reverse=True,
    )
    for directory in directories:
        directory.rmdir()
    path.rmdir()


def _restore_empty_tree(path: Path, snapshot: tuple[bool, tuple[Path, ...]]) -> None:
    existed, directories = snapshot
    if not existed:
        return
    path.mkdir(parents=True, exist_ok=True)
    for relative in directories:
        (path / relative).mkdir(parents=True, exist_ok=True)


class PerceptionAdoptionAdapter:
    """Adopt a isolated perception delivery into formal Stage 03 and Stage 04.

    The adapter only validates, copies and publishes existing delivery artifacts.
    It never invokes segmentation, changes masks, or rebuilds object numbering.
    """

    def __init__(
        self,
        *,
        run_manifest_path: Path,
        scene_config_path: Path,
        perception_attempts_root: Path | None = None,
    ) -> None:
        self.run_manifest_path = Path(run_manifest_path).expanduser().resolve()
        self.perception_attempts_root = Path(
            perception_attempts_root or self.run_manifest_path.parent
        ).expanduser().resolve()
        self.contract = RunContract.from_scene_config(scene_config_path)

    def adopt(self) -> PerceptionAdoptionArtifacts:
        delivery = self._validate_delivery()
        existing = self._reuse_existing(delivery)
        if existing is not None:
            return existing

        stage03 = self.contract.stage03_dir
        stage04 = self.contract.stage04_dir
        stage03_tmp = stage03.parent / f".{stage03.name}.adopt-{uuid.uuid4().hex}.tmp"
        stage04_tmp = stage04.parent / f".{stage04.name}.adopt-{uuid.uuid4().hex}.tmp"
        stage03_tmp.mkdir(parents=True, exist_ok=False)
        stage04_tmp.mkdir(parents=True, exist_ok=False)

        try:
            object_ids = self._materialize(delivery, stage03_tmp, stage04_tmp)
            self._publish_pair(stage03_tmp, stage04_tmp, stage03, stage04)
        except Exception:
            if stage03_tmp.exists():
                shutil.rmtree(stage03_tmp)
            if stage04_tmp.exists():
                shutil.rmtree(stage04_tmp)
            raise

        return PerceptionAdoptionArtifacts(
            stage03_dir=stage03,
            stage04_dir=stage04,
            reviewed_objects_path=stage03 / "reviewed_objects.json",
            sam3d_manifest_path=stage04 / "sam3d_input_manifest.json",
            adoption_report_path=stage03 / "perception_adoption.json",
            delivery_status=delivery.status,
            object_ids=object_ids,
            reused=False,
        )

    @classmethod
    def validate_formal(
        cls,
        scene_config_path: Path,
    ) -> PerceptionAdoptionArtifacts:
        """Validate an already-published Stage 03/04 without attempt-source access.

        The attempt delivery is a source used during adoption, not a lifetime
        dependency of the formal run. Every fact needed for reuse is bound by
        perception_adoption.json and rechecked against the formal copies.
        """

        contract = RunContract.from_scene_config(scene_config_path)
        stage03 = contract.stage03_dir.resolve()
        stage04 = contract.stage04_dir.resolve()
        adoption_path = stage03 / "perception_adoption.json"
        adoption = _read_json(adoption_path, "formal perception adoption")
        status = str(adoption.get("delivery_status") or "")
        if (
            adoption.get("schema_version")
            != "scenemosaic_perception_adoption"
            or adoption.get("scene_name") != contract.scene_name
            or status not in READY_STATUSES
            or adoption.get("sam3d_execution_allowed") is not True
            or adoption.get("human_review_required") is not False
            or adoption.get("renumbered") is not False
        ):
            raise PerceptionAdoptionError(
                "formal perception adoption invariants are invalid"
            )

        formal_records = adoption.get("formal_artifacts")
        if not isinstance(formal_records, list) or not formal_records:
            raise PerceptionAdoptionError(
                "formal perception adoption has no artifact hash closure"
            )
        declared: dict[Path, str] = {}
        for index, record in enumerate(formal_records):
            if not isinstance(record, dict):
                raise PerceptionAdoptionError(
                    f"formal artifact record {index} is invalid"
                )
            path = Path(
                str(record.get("path") or "")
            ).expanduser().resolve()
            digest = record.get("sha256")
            if (
                path in declared
                or not isinstance(digest, str)
                or not (
                    _is_within(path, stage03)
                    or _is_within(path, stage04)
                )
                or path == adoption_path
            ):
                raise PerceptionAdoptionError(
                    f"formal artifact record is unsafe: {path}"
                )
            if (
                not path.is_file()
                or path.is_symlink()
                or path.stat().st_size == 0
                or _sha256(path) != digest
            ):
                raise PerceptionAdoptionError(
                    f"formal perception artifact hash mismatch: {path}"
                )
            declared[path] = digest

        actual: set[Path] = set()
        for root in (stage03, stage04):
            if not root.is_dir() or root.is_symlink():
                raise PerceptionAdoptionError(
                    f"formal perception stage is missing or unsafe: {root}"
                )
            for path in root.rglob("*"):
                if path.is_symlink():
                    raise PerceptionAdoptionError(
                        f"formal perception stage contains a symlink: {path}"
                    )
                if path.is_file() and path != adoption_path:
                    actual.add(path.resolve())
        if actual != set(declared):
            missing = sorted(str(path) for path in set(declared) - actual)
            extra = sorted(str(path) for path in actual - set(declared))
            raise PerceptionAdoptionError(
                "formal perception artifact closure mismatch: "
                f"missing={missing}, extra={extra}"
            )

        reviewed_path = stage03 / "reviewed_objects.json"
        quality_path = stage03 / "quality_report.json"
        manifest_path = stage04 / "sam3d_input_manifest.json"
        delivery_report_path = stage04 / "delivery_report.json"
        run_info_path = stage04 / "sam3d_input_run_info.json"
        for path in (
            reviewed_path,
            quality_path,
            manifest_path,
            delivery_report_path,
            run_info_path,
        ):
            if path not in declared:
                raise PerceptionAdoptionError(
                    "required formal perception artifact is not "
                    f"hash-bound: {path}"
                )
        reviewed = _read_json(
            reviewed_path, "formal reviewed objects"
        )
        _read_json(quality_path, "formal quality report")
        manifest = _read_json(
            manifest_path, "formal Stage 04 manifest"
        )
        delivery_report = _read_json(
            delivery_report_path, "formal delivery report"
        )
        run_info = _read_json(
            run_info_path, "formal Stage 04 run info"
        )
        if (
            manifest.get("delivery_status") != status
            or manifest.get("sam3d_execution_allowed") is not True
            or run_info.get("status") != status
            or run_info.get("sam3d_execution_allowed") is not True
        ):
            raise PerceptionAdoptionError(
                "formal Stage 04 delivery status is inconsistent"
            )

        object_items = manifest.get("objects")
        reviewed_items = reviewed.get("objects")
        report_items = delivery_report.get("objects")
        if not all(
            isinstance(items, list)
            for items in (object_items, reviewed_items, report_items)
        ):
            raise PerceptionAdoptionError(
                "formal perception object lists are invalid"
            )
        assert isinstance(object_items, list)
        assert isinstance(reviewed_items, list)
        assert isinstance(report_items, list)
        object_count = manifest.get("object_count")
        if not isinstance(object_count, int) or object_count <= 0:
            raise PerceptionAdoptionError(
                "formal Stage 04 object_count is invalid"
            )
        expected_ids = tuple(
            f"object_{index:03d}"
            for index in range(1, object_count + 1)
        )

        def ids(items: list[Any], label: str) -> tuple[str, ...]:
            values: list[str] = []
            for index, item in enumerate(items):
                if not isinstance(item, dict):
                    raise PerceptionAdoptionError(
                        f"{label} object {index} is invalid"
                    )
                object_id = str(item.get("object_id") or "")
                if (
                    not OBJECT_ID_RE.fullmatch(object_id)
                    or item.get("sam3d_internal_name") != object_id
                ):
                    raise PerceptionAdoptionError(
                        f"{label} identity mismatch for "
                        f"{object_id or index}"
                    )
                values.append(object_id)
            return tuple(values)

        manifest_ids = ids(object_items, "Stage 04")
        reviewed_ids = ids(reviewed_items, "reviewed")
        report_ids = ids(report_items, "delivery report")
        closure = adoption.get("object_id_closure")
        if (
            manifest_ids != expected_ids
            or reviewed_ids != expected_ids
            or report_ids != expected_ids
            or reviewed.get("object_count") != object_count
            or run_info.get("object_count") != object_count
            or not isinstance(closure, dict)
            or closure.get("closed") is not True
            or tuple(closure.get("manifest", ())) != expected_ids
            or tuple(closure.get("reviewed_objects", ()))
            != expected_ids
            or tuple(closure.get("delivery_report", ()))
            != expected_ids
        ):
            raise PerceptionAdoptionError(
                "formal perception object ID closure is invalid"
            )

        image_path = Path(
            str(manifest.get("image_path") or "")
        ).expanduser().resolve()
        expected_image = stage04 / "image.png"
        if (
            image_path != expected_image
            or image_path not in declared
            or _sha256(image_path) != _sha256(contract.input_image)
        ):
            raise PerceptionAdoptionError(
                "formal Stage 04 image does not match the formal input"
            )
        try:
            with Image.open(image_path) as image:
                image.load()
                image_size = image.size
        except (OSError, ValueError) as exc:
            raise PerceptionAdoptionError(
                "formal Stage 04 image is unreadable"
            ) from exc

        masks_dir = stage04 / "masks"
        actual_masks = (
            {
                path.resolve()
                for path in masks_dir.iterdir()
                if path.is_file() and not path.is_symlink()
            }
            if masks_dir.is_dir() and not masks_dir.is_symlink()
            else set()
        )
        declared_masks: set[Path] = set()
        mask_hashes: dict[str, str] = {}
        for index, item in enumerate(object_items):
            assert isinstance(item, dict)
            object_id = expected_ids[index]
            if item.get("order_index") != index:
                raise PerceptionAdoptionError(
                    "formal Stage 04 order_index mismatch for "
                    f"{object_id}"
                )
            mask_path = Path(
                str(item.get("mask_path") or "")
            ).expanduser().resolve()
            if (
                not _is_within(mask_path, masks_dir)
                or mask_path not in declared
                or mask_path in declared_masks
            ):
                raise PerceptionAdoptionError(
                    "formal Stage 04 mask path is invalid for "
                    f"{object_id}"
                )
            digest = _sha256(mask_path)
            if item.get("mask_sha256") != digest:
                raise PerceptionAdoptionError(
                    "formal Stage 04 mask hash mismatch for "
                    f"{object_id}"
                )
            try:
                with Image.open(mask_path) as mask:
                    mask.load()
                    if (
                        mask.size != image_size
                        or mask.convert("L").getbbox() is None
                    ):
                        raise PerceptionAdoptionError(
                            "formal Stage 04 mask content is invalid "
                            f"for {object_id}"
                        )
            except PerceptionAdoptionError:
                raise
            except (OSError, ValueError) as exc:
                raise PerceptionAdoptionError(
                    f"formal Stage 04 mask is unreadable for {object_id}"
                ) from exc
            declared_masks.add(mask_path)
            mask_hashes[object_id] = digest
        if actual_masks != declared_masks:
            raise PerceptionAdoptionError(
                "formal Stage 04 mask file closure is invalid"
            )

        comparisons = adoption.get("mask_hash_comparison")
        if (
            not isinstance(comparisons, list)
            or len(comparisons) != object_count
        ):
            raise PerceptionAdoptionError(
                "formal perception mask hash comparison is invalid"
            )
        for comparison, object_id in zip(
            comparisons, expected_ids
        ):
            if (
                not isinstance(comparison, dict)
                or comparison.get("object_id") != object_id
                or Path(
                    str(comparison.get("formal_path") or "")
                ).expanduser().resolve()
                not in declared_masks
                or comparison.get("formal_sha256")
                != mask_hashes[object_id]
                or comparison.get("identical") is not True
            ):
                raise PerceptionAdoptionError(
                    f"formal mask hash evidence mismatch for {object_id}"
                )

        return PerceptionAdoptionArtifacts(
            stage03_dir=stage03,
            stage04_dir=stage04,
            reviewed_objects_path=reviewed_path,
            sam3d_manifest_path=manifest_path,
            adoption_report_path=adoption_path,
            delivery_status=status,
            object_ids=expected_ids,
            reused=True,
        )

    def _validate_delivery(self) -> _ValidatedDelivery:
        if not self.run_manifest_path.is_file():
            raise PerceptionAdoptionError(
                f"run_manifest is not a file: {self.run_manifest_path}"
            )
        if not _is_within(self.run_manifest_path, self.perception_attempts_root):
            raise PerceptionAdoptionError(
                f"run_manifest escapes perception attempts root: {self.run_manifest_path}"
            )

        run_manifest = _read_json(self.run_manifest_path, "run_manifest")
        if str(run_manifest.get("quality_status") or "") != "pass":
            raise PerceptionAdoptionError(
                "perception attempt quality_status must be pass before adoption"
            )
        delivery = run_manifest.get("sam3d_delivery")
        if not isinstance(delivery, dict):
            raise PerceptionAdoptionError("run_manifest.sam3d_delivery is missing")
        status = str(delivery.get("status") or "")
        if status not in READY_STATUSES:
            raise PerceptionAdoptionError(
                f"delivery status does not allow adoption: {status or 'missing'}"
            )
        if delivery.get("sam3d_execution_allowed") is not True:
            raise PerceptionAdoptionError("sam3d_execution_allowed must be true")

        base_dir = self.run_manifest_path.parent
        output_dir = _resolve_confined(
            delivery.get("output_dir"),
            base_dir=base_dir,
            trusted_root=self.perception_attempts_root,
            label="sam3d_delivery.output_dir",
        )
        manifest_path = _resolve_confined(
            delivery.get("manifest_path"),
            base_dir=base_dir,
            trusted_root=self.perception_attempts_root,
            label="sam3d_delivery.manifest_path",
        )
        report_path = _resolve_confined(
            delivery.get("delivery_report_path"),
            base_dir=base_dir,
            trusted_root=self.perception_attempts_root,
            label="sam3d_delivery.delivery_report_path",
        )
        if not output_dir.is_dir():
            raise PerceptionAdoptionError(f"delivery output_dir is missing: {output_dir}")
        if manifest_path.parent != output_dir or report_path.parent != output_dir:
            raise PerceptionAdoptionError(
                "delivery manifest and report must be direct children of output_dir"
            )
        if not manifest_path.is_file() or not report_path.is_file():
            raise PerceptionAdoptionError("delivery manifest or report is missing")

        manifest = _read_json(manifest_path, "sam3d_input_manifest")
        report = _read_json(report_path, "delivery_report")
        if str(report.get("source_quality_status") or "") != "pass":
            raise PerceptionAdoptionError(
                "SAM3D delivery source_quality_status must be pass before adoption"
            )
        for label, payload in (("manifest", manifest), ("delivery_report", report)):
            payload_status = payload.get("delivery_status") if label == "manifest" else payload.get("status")
            if payload_status != status:
                raise PerceptionAdoptionError(
                    f"{label} status mismatch: expected {status}, got {payload_status}"
                )
            if payload.get("sam3d_execution_allowed") is not True:
                raise PerceptionAdoptionError(
                    f"{label}.sam3d_execution_allowed must be true"
                )

        manifest_objects = manifest.get("objects")
        report_objects = report.get("objects")
        if not isinstance(manifest_objects, list) or not isinstance(report_objects, list):
            raise PerceptionAdoptionError("delivery objects must be arrays")
        object_count = manifest.get("object_count")
        counts = {
            "run_manifest": delivery.get("object_count"),
            "manifest": object_count,
            "delivery_report": report.get("delivery_object_count"),
            "manifest_objects": len(manifest_objects),
            "report_objects": len(report_objects),
        }
        if not isinstance(object_count, int) or object_count <= 0:
            raise PerceptionAdoptionError("manifest.object_count must be a positive integer")
        if any(value != object_count for value in counts.values()):
            raise PerceptionAdoptionError(f"delivery object_count mismatch: {counts}")

        expected_ids = tuple(f"object_{index:03d}" for index in range(1, object_count + 1))
        manifest_ids: list[str] = []
        report_ids: list[str] = []
        mask_paths: list[Path] = []
        seen_mask_paths: set[Path] = set()
        for index, item in enumerate(manifest_objects):
            if not isinstance(item, dict):
                raise PerceptionAdoptionError(f"manifest object {index} is not an object")
            object_id = str(item.get("object_id") or "")
            if not OBJECT_ID_RE.fullmatch(object_id):
                raise PerceptionAdoptionError(f"invalid object_id: {object_id}")
            if item.get("sam3d_internal_name") != object_id:
                raise PerceptionAdoptionError(
                    f"sam3d_internal_name mismatch for {object_id}"
                )
            if item.get("order_index") != index:
                raise PerceptionAdoptionError(
                    f"order_index mismatch for {object_id}: expected {index}"
                )
            mask_path = _resolve_confined(
                item.get("mask_path"),
                base_dir=manifest_path.parent,
                trusted_root=self.perception_attempts_root,
                label=f"{object_id}.mask_path",
            )
            if mask_path in seen_mask_paths:
                raise PerceptionAdoptionError(f"duplicate mask path: {mask_path}")
            seen_mask_paths.add(mask_path)
            manifest_ids.append(object_id)
            mask_paths.append(mask_path)

        for index, item in enumerate(report_objects):
            if not isinstance(item, dict):
                raise PerceptionAdoptionError(f"delivery report object {index} is not an object")
            object_id = str(item.get("object_id") or "")
            if item.get("sam3d_internal_name") != object_id:
                raise PerceptionAdoptionError(
                    f"delivery report sam3d_internal_name mismatch for {object_id}"
                )
            report_ids.append(object_id)

        if tuple(manifest_ids) != expected_ids or tuple(report_ids) != expected_ids:
            raise PerceptionAdoptionError(
                "object IDs must be unique and continuous from object_001 to object_N"
            )

        image_path = _resolve_confined(
            manifest.get("image_path"),
            base_dir=manifest_path.parent,
            trusted_root=self.perception_attempts_root,
            label="manifest.image_path",
        )
        if not image_path.is_file():
            raise PerceptionAdoptionError(f"delivery image is missing: {image_path}")
        try:
            with Image.open(image_path) as image:
                image.load()
                image_size = image.size
        except (OSError, ValueError) as exc:
            raise PerceptionAdoptionError(f"delivery image is unreadable: {image_path}") from exc
        if list(image_size) != report.get("image_size"):
            raise PerceptionAdoptionError(
                f"delivery image size mismatch: {image_size} vs {report.get('image_size')}"
            )
        if _sha256(image_path) != _sha256(self.contract.input_image):
            raise PerceptionAdoptionError(
                "delivery image does not match formal input/image.png"
            )

        masks_dir = output_dir / "masks"
        if not masks_dir.is_dir():
            raise PerceptionAdoptionError(f"delivery masks directory is missing: {masks_dir}")
        actual_mask_files = tuple(
            sorted(item.resolve() for item in masks_dir.iterdir() if item.is_file())
        )
        if len(actual_mask_files) != object_count or set(actual_mask_files) != set(mask_paths):
            raise PerceptionAdoptionError(
                "actual mask files must exactly match manifest objects"
            )

        report_by_id = {str(item.get("object_id")): item for item in report_objects}
        for item, mask_path in zip(manifest_objects, mask_paths):
            object_id = str(item["object_id"])
            if not mask_path.is_file() or mask_path.stat().st_size == 0:
                raise PerceptionAdoptionError(f"mask is missing or empty: {mask_path}")
            try:
                with Image.open(mask_path) as mask:
                    mask.load()
                    if mask.size != image_size:
                        raise PerceptionAdoptionError(
                            f"mask size mismatch for {object_id}: {mask.size} vs {image_size}"
                        )
                    if mask.convert("L").getbbox() is None:
                        raise PerceptionAdoptionError(f"mask has no foreground: {object_id}")
            except PerceptionAdoptionError:
                raise
            except (OSError, ValueError) as exc:
                raise PerceptionAdoptionError(f"mask is unreadable: {mask_path}") from exc
            actual_sha = _sha256(mask_path)
            report_item = report_by_id[object_id]
            if item.get("mask_sha256") != actual_sha or report_item.get("mask_sha256") != actual_sha:
                raise PerceptionAdoptionError(f"mask SHA-256 mismatch for {object_id}")

        quality_report_path = _resolve_confined(
            manifest.get("source_quality_report"),
            base_dir=manifest_path.parent,
            trusted_root=self.perception_attempts_root,
            label="manifest.source_quality_report",
        )
        if not quality_report_path.is_file():
            raise PerceptionAdoptionError(
                f"source quality report is missing: {quality_report_path}"
            )
        _read_json(quality_report_path, "source quality report")

        run_info_path = output_dir / "sam3d_input_run_info.json"
        if not run_info_path.is_file():
            run_info_path = None

        return _ValidatedDelivery(
            run_manifest_path=self.run_manifest_path,
            run_manifest=run_manifest,
            run_manifest_sha256=_sha256(self.run_manifest_path),
            output_dir=output_dir,
            source_manifest_path=manifest_path,
            source_manifest=manifest,
            delivery_report_path=report_path,
            delivery_report=report,
            run_info_path=run_info_path,
            quality_report_path=quality_report_path,
            image_path=image_path,
            image_size=image_size,
            mask_paths=tuple(mask_paths),
            object_ids=expected_ids,
            status=status,
        )

    def _reuse_existing(
        self, delivery: _ValidatedDelivery
    ) -> PerceptionAdoptionArtifacts | None:
        stage03 = self.contract.stage03_dir
        stage04 = self.contract.stage04_dir
        stage03_has_files = _directory_has_files(stage03)
        stage04_has_files = _directory_has_files(stage04)
        if not stage03_has_files and not stage04_has_files:
            return None
        if stage03_has_files != stage04_has_files:
            raise PerceptionAdoptionError(
                "formal Stage 03/04 are partially populated; refusing implicit recovery"
            )

        adoption_path = stage03 / "perception_adoption.json"
        adoption = _read_json(adoption_path, "existing perception adoption")
        source = adoption.get("source")
        if not isinstance(source, dict) or source.get("run_manifest_sha256") != delivery.run_manifest_sha256:
            raise PerceptionAdoptionError(
                "formal Stage 03/04 already contain a different adoption"
            )
        formal_artifacts = adoption.get("formal_artifacts")
        if not isinstance(formal_artifacts, list):
            raise PerceptionAdoptionError("existing adoption lacks formal artifact hashes")
        for record in formal_artifacts:
            if not isinstance(record, dict):
                raise PerceptionAdoptionError("existing formal artifact record is invalid")
            path = Path(str(record.get("path") or "")).expanduser().resolve()
            expected_sha = record.get("sha256")
            if not path.is_file() or not isinstance(expected_sha, str) or _sha256(path) != expected_sha:
                raise PerceptionAdoptionError(
                    f"existing formal artifact hash mismatch: {path}"
                )

        manifest = _read_json(stage04 / "sam3d_input_manifest.json", "formal Stage 04 manifest")
        existing_ids = tuple(str(item.get("object_id") or "") for item in manifest.get("objects", []))
        if existing_ids != delivery.object_ids:
            raise PerceptionAdoptionError("existing adoption object ID closure mismatch")

        return PerceptionAdoptionArtifacts(
            stage03_dir=stage03,
            stage04_dir=stage04,
            reviewed_objects_path=stage03 / "reviewed_objects.json",
            sam3d_manifest_path=stage04 / "sam3d_input_manifest.json",
            adoption_report_path=adoption_path,
            delivery_status=delivery.status,
            object_ids=delivery.object_ids,
            reused=True,
        )

    def _materialize(
        self,
        delivery: _ValidatedDelivery,
        stage03_tmp: Path,
        stage04_tmp: Path,
    ) -> tuple[str, ...]:
        final_stage03 = self.contract.stage03_dir
        final_stage04 = self.contract.stage04_dir
        stage03_masks = stage03_tmp / "reviewed_masks"
        stage03_overlays = stage03_tmp / "reviewed_overlays"
        stage04_masks = stage04_tmp / "masks"
        stage03_masks.mkdir(parents=True)
        stage03_overlays.mkdir(parents=True)
        stage04_masks.mkdir(parents=True)

        image_tmp = stage04_tmp / "image.png"
        shutil.copy2(delivery.image_path, image_tmp)
        source_image_sha = _sha256(delivery.image_path)
        if _sha256(image_tmp) != source_image_sha:
            raise PerceptionAdoptionError("Stage 04 image copy hash mismatch")

        source_objects = delivery.source_manifest["objects"]
        reviewed_objects: list[dict[str, Any]] = []
        formal_manifest = copy.deepcopy(delivery.source_manifest)
        formal_manifest["image_path"] = str(final_stage04 / "image.png")
        mask_hash_comparison: list[dict[str, Any]] = []

        with Image.open(delivery.image_path) as source_image:
            base_rgba = source_image.convert("RGBA")
            for index, (source_object, source_mask) in enumerate(
                zip(source_objects, delivery.mask_paths)
            ):
                object_id = delivery.object_ids[index]
                formal_mask_tmp = stage04_masks / source_mask.name
                formal_mask_path = final_stage04 / "masks" / source_mask.name
                shutil.copy2(source_mask, formal_mask_tmp)
                source_mask_sha = _sha256(source_mask)
                formal_mask_sha = _sha256(formal_mask_tmp)
                if formal_mask_sha != source_mask_sha:
                    raise PerceptionAdoptionError(
                        f"Stage 04 mask copy hash mismatch for {object_id}"
                    )

                category = re.sub(
                    r"[^0-9A-Za-z]+",
                    "_",
                    str(source_object.get("category") or "object"),
                ).strip("_").lower() or "object"
                reviewed_mask_name = f"{object_id}_{category}_mask.png"
                reviewed_overlay_name = f"{object_id}_{category}_overlay.png"
                reviewed_mask_tmp = stage03_masks / reviewed_mask_name
                reviewed_overlay_tmp = stage03_overlays / reviewed_overlay_name
                shutil.copy2(source_mask, reviewed_mask_tmp)

                with Image.open(source_mask) as mask_image:
                    mask_l = mask_image.convert("L")
                    alpha = mask_l.point(lambda value: 112 if value else 0)
                    tint = Image.new("RGBA", delivery.image_size, (255, 32, 32, 0))
                    tint.putalpha(alpha)
                    overlay = Image.alpha_composite(base_rgba, tint).convert("RGB")
                    overlay.save(
                        reviewed_overlay_tmp,
                        format="PNG",
                        optimize=False,
                        compress_level=1,
                    )

                formal_manifest["objects"][index]["mask_path"] = str(formal_mask_path)
                reviewed = copy.deepcopy(source_object)
                sam3 = reviewed.get("sam3")
                if not isinstance(sam3, dict):
                    sam3 = {}
                sam3 = dict(sam3)
                sam3["mask_path"] = str(formal_mask_path)
                sam3["overlay_path"] = str(
                    final_stage03 / "reviewed_overlays" / reviewed_overlay_name
                )
                reviewed["sam3"] = sam3
                reviewed["review_mode"] = "automatic_perception_delivery"
                reviewed["human_review_required"] = False
                reviewed_objects.append(reviewed)
                mask_hash_comparison.append(
                    {
                        "object_id": object_id,
                        "source_path": str(source_mask),
                        "source_sha256": source_mask_sha,
                        "formal_path": str(formal_mask_path),
                        "formal_sha256": formal_mask_sha,
                        "identical": source_mask_sha == formal_mask_sha,
                    }
                )

        reviewed_payload = {
            "schema_version": "scenemosaic_reviewed_objects",
            "scene_name": self.contract.scene_name,
            "review_mode": "automatic_perception_delivery",
            "human_review_required": False,
            "source_delivery_status": delivery.status,
            "object_count": len(reviewed_objects),
            "objects": reviewed_objects,
        }
        _write_json(stage03_tmp / "reviewed_objects.json", reviewed_payload)
        _write_json(stage04_tmp / "sam3d_input_manifest.json", formal_manifest)
        shutil.copy2(delivery.quality_report_path, stage03_tmp / "quality_report.json")
        shutil.copy2(delivery.delivery_report_path, stage04_tmp / "delivery_report.json")
        if delivery.run_info_path is not None:
            shutil.copy2(delivery.run_info_path, stage04_tmp / "sam3d_input_run_info.json")
        else:
            _write_json(
                stage04_tmp / "sam3d_input_run_info.json",
                {
                    "schema_version": "scenemosaic_sam3d_delivery_run_info",
                    "scene_name": self.contract.scene_name,
                    "status": delivery.status,
                    "sam3d_execution_allowed": True,
                    "object_count": len(reviewed_objects),
                    "created_at": _utc_now(),
                    "source_quality_status": delivery.delivery_report.get("source_quality_status"),
                    "excluded_object_count": len(delivery.delivery_report.get("excluded_objects", [])),
                    "duplicate_collapse_count": len(delivery.delivery_report.get("duplicate_collapses", [])),
                    "infrastructure_error_count": len(delivery.delivery_report.get("infrastructure_errors", [])),
                },
            )

        formal_records: list[dict[str, str]] = []
        for temporary_root, final_root in (
            (stage03_tmp, final_stage03),
            (stage04_tmp, final_stage04),
        ):
            for path in sorted(item for item in temporary_root.rglob("*") if item.is_file()):
                formal_records.append(
                    {
                        "path": str(final_root / path.relative_to(temporary_root)),
                        "sha256": _sha256(path),
                    }
                )

        adoption_report = {
            "schema_version": "scenemosaic_perception_adoption",
            "created_at_utc": _utc_now(),
            "scene_name": self.contract.scene_name,
            "delivery_status": delivery.status,
            "sam3d_execution_allowed": True,
            "review_mode": "automatic_perception_delivery",
            "human_review_required": False,
            "renumbered": False,
            "source": {
                "perception_attempts_root": str(self.perception_attempts_root),
                "run_manifest_path": str(delivery.run_manifest_path),
                "run_manifest_sha256": delivery.run_manifest_sha256,
                "sam3d_input_manifest_path": str(delivery.source_manifest_path),
                "sam3d_input_manifest_sha256": _sha256(delivery.source_manifest_path),
                "delivery_report_path": str(delivery.delivery_report_path),
                "delivery_report_sha256": _sha256(delivery.delivery_report_path),
                "quality_report_path": str(delivery.quality_report_path),
                "quality_report_sha256": _sha256(delivery.quality_report_path),
                "image_path": str(delivery.image_path),
                "image_sha256": source_image_sha,
            },
            "source_models": {
                "pass_a": delivery.run_manifest.get("pass_a_model"),
                "pass_b": delivery.run_manifest.get("pass_b_model"),
            },
            "object_id_closure": {
                "manifest": list(delivery.object_ids),
                "delivery_report": list(delivery.object_ids),
                "reviewed_objects": list(delivery.object_ids),
                "closed": True,
            },
            "mask_hash_comparison": mask_hash_comparison,
            "excluded_objects": delivery.delivery_report.get("excluded_objects", []),
            "duplicate_collapses": delivery.delivery_report.get("duplicate_collapses", []),
            "infrastructure_errors": delivery.delivery_report.get("infrastructure_errors", []),
            "source_warning_count": delivery.delivery_report.get("source_warning_count", 0),
            "per_object_physics_consumed": False,
            "formal_artifacts": formal_records,
        }
        _write_json(stage03_tmp / "perception_adoption.json", adoption_report)
        return delivery.object_ids

    @staticmethod
    def _publish_pair(
        stage03_tmp: Path,
        stage04_tmp: Path,
        stage03: Path,
        stage04: Path,
    ) -> None:
        stage03_snapshot = _empty_tree_snapshot(stage03)
        stage04_snapshot = _empty_tree_snapshot(stage04)
        _remove_empty_tree(stage03)
        _remove_empty_tree(stage04)
        stage03_published = False
        stage04_published = False
        try:
            os.replace(stage03_tmp, stage03)
            stage03_published = True
            os.replace(stage04_tmp, stage04)
            stage04_published = True
        except Exception as exc:
            if stage04_published and stage04.exists():
                os.replace(stage04, stage04_tmp)
            if stage03_published and stage03.exists():
                os.replace(stage03, stage03_tmp)
            _restore_empty_tree(stage03, stage03_snapshot)
            _restore_empty_tree(stage04, stage04_snapshot)
            raise PerceptionAdoptionError(
                "atomic Stage 03/04 publication failed and was rolled back"
            ) from exc
