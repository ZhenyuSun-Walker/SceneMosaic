from __future__ import annotations

import json
import os
import re
import subprocess
from pathlib import Path
from typing import Any


EXPECTED_TRANSFORMERS_VERSION = "5.5.4"
PERCEPTION_DEPENDENCY_MODULES = (
    "PIL",
    "cv2",
    "scipy",
    "transformers",
    "openai",
    "huggingface_hub",
    "typing_extensions",
)
SENSITIVE_ENV_NAME_RE = re.compile(
    r"(?:KEY|TOKEN|SECRET|PASSWORD|CREDENTIAL|AUTHORIZATION|BASE_URL)", re.I
)


def verify_perception_dependencies(
    perception_python: Path,
) -> dict[str, Any]:
    """Import each perception dependency in an isolated CPU-only process."""

    python = Path(perception_python).expanduser().resolve()
    base: dict[str, Any] = {
        "required_modules": list(PERCEPTION_DEPENDENCY_MODULES),
        "api_calls": 0,
        "gpu_used": False,
    }
    if not python.is_file() or python.is_symlink():
        return {**base, "valid": False, "failure": "python missing"}

    environment = {
        name: value
        for name, value in os.environ.items()
        if not SENSITIVE_ENV_NAME_RE.search(name)
    }
    environment["CUDA_VISIBLE_DEVICES"] = ""
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    environment["PYTHONNOUSERSITE"] = "1"
    results: list[dict[str, Any]] = []
    for module_name in PERCEPTION_DEPENDENCY_MODULES:
        try:
            completed = subprocess.run(
                [
                    str(python),
                    "-c",
                    (
                        "import importlib; "
                        f"importlib.import_module({module_name!r})"
                    ),
                ],
                capture_output=True,
                text=True,
                timeout=30,
                check=False,
                env=environment,
            )
            results.append(
                {
                    "module": module_name,
                    "valid": completed.returncode == 0,
                    "returncode": completed.returncode,
                }
            )
        except (OSError, subprocess.SubprocessError) as exc:
            results.append(
                {
                    "module": module_name,
                    "valid": False,
                    "failure": type(exc).__name__,
                }
            )
    return {
        **base,
        "valid": all(item["valid"] for item in results),
        "modules": results,
    }


def verify_sam3_runtime_provider(
    perception_python: Path,
    source_root: Path,
) -> dict[str, Any]:
    # Verify the version and import origin used by the perception runtime.
    base: dict[str, Any] = {
        "provider": "transformers",
        "expected_version": EXPECTED_TRANSFORMERS_VERSION,
        "api_calls": 0,
        "gpu_used": False,
    }
    python = Path(perception_python).expanduser().resolve()
    source = Path(source_root).expanduser().resolve()
    if not python.is_file() or python.is_symlink():
        return {**base, "valid": False, "failure": "python missing"}

    script = """
import importlib.metadata
import importlib.util
import json
from pathlib import Path

distribution = importlib.metadata.distribution("transformers")
spec = importlib.util.find_spec("sam3")
print(json.dumps({
    "version": distribution.version,
    "sam3_origin": (
        str(Path(spec.origin).resolve())
        if spec and spec.origin
        else None
    ),
}))
"""
    environment = {
        name: value
        for name, value in os.environ.items()
        if not SENSITIVE_ENV_NAME_RE.search(name)
    }
    environment["PYTHONPATH"] = str(source)
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    environment["PYTHONNOUSERSITE"] = "1"
    try:
        completed = subprocess.run(
            [str(python), "-c", script],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
            env=environment,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return {**base, "valid": False, "failure": type(exc).__name__}
    if completed.returncode != 0:
        return {
            **base,
            "valid": False,
            "failure": "provider probe failed",
            "returncode": completed.returncode,
        }
    try:
        result = json.loads(completed.stdout)
    except json.JSONDecodeError:
        return {**base, "valid": False, "failure": "invalid provider probe"}

    expected_origin = (source / "sam3" / "__init__.py").resolve()
    origin = result.get("sam3_origin")
    valid = (
        result.get("version") == EXPECTED_TRANSFORMERS_VERSION
        and origin == str(expected_origin)
    )
    return {
        **base,
        "valid": valid,
        "version": result.get("version"),
        "sam3_origin": origin,
        "expected_sam3_origin": str(expected_origin),
        "failures": [] if valid else ["version_or_import_origin"],
    }
