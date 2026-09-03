"""Production SAM3 segmentation runtime used by SceneMosaic perception."""

from .runtime import (
    ASSIGNMENT_POLICIES,
    DEFAULT_MODEL_DIR,
    Sam3Session,
    infer_text_candidates,
    load_session,
    prompt_groups,
    segment_registry,
)

__all__ = [
    "ASSIGNMENT_POLICIES",
    "DEFAULT_MODEL_DIR",
    "Sam3Session",
    "infer_text_candidates",
    "load_session",
    "prompt_groups",
    "segment_registry",
]
