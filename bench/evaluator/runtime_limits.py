"""Lookup and validation for published test-set runtime limits."""

from __future__ import annotations

import json
import math
from hashlib import sha256
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Mapping, Optional


RUNTIME_LIMITS_PATH = (
    Path(__file__).resolve().parents[1]
    / "datasets"
    / "metadata"
    / "test_runtime_limits.json"
)


@dataclass(frozen=True)
class RuntimeLimits:
    """Validated limits for one benchmark episode."""

    max_steps: int
    max_wall_time_s: Optional[float]
    annotation_sha256: str


def canonical_json_sha256(payload: Any) -> str:
    """Hash JSON content canonically so harmless formatting changes do not matter."""

    try:
        encoded = json.dumps(
            payload,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Cannot canonicalize annotation JSON: {exc}") from exc
    return sha256(encoded).hexdigest()


def annotation_json_sha256(path: Path) -> str:
    """Load a public annotation and return its canonical JSON SHA-256."""

    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Public test annotation not found for hash verification: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in public test annotation {path}: {exc}") from exc
    return canonical_json_sha256(payload)


def test_annotation_relative_path(envset_path: Path) -> Optional[str]:
    """Return the stable dataset-relative path for a public test annotation.

    The canonical layout is ``annotations/test/<style>/<embodiment>/...``.  The
    suffix-only fallback also supports users who copy just the test subtree to
    another location while preserving its style/embodiment layout.
    """

    parts = Path(envset_path).parts
    for index in range(len(parts) - 1):
        if parts[index] == "annotations" and parts[index + 1] == "test":
            relative_parts = parts[index + 2 :]
            if relative_parts:
                return Path(*relative_parts).as_posix()
            return None

    # A canonical annotations/<split> path for any other split is never test.
    if any(parts[index] == "annotations" for index in range(len(parts) - 1)):
        return None

    styles = {"original", "concise", "verbose", "first_person"}
    embodiments = {"car", "dog", "human"}
    for index in range(len(parts) - 1):
        if parts[index] in styles and parts[index + 1] in embodiments:
            relative_parts = parts[index:]
            if len(relative_parts) >= 4 and relative_parts[-1].lower().endswith(".json"):
                return Path(*relative_parts).as_posix()
    return None


def test_runtime_limit_key(envset_path: Path, scenario_id: str) -> Optional[str]:
    """Build a stable key from test annotation path and scenario ID."""

    relative_path = test_annotation_relative_path(envset_path)
    if relative_path is None:
        return None
    return f"{relative_path}::{scenario_id}"


def _validate_runtime_limits(key: str, payload: Any) -> RuntimeLimits:
    if not isinstance(payload, dict):
        raise ValueError(f"Runtime limits for '{key}' must be an object")

    max_steps = payload.get("max_steps")
    if isinstance(max_steps, bool) or not isinstance(max_steps, int) or max_steps <= 0:
        raise ValueError(f"Runtime limits for '{key}': max_steps must be a positive integer")

    if "max_wall_time_s" not in payload:
        raise ValueError(f"Runtime limits for '{key}': max_wall_time_s field is required")
    max_wall_time_s = payload["max_wall_time_s"]
    if max_wall_time_s is not None:
        if isinstance(max_wall_time_s, bool) or not isinstance(max_wall_time_s, (int, float)):
            raise ValueError(
                f"Runtime limits for '{key}': max_wall_time_s must be null or a positive number"
            )

        max_wall_time_s = float(max_wall_time_s)
        if not math.isfinite(max_wall_time_s) or max_wall_time_s <= 0:
            raise ValueError(
                f"Runtime limits for '{key}': max_wall_time_s must be null or a positive number"
            )

    annotation_sha256 = payload.get("annotation_sha256")
    if (
        not isinstance(annotation_sha256, str)
        or len(annotation_sha256) != 64
        or any(character not in "0123456789abcdef" for character in annotation_sha256)
    ):
        raise ValueError(
            f"Runtime limits for '{key}': annotation_sha256 must be a lowercase SHA-256 hex digest"
        )

    return RuntimeLimits(
        max_steps=max_steps,
        max_wall_time_s=max_wall_time_s,
        annotation_sha256=annotation_sha256,
    )


def load_runtime_limit_manifest(path: Path = RUNTIME_LIMITS_PATH) -> Dict[str, RuntimeLimits]:
    """Load and validate the versioned runtime-limit manifest."""

    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Test runtime-limit manifest not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in test runtime-limit manifest {path}: {exc}") from exc

    if not isinstance(payload, dict):
        raise ValueError(f"Test runtime-limit manifest must be an object: {path}")
    if payload.get("schema_version") != 1:
        raise ValueError(
            f"Unsupported test runtime-limit schema in {path}: "
            f"expected 1, got {payload.get('schema_version')!r}"
        )

    episodes = payload.get("episodes")
    if not isinstance(episodes, dict):
        raise ValueError(f"Test runtime-limit manifest 'episodes' must be an object: {path}")

    return {key: _validate_runtime_limits(key, value) for key, value in episodes.items()}


@lru_cache(maxsize=1)
def default_runtime_limit_manifest() -> Mapping[str, RuntimeLimits]:
    """Load the repository manifest once per evaluator process."""

    return load_runtime_limit_manifest()


def find_test_runtime_limits(
    envset_path: Optional[Path],
    scenario_id: str,
    *,
    manifest: Optional[Mapping[str, RuntimeLimits]] = None,
) -> Optional[RuntimeLimits]:
    """Find limits for a canonical public test annotation, if present."""

    if envset_path is None:
        return None
    key = test_runtime_limit_key(Path(envset_path), str(scenario_id))
    if key is None:
        return None
    limits = manifest if manifest is not None else default_runtime_limit_manifest()
    runtime_limits = limits.get(key)
    if runtime_limits is None:
        return None

    actual_sha256 = annotation_json_sha256(Path(envset_path))
    if actual_sha256 != runtime_limits.annotation_sha256:
        raise ValueError(
            f"Public test annotation content does not match the published runtime limits for key "
            f"'{key}': expected SHA-256 {runtime_limits.annotation_sha256}, got {actual_sha256}. "
            "Use the test annotations released with this version of OmniNavBench."
        )
    return runtime_limits
