# SPDX-License-Identifier: Apache-2.0
"""Persisted decode-performance curve for DeepSeek V4.1 verification widths.

The curve is the cost input to the width policy. It is machine-measured and
stored, not sampled live, so system load cannot change which draft positions a
request verifies between two runs of the same prompt. A key change invalidates
it, and the key carries the numerics revision so a kernel change cannot reuse a
stale curve.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path

SCHEMA_VERSION = 1
DEFAULT_TTL_S = 3600
TIME_UNIT = "ms"


def curve_key(
    model_id: str,
    machine_id: str,
    parallel_sizes: tuple[int, ...],
    numerics_revision: str,
) -> str:
    """Stable identity of one curve. Any field change yields a new key."""
    payload = "|".join(
        (
            str(SCHEMA_VERSION),
            TIME_UNIT,
            model_id,
            machine_id,
            ",".join(str(size) for size in parallel_sizes),
            numerics_revision,
        )
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _median(values: tuple[float, ...]) -> float:
    ordered = sorted(values)
    count = len(ordered)
    if count == 0:
        raise ValueError("a curve point needs at least one sample")
    middle = count // 2
    if count % 2:
        return ordered[middle]
    return (ordered[middle - 1] + ordered[middle]) / 2.0


@dataclass(frozen=True)
class CurvePoint:
    """Measured samples for one (width, rows) shape; `ms` is their median."""

    width: int
    rows: int
    samples: tuple[float, ...]

    @property
    def ms(self) -> float:
        return _median(self.samples)


@dataclass
class DecodeCurve:
    """A keyed set of curve points, persisted atomically with a TTL."""

    key: str
    points: dict[tuple[int, int], CurvePoint] = field(default_factory=dict)
    measured_at: float = 0.0
    ttl_s: int = DEFAULT_TTL_S

    def observe(self, width: int, rows: int, ms: float) -> None:
        """Join one sample to the point's sample set. Median recomputes."""
        point = self.points.get((width, rows))
        if point is None:
            self.points[(width, rows)] = CurvePoint(width, rows, (float(ms),))
            return
        self.points[(width, rows)] = CurvePoint(
            width, rows, tuple(sorted(point.samples + (float(ms),)))
        )

    def ms(self, width: int, rows: int) -> float | None:
        point = self.points.get((width, rows))
        return None if point is None else point.ms

    def best(self, expected_ms: dict[tuple[int, int], float]) -> tuple[int, int] | None:
        """Shape maximizing expected tokens per millisecond over known points."""
        best_shape: tuple[int, int] | None = None
        best_ratio = 0.0
        for shape in sorted(self.points):
            cost = self.points[shape].ms
            if cost <= 0:
                continue
            ratio = expected_ms.get(shape, 0.0) / cost
            if ratio > best_ratio:
                best_ratio = ratio
                best_shape = shape
        return best_shape

    def to_dict(self) -> dict:
        return {
            "version": SCHEMA_VERSION,
            "time_unit": TIME_UNIT,
            "key": self.key,
            "measured_at": self.measured_at,
            "ttl_s": self.ttl_s,
            "points": [
                {"width": p.width, "rows": p.rows, "samples": list(p.samples)}
                for p in sorted(self.points.values(), key=lambda p: (p.width, p.rows))
            ],
        }

    @classmethod
    def from_dict(cls, data: dict, *, key: str) -> "DecodeCurve | None":
        if not isinstance(data, dict):
            return None
        if data.get("version") != SCHEMA_VERSION or data.get("key") != key:
            return None
        raw_points = data.get("points")
        if not isinstance(raw_points, list):
            return None
        curve = cls(key=key, measured_at=float(data.get("measured_at") or 0.0),
                    ttl_s=int(data.get("ttl_s") or DEFAULT_TTL_S))
        for raw in raw_points:
            if not isinstance(raw, dict):
                return None
            samples = raw.get("samples")
            if not isinstance(samples, list) or not samples:
                return None
            try:
                width = int(raw["width"])
                rows = int(raw["rows"])
                values = tuple(float(v) for v in samples)
            except (KeyError, TypeError, ValueError):
                return None
            curve.points[(width, rows)] = CurvePoint(width, rows, values)
        return curve

    def write(self, path: Path) -> None:
        """Atomic write: tmp in the target directory, replace, fsync parent."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(self.to_dict(), sort_keys=True)
        handle = tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=str(path.parent),
            prefix=path.name + ".", suffix=".tmp", delete=False,
        )
        try:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        finally:
            handle.close()
        os.replace(handle.name, str(path))
        directory_fd = os.open(str(path.parent), os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)

    @classmethod
    def read(
        cls, path: Path, *, key: str, ttl_s: int | None = None, now: float | None = None
    ) -> "DecodeCurve | None":
        """Return the curve when fresh and valid; None when missing, stale, or corrupt."""
        path = Path(path)
        if not path.is_file():
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        curve = cls.from_dict(data, key=key)
        if curve is None:
            return None
        ttl = curve.ttl_s if ttl_s is None else ttl_s
        age = (now if now is not None else time.time()) - curve.measured_at
        if ttl >= 0 and age > ttl:
            return None
        return curve
