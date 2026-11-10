# SPDX-License-Identifier: Apache-2.0
"""Unit 1 predicate: the persisted decode-performance curve."""

from __future__ import annotations

import time

from omlx.patches.deepseek_v41.perf_curve import (
    DEFAULT_TTL_S,
    DecodeCurve,
    curve_key,
)


def _curve(tmp_path, key="key-a"):
    curve = DecodeCurve(key=key, measured_at=time.time(), ttl_s=DEFAULT_TTL_S)
    curve.observe(1, 1, 9.0)
    curve.observe(1, 1, 11.0)
    curve.observe(4, 3, 21.0)
    return curve


def test_round_trip_returns_identical_points(tmp_path):
    curve = _curve(tmp_path)
    path = tmp_path / "curve.json"
    curve.write(path)
    loaded = DecodeCurve.read(path, key="key-a")
    assert loaded is not None
    assert loaded.key == "key-a"
    assert set(loaded.points) == {(1, 1), (4, 3)}
    assert loaded.ms(1, 1) == 10.0
    assert loaded.ms(4, 3) == 21.0


def test_median_merge_is_idempotent(tmp_path):
    curve = _curve(tmp_path)
    before = curve.ms(1, 1)
    curve.observe(1, 1, 11.0)
    curve.observe(1, 1, 9.0)
    assert curve.ms(1, 1) == before
    assert len(curve.points[(1, 1)].samples) == 4


def test_stale_curve_reads_as_missing(tmp_path):
    curve = _curve(tmp_path)
    curve.measured_at = time.time() - DEFAULT_TTL_S - 1
    path = tmp_path / "curve.json"
    curve.write(path)
    assert DecodeCurve.read(path, key="key-a") is None
    assert DecodeCurve.read(path, key="key-a", now=curve.measured_at + 1) is not None


def test_truncated_file_reads_as_missing(tmp_path):
    curve = _curve(tmp_path)
    path = tmp_path / "curve.json"
    curve.write(path)
    payload = path.read_text(encoding="utf-8")
    path.write_text(payload[: len(payload) // 2], encoding="utf-8")
    assert DecodeCurve.read(path, key="key-a") is None


def test_missing_file_reads_as_missing(tmp_path):
    assert DecodeCurve.read(tmp_path / "absent.json", key="key-a") is None


def test_key_changes_with_numerics_revision(tmp_path):
    base = curve_key("model", "machine", (1, 1, 1), "decode-curve-1")
    changed = curve_key("model", "machine", (1, 1, 1), "decode-curve-2")
    assert base != changed
    curve = DecodeCurve(key=base, measured_at=time.time())
    curve.observe(8, 8, 40.0)
    path = tmp_path / "curve.json"
    curve.write(path)
    assert DecodeCurve.read(path, key=changed) is None
    assert DecodeCurve.read(path, key=base) is not None


def test_wrong_key_reads_as_missing(tmp_path):
    curve = _curve(tmp_path, key="key-a")
    path = tmp_path / "curve.json"
    curve.write(path)
    assert DecodeCurve.read(path, key="key-b") is None


def test_best_picks_highest_expected_per_ms():
    curve = DecodeCurve(key="key-a")
    curve.observe(1, 1, 10.0)
    curve.observe(4, 3, 20.0)
    expected = {(1, 1): 4.0, (4, 3): 30.0}
    assert curve.best(expected) == (4, 3)


def test_write_leaves_no_tmp_behind(tmp_path):
    curve = _curve(tmp_path)
    path = tmp_path / "curve.json"
    curve.write(path)
    leftovers = list(tmp_path.glob("curve.json.*.tmp"))
    assert leftovers == []
