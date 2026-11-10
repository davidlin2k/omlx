# SPDX-License-Identifier: Apache-2.0
"""Unit 20 predicate: structured-output rollback bound and failure policy."""

from __future__ import annotations

from omlx.patches.deepseek_v41.so_rollback import (
    SENTINEL_TOKEN,
    SO_FAILED_STATUS,
    can_recompute_after_preemption,
    rollback_cap,
    rollback_plan,
    so_failure_status,
    so_failure_token,
)


def test_rollback_cap_is_nextn_plus_two():
    assert rollback_cap(5) == 7
    assert rollback_cap(1) == 3
    assert rollback_cap(0) == 2
    assert rollback_cap(5, extra=0) == 5


def test_plan_keeps_the_accepted_prefix_and_confirmed_token():
    # advanced 6 over accepted 2 -> keep 3, rewind 3.
    assert rollback_plan(6, 2, 5) == 3
    # a full accept keeps everything.
    assert rollback_plan(6, 5, 5) == 0
    # nothing advanced -> nothing to rewind.
    assert rollback_plan(0, 0, 5) == 0


def test_plan_is_clamped_by_the_bound():
    # advanced 12 over accepted 1 -> raw rewind 10, capped at 7.
    assert rollback_plan(12, 1, 5) == 7


def test_plan_clamps_negative_inputs():
    assert rollback_plan(-3, -1, 5) == 0


def test_failure_policy_values():
    assert so_failure_status() == SO_FAILED_STATUS == -1
    assert so_failure_token() == SENTINEL_TOKEN == 0
    assert can_recompute_after_preemption() is False
