# SPDX-License-Identifier: Apache-2.0
"""Structured-output rollback bound and failure policy.

DeepInfer's grammar matcher advances the automaton over the accepted tokens and
the drafts, then rewinds when a draft is rejected, bounded by
``USE_NEXTN_DECODE + 2``. A structured-output failure kills only that request:
``so_status = -1``, a sentinel token, and a finished-with-failure state. A
preempted structured-output request cannot be recomputed.

Source of the rule: DeepInfer's runtime scheduling report, the structured-output
section, and the C++ scheduler's failure path.
"""

from __future__ import annotations

SENTINEL_TOKEN = 0
SO_FAILED_STATUS = -1


def rollback_cap(nextn: int, extra: int = 2) -> int:
    """Grammar rollback bound, ``USE_NEXTN_DECODE + extra``."""
    return max(1, int(nextn) + max(0, int(extra)))


def rollback_plan(advanced: int, accepted: int, nextn: int) -> int:
    """Tokens to rewind when a draft is rejected.

    The automaton keeps the accepted prefix plus its confirmed token and rewinds
    the rest, capped by the rollback bound. A negative or oversized input is
    clamped, so the plan is always a usable count.
    """
    advanced = max(0, int(advanced))
    accepted = max(0, int(accepted))
    keep = min(advanced, accepted + 1)
    return min(advanced - keep, rollback_cap(nextn))


def so_failure_status() -> int:
    """The status a structured-output failure reports to the caller."""
    return SO_FAILED_STATUS


def so_failure_token() -> int:
    """The sentinel token emitted when a structured-output request fails."""
    return SENTINEL_TOKEN


def can_recompute_after_preemption() -> bool:
    """A preempted structured-output request cannot be recomputed."""
    return False
