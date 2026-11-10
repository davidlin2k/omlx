# SPDX-License-Identifier: Apache-2.0
"""Greedy residency placement for one host, mapped from DeepInfer's EPLB policy.

DeepInfer places expert replicas across EP ranks to minimize the worst rank load
per historical step. On one host the same objective applies to residency slots:
which experts stay resident, and in which order the streaming reads land. Routing
is never changed by this policy, only when an expert's weights are read.
"""

from __future__ import annotations


def _slot_loads(trace, placement, num_slots):
    loads = [[0.0] * num_slots for _ in trace]
    for step, row in enumerate(trace):
        for expert, tokens in enumerate(row):
            loads[step][placement[expert]] += tokens
    return loads


def moe_lb(trace, placement, num_slots) -> float:
    """Slowest slot over mean, the same ratio DeepInfer reports as moe_lb."""
    loads = _slot_loads(trace, placement, num_slots)
    slowest = sum(max(step) for step in loads)
    mean = sum(sum(step) / num_slots for step in loads)
    return slowest / max(mean, 1e-9)


def naive_placement(num_experts: int, num_slots: int) -> tuple[int, ...]:
    """Round-robin, the layout to beat."""
    return tuple(expert % num_slots for expert in range(num_experts))


def rebalance(trace, num_slots: int, capacity_per_slot: int) -> tuple[int, ...]:
    """Slot per expert, heaviest first, minimizing the worst historical step.

    The tie rule prefers the lower slot, so the result is deterministic.
    """
    experts = len(trace[0])
    if num_slots * capacity_per_slot < experts:
        raise ValueError("capacity cannot hold every expert")
    totals = [sum(row[expert] for row in trace) for expert in range(experts)]
    order = sorted(range(experts), key=lambda expert: (-totals[expert], expert))
    placement = [0] * experts
    occupancy = [0] * num_slots
    loads = [[0.0] * num_slots for _ in trace]
    for expert in order:
        best_slot, best_score = None, None
        for slot in range(num_slots):
            if occupancy[slot] >= capacity_per_slot:
                continue
            score = sum(
                max(loads[step][slot] + row[expert], 0.0)
                for step, row in enumerate(trace)
            )
            if best_score is None or score < best_score:
                best_slot, best_score = slot, score
        placement[expert] = best_slot
        occupancy[best_slot] += 1
        for step, row in enumerate(trace):
            loads[step][best_slot] += row[expert]
    return tuple(placement)
