#!/usr/bin/env python3
"""Replay one exact Qwen3-Next route trace against feasible cache policies."""

from __future__ import annotations

import argparse
import json
from collections import Counter, OrderedDict, defaultdict, deque
from pathlib import Path
from typing import Any


LAYERS = 48
EXPERTS = 512
GATE_BYTES = 589_824
UP_BYTES = 589_824
Q4_DOWN_BYTES = 589_824
Q6_DOWN_BYTES = 860_160
Q6_DOWN_LAYERS = {
    0, 1, 2, 3, 4, 5, 8, 11, 14, 17, 20, 23,
    26, 29, 32, 35, 38, 41, 42, 43, 44, 45, 46, 47,
}


def bundle_bytes(layer: int) -> int:
    down = Q6_DOWN_BYTES if layer in Q6_DOWN_LAYERS else Q4_DOWN_BYTES
    return GATE_BYTES + UP_BYTES + down


def accesses(events: list[dict[str, Any]]) -> list[tuple[str, int, int]]:
    result: list[tuple[str, int, int]] = []
    for event in events:
        phase = "prefill" if int(event["step"]) == 0 else "decode"
        layer = int(event["layer"])
        if not 0 <= layer < LAYERS:
            raise ValueError(f"invalid layer {layer}")
        for expert in event["selected_unique"]:
            expert = int(expert)
            if not 0 <= expert < EXPERTS:
                raise ValueError(f"invalid expert {expert}")
            result.append((phase, layer, expert))
    return result


def empty_counts() -> dict[str, int]:
    return {"prefill_misses": 0, "decode_misses": 0, "reload_misses": 0}


def add_miss(result: dict[str, int], phase: str, layer: int) -> None:
    result[f"{phase}_misses"] += 1
    result[f"{phase}_logical_bytes"] = (
        result.get(f"{phase}_logical_bytes", 0) + bundle_bytes(layer)
    )


def replay_lru(sequence: list[tuple[str, int, int]], capacity: int) -> dict[str, int]:
    result = empty_counts()
    caches = [OrderedDict() for _ in range(LAYERS)]
    for phase, layer, expert in sequence:
        cache = caches[layer]
        if expert in cache:
            cache.move_to_end(expert)
            continue
        add_miss(result, phase, layer)
        if capacity:
            if len(cache) == capacity:
                cache.popitem(last=False)
            cache[expert] = None
    return result


def replay_lfu(sequence: list[tuple[str, int, int]], capacity: int) -> dict[str, int]:
    result = empty_counts()
    clock = 0
    caches: list[dict[int, tuple[int, int]]] = [dict() for _ in range(LAYERS)]
    for phase, layer, expert in sequence:
        clock += 1
        cache = caches[layer]
        if expert in cache:
            count, _ = cache[expert]
            cache[expert] = (count + 1, clock)
            continue
        add_miss(result, phase, layer)
        if capacity:
            if len(cache) == capacity:
                victim = min(cache, key=lambda key: (cache[key][0], cache[key][1], key))
                del cache[victim]
            cache[expert] = (1, clock)
    return result


def replay_static_frequency(
    sequence: list[tuple[str, int, int]], capacity: int
) -> dict[str, int]:
    result = empty_counts()
    frequencies = [Counter() for _ in range(LAYERS)]
    for phase, layer, expert in sequence:
        if phase == "prefill":
            add_miss(result, phase, layer)
            frequencies[layer][expert] += 1
    pinned: list[set[int]] = []
    for layer, counts in enumerate(frequencies):
        chosen = {
            expert for expert, _ in sorted(
                counts.items(), key=lambda item: (-item[1], item[0])
            )[:capacity]
        }
        pinned.append(chosen)
        for _ in chosen:
            add_miss(result, "reload", layer)
    for phase, layer, expert in sequence:
        if phase == "decode" and expert not in pinned[layer]:
            add_miss(result, phase, layer)
    return result


def replay_belady(sequence: list[tuple[str, int, int]], capacity: int) -> dict[str, int]:
    result = empty_counts()
    future: dict[tuple[int, int], deque[int]] = defaultdict(deque)
    for index, (_, layer, expert) in enumerate(sequence):
        future[(layer, expert)].append(index)
    caches = [set() for _ in range(LAYERS)]
    infinity = len(sequence) + 1
    for index, (phase, layer, expert) in enumerate(sequence):
        positions = future[(layer, expert)]
        if not positions or positions[0] != index:
            raise ValueError("future-use index is inconsistent")
        positions.popleft()
        cache = caches[layer]
        if expert in cache:
            continue
        add_miss(result, phase, layer)
        if capacity:
            if len(cache) == capacity:
                victim = max(
                    cache,
                    key=lambda item: future[(layer, item)][0]
                    if future[(layer, item)] else infinity,
                )
                cache.remove(victim)
            cache.add(expert)
    return result


def complete(
    result: dict[str, int], arm: dict[str, Any], physical_ratio: float
) -> dict[str, int | float]:
    generated = int(arm["workload"]["generated_token_count"])
    decode_outputs = max(0, generated - 1)
    for phase in ("prefill", "decode", "reload"):
        logical = result.get(f"{phase}_logical_bytes", 0)
        result[f"{phase}_physical_bytes_estimate"] = round(logical * physical_ratio)
    measured_decode = arm["plan"]["phase_source"]["decode"]
    measured_physical = int(measured_decode["physical_read_bytes"])
    measured_exposed_seconds = int(measured_decode["exposed_wait_ns"]) / 1e9
    exposed_per_physical_byte = (
        measured_exposed_seconds / measured_physical if measured_physical else 0.0
    )
    measured_decode_seconds = int(arm["timing_ns"]["decode"]) / 1e9
    non_source_decode_seconds = max(0.0, measured_decode_seconds - measured_exposed_seconds)
    policy_exposed_seconds = (
        int(result["decode_physical_bytes_estimate"]) * exposed_per_physical_byte
    )
    favorable_seconds = non_source_decode_seconds + policy_exposed_seconds
    result["decode_outputs"] = decode_outputs
    result["decode_physical_bytes_per_output"] = (
        int(result["decode_physical_bytes_estimate"]) / decode_outputs
        if decode_outputs else 0.0
    )
    result["compute_inclusive_favorable_output_tokens_per_second"] = (
        decode_outputs / favorable_seconds if favorable_seconds else 0.0
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--capacities", type=int, nargs="+", default=[0, 64, 128, 192, 224])
    args = parser.parse_args()
    arm = json.loads(args.trace.read_text(encoding="utf-8"))
    events = arm["plan"]["route_events"]
    sequence = accesses(events)
    source = arm["plan"]["bounded_source"]
    logical = int(source["logical_bytes"])
    physical = int(source["physical_read_bytes"])
    physical_ratio = physical / logical if logical else 1.0
    policies = {
        "static_frequency": replay_static_frequency,
        "lru": replay_lru,
        "lfu": replay_lfu,
        "belady_oracle": replay_belady,
    }
    rows = []
    for capacity in args.capacities:
        if not 0 <= capacity <= EXPERTS:
            raise ValueError(f"invalid capacity {capacity}")
        for name, replay in policies.items():
            stats = complete(replay(sequence, capacity), arm, physical_ratio)
            rows.append({"capacity": capacity, "policy": name, **stats})
    output = {
        "schema_version": 1,
        "artifact": "qwen3_next_exact_trace_cache_replay",
        "classification": "exploratory",
        "trace": str(args.trace.resolve()),
        "physical_to_logical_source_ratio": physical_ratio,
        "rows": rows,
        "claim_boundary": "Implementable online policies and the Belady oracle are reported separately. Physical bytes and favorable rates are trace-scaled estimates, not executed cache measurements.",
    }
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
