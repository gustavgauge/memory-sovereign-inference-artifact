#!/usr/bin/env python3
"""Audit Qwen3-Next semantic expert demand against canonical GGUF extents.

The recorded route events are copied from the stock ``ffn_moe_topk`` tensor
before Plan scheduling.  This tool joins those semantic layer/expert identities
to the native gate, up, and down tensor extents and reports a deduplicated
canonical-byte union.  It intentionally keeps cumulative reads and alignment
padding outside the unique-demand numerator.

The ``gguf`` Python package from the pinned llama.cpp runtime is required.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any

EXPERT_TENSOR = re.compile(
    r"^blk\.(?P<layer>\d+)\.ffn_(?P<component>down|gate|up)_exps\.weight$"
)
LAYERS = 48
EXPERTS = 512


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--model-sha256", required=True)
    parser.add_argument("--result", type=Path, required=True)
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def main() -> None:
    args = parse_args()
    try:
        from gguf import GGUFReader
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "gguf is required for execution; add the pinned llama.cpp gguf-py "
            "directory to PYTHONPATH"
        ) from exc
    run = load_json(args.result)
    events = run["plan"]["route_events"]
    prefill = [event for event in events if int(event["step"]) == 0]
    demanded = {
        (int(event["layer"]), int(expert))
        for event in prefill
        for expert in event["selected_unique"]
    }

    if len(prefill) != 8 * LAYERS:
        raise ValueError(f"expected 384 prefill events, found {len(prefill)}")
    if demanded != {(layer, expert) for layer in range(LAYERS) for expert in range(EXPERTS)}:
        raise ValueError("prefill trace does not cover every layer/expert identity")

    reader = GGUFReader(args.model, "r")
    tensors: list[dict[str, Any]] = []
    intervals: list[tuple[str, int, int]] = []
    for tensor in reader.tensors:
        match = EXPERT_TENSOR.match(tensor.name)
        if match is None:
            continue
        layer = int(match.group("layer"))
        component = match.group("component")
        tensor_bytes = int(tensor.n_bytes)
        if tensor_bytes % EXPERTS != 0:
            raise ValueError(f"expert tensor is not divisible by {EXPERTS}: {tensor.name}")
        expert_bytes = tensor_bytes // EXPERTS
        tensor_record = {
            "layer": layer,
            "component": component,
            "source_offset": int(tensor.data_offset),
            "tensor_bytes": tensor_bytes,
            "expert_bytes": expert_bytes,
            "ggml_type": int(tensor.tensor_type),
        }
        tensors.append(tensor_record)
        for expert in range(EXPERTS):
            if (layer, expert) not in demanded:
                continue
            intervals.append(
                (
                    args.model_sha256,
                    int(tensor.data_offset) + expert * expert_bytes,
                    expert_bytes,
                )
            )

    if len(tensors) != LAYERS * 3:
        raise ValueError(f"expected 144 managed expert tensors, found {len(tensors)}")
    if len(intervals) != LAYERS * EXPERTS * 3:
        raise ValueError(f"expected 73,728 component intervals, found {len(intervals)}")

    ordered = sorted(intervals)
    duplicate_count = len(ordered) - len(set(ordered))
    overlap_count = 0
    previous: tuple[str, int, int] | None = None
    for interval in ordered:
        if (
            previous is not None
            and interval[0] == previous[0]
            and interval[1] < previous[1] + previous[2]
        ):
            overlap_count += 1
        if (
            previous is None
            or interval[0] != previous[0]
            or interval[1] + interval[2] > previous[1] + previous[2]
        ):
            previous = interval

    canonical = "".join(
        f"{source}\t{offset}\t{length}\n" for source, offset, length in ordered
    ).encode("ascii")
    union_bytes = sum(length for _, _, length in set(ordered))
    gate_sizes = sorted({row["expert_bytes"] for row in tensors if row["component"] == "gate"})
    up_sizes = sorted({row["expert_bytes"] for row in tensors if row["component"] == "up"})
    down_sizes = sorted({row["expert_bytes"] for row in tensors if row["component"] == "down"})
    layer_bundle_sizes = sorted(
        {
            sum(
                row["expert_bytes"]
                for row in tensors
                if row["layer"] == layer
            )
            for layer in range(LAYERS)
        }
    )

    report = {
        "schema_version": 1,
        "artifact": "qwen3_next_semantic_demand_extent_audit",
        "demand_source": "stock ffn_moe_topk tensor captured before Plan scheduling",
        "inclusion_rule": "at least one authoritative routed token for the layer/expert identity",
        "prefill_events": len(prefill),
        "semantic_layer_expert_objects": len(demanded),
        "minimum_known_routed_token_count_per_object": 1,
        "managed_expert_tensors": len(tensors),
        "component_intervals": len(intervals),
        "duplicate_component_intervals": duplicate_count,
        "overlapping_component_intervals": overlap_count,
        "deduplicated_union_bytes": union_bytes,
        "canonical_serialization": "sorted lines: model_sha256<TAB>offset<TAB>length<LF>",
        "sorted_union_sha256": hashlib.sha256(canonical).hexdigest(),
        "bundle_composition": {
            "components": ["gate", "up", "down"],
            "gate_expert_bytes": gate_sizes,
            "up_expert_bytes": up_sizes,
            "down_expert_bytes": down_sizes,
            "logical_bundle_bytes": layer_bundle_sizes,
            "q4_down_layers": sum(
                1
                for row in tensors
                if row["component"] == "down" and row["expert_bytes"] == 589824
            ),
            "q6_down_layers": sum(
                1
                for row in tensors
                if row["component"] == "down" and row["expert_bytes"] == 860160
            ),
        },
    }
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
