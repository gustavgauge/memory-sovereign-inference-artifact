#!/usr/bin/env python3
"""Validate one bounded Qwen3-Next prefill/decode measurement cell."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from validate_qwen3_next_canary import (
    digest,
    lifecycle_checks,
    load,
    resource_checks,
    write_atomic,
)


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--resource", type=Path, required=True)
    parser.add_argument("--expected-prompt-tokens", type=int, required=True)
    parser.add_argument("--budget-key", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    parsed = arguments()
    config = load(parsed.config)
    result = load(parsed.result)
    resource = load(parsed.resource)
    runtime = config["runtime"]
    hardware = config["hardware"]
    budget = config["cell_budgets"][parsed.budget_key]
    plan = result["plan"]
    source = plan["bounded_source"]
    phase = plan["phase_source"]
    checks = lifecycle_checks("cell", result)
    for name in (
        "adapter",
        "observer",
        "bounded_launcher",
        "canary_validator",
        "cell_validator",
    ):
        source_path = parsed.config.parent.parent / runtime[f"{name}_source"]
        checks[f"{name}_source_identity"] = digest(source_path) == runtime[
            f"{name}_source_sha256"
        ]
    checks.update(
        resource_checks(
            "cell",
            resource,
            int(hardware["host_cgroup_max_bytes"]),
            int(hardware["gpu_attributable_max_bytes"]),
            min(
                int(hardware["minimum_host_margin_bytes"]),
                int(hardware["minimum_gpu_margin_bytes"]),
            ),
        )
    )
    checks["workload_shape"] = (
        result["workload"]["prompt_tokens"] == parsed.expected_prompt_tokens
        and result["workload"]["generated_token_count"] >= 8
    )
    checks["direct_bounded_source"] = (
        source["enabled"] is True
        and source["direct_io"] is True
        and source["in_flight_limit"] == 8
    )
    checks["source_balance"] = (
        source["submissions"] == source["completions"]
        and source["h2d_issued_bytes"] == source["h2d_completed_bytes"]
        and source["active_tickets"] == 0
        and source["failures"] == 0
        and source["lifecycle_rejections"] == 0
        and source["queue_rejections"] == 0
    )
    checks["phase_balance"] = (
        phase["prefill"]["logical_bytes"] + phase["decode"]["logical_bytes"]
        == source["logical_bytes"]
        and phase["prefill"]["physical_read_bytes"]
        + phase["decode"]["physical_read_bytes"]
        == source["physical_read_bytes"]
    )
    total_seconds = sum(int(value) for value in result["timing_ns"].values()) / 1e9
    checks["source_budget"] = source["physical_read_bytes"] <= int(
        budget["maximum_physical_source_bytes"]
    )
    checks["wall_budget"] = total_seconds <= float(budget["maximum_wall_seconds"])

    route = plan["route_events"]
    prefill = [row for row in route if int(row["step"]) == 0]
    decode = [row for row in route if int(row["step"]) > 0]
    chunks = sorted({int(row["chunk"]) for row in prefill})
    layers_by_chunk = {
        chunk: {int(row["layer"]) for row in prefill if int(row["chunk"]) == chunk}
        for chunk in chunks
    }
    checks["complete_prefill_chunks"] = bool(chunks) and all(
        layers == set(range(48)) for layers in layers_by_chunk.values()
    )
    decode_outputs = len({int(row["step"]) for row in decode})
    unique_counts = [len(row["selected_unique"]) for row in prefill]
    rereads = 0
    for layer in range(48):
        layer_events = [row for row in prefill if int(row["layer"]) == layer]
        total = sum(len(row["selected_unique"]) for row in layer_events)
        union = {expert for row in layer_events for expert in row["selected_unique"]}
        rereads += total - len(union)

    prefill_seconds = result["timing_ns"]["prefill"] / 1e9
    decode_seconds = result["timing_ns"]["decode"] / 1e9
    metrics = {
        "prompt_tokens": parsed.expected_prompt_tokens,
        "prefill_chunks": len(chunks),
        "prefill_seconds": prefill_seconds,
        "prefill_exposed_source_seconds": phase["prefill"]["exposed_wait_ns"] / 1e9,
        "prefill_physical_source_bytes": phase["prefill"]["physical_read_bytes"],
        "prefill_unique_experts_per_event_min": min(unique_counts),
        "prefill_unique_experts_per_event_max": max(unique_counts),
        "prefill_expert_rereads_across_chunks": rereads,
        "linear_32k_prefill_projection_seconds": prefill_seconds
        * 32768
        / parsed.expected_prompt_tokens,
        "decode_outputs": decode_outputs,
        "decode_seconds": decode_seconds,
        "zero_cache_decode_tokens_per_second": decode_outputs / decode_seconds,
        "decode_physical_source_bytes_per_output": phase["decode"][
            "physical_read_bytes"
        ]
        / decode_outputs,
        "decode_exposed_source_seconds_per_output": phase["decode"]["exposed_wait_ns"]
        / 1e9
        / decode_outputs,
    }
    valid = all(checks.values())
    certificate = {
        "schema_version": 1,
        "artifact": "qwen3_next_bounded_measurement_cell_certificate",
        "valid": valid,
        "budget_key": parsed.budget_key,
        "checks": checks,
        "metrics": metrics,
        "inputs": {
            "config": str(parsed.config.resolve()),
            "result": str(parsed.result.resolve()),
            "resource": str(parsed.resource.resolve()),
        },
    }
    write_atomic(parsed.output, certificate)
    print(json.dumps(certificate, indent=2, sort_keys=True))
    return 0 if valid else 1


if __name__ == "__main__":
    raise SystemExit(main())
