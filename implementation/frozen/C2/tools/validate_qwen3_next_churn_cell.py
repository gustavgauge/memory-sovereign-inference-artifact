#!/usr/bin/env python3
"""Validate one frozen Qwen3-Next matched-control churn cell.

Scientific negatives (capacity, exactness, lifecycle, or churn predicates) are
recorded as outcomes.  The validator fails only when the evidence envelope is
malformed, contaminated, or does not instantiate the frozen comparison.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import tempfile
from pathlib import Path
from typing import Any


ARMS = ("kernel_page_cache", "minimal_direct", "full_msi")
SOURCE_MODES = {
    "kernel_page_cache": "kernel_page_cache",
    "minimal_direct": "minimal_synchronous_odirect",
    "full_msi": "full_msi_async_odirect",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--profile", choices=("qualification", "campaign"), required=True)
    parser.add_argument("--arm", choices=ARMS, required=True)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--logits", type=Path, required=True)
    parser.add_argument("--resource", type=Path, required=True)
    parser.add_argument("--control", type=Path, required=True)
    parser.add_argument("--oracle-result", type=Path, required=True)
    parser.add_argument("--oracle-logits", type=Path, required=True)
    parser.add_argument("--oracle-resource", type=Path, required=True)
    parser.add_argument("--oracle-control", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def write_atomic(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, delete=False
    ) as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")
        temporary = Path(stream.name)
    os.replace(temporary, path)


def complete_route_shape(result: dict[str, Any], prompt_tokens: int, outputs: int,
                         n_ubatch: int) -> bool:
    events = result["plan"]["route_events"]
    prefill_chunks = math.ceil(prompt_tokens / n_ubatch)
    if len(events) != 48 * (prefill_chunks + outputs - 1):
        return False
    observed: set[tuple[int, int, int]] = set()
    for row in events:
        step = int(row["step"])
        chunk = int(row["chunk"])
        layer = int(row["layer"])
        selected = row["selected_unique"]
        if not (0 <= step < outputs and 0 <= layer < 48):
            return False
        if step == 0:
            if not (0 <= chunk < prefill_chunks):
                return False
        elif chunk != 0:
            return False
        if not isinstance(selected, list) or not selected:
            return False
        if len(selected) != len(set(int(value) for value in selected)):
            return False
        if any(not 0 <= int(value) < 512 for value in selected):
            return False
        observed.add((step, chunk, layer))
    expected = {
        (0, chunk, layer)
        for chunk in range(prefill_chunks)
        for layer in range(48)
    }
    expected.update(
        (step, 0, layer) for step in range(1, outputs) for layer in range(48)
    )
    return observed == expected


def source_balance(plan: dict[str, Any]) -> bool:
    source = plan["bounded_source"]
    phase = plan["phase_source"]
    return (
        int(source["submissions"]) == int(source["completions"])
        and int(source["h2d_issued_bytes"]) == int(source["h2d_completed_bytes"])
        and int(phase["prefill"]["logical_bytes"])
        + int(phase["decode"]["logical_bytes"])
        == int(source["logical_bytes"])
        and int(phase["prefill"]["physical_read_bytes"])
        + int(phase["decode"]["physical_read_bytes"])
        == int(source["physical_read_bytes"])
        and int(phase["prefill"]["h2d_completed_bytes"])
        + int(phase["decode"]["h2d_completed_bytes"])
        == int(source["h2d_completed_bytes"])
    )


def cache_shape(plan: dict[str, Any]) -> bool:
    cache = plan["cache"]
    final_state = cache["final_state"]
    return (
        cache["enabled"] is True
        and cache["policy"] == "per_layer_lru"
        and int(cache["capacity_per_layer"]) == 64
        and len(final_state) == 48
        and all(len(layer) == 64 for layer in final_state)
        and cache["final_occupancy_per_layer"] == [64] * 48
        and all(int(value) <= 64 for value in cache["max_occupancy_per_layer"])
        and int(cache["accesses"]) == int(cache["hits"]) + int(cache["misses"])
        and int(cache["decode_accesses"])
        == int(cache["decode_hits"]) + int(cache["decode_misses"])
        and int(cache["decode_populations"]) == int(cache["decode_misses"])
        and int(cache["stale_entry_rejections"]) == 0
        and int(cache["premature_reuse_rejections"]) == 0
    )


def full_msi_lifecycle(plan: dict[str, Any]) -> bool:
    source = plan["bounded_source"]
    before = plan["pre_shutdown_snapshot"]
    after = plan["shutdown_snapshot"]
    telemetry = after["telemetry"]
    rejection_keys = (
        "capacity_rejections",
        "lifecycle_rejections",
        "premature_reuse_rejections",
        "stale_generation_rejections",
    )
    source_zero = (
        "active_tickets",
        "cancellations",
        "failures",
        "filling_windows",
        "copying_windows",
        "ready_windows",
        "retirable_windows",
        "lifecycle_rejections",
        "queue_rejections",
        "dynamic_direct_allocations",
    )
    return (
        plan["component_manifest_valid"] is True
        and plan["lifecycle_profile"] == "full_msi"
        and source["direct_io"] is True
        and source["registered_host_windows"] is True
        and int(source["in_flight_limit"]) == 8
        and int(source["peak_in_flight"]) == 8
        and int(source["free_windows"]) == 8
        and int(source["fixed_direct_reads"]) == 3 * int(source["submissions"])
        and all(int(source[key]) == 0 for key in source_zero)
        and int(before["free_windows"]) == 8
        and int(before["bound_slots"]) == 0
        and int(before["live_consumers"]) == 0
        and int(before["ready_slots"]) == 0
        and int(after["free_windows"]) == 8
        and int(after["bound_slots"]) == 0
        and int(after["live_consumers"]) == 0
        and int(after["ready_slots"]) == 0
        and int(telemetry["requests_begun"]) == int(telemetry["requests_finished"])
        and int(telemetry["source_reads_issued"])
        == int(telemetry["source_reads_completed"])
        == int(telemetry["scheduled_objects"])
        and int(telemetry["bindings"])
        == int(telemetry["readiness_events"])
        == int(telemetry["consumer_acquires"])
        == int(telemetry["consumer_completions"])
        == int(telemetry["slot_releases"])
        == int(telemetry["window_recycles"])
        and int(telemetry["completed_application_read_bytes"])
        == int(telemetry["h2d_completed_bytes"])
        == int(plan["h2d_bytes"])
        and all(int(telemetry[key]) == 0 for key in rejection_keys)
        and int(telemetry["resets"]) == 1
        and int(telemetry["shutdowns"]) == 1
    )


def minimal_lifecycle(arm: str, plan: dict[str, Any]) -> bool:
    source = plan["bounded_source"]
    direct = arm == "minimal_direct"
    return (
        plan["component_manifest_valid"] is True
        and plan["lifecycle_profile"] == "minimal_synchronous"
        and source["direct_io"] is direct
        and source["registered_host_windows"] is True
        and int(source["in_flight_limit"]) == 1
        and int(source["peak_in_flight"]) == 1
        and int(source["active_tickets"]) == 0
        and int(source["free_windows"]) == 1
        and int(source["dynamic_direct_allocations"]) == 0
        and int(source["fixed_direct_reads"])
        == (3 * int(source["submissions"]) if direct else 0)
        and plan["pre_shutdown_snapshot"] == {
            "lifecycle": "minimal_synchronous",
            "live_consumers": 0,
            "active_operations": 0,
        }
        and plan["shutdown_snapshot"] == {
            "lifecycle": "shutdown",
            "live_consumers": 0,
            "active_operations": 0,
        }
    )


def churn_outcome(result: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    cache = result["plan"]["cache"]
    thresholds = config["churn_predicates"]
    per_layer = [int(value) for value in cache["decode_populations_per_layer"]]
    recycled = int(cache["decode_recycled_slots"])
    checks = {
        "cache_population_exact": len(cache["final_state"]) * 64
        == int(thresholds["cache_population_slots"]),
        "aggregate_decode_generations": int(cache["decode_populations"])
        >= int(thresholds["minimum_aggregate_decode_generations"]),
        "every_layer_decode_generations": len(per_layer) == 48
        and min(per_layer) >= int(thresholds["minimum_decode_generations_per_layer"]),
        "recycled_slot_fraction": recycled / int(thresholds["cache_population_slots"])
        >= float(thresholds["minimum_recycled_slot_fraction"]),
    }
    return {
        "passes": all(checks.values()),
        "checks": checks,
        "aggregate_decode_generations": int(cache["decode_populations"]),
        "minimum_layer_decode_generations": min(per_layer),
        "maximum_layer_decode_generations": max(per_layer),
        "recycled_slots": recycled,
        "recycled_slot_fraction": recycled
        / int(thresholds["cache_population_slots"]),
    }


def main() -> int:
    parsed = parse_args()
    config = load(parsed.config)
    result = load(parsed.result)
    resource = load(parsed.resource)
    control = load(parsed.control)
    oracle = load(parsed.oracle_result)
    oracle_resource = load(parsed.oracle_resource)
    oracle_control = load(parsed.oracle_control)
    workload = config["workloads"][parsed.profile]
    runtime = config["runtime"]
    hardware = config["hardware"]
    prompt_tokens = int(workload["prompt_tokens"])
    outputs = int(workload["output_tokens"])
    n_ubatch = int(workload["n_ubatch"])
    checks: dict[str, bool] = {}

    root = parsed.config.parent.parent
    for name in ("adapter", "bounded_source", "observer", "bounded_launcher"):
        source = root / runtime[f"{name}_source"]
        checks[f"{name}_identity"] = digest(source) == runtime[f"{name}_sha256"]
    binary = Path(runtime["binary_path"])
    checks["binary_identity"] = (
        digest(binary) == runtime["binary_sha256"]
        and Path(control["command"][0]) == binary
        and Path(oracle_control["command"][0]) == binary
    )

    expected_source_identity = config["source_object_identity"]
    candidate_workload = result["workload"]
    oracle_workload = oracle["workload"]
    prompt_path = Path(workload["prompt_path"])
    if not prompt_path.is_absolute():
        prompt_path = root / prompt_path
    checks.update({
        "candidate_launch": control.get("valid") is True
        and control.get("return_code") == 0,
        "oracle_launch": oracle_control.get("valid") is True
        and oracle_control.get("return_code") == 0,
        "candidate_observer": resource.get("valid") is True
        and resource.get("process_return_code") == 0,
        "oracle_observer": oracle_resource.get("valid") is True
        and oracle_resource.get("process_return_code") == 0,
        "no_foreign_compute": resource.get("foreign_compute_pids") == []
        and oracle_resource.get("foreign_compute_pids") == [],
        "hardware_identity": resource.get("gpu_uuid") == hardware["gpu_uuid"],
        "prompt_identity": digest(prompt_path) == workload["prompt_sha256"]
        and candidate_workload["prompt_file"] == workload["prompt_path"]
        and oracle_workload["prompt_file"] == workload["prompt_path"],
        "candidate_limit_identity": int(resource["cgroup_memory_max_bytes"])
        == int(hardware["host_cgroup_max_bytes"])
        and int(resource["cgroup_memory_swap_max_bytes"]) == 0,
        "arm_identity": result.get("arm") == parsed.arm
        and result["plan"]["source_mode"] == SOURCE_MODES[parsed.arm],
        "oracle_identity": oracle.get("arm") == "sync_oracle"
        and oracle["plan"]["source_mode"] == "sync_oracle_buffered_discard",
        "workload_shape": candidate_workload["prompt_tokens"] == prompt_tokens
        and candidate_workload["prompt_token_limit"] == prompt_tokens
        and candidate_workload["generated_token_count"] == outputs
        and candidate_workload["n_predict_limit"] == outputs
        and candidate_workload["n_ubatch"] == n_ubatch
        and candidate_workload["use_mmap"] is False
        and candidate_workload["sampling"] == "greedy_eog_masked_fixed_horizon"
        and int(candidate_workload["eog_tokens_masked"]) > 0
        and oracle_workload["prompt_tokens"] == prompt_tokens
        and oracle_workload["generated_token_count"] == outputs
        and oracle_workload["n_ubatch"] == n_ubatch
        and oracle_workload["sampling"] == "greedy_eog_masked_fixed_horizon"
        and oracle_workload["eog_tokens_masked"]
        == candidate_workload["eog_tokens_masked"],
        "logits_shape": parsed.logits.stat().st_size
        == parsed.oracle_logits.stat().st_size
        == outputs * int(candidate_workload["vocabulary_size"]) * 4
        == int(candidate_workload["logits_bytes"]),
        "candidate_route_complete": complete_route_shape(
            result, prompt_tokens, outputs, n_ubatch
        ),
        "oracle_route_complete": complete_route_shape(
            oracle, prompt_tokens, outputs, n_ubatch
        ),
        "source_object_identity": result["plan"]["source_object_identity"]
        == oracle["plan"]["source_object_identity"]
        == expected_source_identity,
        "source_inventory_identity": int(result["plan"]["bounded_source"]["source_file_bytes"])
        == int(oracle["plan"]["bounded_source"]["source_file_bytes"])
        == int(config["model"]["bytes"])
        and int(result["plan"]["bounded_source"]["managed_expert_inventory_bytes"])
        == int(oracle["plan"]["bounded_source"]["managed_expert_inventory_bytes"])
        == int(config["model"]["managed_expert_inventory_bytes"]),
        "consumer_identity": result["consumer"] == oracle["consumer"],
        "runtime_identity": result["consumer"]["base_runtime_commit"]
        == oracle["consumer"]["base_runtime_commit"]
        == config["runtime_identity"]["base_commit"]
        and result["consumer"]["runtime_adapter_commit"]
        == oracle["consumer"]["runtime_adapter_commit"]
        == config["runtime_identity"]["adapter_commit"],
        "model_memory_identity": result["model_memory"] == oracle["model_memory"],
        "cache_shape": cache_shape(result["plan"]) and cache_shape(oracle["plan"]),
        "source_balance": source_balance(result["plan"]),
    })

    lifecycle_pass = (
        full_msi_lifecycle(result["plan"])
        if parsed.arm == "full_msi"
        else minimal_lifecycle(parsed.arm, result["plan"])
    )
    exactness = {
        "tokens": candidate_workload["generated_tokens"]
        == oracle_workload["generated_tokens"],
        "pieces": candidate_workload["generated_pieces"]
        == oracle_workload["generated_pieces"],
        "logits": digest(parsed.logits) == digest(parsed.oracle_logits),
        "routes": result["plan"]["route_events"] == oracle["plan"]["route_events"],
        "final_cache_state": result["plan"]["cache"]["final_state"]
        == oracle["plan"]["cache"]["final_state"],
    }
    events = resource["cgroup_memory_event_deltas"]
    host_peak = int(resource["cgroup_memory_peak_bytes"])
    gpu_peak = int(resource["device_memory_used_peak_bytes"])
    margin = int(hardware["minimum_margin_bytes"])
    resource_predicates = {
        "host_margin": host_peak <= int(hardware["host_cgroup_max_bytes"]) - margin,
        "gpu_margin": gpu_peak <= int(hardware["gpu_board_budget_bytes"]) - margin,
        "zero_swap": int(resource["cgroup_memory_swap_peak_bytes"]) == 0,
        "zero_memory_events": bool(events)
        and all(int(value) == 0 for value in events.values()),
    }
    churn = (
        churn_outcome(result, config)
        if parsed.profile == "campaign"
        else {"passes": None, "checks": {}}
    )
    evidence_valid = all(checks.values())
    outcome = {
        "correctness_exact": all(exactness.values()),
        "exactness": exactness,
        "resource_admissible": all(resource_predicates.values()),
        "resource_predicates": resource_predicates,
        "lifecycle_admissible": lifecycle_pass,
        "churn": churn,
        "performance_eligible": all(exactness.values())
        and all(resource_predicates.values())
        and lifecycle_pass,
    }
    certificate = {
        "schema_version": 1,
        "artifact": "qwen3_next_matched_control_churn_cell_certificate",
        "valid": evidence_valid,
        "profile": parsed.profile,
        "arm": parsed.arm,
        "checks": checks,
        "outcome": outcome,
        "metrics": {
            "complete_request_wall_seconds": int(
                candidate_workload["output_ready_request_elapsed_ns"][-1]
            ) / 1e9,
            "prefill_seconds": int(result["timing_ns"]["prefill"]) / 1e9,
            "decode_seconds": int(result["timing_ns"]["decode"]) / 1e9,
            "decode_outputs_per_second": (outputs - 1)
            / (int(result["timing_ns"]["decode"]) / 1e9),
            "decode_source_wait_seconds": int(
                result["plan"]["phase_source"]["decode"]["exposed_wait_ns"]
            ) / 1e9,
            "decode_physical_source_bytes_per_output": int(
                result["plan"]["phase_source"]["decode"]["physical_read_bytes"]
            ) / max(1, outputs - 1),
            "host_peak_bytes": host_peak,
            "gpu_board_peak_bytes": gpu_peak,
        },
        "inputs": {
            "config": str(parsed.config.resolve()),
            "result": str(parsed.result.resolve()),
            "oracle_result": str(parsed.oracle_result.resolve()),
        },
    }
    write_atomic(parsed.output, certificate)
    print(json.dumps(certificate, indent=2, sort_keys=True))
    return 0 if evidence_valid else 1


if __name__ == "__main__":
    raise SystemExit(main())
