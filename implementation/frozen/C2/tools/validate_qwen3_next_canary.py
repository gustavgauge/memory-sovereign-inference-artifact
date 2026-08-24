#!/usr/bin/env python3
"""Validate the matched Qwen3-Next source-path correctness canary."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any


def args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--candidate-resource", type=Path, required=True)
    parser.add_argument("--control", type=Path, required=True)
    parser.add_argument("--control-resource", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def digest(path: Path) -> str:
    sha = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            sha.update(block)
    return sha.hexdigest()


def write_atomic(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, delete=False
    ) as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")
        temporary = Path(stream.name)
    os.replace(temporary, path)


def lifecycle_checks(prefix: str, result: dict[str, Any]) -> dict[str, bool]:
    plan = result["plan"]
    before = plan["pre_shutdown_snapshot"]
    after = plan["shutdown_snapshot"]
    telemetry = after["telemetry"]
    rejection_keys = (
        "capacity_rejections",
        "lifecycle_rejections",
        "premature_reuse_rejections",
        "stale_generation_rejections",
    )
    return {
        f"{prefix}_component_manifest": plan.get("component_manifest_valid") is True,
        f"{prefix}_request_balance": telemetry["requests_begun"]
        == telemetry["requests_finished"],
        f"{prefix}_source_balance": telemetry["source_reads_issued"]
        == telemetry["source_reads_completed"]
        == telemetry["scheduled_objects"],
        f"{prefix}_consumer_balance": telemetry["bindings"]
        == telemetry["readiness_events"]
        == telemetry["consumer_acquires"]
        == telemetry["consumer_completions"]
        == telemetry["slot_releases"]
        == telemetry["window_recycles"],
        f"{prefix}_byte_balance": telemetry["completed_application_read_bytes"]
        == telemetry["h2d_completed_bytes"]
        == plan["h2d_bytes"]
        and plan["d2d_scatter_bytes"]
        == plan["h2d_bytes"] + plan["cache"]["logical_hit_bytes"],
        f"{prefix}_quiescent_before_shutdown": before["bound_slots"] == 0
        and before["live_consumers"] == 0
        and before["ready_slots"] == 0,
        f"{prefix}_reset_shutdown": telemetry["resets"] == 1
        and telemetry["shutdowns"] == 1,
        f"{prefix}_no_plan_rejections": all(telemetry[key] == 0 for key in rejection_keys),
    }


def resource_checks(
    prefix: str, resource: dict[str, Any], host_max: int, gpu_max: int, margin: int
) -> dict[str, bool]:
    events = resource.get("cgroup_memory_event_deltas", {})
    return {
        f"{prefix}_observer_valid": resource.get("valid") is True
        and resource.get("process_return_code") == 0,
        f"{prefix}_no_foreign_compute": resource.get("foreign_compute_pids") == [],
        f"{prefix}_host_margin": 0
        <= int(resource["cgroup_memory_peak_bytes"])
        <= host_max - margin,
        f"{prefix}_gpu_margin": 0
        < int(resource["device_memory_used_peak_bytes"])
        <= gpu_max - margin,
        f"{prefix}_zero_swap": resource.get("cgroup_memory_swap_peak_bytes") == 0,
        f"{prefix}_zero_memory_events": bool(events)
        and all(int(value) == 0 for value in events.values()),
    }


def main() -> int:
    parsed = args()
    config = load(parsed.config)
    candidate = load(parsed.candidate)
    control = load(parsed.control)
    candidate_resource = load(parsed.candidate_resource)
    control_resource = load(parsed.control_resource)
    hardware = config["hardware"]
    runtime = config["runtime"]
    checks: dict[str, bool] = {}

    for name in ("adapter", "observer", "bounded_launcher", "canary_validator"):
        source = parsed.config.parent.parent / runtime[f"{name}_source"]
        checks[f"{name}_source_identity"] = digest(source) == runtime[f"{name}_source_sha256"]
    checks.update(lifecycle_checks("candidate", candidate))
    checks.update(lifecycle_checks("control", control))
    checks.update(
        resource_checks(
            "candidate",
            candidate_resource,
            int(hardware["host_cgroup_max_bytes"]),
            int(hardware["gpu_attributable_max_bytes"]),
            int(hardware["minimum_host_margin_bytes"]),
        )
    )
    checks.update(
        resource_checks(
            "control",
            control_resource,
            int(hardware["host_cgroup_max_bytes"]),
            int(hardware["gpu_attributable_max_bytes"]),
            int(hardware["minimum_host_margin_bytes"]),
        )
    )

    candidate_plan = candidate["plan"]
    candidate_source = candidate_plan["bounded_source"]
    control_source = control["plan"]["bounded_source"]
    phase = candidate_plan["phase_source"]
    checks["candidate_direct_source"] = candidate_source["enabled"] is True and candidate_source[
        "direct_io"
    ] is True and candidate_source["in_flight_limit"] == 8
    checks["candidate_source_service_balance"] = candidate_source["submissions"] == candidate_source[
        "completions"
    ] and candidate_source["h2d_issued_bytes"] == candidate_source["h2d_completed_bytes"]
    checks["candidate_source_service_quiescent"] = candidate_source["active_tickets"] == 0 and all(
        candidate_source[key] == 0
        for key in (
            "cancellations",
            "failures",
            "filling_windows",
            "copying_windows",
            "ready_windows",
            "retirable_windows",
            "lifecycle_rejections",
            "queue_rejections",
        )
    )
    checks["candidate_phase_balance"] = phase["prefill"]["logical_bytes"] + phase["decode"][
        "logical_bytes"
    ] == candidate_source["logical_bytes"] and phase["prefill"]["physical_read_bytes"] + phase[
        "decode"
    ]["physical_read_bytes"] == candidate_source["physical_read_bytes"]
    checks["control_synchronous_source"] = control_source["enabled"] is False
    checks["prompt_identity"] = candidate["workload"]["prompt_tokens"] == control["workload"][
        "prompt_tokens"
    ] == 256 and candidate["workload"]["n_ubatch"] == control["workload"]["n_ubatch"] == 256
    checks["token_identity"] = candidate["workload"]["generated_tokens"] == control["workload"][
        "generated_tokens"
    ]
    checks["response_identity"] = candidate["workload"]["response"] == control["workload"][
        "response"
    ]
    checks["logits_identity"] = digest(Path(candidate["workload"]["logits_path"])) == digest(
        Path(control["workload"]["logits_path"])
    )
    checks["route_identity"] = candidate_plan["route_events"] == control["plan"]["route_events"]
    checks["consumer_identity"] = candidate["consumer"] == control["consumer"]
    checks["destination_identity"] = candidate["model_memory"] == control["model_memory"]
    checks["source_budget"] = candidate_source["physical_read_bytes"] <= int(
        config["cell_budgets"]["canary"]["maximum_physical_source_bytes"]
    )

    route_events = candidate_plan["route_events"]
    decode_events = [row for row in route_events if int(row["step"]) > 0]
    decode_outputs = len({int(row["step"]) for row in decode_events})
    metrics = {
        "prefill_seconds": candidate["timing_ns"]["prefill"] / 1e9,
        "decode_outputs": decode_outputs,
        "decode_seconds": candidate["timing_ns"]["decode"] / 1e9,
        "zero_cache_decode_tokens_per_second": decode_outputs
        / (candidate["timing_ns"]["decode"] / 1e9),
        "decode_physical_source_bytes_per_output": phase["decode"]["physical_read_bytes"]
        / decode_outputs,
        "decode_exposed_source_seconds_per_output": phase["decode"]["exposed_wait_ns"]
        / 1e9
        / decode_outputs,
        "prefill_unique_experts_per_layer_min": min(
            len(row["selected_unique"]) for row in route_events if int(row["step"]) == 0
        ),
        "prefill_unique_experts_per_layer_max": max(
            len(row["selected_unique"]) for row in route_events if int(row["step"]) == 0
        ),
    }
    valid = all(checks.values())
    certificate = {
        "schema_version": 1,
        "artifact": "qwen3_next_minimal_correctness_canary_certificate",
        "valid": valid,
        "checks": checks,
        "metrics": metrics,
        "inputs": {
            "config": str(parsed.config.resolve()),
            "candidate": str(parsed.candidate.resolve()),
            "control": str(parsed.control.resolve()),
        },
    }
    write_atomic(parsed.output, certificate)
    print(json.dumps(certificate, indent=2, sort_keys=True))
    return 0 if valid else 1


if __name__ == "__main__":
    raise SystemExit(main())
