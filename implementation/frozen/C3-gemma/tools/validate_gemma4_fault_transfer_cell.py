#!/usr/bin/env python3
"""Validate one frozen Gemma 4 representative integrated-fault cell."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


FATAL_FAULTS = {
    "short_read": "short_successful_read",
    "stale_completion": "late_stale_completion_after_reassignment",
}
RECOVERABLE_FAULTS = {
    "held_consumer": "recycle_with_held_consumer",
}


def load(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def resource_checks(resource: dict[str, Any], config: dict[str, Any]) -> dict[str, bool]:
    hardware = config["hardware"]
    events = resource["cgroup_memory_event_deltas"]
    return {
        "observer_valid": resource.get("valid") is True,
        "process_succeeded": resource.get("process_return_code") == 0,
        "zero_foreign_compute": resource.get("foreign_compute_pids") == [],
        "exact_cgroup_max": resource.get("cgroup_memory_max_bytes")
        == hardware["host_cgroup_max_bytes"],
        "exact_swap_max": resource.get("cgroup_memory_swap_max_bytes") == 0,
        "host_margin": resource.get("cgroup_memory_peak_bytes", 2**64)
        <= hardware["host_cgroup_max_bytes"] - hardware["minimum_margin_bytes"],
        "gpu_margin": resource.get("device_memory_used_peak_bytes", 2**64)
        <= hardware["gpu_board_budget_bytes"] - hardware["minimum_margin_bytes"],
        "zero_swap": resource.get("cgroup_memory_swap_peak_bytes") == 0,
        "zero_memory_events": all(
            events.get(key, 0) == 0
            for key in ("max", "oom", "oom_kill", "oom_group_kill")
        ),
    }


def clean_shutdown(result: dict[str, Any]) -> bool:
    plan = result["plan"]
    source = plan["bounded_source"]
    shutdown = plan["shutdown_snapshot"]
    telemetry = shutdown["telemetry"]
    return all(
        (
            source["submissions"] == source["completions"],
            source["failures"] == 0,
            source["active_tickets"] == 0,
            source["free_windows"] == 8,
            source["h2d_issued_bytes"] == source["h2d_completed_bytes"],
            shutdown["lifecycle"] == 3,
            shutdown["bound_slots"] == 0,
            shutdown["live_consumers"] == 0,
            shutdown["free_windows"] == 8,
            shutdown["filling_windows"] == 0,
            shutdown["ready_windows"] == 0,
            shutdown["copying_windows"] == 0,
            telemetry["source_reads_issued"] == telemetry["source_reads_completed"],
            telemetry["h2d_issued_bytes"] == telemetry["h2d_completed_bytes"],
            telemetry["consumer_acquires"] == telemetry["consumer_completions"],
        )
    )


def exact_against_control(
    result: dict[str, Any], logits: Path, control: dict[str, Any], control_logits: Path
) -> dict[str, bool]:
    return {
        "tokens": result["workload"]["generated_tokens"]
        == control["workload"]["generated_tokens"],
        "logits": digest(logits) == digest(control_logits),
        "routes": result["plan"]["route_events"] == control["plan"]["route_events"],
        "materializations": result["plan"]["materializations"]
        == control["plan"]["materializations"],
        "cache_state": result["plan"]["cache"]["final_occupancy_per_layer"]
        == control["plan"]["cache"]["final_occupancy_per_layer"],
    }


def fatal_checks(result: dict[str, Any], logits: Path, fault: str) -> dict[str, bool]:
    report = result["fault"]
    plan_terminal = report["terminal_plan_snapshot"]
    source_terminal = report["terminal_source_snapshot"]
    telemetry = plan_terminal["telemetry"]
    before = report["details"]["before"]
    after = report["details"]["after"]
    checks = {
        "fault_triggered": report["triggered"] is True,
        "fatal_classified": report["fatal"] is True,
        "expected_stage": report["stage"] == FATAL_FAULTS[fault],
        "zero_generated_tokens": result["workload"]["generated_token_count"] == 0,
        "zero_logits": result["workload"]["logits_bytes"] == 0
        and logits.stat().st_size == 0,
        "zero_accepted_after_fault": result["fault_acceptance"]
        == {
            "fatal_fault": True,
            "accepted_tokens_after_fault": 0,
            "accepted_logits_after_fault": 0,
        },
        "no_invalid_publication": (
            after["telemetry"]["h2d_issued_bytes"]
            == before["telemetry"]["h2d_issued_bytes"]
            and after["telemetry"]["h2d_completed_bytes"]
            == before["telemetry"]["h2d_completed_bytes"]
            and after["telemetry"]["readiness_events"]
            == before["telemetry"]["readiness_events"]
            and after["ready_slots"] == before["ready_slots"]
        ),
        "plan_fault_window_accounted": plan_terminal["filling_windows"] == 1
        and plan_terminal["ready_windows"] == 0
        and plan_terminal["copying_windows"] == 0
        and plan_terminal["live_consumers"] == 0,
        "source_drained": source_terminal["active_tickets"] == 0
        and source_terminal["free_windows"] == 8
        and source_terminal["filling_windows"] == 0
        and source_terminal["ready_windows"] == 0
        and source_terminal["copying_windows"] == 0
        and source_terminal["retirable_windows"] == 0,
    }
    if fault == "short_read":
        checks["short_completion_recorded"] = (
            source_terminal["injected_short_completions"] == 1
            and source_terminal["failures"] == 1
        )
    else:
        checks["stale_rejection_recorded"] = (
            telemetry["stale_generation_rejections"] == 1
        )
        checks["generation_advanced"] = (
            report["details"]["current_generation"]
            > report["details"]["stale_generation"]
        )
    return checks


def held_consumer_checks(
    result: dict[str, Any], logits: Path, control: dict[str, Any], control_logits: Path
) -> dict[str, bool]:
    report = result["fault"]
    terminal = report["terminal_plan_snapshot"]
    checks = {
        "fault_triggered": report["triggered"] is True,
        "recoverable_classified": report["fatal"] is False,
        "expected_stage": report["stage"] == RECOVERABLE_FAULTS["held_consumer"],
        "eight_outputs": result["workload"]["generated_token_count"] == 8,
        "lifecycle_closed": clean_shutdown(result),
        "held_consumer_observed": terminal["live_consumers"] == 1,
        "recycle_rejected": terminal["telemetry"]["premature_reuse_rejections"] == 1,
    }
    checks.update(
        {
            f"exact_{key}": value
            for key, value in exact_against_control(
                result, logits, control, control_logits
            ).items()
        }
    )
    return checks


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--fault", required=True)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--logits", type=Path, required=True)
    parser.add_argument("--resource", type=Path, required=True)
    parser.add_argument("--control-result", type=Path, required=True)
    parser.add_argument("--control-logits", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if args.fault not in {"none", *FATAL_FAULTS, *RECOVERABLE_FAULTS}:
        raise ValueError(f"unsupported fault: {args.fault}")
    config = load(args.config)
    result = load(args.result)
    resource = load(args.resource)
    control = load(args.control_result)
    checks = {
        "decision_identity": config.get("decision_id")
        == "gemma4-integrated-fault-transfer",
        "arm_identity": result.get("arm") == "fault_canary",
        "fault_identity": result.get("fault", {}).get("requested") == args.fault,
        "prompt_identity": result["workload"]["prompt"] == config["workload"]["prompt"],
        "workload_shape": result["workload"]["prompt_tokens"] == 42
        and result["workload"]["n_predict_limit"] == 8,
        "zero_cache": result["plan"]["cache"]["enabled"] is False
        and result["plan"]["cache"]["capacity_per_layer"] == 0,
        "real_consumer": result["consumer"]["graph"] == "stock_gemma4_llama_decode"
        and result["consumer"]["expert_consumer"] == "stock_ggml_mul_mat_id_q4_0",
        "eight_window_direct_source": result["plan"]["bounded_source"]["direct_io"] is True
        and result["plan"]["bounded_source"]["in_flight_limit"] == 8,
    }
    checks.update(
        {
            f"resource_{key}": value
            for key, value in resource_checks(resource, config).items()
        }
    )

    if args.fault == "none":
        checks.update(
            {
                "fault_not_triggered": result["fault"]["triggered"] is False,
                "eight_outputs": result["workload"]["generated_token_count"] == 8,
                "lifecycle_closed": clean_shutdown(result),
            }
        )
    elif args.fault in FATAL_FAULTS:
        checks.update(fatal_checks(result, args.logits, args.fault))
    else:
        checks.update(
            held_consumer_checks(
                result, args.logits, control, args.control_logits
            )
        )

    valid = all(checks.values())
    certificate = {
        "schema_version": 1,
        "artifact": "gemma4_integrated_fault_transfer_cell_certificate",
        "fault": args.fault,
        "valid": valid,
        "checks": checks,
        "inputs": {
            "config": {"path": str(args.config), "sha256": digest(args.config)},
            "result": {"path": str(args.result), "sha256": digest(args.result)},
            "logits": {"path": str(args.logits), "sha256": digest(args.logits)},
            "resource": {"path": str(args.resource), "sha256": digest(args.resource)},
            "control_result": {
                "path": str(args.control_result),
                "sha256": digest(args.control_result),
            },
            "control_logits": {
                "path": str(args.control_logits),
                "sha256": digest(args.control_logits),
            },
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(certificate, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return 0 if valid else 1


if __name__ == "__main__":
    raise SystemExit(main())
