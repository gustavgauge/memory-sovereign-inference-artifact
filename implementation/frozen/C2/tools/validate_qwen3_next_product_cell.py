#!/usr/bin/env python3
"""Validate one paired repository-task process without grading its answer."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from validate_qwen3_next_canary import digest, lifecycle_checks, load, resource_checks, write_atomic


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--population", type=Path, required=True)
    parser.add_argument("--task-id", required=True)
    parser.add_argument("--arm", choices=("msi_qwen3_next", "resident_qwen36"), required=True)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--resource", type=Path, required=True)
    parser.add_argument("--stderr", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    parsed = arguments()
    config = load(parsed.config)
    population = load(parsed.population)
    result = load(parsed.result)
    resource = load(parsed.resource)
    task = next(row for row in population["tasks"] if row["id"] == parsed.task_id)
    runtime = config["runtime"]
    hardware = config["hardware"]
    gpu_board_max = int(
        hardware["gpu_physical_board_max_bytes"]
        if "gpu_physical_board_max_bytes" in hardware
        else hardware["gpu_attributable_max_bytes"]
    )
    checks: dict[str, bool] = {}
    checks["population_identity"] = digest(parsed.population) == config["population"]["sha256"]
    checks["task_identity"] = digest(Path(task["path"])) == task["sha256"]
    for name in (
        "adapter",
        "observer",
        "bounded_launcher",
        "resource_validator",
        "product_validator",
    ):
        source = parsed.config.parent.parent / runtime[f"{name}_source"]
        checks[f"{name}_source_identity"] = digest(source) == runtime[f"{name}_source_sha256"]
    checks.update(
        resource_checks(
            "cell",
            resource,
            int(hardware["host_cgroup_max_bytes"]),
            gpu_board_max,
            min(
                int(hardware["minimum_host_margin_bytes"]),
                int(hardware["minimum_gpu_margin_bytes"]),
            ),
        )
    )
    workload = result["workload"]
    checks["workload_identity"] = (
        Path(workload["prompt_file"]).resolve() == Path(task["path"]).resolve()
        and int(workload["prompt_token_limit"]) == 0
        and int(workload["prompt_tokens"]) >= 8
        and int(workload["generated_token_count"]) >= 1
    )
    count = int(workload["generated_token_count"])
    checks["token_timing_balance"] = (
        len(workload["generated_tokens"]) == count
        and len(workload["generated_pieces"]) == count
        and len(workload["output_ready_request_elapsed_ns"]) == count
        and all(
            int(right) > int(left)
            for left, right in zip(
                [0, *workload["output_ready_request_elapsed_ns"][:-1]],
                workload["output_ready_request_elapsed_ns"],
            )
        )
    )
    timing = result["timing_ns"]
    checks["positive_timing"] = int(timing["prefill"]) > 0 and int(timing["initialization"]) > 0
    if parsed.arm == "msi_qwen3_next":
        checks["arm_identity"] = result["arm"] == "plan_cached_candidate"
        checks.update(lifecycle_checks("plan", result))
        source = result["plan"]["bounded_source"]
        checks["bounded_source_balance"] = (
            source["enabled"] is True
            and source["direct_io"] is True
            and int(source["submissions"]) == int(source["completions"])
            and int(source["active_tickets"]) == 0
            and int(source["failures"]) == 0
        )
        checks["cache_identity"] = (
            result["plan"]["cache"]["policy"] == "per_layer_lru"
            and int(result["plan"]["cache"]["capacity_per_layer"]) == 64
        )
    else:
        checks["arm_identity"] = result["arm"] == "resident_control"
        checks["resident_has_no_plan"] = result["plan"] is None
        requested_load_mode = config["arms"]["resident_qwen36"]["load_mode"]
        requested_thinking_mode = config["arms"]["resident_qwen36"].get(
            "thinking_mode", "template_default"
        )
        stderr = parsed.stderr.read_text(errors="replace")
        checks["resident_load_mode"] = workload["model_load_mode"] == requested_load_mode
        checks["resident_thinking_mode"] = (
            workload.get("thinking_mode", "template_default") == requested_thinking_mode
        )
        checks["resident_direct_io_physical"] = (
            requested_load_mode == "direct_io"
            and "Falling back to buffered" not in stderr
            and int(result["process_io"]["physical_storage_read_bytes"])
            >= int(config["arms"]["resident_qwen36"]["tensor_payload_bytes"])
        )
    valid = all(checks.values())
    certificate = {
        "schema_version": 1,
        "artifact": "qwen3_next_p2_product_task_process_certificate",
        "valid": valid,
        "task_id": parsed.task_id,
        "arm": parsed.arm,
        "checks": checks,
        "metrics": {
            "prompt_tokens": int(workload["prompt_tokens"]),
            "generated_tokens": count,
            "initialization_seconds": int(timing["initialization"]) / 1e9,
            "prefill_seconds": int(timing["prefill"]) / 1e9,
            "decode_seconds": int(timing["decode"]) / 1e9,
            "first_output_request_seconds": int(workload["output_ready_request_elapsed_ns"][0]) / 1e9,
            "host_peak_bytes": int(resource["cgroup_memory_peak_bytes"]),
            "gpu_board_peak_bytes": int(resource["device_memory_used_peak_bytes"]),
            "gpu_process_peak_bytes": int(resource["candidate_process_memory_peak_bytes"]),
        },
        "response": workload["response"],
        "inputs": {
            "config": str(parsed.config.resolve()),
            "population": str(parsed.population.resolve()),
            "result": str(parsed.result.resolve()),
            "resource": str(parsed.resource.resolve()),
            "stderr": str(parsed.stderr.resolve()),
        },
    }
    write_atomic(parsed.output, certificate)
    print(json.dumps(certificate, indent=2, sort_keys=True))
    return 0 if valid else 1


if __name__ == "__main__":
    raise SystemExit(main())
