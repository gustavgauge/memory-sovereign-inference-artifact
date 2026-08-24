#!/usr/bin/env python3
"""Validate the single frozen Qwen3-Next 32K+64 zero-cache oracle confirmation."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import subprocess
import tempfile
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


def load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            hasher.update(block)
    return hasher.hexdigest()


def resolved(path: str) -> Path:
    value = Path(path)
    return value if value.is_absolute() else ROOT / value


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, delete=False
    ) as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
        temporary = Path(handle.name)
    os.replace(temporary, path)


def file_identity(
    checks: dict[str, bool], observed: dict[str, dict[str, Any]], name: str, row: dict[str, Any]
) -> None:
    path = resolved(str(row["path"]))
    exists = path.is_file()
    observed_digest = digest(path) if exists else None
    checks[f"identity_{name}"] = exists and observed_digest == row["sha256"]
    if "bytes" in row:
        checks[f"identity_{name}_bytes"] = exists and path.stat().st_size == int(row["bytes"])
    observed[name] = {
        "path": str(path),
        "bytes": path.stat().st_size if exists else None,
        "sha256": observed_digest,
    }


def frozen_identity_checks(
    config_path: Path, config: dict[str, Any]
) -> tuple[dict[str, bool], dict[str, dict[str, Any]]]:
    checks: dict[str, bool] = {
        "decision_identity": config.get("decision_id")
        == "qwen3-next-full64-oracle-confirmation",
        "one_shot_status": config.get("status") == "FROZEN_FOR_ONE_TERMINAL_CONFIRMATION",
        "one_material_repetition": config["execution"]["repetition_rule"]
        == "one material zero-cache oracle; no outcome-selected repetition",
    }
    observed: dict[str, dict[str, Any]] = {}
    file_identity(checks, observed, "model", config["model"])
    file_identity(
        checks,
        observed,
        "prompt",
        {
            "path": config["workload"]["prompt_path"],
            "bytes": config["workload"]["prompt_bytes"],
            "sha256": config["workload"]["prompt_sha256"],
        },
    )
    file_identity(checks, observed, "adapter_binary", config["runtime"]["binary"])
    for index, row in enumerate(config["runtime"]["backend_libraries"]):
        file_identity(checks, observed, f"backend_{index}", row)
    for index, row in enumerate(config["runtime"]["historical_sources"]):
        file_identity(checks, observed, f"historical_source_{index}", row)
    for prefix, section, names in (
        (
            "candidate",
            config["candidate_product_cell"],
            ("result", "logits", "certificate", "resource", "launch_control"),
        ),
        (
            "first8",
            config["historical_first8_oracle"],
            ("result", "logits", "certificate"),
        ),
    ):
        for name in names:
            file_identity(checks, observed, f"{prefix}_{name}", section[name])
    file_identity(
        checks,
        observed,
        "separate_canary_certificate",
        {
            "path": config["separate_evidence"]["source_path_canary"]["certificate_path"],
            "sha256": config["separate_evidence"]["source_path_canary"]["certificate_sha256"],
        },
    )
    file_identity(
        checks,
        observed,
        "separate_full128_result",
        {
            "path": config["separate_evidence"]["full128_population"]["result_path"],
            "sha256": config["separate_evidence"]["full128_population"]["result_sha256"],
        },
    )
    audit = config["applicable_fail_closed_audit"]
    file_identity(
        checks,
        observed,
        "fault_config",
        {"path": audit["config_path"], "sha256": audit["config_sha256"]},
    )
    file_identity(
        checks,
        observed,
        "fault_terminal_result",
        {
            "path": audit["terminal_result_path"],
            "sha256": audit["terminal_result_sha256"],
        },
    )
    file_identity(
        checks,
        observed,
        "validator",
        {
            "path": config["validation"]["source_path"],
            "sha256": config["validation"]["source_sha256"],
        },
    )
    checks["config_is_selected_path"] = config_path.resolve() == (
        ROOT / "configs" / "qwen3-next-full64-oracle-confirmation.json"
    ).resolve()
    return checks, observed


def adapter_command(config: dict[str, Any], output: Path, logits: Path) -> list[str]:
    workload = config["workload"]
    runtime = config["runtime"]
    return [
        runtime["binary"]["path"],
        "--arm",
        config["execution"]["arm"],
        "--model",
        config["model"]["path"],
        "--backend-dir",
        runtime["backend_dir"],
        "--prompt-file",
        workload["prompt_path"],
        "--prompt-token-limit",
        str(workload["prompt_tokens"]),
        "--output",
        str(output),
        "--logits",
        str(logits),
        "--n-predict",
        str(workload["output_tokens"]),
        "--cache-capacity",
        str(workload["oracle_cache"]["capacity_per_layer"]),
        "--n-gpu-layers",
        str(workload["n_gpu_layers"]),
        "--n-ubatch",
        str(workload["n_ubatch"]),
        "--no-mmap",
        "--bounded-source",
        "--source-direct",
        "--source-in-flight",
        str(workload["source"]["in_flight"]),
    ]


def research_job_shell(config: dict[str, Any]) -> str:
    runtime = config["runtime"]
    hardware = config["hardware"]
    execution = config["execution"]
    launcher = next(
        row["path"]
        for row in runtime["historical_sources"]
        if row["path"].endswith("run_qwen3_next_bounded_cell.sh")
    )
    fixed_prefix = [
        launcher,
        "--memory-max-bytes",
        str(hardware["host_cgroup_max_bytes"]),
        "--memory-swap-max-bytes",
        str(hardware["host_cgroup_swap_max_bytes"]),
        "--",
        "--gpu-index",
        str(hardware["gpu_index"]),
        "--interval-seconds",
        str(execution["observer_interval_seconds"]),
    ]
    adapter = adapter_command(config, Path("RESULT_PLACEHOLDER"), Path("LOGITS_PLACEHOLDER"))
    adapter[adapter.index("RESULT_PLACEHOLDER")] = '"$RESEARCH_JOB_DIR/result.json"'
    adapter[adapter.index("LOGITS_PLACEHOLDER")] = '"$RESEARCH_JOB_DIR/logits.bin"'
    dynamic = [
        "--samples",
        '"$RESEARCH_JOB_DIR/resource-samples.jsonl"',
        "--summary",
        '"$RESEARCH_JOB_DIR/resource-summary.json"',
        "--control",
        '"$RESEARCH_JOB_DIR/launch-control.json"',
        "--backing-path",
        config["model"]["path"],
        "--",
        *adapter,
    ]
    tokens = [shlex.quote(value) for value in fixed_prefix]
    tokens.extend(value if value.startswith('"$RESEARCH_JOB_DIR/') else shlex.quote(value) for value in dynamic)
    return "exec " + " ".join(tokens)


def research_job_command(config: dict[str, Any]) -> list[str]:
    return ["/bin/bash", "-lc", research_job_shell(config)]


def run_host_preflight(config: dict[str, Any]) -> tuple[dict[str, bool], dict[str, Any]]:
    with tempfile.TemporaryDirectory(prefix="msi-qwen-full64-preflight-") as directory:
        root = Path(directory)
        command = adapter_command(config, root / "preflight.json", root / "unused-logits.bin")
        command.insert(1, "--preflight")
        completed = subprocess.run(command, capture_output=True, text=True, timeout=120)
        report = load(root / "preflight.json") if (root / "preflight.json").is_file() else {}
        checks = {
            "adapter_preflight_exit_zero": completed.returncode == 0,
            "adapter_preflight_pass": report.get("status") == "PASS",
            "adapter_preflight_arm": report.get("arm") == config["execution"]["arm"],
            "adapter_preflight_geometry": report.get("geometry", {}).get(
                "managed_expert_inventory_bytes"
            )
            == config["model"]["managed_expert_inventory_bytes"],
            "adapter_preflight_workload": report.get("n_predict")
            == config["workload"]["output_tokens"]
            and report.get("n_ubatch") == config["workload"]["n_ubatch"]
            and report.get("cache_capacity") == 0,
            "adapter_preflight_source": report.get("bounded_source") is True
            and report.get("source_direct") is True
            and report.get("source_in_flight") == 8,
        }
        details = {
            "command": command,
            "return_code": completed.returncode,
            "stdout": completed.stdout,
            "stderr": completed.stderr,
            "report": report,
        }
        return checks, details


def resource_checks(
    prefix: str, resource: dict[str, Any], config: dict[str, Any]
) -> dict[str, bool]:
    hardware = config["hardware"]
    events = resource.get("cgroup_memory_event_deltas", {})
    return {
        f"{prefix}_observer_valid": resource.get("valid") is True
        and resource.get("process_return_code") == 0,
        f"{prefix}_gpu_identity": resource.get("gpu_uuid") == hardware["gpu_uuid"]
        and resource.get("hardware", {}).get("gpu_product") == hardware["gpu_product"],
        f"{prefix}_whole_board_margin": 0
        < int(resource.get("device_memory_used_peak_bytes", 2**64))
        <= int(hardware["gpu_board_budget_bytes"])
        - int(hardware["minimum_gpu_margin_bytes"]),
        f"{prefix}_complete_cgroup_margin": 0
        < int(resource.get("cgroup_memory_peak_bytes", 2**64))
        <= int(hardware["host_cgroup_max_bytes"])
        - int(hardware["minimum_host_margin_bytes"]),
        f"{prefix}_zero_swap": resource.get("cgroup_memory_swap_peak_bytes") == 0,
        f"{prefix}_zero_memory_events": bool(events)
        and all(int(value) == 0 for value in events.values()),
        f"{prefix}_zero_foreign_compute": resource.get("foreign_compute_pids")
        == hardware["required_foreign_compute_pids"],
    }


def lifecycle_checks(prefix: str, result: dict[str, Any]) -> dict[str, bool]:
    plan = result["plan"]
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
    return {
        f"{prefix}_component_manifest": plan.get("component_manifest_valid") is True,
        f"{prefix}_direct_eight_window_source": source.get("enabled") is True
        and source.get("direct_io") is True
        and source.get("registered_host_windows") is True
        and source.get("in_flight_limit") == 8
        and source.get("peak_in_flight") == 8,
        f"{prefix}_source_balance": source["submissions"] == source["completions"]
        and source["h2d_issued_bytes"] == source["h2d_completed_bytes"]
        and source["logical_bytes"] == plan["h2d_bytes"],
        f"{prefix}_source_quiescent": source["active_tickets"] == 0
        and source["free_windows"] == 8
        and all(
            source[key] == 0
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
        ),
        f"{prefix}_request_balance": telemetry["requests_begun"]
        == telemetry["requests_finished"],
        f"{prefix}_gpu_slot_consumer_balance": telemetry["bindings"]
        == telemetry["readiness_events"]
        == telemetry["consumer_acquires"]
        == telemetry["consumer_completions"]
        == telemetry["slot_releases"]
        == telemetry["window_recycles"],
        f"{prefix}_plan_source_balance": telemetry["scheduled_objects"]
        == telemetry["source_reads_issued"]
        == telemetry["source_reads_completed"],
        f"{prefix}_plan_byte_balance": telemetry["completed_application_read_bytes"]
        == telemetry["h2d_completed_bytes"]
        == plan["h2d_bytes"]
        and plan["d2d_scatter_bytes"]
        == plan["h2d_bytes"] + plan["cache"]["logical_hit_bytes"],
        f"{prefix}_quiescent_before_shutdown": before["bound_slots"] == 0
        and before["live_consumers"] == 0
        and before["ready_slots"] == 0
        and before["free_windows"] == 8,
        f"{prefix}_shutdown_closed": after["lifecycle"] == 3
        and after["bound_slots"] == 0
        and after["live_consumers"] == 0
        and after["ready_slots"] == 0
        and after["free_windows"] == 8
        and telemetry["resets"] == 1
        and telemetry["shutdowns"] == 1,
        f"{prefix}_zero_plan_rejections": all(telemetry[key] == 0 for key in rejection_keys),
    }


def route_coverage(result: dict[str, Any]) -> tuple[dict[str, bool], dict[str, Any]]:
    events = result["plan"]["route_events"]
    prefill = [row for row in events if int(row["step"]) == 0]
    chunks = sorted({int(row["chunk"]) for row in prefill})
    layers_by_chunk = {
        chunk: {int(row["layer"]) for row in prefill if int(row["chunk"]) == chunk}
        for chunk in chunks
    }
    unique_counts = [len(row["selected_unique"]) for row in prefill]
    coverage = {
        "route_event_count": len(events),
        "prefill_event_count": len(prefill),
        "prefill_chunks": chunks,
        "prefill_unique_experts_min": min(unique_counts, default=0),
        "prefill_unique_experts_max": max(unique_counts, default=0),
        "decode_steps": sorted({int(row["step"]) for row in events if int(row["step"]) > 0}),
    }
    checks = {
        "eight_complete_prefill_chunks": chunks == list(range(8))
        and all(layers_by_chunk[chunk] == set(range(48)) for chunk in chunks),
        "all_experts_demanded_each_prefill_event": bool(unique_counts)
        and min(unique_counts) == max(unique_counts) == 512,
        "complete_decode_route_horizon": coverage["decode_steps"] == list(range(1, 64)),
    }
    return checks, coverage


def row_digests(path: Path, row_bytes: int, rows: int) -> list[str]:
    if path.stat().st_size != row_bytes * rows:
        raise ValueError(
            f"logits size {path.stat().st_size} != {rows} * {row_bytes}: {path}"
        )
    values: list[str] = []
    with path.open("rb") as handle:
        for _ in range(rows):
            row = handle.read(row_bytes)
            if len(row) != row_bytes:
                raise ValueError(f"short logits row: {path}")
            values.append(hashlib.sha256(row).hexdigest())
        if handle.read(1):
            raise ValueError(f"trailing logits bytes: {path}")
    return values


def exactness_checks(
    config: dict[str, Any], oracle: dict[str, Any], oracle_logits: Path
) -> tuple[dict[str, bool], dict[str, Any]]:
    candidate = load(Path(config["candidate_product_cell"]["result"]["path"]))
    candidate_logits = Path(config["candidate_product_cell"]["logits"]["path"])
    first8 = load(Path(config["historical_first8_oracle"]["result"]["path"]))
    first8_logits = Path(config["historical_first8_oracle"]["logits"]["path"])
    rows = int(config["workload"]["output_tokens"])
    row_bytes = int(config["workload"]["vocabulary_size"]) * 4
    candidate_rows = row_digests(candidate_logits, row_bytes, rows)
    oracle_rows = row_digests(oracle_logits, row_bytes, rows)
    first8_rows = row_digests(first8_logits, row_bytes, 8)
    mismatches = [index for index, pair in enumerate(zip(candidate_rows, oracle_rows)) if pair[0] != pair[1]]
    route_count_first8 = len(first8["plan"]["route_events"])
    checks = {
        "all64_tokens_exact": oracle["workload"]["generated_tokens"]
        == candidate["workload"]["generated_tokens"]
        and len(oracle["workload"]["generated_tokens"]) == 64,
        "all64_logit_rows_byte_exact": not mismatches,
        "whole_logits_file_byte_exact": digest(oracle_logits) == digest(candidate_logits),
        "all_routes_exact": oracle["plan"]["route_events"]
        == candidate["plan"]["route_events"],
        "response_exact": oracle["workload"]["response"] == candidate["workload"]["response"],
        "consumer_exact": oracle["consumer"] == candidate["consumer"],
        "destination_exact": oracle["model_memory"] == candidate["model_memory"],
        "first8_tokens_preserve_historical_oracle": oracle["workload"]["generated_tokens"][:8]
        == first8["workload"]["generated_tokens"],
        "first8_logits_preserve_historical_oracle": oracle_rows[:8] == first8_rows,
        "first8_routes_preserve_historical_oracle": oracle["plan"]["route_events"][
            :route_count_first8
        ]
        == first8["plan"]["route_events"],
    }
    details = {
        "predicate": config["correctness_predicates"],
        "rows": rows,
        "vocabulary_size": config["workload"]["vocabulary_size"],
        "bytes_per_row": row_bytes,
        "candidate_logits_sha256": digest(candidate_logits),
        "oracle_logits_sha256": digest(oracle_logits),
        "candidate_row_sha256": candidate_rows,
        "oracle_row_sha256": oracle_rows,
        "mismatch_zero_based_steps": mismatches,
        "hybrid_rebased_state": {
            "standing": "UNAVAILABLE_NOT_CLAIMED",
            "reason": config["correctness_predicates"]["hybrid_rebased_state"],
        },
    }
    return checks, details


def workload_checks(config: dict[str, Any], result: dict[str, Any]) -> dict[str, bool]:
    workload = config["workload"]
    observed = result["workload"]
    plan = result["plan"]
    return {
        "oracle_arm": result.get("arm") == config["execution"]["arm"],
        "workload_identity": observed["prompt_file"] == workload["prompt_path"]
        and observed["prompt_formatter"] == workload["prompt_formatter"]
        and observed["prompt_tokens"] == workload["prompt_tokens"]
        and observed["prompt_token_limit"] == workload["prompt_tokens"]
        and observed["generated_token_count"] == workload["output_tokens"]
        and observed["n_predict_limit"] == workload["output_tokens"]
        and observed["vocabulary_size"] == workload["vocabulary_size"]
        and observed["n_ubatch"] == workload["n_ubatch"]
        and observed["n_gpu_layers"] == workload["n_gpu_layers"]
        and observed["use_mmap"] is False,
        "zero_cache_oracle": plan["cache"]["enabled"] is False
        and plan["cache"]["capacity_per_layer"] == 0
        and plan["cache"]["policy"] == "disabled"
        and plan["cache"]["logical_hit_bytes"] == 0
        and plan["cache"]["final_occupancy_per_layer"] == [0] * 48,
        "native_consumer": result["consumer"]
        == {
            "base_runtime_commit": config["runtime"]["base_commit"],
            "expert_consumer": "stock_ggml_mul_mat_id_q4_k_q6_k",
            "graph": "stock_qwen3_next_llama_decode",
            "reduction": "stock_qwen3_next_moe_reduction",
            "router": "stock_ffn_moe_topk",
            "runtime_adapter_commit": config["runtime"]["adapter_commit"],
            "scheduler_boundary": "pause_after_each_ffn_moe_topk",
        },
        "source_budget": plan["bounded_source"]["physical_read_bytes"]
        <= config["execution"]["maximum_physical_source_bytes"],
    }


def adjacent_fault_audit(config: dict[str, Any]) -> tuple[dict[str, bool], dict[str, Any]]:
    audit = config["applicable_fail_closed_audit"]
    fault_config = load(resolved(audit["config_path"]))
    terminal = load(resolved(audit["terminal_result_path"]))
    qwen = terminal["campaign_2"]["qwen_contract"]
    checks = {
        "named_fault_population_identity": fault_config["ordered_cells"]
        == audit["named_qwen_cells"],
        "named_fault_population_terminal": qwen["cells"] == qwen["valid_cells"] == 10
        and qwen["fatal_cells"] == 5
        and qwen["recoverable_cells"] == 4
        and qwen["controls"] == 1,
        "successor_binary_distinguished": fault_config["implementation"]["binary_sha256"]
        == audit["successor_binary_sha256"]
        and audit["successor_binary_sha256"] != config["runtime"]["binary"]["sha256"],
    }
    details = {
        "standing": "INHERITED_ADJACENT_SUCCESSOR_BINARY_NOT_REEXECUTED",
        "role": audit["role"],
        "named_cells": audit["named_qwen_cells"],
        "historical_product_binary_sha256": config["runtime"]["binary"]["sha256"],
        "fault_successor_binary_sha256": audit["successor_binary_sha256"],
        "fatal_cells": qwen["fatal_cells"],
        "recoverable_cells": qwen["recoverable_cells"],
        "controls": qwen["controls"],
    }
    return checks, details


def final_validation(
    config_path: Path, config: dict[str, Any], run_dir: Path
) -> dict[str, Any]:
    checks, identities = frozen_identity_checks(config_path, config)
    oracle_path = run_dir / "result.json"
    oracle_logits = run_dir / "logits.bin"
    resource_path = run_dir / "resource-summary.json"
    control_path = run_dir / "launch-control.json"
    job_path = run_dir / "job.json"
    for name, path in (
        ("oracle_result_exists", oracle_path),
        ("oracle_logits_exists", oracle_logits),
        ("oracle_resource_exists", resource_path),
        ("oracle_launch_control_exists", control_path),
        ("research_job_record_exists", job_path),
    ):
        checks[name] = path.is_file()
    if not all(checks[name] for name in (
        "oracle_result_exists",
        "oracle_logits_exists",
        "oracle_resource_exists",
        "oracle_launch_control_exists",
        "research_job_record_exists",
    )):
        return {
            "schema_version": 1,
            "artifact": "qwen3_next_full64_oracle_certificate",
            "valid": False,
            "checks": checks,
            "frozen_identities": identities,
        }
    oracle = load(oracle_path)
    resource = load(resource_path)
    control = load(control_path)
    job = load(job_path)
    checks["research_job_execution_command"] = job.get("command") == research_job_command(config)
    checks["research_job_cwd"] = job.get("cwd") == config["runtime"][
        "historical_project_worktree"
    ]
    checks["research_job_git_identity"] = job.get("git", {}).get("commit") == config[
        "runtime"
    ]["historical_project_commit"] and job.get("git", {}).get("dirty") is False
    checks["research_job_shape"] = job.get("project") == "memory-sovereign-inference"
    checks["research_job_risk"] = job.get("risk") == "bounded"
    checks["research_job_stage"] = job.get("stage_key") == config["execution"][
        "research_job_stage_key"
    ]
    checks["launch_control_valid"] = control.get("valid") is True and control.get(
        "return_code"
    ) == 0
    checks["adapter_execution_command"] = control.get("command") == adapter_command(
        config, oracle_path, oracle_logits
    )
    checks.update(workload_checks(config, oracle))
    checks.update(resource_checks("oracle", resource, config))
    candidate_resource = load(
        Path(config["candidate_product_cell"]["resource"]["path"])
    )
    checks.update(resource_checks("candidate", candidate_resource, config))
    checks.update(lifecycle_checks("oracle", oracle))
    candidate = load(Path(config["candidate_product_cell"]["result"]["path"]))
    checks.update(lifecycle_checks("candidate", candidate))
    coverage_checks, coverage = route_coverage(oracle)
    checks.update(coverage_checks)
    exact_checks, exactness = exactness_checks(config, oracle, oracle_logits)
    checks.update(exact_checks)
    fault_checks, fault_details = adjacent_fault_audit(config)
    checks.update(fault_checks)
    valid = all(checks.values())
    return {
        "schema_version": 1,
        "artifact": "qwen3_next_full64_oracle_certificate",
        "decision_id": config["decision_id"],
        "classification": config["classification"],
        "valid": valid,
        "outcome": "PASS_ALL64_EXACT" if valid else "FAIL_OR_INVALID_REQUIRES_CLASSIFICATION",
        "checks": checks,
        "exactness": exactness,
        "route_and_demand_coverage": coverage,
        "resources": {
            "contract": config["hardware"],
            "candidate": candidate_resource,
            "oracle": resource,
        },
        "lifecycle_contract": config["resource_and_lifecycle_predicates"],
        "applicable_fail_closed_audit": fault_details,
        "frozen_identities": identities,
        "raw_inputs": {
            "config": {"path": str(config_path.resolve()), "sha256": digest(config_path)},
            "oracle_result": {"path": str(oracle_path), "sha256": digest(oracle_path)},
            "oracle_logits": {"path": str(oracle_logits), "sha256": digest(oracle_logits)},
            "oracle_resource": {"path": str(resource_path), "sha256": digest(resource_path)},
            "oracle_launch_control": {"path": str(control_path), "sha256": digest(control_path)},
            "research_job": {"path": str(job_path), "sha256": digest(job_path)},
        },
        "claim_boundary": config["claim_boundary"],
        "stopping_decision": "STOP_AFTER_SINGLE_TERMINAL_CONFIRMATION",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()
    config = load(args.config)
    if args.preflight:
        checks, identities = frozen_identity_checks(args.config, config)
        adapter_checks, adapter = run_host_preflight(config)
        checks.update(adapter_checks)
        report = {
            "schema_version": 1,
            "artifact": "qwen3_next_full64_oracle_host_preflight",
            "decision_id": config["decision_id"],
            "valid": all(checks.values()),
            "checks": checks,
            "frozen_identities": identities,
            "adapter_preflight": adapter,
        }
    else:
        if args.run_dir is None:
            parser.error("--run-dir is required without --preflight")
        report = final_validation(args.config, config, args.run_dir.resolve())
    atomic_json(args.output, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
