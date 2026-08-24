#!/usr/bin/env python3
"""Low-overhead NVML observer and optional launcher for the Qwen3-Next consumer."""

from __future__ import annotations

import argparse
import hashlib
import json
import signal
import subprocess
import time
from pathlib import Path
from typing import Any

import pynvml


STOP = False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu-index", type=int, required=True)
    parser.add_argument("--candidate-pid", type=int)
    parser.add_argument("--candidate-cgroup", type=Path)
    parser.add_argument("--interval-seconds", type=float, required=True)
    parser.add_argument("--samples", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--boundary", type=Path)
    parser.add_argument("--control", type=Path)
    parser.add_argument("--backing-path", type=Path)
    parser.add_argument("--phase-authority", type=Path)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    return parser.parse_args()


def request_stop(_signum: int, _frame: Any) -> None:
    global STOP
    STOP = True


def cgroup_pids(path: Path, fallback: int) -> set[int]:
    try:
        values = {int(value) for value in (path / "cgroup.procs").read_text().split()}
        return values or {fallback}
    except (OSError, ValueError):
        return {fallback}


def integer(path: Path) -> int | None:
    try:
        return int(path.read_text().strip())
    except (OSError, ValueError):
        return None


def integers(path: Path) -> dict[str, int] | None:
    try:
        return {
            key: int(value)
            for key, value in (
                line.split(maxsplit=1) for line in path.read_text().splitlines()
            )
        }
    except (OSError, ValueError):
        return None


def process_rows(handle: Any) -> list[dict[str, int | None]]:
    for name in (
        "nvmlDeviceGetComputeRunningProcesses_v3",
        "nvmlDeviceGetComputeRunningProcesses_v2",
        "nvmlDeviceGetComputeRunningProcesses",
    ):
        function = getattr(pynvml, name, None)
        if function is None:
            continue
        return [
            {"pid": int(row.pid), "used_gpu_memory_bytes": int(row.usedGpuMemory)}
            for row in function(handle)
        ]
    raise RuntimeError("NVML compute-process API is unavailable")


def sample(handle: Any, args: argparse.Namespace) -> dict[str, Any]:
    if args.candidate_cgroup is None:
        pids = {args.candidate_pid} if args.candidate_pid is not None else set()
    else:
        pids = cgroup_pids(args.candidate_cgroup, args.candidate_pid)
    rows = process_rows(handle)
    candidate_bytes = sum(
        int(row["used_gpu_memory_bytes"] or 0) for row in rows if row["pid"] in pids
    )
    memory = pynvml.nvmlDeviceGetMemoryInfo(handle)
    return {
        "monotonic_ns": time.monotonic_ns(),
        "wall_time_ns": time.time_ns(),
        "candidate_pids": sorted(pids),
        "candidate_process_memory_bytes": candidate_bytes,
        "device_memory_used_bytes": int(memory.used),
        "cgroup_memory_current_bytes": (
            integer(args.candidate_cgroup / "memory.current")
            if args.candidate_cgroup is not None
            else None
        ),
        "cgroup_memory_peak_bytes": (
            integer(args.candidate_cgroup / "memory.peak")
            if args.candidate_cgroup is not None
            else None
        ),
        "cgroup_memory_swap_current_bytes": (
            integer(args.candidate_cgroup / "memory.swap.current")
            if args.candidate_cgroup is not None
            else None
        ),
        "cgroup_memory_events": (
            integers(args.candidate_cgroup / "memory.events")
            if args.candidate_cgroup is not None
            else None
        ),
        "compute_processes": rows,
    }


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    args = parse_args()
    if args.interval_seconds <= 0:
        raise ValueError("interval must be positive")
    command = args.command
    if command and command[0] == "--":
        command = command[1:]
    launch_mode = bool(command)
    if launch_mode and args.candidate_pid is not None:
        raise ValueError("command launch and --candidate-pid are mutually exclusive")
    if not launch_mode and args.candidate_pid is None:
        raise ValueError("provide a command or --candidate-pid")
    if launch_mode and args.control is None:
        raise ValueError("--control is required for command launch")
    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)
    pynvml.nvmlInit()
    try:
        handle = pynvml.nvmlDeviceGetHandleByIndex(args.gpu_index)
        product = pynvml.nvmlDeviceGetName(handle)
        if isinstance(product, bytes):
            product = product.decode()
        uuid = pynvml.nvmlDeviceGetUUID(handle)
        if isinstance(uuid, bytes):
            uuid = uuid.decode()
        baseline_memory = pynvml.nvmlDeviceGetMemoryInfo(handle)
        baseline_device_bytes = int(baseline_memory.used)
        baseline_cgroup = sample(handle, args)
        process: subprocess.Popen[str] | None = None
        stdout_stream = None
        stderr_stream = None
        stdout_path = None
        stderr_path = None
        if launch_mode:
            stdout_path = args.control.with_name(args.control.stem + ".stdout.log")
            stderr_path = args.control.with_name(args.control.stem + ".stderr.log")
            stdout_path.parent.mkdir(parents=True, exist_ok=True)
            stdout_stream = stdout_path.open("w+", encoding="utf-8")
            stderr_stream = stderr_path.open("w+", encoding="utf-8")
            process = subprocess.Popen(
                command,
                stdout=stdout_stream,
                stderr=stderr_stream,
                text=True,
            )
            args.candidate_pid = process.pid
        samples: list[dict[str, Any]] = []
        args.samples.parent.mkdir(parents=True, exist_ok=True)
        with args.samples.open("w", encoding="utf-8") as stream:
            while True:
                row = sample(handle, args)
                samples.append(row)
                stream.write(json.dumps(row, sort_keys=True) + "\n")
                stream.flush()
                if STOP or (process is not None and process.poll() is not None):
                    break
                time.sleep(args.interval_seconds)
        return_code = None
        stdout = ""
        stderr = ""
        if process is not None:
            return_code = process.wait()
            assert stdout_stream is not None and stderr_stream is not None
            assert stdout_path is not None and stderr_path is not None
            stdout_stream.flush()
            stderr_stream.flush()
            stdout_stream.seek(0)
            stderr_stream.seek(0)
            stdout = stdout_stream.read()
            stderr = stderr_stream.read()
            stdout_stream.close()
            stderr_stream.close()
            return_code = process.returncode
            control = {
                "schema_version": 1,
                "artifact": "qwen3_next_plan_consumer_launch_control",
                "valid": return_code == 0,
                "command": command,
                "candidate_pid": process.pid,
                "return_code": return_code,
                "stdout_path": str(stdout_path),
                "stdout_sha256": digest(stdout_path),
                "stdout_tail": stdout[-32768:],
                "stderr_path": str(stderr_path),
                "stderr_sha256": digest(stderr_path),
                "stderr_tail": stderr[-32768:],
            }
            write_json(args.control, control)
        candidate_peak = max(
            int(row["candidate_process_memory_bytes"]) for row in samples
        )
        device_peak = max(int(row["device_memory_used_bytes"]) for row in samples)
        cgroup_values = [
            int(row["cgroup_memory_current_bytes"])
            for row in samples
            if row["cgroup_memory_current_bytes"] is not None
        ]
        cgroup_peak_values = [
            int(row["cgroup_memory_peak_bytes"])
            for row in samples
            if row["cgroup_memory_peak_bytes"] is not None
        ]
        cgroup_swap_values = [
            int(row["cgroup_memory_swap_current_bytes"])
            for row in samples
            if row["cgroup_memory_swap_current_bytes"] is not None
        ]
        baseline_events = baseline_cgroup.get("cgroup_memory_events") or {}
        final_events = samples[-1].get("cgroup_memory_events") or {}
        summary = {
            "schema_version": 1,
            "artifact": "qwen3_next_plan_consumer_observer",
            "valid": len(samples) >= 2 and candidate_peak > 0,
            "gpu_index": args.gpu_index,
            "gpu_uuid": uuid,
            "hardware": {"gpu_product": product},
            "sample_count": len(samples),
            "interval_seconds": args.interval_seconds,
            "launch_mode": launch_mode,
            "candidate_pid": args.candidate_pid,
            "process_return_code": return_code,
            "device_memory_used_baseline_bytes": baseline_device_bytes,
            "candidate_process_memory_peak_bytes": candidate_peak,
            "device_memory_used_peak_bytes": device_peak,
            "cgroup_memory_current_sample_peak_bytes": max(cgroup_values, default=None),
            "cgroup_memory_peak_bytes": max(cgroup_peak_values, default=None),
            "cgroup_memory_max_bytes": (
                integer(args.candidate_cgroup / "memory.max")
                if args.candidate_cgroup is not None
                else None
            ),
            "cgroup_memory_swap_peak_bytes": max(cgroup_swap_values, default=None),
            "cgroup_memory_swap_max_bytes": (
                integer(args.candidate_cgroup / "memory.swap.max")
                if args.candidate_cgroup is not None
                else None
            ),
            "cgroup_memory_event_deltas": {
                key: int(value) - int(baseline_events.get(key, 0))
                for key, value in final_events.items()
            },
            "foreign_compute_pids": sorted(
                {
                    int(row["pid"])
                    for sample_row in samples
                    for row in sample_row["compute_processes"]
                    if int(row["pid"])
                    not in {
                        int(pid)
                        for candidate_row in samples
                        for pid in candidate_row["candidate_pids"]
                    }
                }
            ),
        }
        summary["valid"] = summary["valid"] and (
            return_code == 0 if launch_mode else True
        )
        write_json(args.summary, summary)
        return 0 if summary["valid"] else 1
    finally:
        pynvml.nvmlShutdown()


if __name__ == "__main__":
    raise SystemExit(main())
