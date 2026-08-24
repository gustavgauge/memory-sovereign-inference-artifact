from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "qwen3_next_full64_oracle_validation",
    ROOT / "tools" / "validate_qwen3_next_full64_oracle.py",
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_row_digests_preserve_per_step_boundaries(tmp_path: Path) -> None:
    logits = tmp_path / "logits.bin"
    logits.write_bytes(b"abcd" + b"efgh")

    observed = MODULE.row_digests(logits, 4, 2)

    assert observed == [
        "88d4266fd4e6338d13b845fcf289579d209c897823b9217da3e161936f031589",
        "e5e088a0b66163a0a26a5e053d2a4496dc16ab6e0e3dd1adf2d16aa84a078c9d",
    ]


def test_resource_contract_uses_whole_board_peak() -> None:
    config = {
        "hardware": {
            "gpu_uuid": "gpu",
            "gpu_product": "board",
            "gpu_board_budget_bytes": 1000,
            "minimum_gpu_margin_bytes": 100,
            "host_cgroup_max_bytes": 1000,
            "minimum_host_margin_bytes": 100,
            "required_foreign_compute_pids": [],
        }
    }
    resource = {
        "valid": True,
        "process_return_code": 0,
        "gpu_uuid": "gpu",
        "hardware": {"gpu_product": "board"},
        "candidate_process_memory_peak_bytes": 700,
        "device_memory_used_peak_bytes": 950,
        "cgroup_memory_peak_bytes": 800,
        "cgroup_memory_swap_peak_bytes": 0,
        "cgroup_memory_event_deltas": {"max": 0, "oom": 0},
        "foreign_compute_pids": [],
    }

    checks = MODULE.resource_checks("cell", resource, config)

    assert checks["cell_complete_cgroup_margin"] is True
    assert checks["cell_whole_board_margin"] is False
