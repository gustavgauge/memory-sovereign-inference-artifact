#!/usr/bin/env bash
set -euo pipefail

msi_preflight=0
if [[ ${1:-} == "--preflight" ]]; then
  msi_preflight=1
  shift
fi

if [[ ${1:-} != "--memory-max-bytes" || -z ${2:-} ]]; then
  echo "missing --memory-max-bytes" >&2
  exit 2
fi
msi_memory_max=$2
shift 2
if [[ ${1:-} != "--memory-swap-max-bytes" || -z ${2:-} ]]; then
  echo "missing --memory-swap-max-bytes" >&2
  exit 2
fi
msi_memory_swap_max=$2
shift 2
if [[ ${1:-} != "--" ]]; then
  echo "missing command separator" >&2
  exit 2
fi
shift
if [[ $# -eq 0 ]]; then
  echo "missing bounded command" >&2
  exit 2
fi

msi_cgroup_rel=$(awk -F: '$1 == "0" { print $3 }' /proc/self/cgroup)
msi_cgroup=/sys/fs/cgroup${msi_cgroup_rel}
msi_unit=${msi_cgroup_rel##*/}
if [[ ! ${msi_unit} =~ ^research-job-[a-z0-9]+\.service$ ]]; then
  echo "not running in a research-job systemd unit: ${msi_unit}" >&2
  exit 2
fi

systemctl --user set-property --runtime "${msi_unit}" \
  "MemoryMax=${msi_memory_max}" "MemorySwapMax=${msi_memory_swap_max}"
[[ $(<"${msi_cgroup}/memory.max") == "${msi_memory_max}" ]]
[[ $(<"${msi_cgroup}/memory.swap.max") == "${msi_memory_swap_max}" ]]

if [[ ${msi_preflight} -eq 1 ]]; then
  exec "$@"
fi

exec /home/krooksn/.local/bin/uv run python \
  tools/observe_qwen3_next_plan_consumer.py \
  --candidate-cgroup "${msi_cgroup}" "$@"
