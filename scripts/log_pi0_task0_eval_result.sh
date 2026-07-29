#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   scripts/log_pi0_task0_eval_result.sh <run_name> <checkpoint> <successes> <total> [notes]

if [[ $# -lt 4 || $# -gt 5 ]]; then
  echo "Usage: $0 <run_name> <checkpoint> <successes> <total> [notes]" >&2
  exit 2
fi

run_name="$1"
checkpoint="$2"
successes="$3"
total="$4"
notes="${5:-}"

if ! [[ "${successes}" =~ ^[0-9]+$ && "${total}" =~ ^[1-9][0-9]*$ ]]; then
  echo "successes and total must be non-negative/positive integers" >&2
  exit 2
fi

if (( successes > total )); then
  echo "successes cannot be greater than total" >&2
  exit 2
fi

results_dir="/m2v_intern/tujiahang/Projects/lerobot/output_lerobot/pi0/task0_lora_ablation"
results_file="${results_dir}/real_robot_results.csv"
mkdir -p "${results_dir}"

if [[ ! -f "${results_file}" ]]; then
  echo "timestamp_utc,run_name,checkpoint,successes,total,success_rate,notes" > "${results_file}"
fi

success_rate="$(awk -v s="${successes}" -v t="${total}" 'BEGIN { printf "%.4f", s / t }')"
timestamp_utc="$(date -u +%Y-%m-%dT%H:%M:%SZ)"

# Replace CSV-breaking characters in free-form fields.
run_name="${run_name//,/;}"
checkpoint="${checkpoint//,/;}"
notes="${notes//,/;}"
notes="${notes//$'\n'/ }"

printf '%s,%s,%s,%s,%s,%s,%s\n' \
  "${timestamp_utc}" "${run_name}" "${checkpoint}" \
  "${successes}" "${total}" "${success_rate}" "${notes}" >> "${results_file}"

echo "Saved: ${results_file}"
