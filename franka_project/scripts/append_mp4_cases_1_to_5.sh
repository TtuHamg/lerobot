#!/usr/bin/env bash
set -e

case_prefix="${1:?Usage: $0 CASE_PREFIX [OUTPUT_NAME]}"
output_name="${2:-output.mp4}"

for case_id in {4..13}; do
    case_dir="${case_prefix}${case_id}"
    output_file="$case_dir/$output_name"
    python3 "$(dirname "$0")/append_mp4_videos.py" "$case_dir" "$output_file"
done
