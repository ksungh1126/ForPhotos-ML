#!/usr/bin/env bash
# Run the recommender CLI 10 times and store each JSON response.
set -euo pipefail

if [ "$#" -lt 2 ]; then
  echo "Usage: $0 <output_dir> <cli-args...>" >&2
  echo "Example: $0 runs photo1.jpg --hint '따뜻한 감성'" >&2
  exit 1
fi

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd "$script_dir/.." && pwd)

out_dir=$1
shift
mkdir -p "$out_dir"

for i in $(seq 1 10); do
  timestamp=$(date +%Y%m%d_%H%M%S)
  outfile=$(printf "%s/run_%02d_%s.json" "$out_dir" "$i" "$timestamp")
  (cd "$repo_root" && PYTHONPATH="$repo_root" python -m emotion.recommender.cli "$@" >"$outfile")
  echo "Saved $outfile"
  sleep 1
done
