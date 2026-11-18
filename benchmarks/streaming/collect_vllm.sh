#!/usr/bin/env bash
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
url=$1
server=$2
cpu_python=${CLIENT_PYTHON:-/usr/bin/python3}
copy=(collect_vllm.sh check_vllm.py client.py freeze_fixtures.py W1.json W2.json README.md)
for file in "${copy[@]}"; do
    cp "$here/$file" .
done
sha256sum "${copy[@]}" run_vllm.sh vllm_observer.py >inputs.sha256
printf '%s  %s\n' \
    1acaa09df4d2f8bfbd2562f501c837c1fa3b6889238f506ff81511acabe5633c W1.json \
    314fb913d558ac717a0a0a2ac86c7d310ea06dd0b4558ce160a141dd18564c43 W2.json \
    | sha256sum --check
post() {
    curl -fsS --max-time 120 -H 'Content-Type: application/json' -d "$2" "$url/$1"
}
post collective_rpc '{"method":"baseline_snapshot","timeout":90}' >effective-worker.json
"$cpu_python" -IB check_vllm.py audit effective-worker.json
curl -fsS --max-time 4 "$url/metrics" >metrics-before.txt
gpu=$CUDA_VISIBLE_DEVICES
fields=timestamp,uuid,memory.used,utilization.gpu,temperature.gpu,clocks.current.sm,power.draw
nvidia-smi -i "$gpu" --query-gpu="$fields" --format=csv -lms 500 >gpu-samples.csv &
monitor=$!
trap 'kill "$monitor" 2>/dev/null || true; wait "$monitor" 2>/dev/null || true' EXIT
command=("$cpu_python" -IB client.py --url "$url" --output measurement --freeze-baseline)
printf '%q ' "${command[@]}" >client-command.txt
printf '\n' >>client-command.txt
date -u +%FT%TZ >measurement-start-utc.txt
"${command[@]}"
date -u +%FT%TZ >measurement-done-utc.txt
kill "$monitor"
wait "$monitor" 2>/dev/null || true
trap - EXIT
nvidia-smi -i "$gpu" --query-compute-apps=pid,process_name,used_memory \
    --format=csv,noheader >compute-during.csv
curl -fsS --max-time 5 "$url/metrics" >metrics-after.txt

for name in W1 W2; do
    kill -0 "$server"
    profile="{\"profile_prefix\":\"$name\",\"max_iterations\":3,"
    profile+='"profiler_kwargs":{"with_stack":false}}'
    post start_profile "$profile" >"$name.profile-start.txt"
    "$cpu_python" -IB client.py --url "$url" --output "probe-$name" --probe "$name" \
        --reference measurement/summary.json
    post stop_profile '{}' >"$name.profile-stop.txt"
    "$cpu_python" -IB check_vllm.py trace profile "$name" >"$name.dispatch.json"
done
