#!/bin/bash
# GA sim chain: self-healing resume loop for the n-series sims.
# Finds the newest live3d_nN sim, resumes from its checkpoint into nN+1,
# and keeps doing that until the target step is reached. Designed to run
# under a systemd user unit (no hermes-worker cgroup memory caps).
set -u
BASE="$HOME/Projects/generative_agents"
STORAGE="$BASE/environment/frontend_server/storage"
VENV="$BASE/venv/bin/python"
TARGET="${GA_TARGET:-25920}"
FRONTEND_SIM_FILE="$BASE/environment/frontend_server/temp_storage/curr_sim_code.json"

while true; do
  latest=$(ls -d "$STORAGE"/live3d_n1[0-9] 2>/dev/null | sort -V | tail -1)
  [ -z "$latest" ] && { echo "FATAL: no live3d_n1x sim found"; exit 1; }
  sim=$(basename "$latest")

  step=$(python3 -c "import json;print(json.load(open('$latest/reverie/meta.json'))['step'])" 2>/dev/null)
  [ -z "$step" ] && { echo "FATAL: no meta.json in $sim"; exit 1; }

  lastmov=$(ls "$latest/movement" 2>/dev/null | sed 's/\.json//' | sort -n | tail -1)
  lastmov=${lastmov:-0}
  echo "chain: latest=$sim checkpoint=$step last_movement=$lastmov target=$TARGET"

  if [ "$lastmov" -ge $((TARGET - 2)) ]; then
    echo "RUN COMPLETE at $sim (step $lastmov)"
    exit 0
  fi

  n=$(echo "$sim" | grep -o '[0-9]*$')
  newsim="live3d_n$((10#$n + 1))"
  if [ -d "$STORAGE/$newsim" ]; then
    echo "chain: wiping partial fork $newsim"
    rm -rf "$STORAGE/$newsim"
  fi

  # point the frontend at the new sim
  printf '{\n  "sim_code": "%s"\n}' "$newsim" > "$FRONTEND_SIM_FILE"
  echo "chain: resuming $sim@$step -> $newsim (+$((TARGET - step)) steps)"

  "$VENV" "$BASE/run_live3d_resume.py" "$sim" "$newsim" "$step" "$TARGET"
  code=$?
  echo "chain: supervisor exited code=$code"
  [ "$code" -eq 0 ] && exit 0
  echo "chain: resuming from newest checkpoint in 15s..."
  sleep 15
done