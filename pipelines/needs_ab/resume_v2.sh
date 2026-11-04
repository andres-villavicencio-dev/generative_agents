#!/usr/bin/env bash
# Resume a v2-treatment sim in place from its last checkpoint (picks up new code). usage: resume_v2.sh <sim> [target_steps]
cd /home/andus/Projects/generative_agents; source pipelines/needs_ab/common_env.sh
SIM=${1:?sim}; TARGET=${2:-17280}
S=environment/frontend_server/storage/$SIM
systemctl --user stop ga-$SIM ga-${SIM}r 2>/dev/null; sleep 3
CK=$(python3 -c "import json;print(json.load(open('$S/reverie/meta.json'))['step'])")
python3 - <<P
import os
S="$S"; ck=$CK
for d in ("movement","environment"):
    for f in os.listdir(f"{S}/{d}"):
        if f.endswith(".json") and f[:-5].isdigit() and int(f[:-5])>ck: os.remove(f"{S}/{d}/{f}")
P
systemd-run --user --unit ga-${SIM}r --working-directory=$PWD $ENVS \
  --setenv=GA_TRACE_DIR=$PWD/ga_traces_$SIM --setenv=GA_DECOMP_CACHE_FILE=$PWD/ga_traces_$SIM/decomp_cache.json \
  --setenv=GA_NEEDS_V2=1 --setenv=GA_NEEDS_LAYA=1 --setenv=GA_NEEDS_CRISIS=1 --setenv=GA_NEEDS_LAYA_URL=http://127.0.0.1:8020 \
  --setenv=GA_FORK_SKIP_COPY=1 venv/bin/python run_live3d_resume.py $SIM $SIM $CK $TARGET
echo "resumed $SIM from checkpoint $CK"
