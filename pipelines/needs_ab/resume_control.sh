#!/usr/bin/env bash
# Control: resume n56 (needs engine v1, v7e) from its last checkpoint WITH the new code, so it also logs the hourly needs series.
cd /home/andus/Projects/generative_agents; source pipelines/needs_ab/common_env.sh
S=environment/frontend_server/storage/n56
CK=$(python3 -c "import json;print(json.load(open('$S/reverie/meta.json'))['step'])")
python3 - <<P
import os
S="$S"; ck=$CK
for d in ("movement","environment"):
    for f in os.listdir(f"{S}/{d}"):
        if f.endswith(".json") and f[:-5].isdigit() and int(f[:-5])>ck: os.remove(f"{S}/{d}/{f}")
P
systemd-run --user --unit ga-n56r --working-directory=$PWD $ENVS \
  --setenv=GA_TRACE_DIR=$PWD/ga_traces_n56 --setenv=GA_DECOMP_CACHE_FILE=$PWD/ga_traces_n56/decomp_cache.json \
  --setenv=GA_FORK_SKIP_COPY=1 venv/bin/python run_live3d_resume.py n56 n56 $CK 60480
