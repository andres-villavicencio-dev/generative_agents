#!/usr/bin/env bash
# Treatment: needs engine v2 + crises + Laya need-intent (served separately on :8020), clean base, 2 sim days.
cd /home/andus/Projects/generative_agents; source pipelines/needs_ab/common_env.sh
SIM=${1:-n57}; STEPS=${2:-17280}
mkdir -p ga_traces_$SIM
systemd-run --user --unit ga-$SIM --working-directory=$PWD $ENVS \
  --setenv=GA_TRACE_DIR=$PWD/ga_traces_$SIM --setenv=GA_DECOMP_CACHE_FILE=$PWD/ga_traces_$SIM/decomp_cache.json \
  --setenv=GA_NEEDS_V2=1 --setenv=GA_NEEDS_LAYA=1 --setenv=GA_NEEDS_CRISIS=1 --setenv=GA_NEEDS_LAYA_URL=http://127.0.0.1:8020 \
  venv/bin/python run_live3d.py $SIM $STEPS base_the_ville_n55
