#!/usr/bin/env bash
# Shared sim environment for the needs A/B: n55's settings minus per-run overrides.
ENVS=$(grep '^Environment=' ~/.config/systemd/user/ga-n55.service | sed 's/^Environment=//' \
  | grep -v -E '^(GA_FORK_SKIP_COPY|GA_TRACE_DIR)=' | sed 's/^/--setenv=/' | tr '\n' ' ')
