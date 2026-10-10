# Copyright (c) 2026 Andus VU — MIT license
"""Standing trace persistence for GA agent decisions (the Overmind loop,
miniaturized — 2026-10-05).

Every poignancy grading and typed reflex decision (talk/react) appends one
JSON record to ga_traces/*.jsonl under the repo root. When low-confidence
density crosses a threshold, a DISTILL_RECOMMENDED.json sentinel is written
— the operator's cue to harvest teacher labels (cloud-jev) and run a
re-distillation. No threads, no network, fail-open everywhere: capture
must never cost the sim a step.

Env gates:
  GA_TRACE           (default 1)  — 0 disables capture entirely
  GA_TRACE_DIR       (default <repo>/ga_traces)
  GA_TRACE_ROT_MB    (default 50) — rotate .jsonl to .jsonl.1 past this size
  GA_DISTILL_WINDOW  (default 2000) — rolling decision window for the trigger
  GA_DISTILL_LOWCONF (default 0.25) — low-conf share that trips the sentinel
"""
import json
import os
import threading
import time
from collections import deque

_REPO_ROOT = os.path.abspath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", ".."))

_lock = threading.Lock()
_handles = {}
_window = deque(maxlen=int(os.environ.get("GA_DISTILL_WINDOW", "2000")))
_lowconf_threshold = float(os.environ.get("GA_DISTILL_LOWCONF", "0.25"))
_sentinel_done = False


def _enabled():
    return os.environ.get("GA_TRACE", "1") == "1"


def _trace_dir():
    return os.path.abspath(os.environ.get(
        "GA_TRACE_DIR", os.path.join(_REPO_ROOT, "ga_traces")))


def _rot_bytes():
    try:
        return int(os.environ.get("GA_TRACE_ROT_MB", "50")) * 1024 * 1024
    except Exception:
        return 50 * 1024 * 1024


def _is_lowconf(kind, conf):
    if kind == "poignancy":
        return conf is not None and conf < 0.35
    # typed choices: a coin-flip decision is the shaky one
    return conf is not None and conf < 0.60


def record(kind, persona, payload):
    """Append one decision trace. kind: 'poignancy' | 'talk' | 'react'.
    payload: dict with the decision inputs/outputs (conf required).
    Fail-open: returns silently on any error."""
    global _sentinel_done
    if not _enabled():
        return
    try:
        rec = {"t": time.time(), "kind": kind,
               "persona": persona,
               "conf": payload.get("conf"),
               "lowconf": _is_lowconf(kind, payload.get("conf"))}
        # Full context so a distill set can be built directly from these files
        # (jev_harvest decisions.jsonl shape: ts/q/state/questions) —
        # GA_TRACE_FULL=0 to store conf-only lines (tiny files).
        if os.environ.get("GA_TRACE_FULL", "1") == "1":
            rec.update({k: v for k, v in payload.items()
                        if k in ("state", "questions", "options")})
        rec.update({k: v for k, v in payload.items() if k != "conf"})
        low = rec["lowconf"]
        flagged = False
        with _lock:
            d = _trace_dir()
            os.makedirs(d, exist_ok=True)
            path = os.path.join(d, kind + ".jsonl")
            h = _handles.get(path)
            try:
                if h is not None and h.closed:
                    h = None
                if h is None and os.path.exists(path) \
                        and os.path.getsize(path) > _rot_bytes():
                    # simple two-generation rotation
                    old = path + ".1"
                    if os.path.exists(old):
                        os.remove(old)
                    os.replace(path, old)
                    h = None
            except Exception:
                h = None
            if h is None:
                h = open(path, "a", encoding="utf-8")
                _handles[path] = h
            h.write(json.dumps(rec, ensure_ascii=False,
                               default=str) + "\n")
            h.flush()
            _window.append(low)
            # Sentinel: fire when the rolling window is FULL (density over a
            # full window is the meaningful statistic; the earlier hard-coded
            # 200 ignored GA_DISTILL_WINDOW — 2026-10-05 fix). Hysteresis:
            # re-arm when share falls back under half the threshold, so the
            # system can re-recommend after a re-distillation.
            if len(_window) >= _window.maxlen:
                share = sum(1 for x in _window if x) / len(_window)
                if not _sentinel_done and share >= _lowconf_threshold:
                    senti = {"t": time.time(),
                             "kind": kind,
                             "lowconf_share": round(share, 3),
                             "window": len(_window),
                             "message": ("low-confidence density crossed "
                                         "threshold — harvest teacher "
                                         "labels (cloud-jev) and "
                                         "re-distill laya")}
                    with open(os.path.join(d, "DISTILL_RECOMMENDED.json"),
                              "w") as f:
                        json.dump(senti, f, indent=2)
                    _sentinel_done = True
                    flagged = True
                elif _sentinel_done and share < _lowconf_threshold / 2.0:
                    _sentinel_done = False
        if flagged:
            print("[GA_TRACE] low-confidence density crossed "
                  f"{_lowconf_threshold:.2f} — DISTILL_RECOMMENDED.json "
                  "written (see ga_traces/)")
    except Exception:
        pass


def stats():
    """In-memory capture summary for probes/reports. Never raises."""
    try:
        with _lock:
            w = list(_window)
        return {"window": len(w),
                "lowconf_share": (round(sum(1 for x in w if x) / len(w), 3)
                                  if w else 0.0),
                "files": sorted(os.listdir(_trace_dir()))
                if os.path.isdir(_trace_dir()) else []}
    except Exception:
        return {"window": 0, "lowconf_share": 0.0, "files": []}