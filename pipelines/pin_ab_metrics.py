#!/usr/bin/env python3
"""Metrics for pipelines/pin_ab.py (control=pinctl pin off, treatment=pinexp pin on)."""
import glob, json, os, re, sys
REPO = "/home/andus/Projects/generative_agents"
ST = f"{REPO}/environment/frontend_server/storage"
ARMS = {"control (no pin)": "pinctl", "treatment (pin)": "pinexp"}
CKPT = 10800

def moves(sim):
    for p in sorted(glob.glob(f"{ST}/{sim}/movement/*.json"), key=lambda x: int(os.path.basename(x)[:-5])):
        n = int(os.path.basename(p)[:-5])
        if n > CKPT:
            yield n, json.load(open(p))

def run(sim):
    at_cafe, party_agents, first_party, sched_bad = {}, set(), None, 0
    last = None
    for n, d in moves(sim):
        t = d["meta"]["curr_time"]
        m = re.search(r"(\d+):(\d+):\d+", t)
        h = int(m.group(1)) + (12 if "PM" in t.upper() and int(m.group(1)) < 12 else 0) if "AM" in t.upper() or "PM" in t.upper() else int(m.group(1))
        if "February 14" not in t: continue
        last = t
        if 17 <= h < 19:
            for name, v in d["persona"].items():
                if name != "Isabella Rodriguez" and "Hobbs Cafe" in v.get("description", ""):
                    party_agents.add(name)
    log = f"/tmp/{sim}_backend.log"
    pins = [l for l in open(log, errors="ignore") if "[PinCommit]" in l] if os.path.exists(log) else []
    pins_ok = [l for l in pins if "skipped" not in l]
    totals_bad = [l for l in pins_ok if not l.strip().endswith("total=1440m")]
    tb = sum("Traceback" in l for l in open(log, errors="ignore")) if os.path.exists(log) else 0
    return {"last_time": last, "agents_at_Hobbs_17-19": sorted(party_agents),
            "n_at_Hobbs": len(party_agents), "pins": len(pins_ok), "pin_skipped": len(pins) - len(pins_ok),
            "pins_total_not_1440": len(totals_bad), "tracebacks": tb,
            "sample_pins": [l.strip()[:200] for l in pins_ok[:6]]}

if __name__ == "__main__":
    res = {k: run(v) for k, v in ARMS.items()}
    for k, v in res.items():
        print(f"\n== {k}\n" + json.dumps(v, indent=1))
    json.dump(res, open(f"{REPO}/pin_ab_results.json", "w"), indent=1)
