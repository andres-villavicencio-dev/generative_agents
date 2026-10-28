#!/usr/bin/env python3
"""Pull candidate residents for the 50-agent Smallville from the synthetic
nvidia/Nemotron-Personas-USA dataset (CC BY 4.0; fully synthetic, grounded in
US census statistics, not real people) through the HF datasets-server search
API. One query per wanted role; keeps rows in the wanted age band.
Output: pipelines/persona_pool/candidates.json"""
import json, urllib.parse, urllib.request, re, sys

API = "https://datasets-server.huggingface.co/search"
ROLES = [  # (slot, query, age_lo, age_hi)
 ("family doctor", "physician", 35, 60), ("paramedic / nurse", "paramedic", 28, 45),
 ("police officer", "police officer", 28, 48), ("college lecturer", "college lecturer literature", 40, 58),
 ("college librarian", "librarian", 45, 65), ("teen student", "high school student", 14, 17),
 ("baker", "baker", 30, 50), ("barista", "barista", 25, 38), ("night cook", "line cook", 24, 36),
 ("pub waitress / grad student", "waitress graduate student", 22, 28), ("market gardener", "farmer", 55, 70),
 ("delivery driver", "delivery driver", 35, 52), ("handyman / plumber", "plumber", 48, 62),
 ("mechanic", "auto mechanic", 40, 55), ("junior accountant", "accountant", 24, 30),
 ("newspaper editor", "newspaper editor", 50, 64), ("mayor", "city council member", 48, 60),
 ("retired contractor", "retired construction", 56, 66), ("chaplain / counsellor", "chaplain", 52, 66),
 ("fitness trainer", "fitness trainer", 28, 38), ("piano teacher / busker", "piano teacher musician", 35, 46),
 ("retired nurse (elder)", "retired nurse", 72, 85), ("retired elder", "retired", 68, 78),
 ("newcomer job seeker", "unemployed looking for work", 25, 35), ("hairdresser", "hairdresser", 38, 52),
]

def q(query, n=60):
    u = API + "?" + urllib.parse.urlencode({"dataset": "nvidia/Nemotron-Personas-USA",
        "config": "default", "split": "train", "query": query, "length": n})
    return json.load(urllib.request.urlopen(u, timeout=60)).get("rows", [])

out = {}
for slot, query, lo, hi in ROLES:
    try:
        rows = [r["row"] for r in q(query)]
    except Exception as e:
        print(slot, "ERR", e); continue
    ok = [r for r in rows if lo <= int(r["age"]) <= hi]
    out[slot] = ok[:6]
    print(f"{slot:28s} {len(rows):3d} rows, {len(ok):3d} in age band", flush=True)
json.dump(out, open("pipelines/persona_pool/candidates.json", "w"), indent=1)
