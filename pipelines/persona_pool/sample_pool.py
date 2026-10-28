#!/usr/bin/env python3
"""Sample N rows of nvidia/Nemotron-Personas-USA (synthetic, CC BY 4.0) via the
HF datasets-server /rows API at random offsets -> persona_pool/pool.jsonl.
Usage: sample_pool.py [requests]  (100 rows per request, 1 s apart)"""
import json, random, sys, time, urllib.request
N = int(sys.argv[1]) if len(sys.argv) > 1 else 80
rnd = random.Random(7)
out = open("pipelines/persona_pool/pool.jsonl", "a")
got = 0
for i in range(N):
    off = rnd.randrange(0, 999_900)
    u = ("https://datasets-server.huggingface.co/rows?dataset=nvidia/Nemotron-Personas-USA"
         f"&config=default&split=train&offset={off}&length=100")
    for attempt in range(4):
        try:
            rows = json.load(urllib.request.urlopen(u, timeout=60))["rows"]
            for r in rows:
                out.write(json.dumps(r["row"]) + "\n")
            got += len(rows); break
        except Exception as e:
            time.sleep(3 * (attempt + 1))
    out.flush(); time.sleep(1)
print("rows", got)
