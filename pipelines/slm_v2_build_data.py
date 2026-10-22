#!/usr/bin/env python3
"""Build the qwen2b-ga-chat-v2 SFT set from captured minimax-m3.1-flash calls.

Reads SNAPSHOT copies (in DATA_DIR) of ga_traces/calls/minimax-m3.1-flash_cloud*.jsonl.
Filters: empty content, done_reason=length, 70 flagged task-decomp example leaks
(ts+persona join), teacher outputs failing the family's format validator, exact
(system,prompt,content) duplicates, token length > MAX_TOK.
Split: family-stratified ~5% test (grouped by prompt so a prompt never straddles),
capped per family. Train: per-family cap (dominant families) + 2x oversample of
tiny families. Answers only (teacher `thinking` is NOT used as a target).

usage: slm_v2_build_data.py <data_dir>
"""
import collections
import glob
import hashlib
import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ga_chat_formats import family_of, validate  # noqa: E402

DATA = sys.argv[1] if len(sys.argv) > 1 else "/home/andus/Projects/sLM-train/data"
MAX_TOK = 4096
TEST_FRAC = 0.05
TEST_MIN, TEST_MAX = 4, 40
TRAIN_CAP = {"run_gpt_prompt_action_location": 1100, "generate_one_utterance": 1000}
SMALL_OVERSAMPLE_BELOW = 60
SEED = 20261022


def main():
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-2B")
    rows = []
    for f in sorted(glob.glob(os.path.join(DATA, "minimax-m3.1-flash_cloud*.jsonl"))):
        for line in open(f):
            try:
                rows.append(json.loads(line))
            except ValueError:
                pass
    leak = {(round(x["ts"], 3), x["persona"]) for x in map(json.loads, open(os.path.join(DATA, "task_decomp_example_leak.jsonl")))}
    drop = collections.Counter()
    fam_drop = collections.defaultdict(collections.Counter)
    seen = set()
    kept = []
    for r in rows:
        fam = family_of(r)
        content = (r.get("content") or "").strip()
        if not content:
            drop["empty_content"] += 1; fam_drop[fam]["empty"] += 1; continue
        if r.get("done_reason") == "length":
            drop["done_length"] += 1; fam_drop[fam]["length"] += 1; continue
        if (round(r["ts"], 3), r.get("persona")) in leak:
            drop["example_leak"] += 1; fam_drop[fam]["leak"] += 1; continue
        ok, why = validate(fam, r.get("prompt") or "", content)
        if not ok:
            drop["invalid_format"] += 1; fam_drop[fam]["invalid:" + why] += 1; continue
        key = hashlib.sha1(json.dumps([r.get("system"), r.get("prompt"), content]).encode()).hexdigest()
        if key in seen:
            drop["duplicate"] += 1; fam_drop[fam]["dup"] += 1; continue
        seen.add(key)
        msgs = []
        if r.get("system"):
            msgs.append({"role": "system", "content": r["system"]})
        msgs.append({"role": "user", "content": r["prompt"]})
        msgs.append({"role": "assistant", "content": content})
        n_tok = len(tok(tok.apply_chat_template(msgs, tokenize=False), add_special_tokens=False)["input_ids"])
        if n_tok > MAX_TOK:
            drop["too_long"] += 1; fam_drop[fam]["too_long"] += 1; continue
        kept.append({"family": fam, "ts": r["ts"], "persona": r.get("persona"), "think": r.get("think"),
                     "temperature": (r.get("options") or {}).get("temperature", 0.6),
                     "n_tok": n_tok, "messages": msgs,
                     "pkey": hashlib.sha1((r.get("system", "") + r["prompt"]).encode()).hexdigest()})
    rng = random.Random(SEED)
    by_fam = collections.defaultdict(list)
    for k in kept:
        by_fam[k["family"]].append(k)
    train, test = [], []
    stats = {}
    for fam, its in sorted(by_fam.items()):
        pkeys = sorted({i["pkey"] for i in its})
        rng.shuffle(pkeys)
        n_test = 0 if len(its) < 15 else min(TEST_MAX, max(TEST_MIN, round(TEST_FRAC * len(its))))
        tk = set()
        cnt = 0
        for pk in pkeys:
            if cnt >= n_test:
                break
            tk.add(pk)
            cnt += sum(1 for i in its if i["pkey"] == pk)
        te = [i for i in its if i["pkey"] in tk]
        tr = [i for i in its if i["pkey"] not in tk]
        rng.shuffle(tr)
        cap = TRAIN_CAP.get(fam)
        tr_used = tr[:cap] if cap else tr
        mult = 2 if len(tr_used) < SMALL_OVERSAMPLE_BELOW else 1
        train += tr_used * mult
        test += te
        stats[fam] = {"kept": len(its), "test": len(te), "train_unique": len(tr_used),
                      "train_rows": len(tr_used) * mult, "dropped": dict(fam_drop[fam])}
    rng.shuffle(train)
    with open(os.path.join(DATA, "sft_train.jsonl"), "w") as f:
        for i in train:
            f.write(json.dumps(i) + "\n")
    with open(os.path.join(DATA, "sft_test.jsonl"), "w") as f:
        for i in test:
            f.write(json.dumps(i) + "\n")
    toks = sorted(i["n_tok"] for i in kept)
    q = lambda p: toks[int(p * (len(toks) - 1))]  # noqa: E731
    rep = {"raw_rows": len(rows), "kept": len(kept), "dropped": dict(drop), "train_rows": len(train),
           "test_rows": len(test), "tok_p50": q(0.5), "tok_p90": q(0.9), "tok_p99": q(0.99), "tok_max": toks[-1],
           "families": stats}
    json.dump(rep, open(os.path.join(DATA, "build_report.json"), "w"), indent=1)
    print(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
