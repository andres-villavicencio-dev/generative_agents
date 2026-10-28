#!/usr/bin/env python3
"""Test run: 35 agents on the 200x130 map (verify_n35 @17280), free choice on and
laya v5 making the comply/pull/own decision. Usage: AB_STEPS=3000 n35_run.py [suffix]"""
import json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import chatfix_ab as c

c.ORIGIN, c.CKPT, c.ORIGIN_LAST_MV = "verify_n35", 17280, 17279
c.STEPS = int(os.environ.get("AB_STEPS", "3000"))
c.END_STEP = c.CKPT + c.STEPS
c.MAX_WALL_S = int(float(os.environ.get("AB_MAX_HOURS", "8")) * 3600)
c.ENV_BASE += ("Environment=GA_COMMUNITY=1\nEnvironment=GA_PAYDAY=1\nEnvironment=GA_JOBS=1\n"
               "Environment=GA_LESSONS=1\nEnvironment=GA_DIVIDEND_WEEKDAY=2\n"
               "Environment=GA_DIVIDEND_HOUR=21\nEnvironment=GA_HOME_FRIDGES=1\n"
               "Environment=GA_FREE_CHOICE=1\nEnvironment=GA_CHOICE_LAYA=1\n"
               "Environment=GA_LAYA_URL=http://127.0.0.1:8019\n"
               f"Environment=GA_LOC_LAYA={os.environ.get('LOC_MODE', 'off')}\n"
               f"Environment=GA_DECOMP_CACHE={os.environ.get('DECOMP', '0')}\n")
sim = "n35laya" + (sys.argv[1] if len(sys.argv) > 1 else "1")
res = c.run_arm(sim, "ga-" + sim, "0")
json.dump(res, open(f"{c.REPO}/{sim}_run.json", "w"), indent=1)
print("DONE", res, flush=True)
