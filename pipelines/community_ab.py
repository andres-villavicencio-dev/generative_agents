#!/usr/bin/env python3
"""Live A/B: community fund + jobs + lessons (GA_COMMUNITY/GA_PAYDAY/GA_JOBS/GA_LESSONS).

Both arms fork verify_1 @17280 (Wed Feb 15 05:52) and run the same steps
SEQUENTIALLY on the shared model lane.
  communityctl : all four flags off
  communityexp : all four flags on; the dividend fires Wed 21:00 (a real Monday
                 payday is unreachable in a one-day window)
Metrics: pipelines/adapt_metrics.py <sim> --from-step 17280
"""
import json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import chatfix_ab as c

c.ORIGIN, c.CKPT, c.ORIGIN_LAST_MV = "verify_1", 17280, 17279
c.STEPS = int(os.environ.get("AB_STEPS", "6000"))
c.END_STEP = c.CKPT + c.STEPS
c.MAX_WALL_S = int(float(os.environ.get("AB_MAX_HOURS", "6")) * 3600)
EXP = ("Environment=GA_COMMUNITY=1\nEnvironment=GA_PAYDAY=1\nEnvironment=GA_JOBS=1\n"
       "Environment=GA_LESSONS=1\nEnvironment=GA_DIVIDEND_WEEKDAY=2\n"
       "Environment=GA_DIVIDEND_HOUR=21\nEnvironment=GA_HOME_FRIDGES=1\n")

if __name__ == "__main__":
    res = []
    sfx = os.environ.get("AB_SUFFIX", "")      # e.g. "2" -> communityexp2
    arms = (("communityctl", "ga-communityctl", False),
            ("communityexp" + sfx, "ga-communityexp" + sfx, True))
    if os.environ.get("AB_ONLY_EXP") == "1":   # baseline = existing adaptctl/verify_1
        arms = arms[1:]
    for sim, unit, exp in arms:
        base = c.ENV_BASE
        c.ENV_BASE = base + (EXP if exp else "")
        res.append(c.run_arm(sim, unit, "0"))
        c.ENV_BASE = base
        json.dump(res, open(f"{c.REPO}/community_ab_runs{os.environ.get('AB_SUFFIX', '')}.json", "w"), indent=1)
    print("DONE", res, flush=True)
