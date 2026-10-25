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
c.MAX_WALL_S = 6 * 3600
EXP = ("Environment=GA_COMMUNITY=1\nEnvironment=GA_PAYDAY=1\nEnvironment=GA_JOBS=1\n"
       "Environment=GA_LESSONS=1\nEnvironment=GA_DIVIDEND_WEEKDAY=2\n"
       "Environment=GA_DIVIDEND_HOUR=21\n")

if __name__ == "__main__":
    res = []
    for sim, unit, exp in (("communityctl", "ga-communityctl", False),
                           ("communityexp", "ga-communityexp", True)):
        base = c.ENV_BASE
        c.ENV_BASE = base + (EXP if exp else "")
        res.append(c.run_arm(sim, unit, "0"))
        c.ENV_BASE = base
        json.dump(res, open(f"{c.REPO}/community_ab_runs.json", "w"), indent=1)
    print("DONE", res, flush=True)
