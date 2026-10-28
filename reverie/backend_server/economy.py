"""Economy v2 (Oct 2026): the 25-agent town economy.

Four commercial venues, each owned by a resident who lives off the
business (and nothing else):
  - Hobbs Cafe                     -> Isabella Rodriguez
  - The Rose and Crown Pub          -> Arthur Burton
  - Harvey Oak Supply Store        -> Carmen Ortiz
  - The Willows Market and Pharmacy -> John Lin

Income flows (all logged, all persisted):
  - SALES: purchases credit the VENUE's till (weekly owner draw of the
    surplus above the float — owners do NOT get paid per sale anymore).
  - RESTOCK: the 6 AM wholesale delivery is Tom Moreno's paid job; his
    wage = 10% of the previous day's combined sales, paid proportionally
    from each venue's till he restocked.
  - STIPENDS: Oak Hill students get a weekly stipend (Mon 6 AM).
  - EMPLOYMENT: owners may hire up to 2 employees. Wages are market-set:
    10% of the venue's daily sales intake, paid at 10 PM from the till
    (owner tops up from wallet if the till is under float).

Hiring market (LLM-driven, board-mediated):
  1. Daily (7 AM) each owner's LLM decides whether to post a
     "help wanted" opening (prompted with venue sales + staffing) —
     daily so broke agents can re-enter the economy within a day.
  2. A posting goes on the community bulletin board (Johnson Park tile
     event) — any money-stressed agent (financial_stress >= 0.4 or
     wallet < $30) is an eligible applicant.
  3. When an applicant is co-located with the owner, the apply-encounter
     fires: the OWNER's LLM decides hire / decline (persona-prompted).
  4. On hire: bulletin announces it, the employee's scratch.job is set,
     and daily wage payments begin.

Fail-open everywhere: any economy error logs and degrades — never
crashes the sim.
"""
import json
import os
import re
import threading
from collections import defaultdict

# ---------------------------------------------------------------------------
# Static configuration
# ---------------------------------------------------------------------------

VENUES = {
    "cafe": {
        "name": "Hobbs Cafe",
        "owner": "Isabella Rodriguez",
        "address_key": "hobbs cafe",
        "float": 60.0,   # wages and owner draws come only from the till above this; 200 left the cafe unable to pay anyone
        "max_employees": 2,
        "restock": {"coffee_beans": 30, "sandwiches": 20, "pastries": 20},
        "wholesale": {"coffee_beans": 1.0, "sandwiches": 2.25, "pastries": 1.50},
    },
    "pub": {
        "name": "The Rose and Crown Pub",
        "owner": "Arthur Burton",
        "address_key": "rose and crown pub",
        "float": 200.0,
        "max_employees": 2,
        "restock": {"beer": 40, "wine": 15, "juice": 20},
        "wholesale": {"beer": 1.75, "wine": 2.25, "juice": 1.00},
    },
    "grocery": {
        "name": "Harvey Oak Supply Store",
        "owner": "Carmen Ortiz",
        "address_key": "harvey oak supply store",
        "float": 200.0,
        "max_employees": 2,
        "restock": {"eggs": 100, "bread": 50, "milk": 40, "coffee_beans": 30},
        "wholesale": {"eggs": 0.25, "bread": 0.75, "milk": 0.60,
                      "coffee_beans": 1.00},
    },
    "market": {
        "name": "The Willows Market and Pharmacy",
        "owner": "John Lin",
        "address_key": "willows market and pharmacy",
        "float": 200.0,
        "max_employees": 2,
        "restock": {"eggs": 60, "bread": 30, "milk": 25, "medicine": 15},
        "wholesale": {"eggs": 0.25, "bread": 0.75, "milk": 0.60,
                      "medicine": 4.00},
    },
}

# Retail prices for items the economy charges for (extends
# resource_manager.ITEM_PRICES; pub + pharmacy additions).
ECONOMY_ITEM_PRICES = {
    "beer": 3.50, "wine": 4.50, "juice": 2.50, "medicine": 8.00,
}

# Luxury goods (Oct 22 2026): sold only at The Willows Market's luxury
# display, priced so that only agents who have saved considerably can buy
# one (typical wallets are ~$40-$180). Buying one satisfies the "status"
# need — a slow craving that never triggers an emergency replan but keeps
# surfacing in the agent's thoughts and daily planning.
LUXURY_GOODS = {
    "silk scarf": 180.00,
    "gold watch": 320.00,
    "designer handbag": 420.00,
    "diamond necklace": 650.00,
}
LUXURY_WHOLESALE_SHARE = 0.55   # supplier cost as a share of retail
LUXURY_RESERVE = 60.0           # never spend the last $60 on a luxury
LUXURY_STOCK = 2                # units of each on display after delivery
ECONOMY_ITEM_PRICES.update(LUXURY_GOODS)

RESTOCK_WORKER = "Tom Moreno"
RESTOCK_WAGE_SHARE = 0.10     # of previous day's combined sales
EMPLOYEE_WAGE_SHARE = 0.10    # of venue's daily sales, per employee
EMPLOYE_WAGE_FLOOR = 5.0      # a shift always pays at least this
SHIFT_START, SHIFT_END = 10, 14   # GA_JOBS: paid shift window (sim hours)
MIN_SHIFT_PRESENCE_MIN = 45       # GA_JOBS: minutes on site to earn the wage (two 30-min goals minus the walk in; 60 was unreachable: Yuriko 50, Hailey 59.x)
CIVIC_PRESENCE_MIN = 30           # GA_COMMUNITY: civic task wage (agents reached sites ~60-100 min at most)


def jobs_enabled():
    import os
    return os.environ.get("GA_JOBS", "1") == "1"

# Community fund (GA_COMMUNITY): tax accumulates, community work and the
# Monday dividend are paid out of it — nothing is created from nothing.
FUND_SALES_TAX = 0.05         # 5% of every venue sale, taken at the register (buyer pays the same price)
FUND_DRAW_TAX = 0.20          # of every owner draw
FUND_SALES_TAX_MAX = 0.30     # ceiling of the dynamic sales tax
CIVIC_EXPECTED_HOURS = 1.0    # paid hours assumed per signed-up civic worker (two 30-min goals)
FUND_SEED = 150.0             # starting balance of a new fund (a town treasury)
FUND_DIVIDEND_SHARE = 0.5     # of the fund shared out each Monday
FUND_DIVIDEND_CAP = 40.0      # per agent
CIVIC_START, CIVIC_END = 10, 16
CIVIC_ELIGIBLE_WALLET = 60.0  # below this (and no job) an agent takes civic work
CIVIC_TASKS = {
    "park": {"name": "park upkeep at Johnson Park",
             "keys": ("johnson park",), "rate": 6.0, "slots": 99},
    "library": {"name": "helping out at the Oak Hill College library",
                "keys": ("oak hill college", "library"), "rate": 6.0, "slots": 99},
    "tutor": {"name": "tutoring at the Oak Hill College classroom",
              "keys": ("oak hill college", "classroom"), "rate": 7.0, "slots": 99},
    # town services: assigned to named residents (PUBLIC_JOBS), never picked at random
    "clinic": {"name": "working at the Willow Clinic", "keys": ("willow clinic",),
               "rate": 9.0, "slots": 99, "public": True},
    "police": {"name": "working at the Police Station", "keys": ("police station",),
               "rate": 8.0, "slots": 99, "public": True},
    "gazette": {"name": "working at the Smallville Gazette", "keys": ("smallville gazette",),
                "rate": 7.0, "slots": 99, "public": True},
    "townhall": {"name": "working at Town Hall", "keys": ("town hall",),
                 "rate": 8.0, "slots": 99, "public": True},
    "records": {"name": "staffing the Oak Hill College library desk",
                "keys": ("oak hill college", "library"), "rate": 7.0, "slots": 99, "public": True},
}
PUBLIC_JOBS = {"Chad Kibler": "clinic", "Hua Lai": "police", "Lesbia Martinez": "gazette",
               "Carl Demissie": "townhall", "Andrea Lovetro": "records"}


def community_enabled():
    import os
    return os.environ.get("GA_COMMUNITY", "1") == "1"


WAGE_HOUR = 22                # 10 PM: employee payday + daily owner draw
RESTOCK_HOUR = 6              # 6 AM: delivery wage + daily-intake rollover
HIRING_HOUR = 7               # 7 AM: owners consider posting openings
STIPEND_DAY_HOUR = (0, 6)     # Mondays 6 AM
STIPEND_AMOUNT = 60.0
STIPEND_STUDENTS = [
    "Klaus Mueller", "Eddy Lin", "Maria Lopez",
    "Wolfgang Schulz", "Ayesha Khan",
]
STRESS_APPLY_THRESHOLD = 0.4   # financial_stress to be an applicant
BROKE_APPLY_WALLET = 30.0      # ...or wallet below this

_LEDGER_LOCK = threading.Lock()


class EconomyManager:
    """Per-sim economy state machine. All flows logged to
    <sim>/resources/economy_v2.json; fail-open on every path."""

    def __init__(self, sim_folder):
        self.sim_folder = sim_folder
        self.file = os.path.join(sim_folder, "resources", "economy_v2.json")
        self._llm = None           # injected: callable(prompt) -> str
        self._personas = None       # injected: {name: persona}
        self._bulletin = None       # injected: callable(announcement_str)
        self.state = {
            "tills": {k: 200.0 for k in VENUES},
            "daily_intake": {k: 0.0 for k in VENUES},   # today's sales
            "yesterday_intake": {k: 0.0 for k in VENUES},
            "employees": {k: [] for k in VENUES},       # lists of names
            "postings": {},            # venue -> {"posted_day": N, "reason": str}
            "hire_log": [],            # [{day, venue, owner, applicant, hired}]
            "pay_log": [],             # [{day, kind, who, amount, venue}]
            "last_stipend_day": -1,
            "last_draw_day": -1,
            "last_hirepost_day": -1,
            "last_restock_wage_day": -1,
            "last_wage_day": -1,
            "yesterday_sales": 0.0,
            "fund": {"balance": 0.0, "in": 0.0, "out": 0.0, "seeded": False},
            "civic": {},               # persona -> task id taken today
            "civic_day": -1,
            "civic_paid_day": -1,
            "last_dividend_day": -1,
        }
        self._load()
        self._apply_init = False

    # -- persistence --------------------------------------------------------
    def _load(self):
        try:
            with open(self.file) as f:
                data = json.load(f)
            for k in self.state:
                if k in data:
                    self.state[k] = data[k]
        except (FileNotFoundError, json.JSONDecodeError):
            pass
        except Exception as e:
            print(f"[Economy] load failed ({e.__class__.__name__}) — defaults")

    def save(self):
        try:
            os.makedirs(os.path.dirname(self.file), exist_ok=True)
            with _LEDGER_LOCK, open(self.file, "w") as f:
                json.dump(self.state, f, indent=2)
        except Exception as e:
            print(f"[Economy] save failed: {e}")

    # -- injections ---------------------------------------------------------
    def attach(self, llm=None, personas=None, bulletin=None):
        self._llm = llm
        self._personas = personas
        self._bulletin = bulletin

    # -- venue routing ------------------------------------------------------
    def venue_of_address(self, address):
        """Map an act_address / world_state key to a venue id, or None."""
        if not address:
            return None
        a = address.lower()
        for vid, cfg in VENUES.items():
            if cfg["address_key"] in a:
                return vid
        return None

    def item_price(self, item):
        """Retail price (economy additions first, then resource_manager)."""
        if item in ECONOMY_ITEM_PRICES:
            return ECONOMY_ITEM_PRICES[item]
        try:
            from resource_manager import ITEM_PRICES
            return ITEM_PRICES.get(item, 1.0)
        except Exception:
            return 1.0

    # -- core flows ---------------------------------------------------------
    def credit_sale(self, venue_id, item, amount, buyer_name, price):
        """A sale at a venue: money buyer->till, tracked as daily intake."""
        try:
            vid = venue_id if venue_id in VENUES else None
            if vid is None:
                return
            tax = self._tax_to_fund(price, self.sales_tax_rate(), vid, "sales tax")
            self.state["tills"][vid] = round(
                self.state["tills"][vid] + price - tax, 2)
            self.state["daily_intake"][vid] = round(
                self.state["daily_intake"][vid] + price, 2)
            self._log_pay("sale", buyer_name, price, vid,
                          note=f"{amount}x {item}")
            self.save()
        except Exception as e:
            print(f"[Economy] credit_sale failed: {e}")

    def do_restock(self, venue_id, sim_time):
        """6 AM wholesale delivery to one venue: debit till, top stock.
        Returns (delivered_items, wholesale_cost)."""
        try:
            if venue_id not in VENUES:
                return {}, 0.0
            cost = 0.0
            delivered = {}
            for item, max_stock in VENUES[venue_id]["restock"].items():
                w = VENUES[venue_id]["wholesale"].get(item, 1.0)
                delivered[item] = {"max": max_stock, "wholesale": w}
                cost += max_stock * w * 0.0   # top-up computed by caller
            # The caller (resource_manager) decides the delivered quantity
            # against actual stock; we debit when told the cost below.
            return delivered, cost
        except Exception as e:
            print(f"[Economy] do_restock failed: {e}")
            return {}, 0.0

    def debit_restock(self, venue_id, wholesale_cost, sim_time):
        """Pay the wholesale bill from the venue till."""
        try:
            if venue_id not in VENUES or wholesale_cost <= 0:
                return
            self.state["tills"][venue_id] = round(
                self.state["tills"][venue_id] - wholesale_cost, 2)
            self._log_pay("wholesale", VENUES[venue_id]["name"],
                          -wholesale_cost, venue_id)
            self.save()
        except Exception as e:
            print(f"[Economy] debit_restock failed: {e}")

    def pay_restock_worker(self, sim_time):
        """Tom's delivery wage: 10% of yesterday's combined sales, split
        across venues proportional to their intake. 6 AM."""
        try:
            # Hour gate (Oct 22 2026): this ran on the FIRST step after
            # midnight and reset daily_intake there — so pay_employees, called
            # right after it in the same step, always saw intake 0 and paid no
            # wages, ever. ">=" (not "==") so a missed step at the exact hour
            # can't skip the day's payment.
            if sim_time.hour < RESTOCK_HOUR:
                return
            day = sim_time.toordinal()
            if self.state["last_restock_wage_day"] == day:
                return
            self.state["last_restock_wage_day"] = day
            self.state["yesterday_intake"] = dict(self.state["daily_intake"])
            sales = sum(self.state["daily_intake"].values())
            self.state["yesterday_sales"] = round(sales, 2)
            wage = round(RESTOCK_WAGE_SHARE * sales, 2)
            if wage > 0 and self._personas and RESTOCK_WORKER in self._personas:
                paid = 0.0
                for vid, intake in self.state["daily_intake"].items():
                    if intake <= 0:
                        continue
                    share = round(wage * intake / sales, 2)
                    # pay proportionally from each venue till
                    # The 6 AM wholesale bill is debited just before this
                    # runs and drains the till to the float or below, so a
                    # float-gated wage was $0 (communityexp5); the owner's
                    # draw already left this wage in the till.
                    pay = round(min(share, max(0.0, self.state["tills"][vid])), 2)
                    if pay > 0:
                        self.state["tills"][vid] = round(
                            self.state["tills"][vid] - pay, 2)
                        paid += pay
                if paid > 0:
                    s = self._personas[RESTOCK_WORKER].scratch
                    s.wallet = round(getattr(s, "wallet", 0.0) + paid, 2)
                    self._relax_stress(s)
                    self._log_pay("restock_wage", RESTOCK_WORKER, paid, None)
                    print(f"[Economy] {RESTOCK_WORKER} earned ${paid:.2f} "
                          f"delivery wage (wallet ${s.wallet:.0f})")
            # reset daily intake for the new day
            for vid in self.state["daily_intake"]:
                self.state["daily_intake"][vid] = 0.0
            self.save()
        except Exception as e:
            print(f"[Economy] restock wage failed: {e}")

    def record_presence(self, sim_time):
        """GA_JOBS: accumulate on-site minutes for employees during their shift;
        GA_COMMUNITY: likewise for the civic task each agent took today."""
        try:
            last = getattr(self, "_presence_t", None)
            self._presence_t = sim_time
            if last is None or not (jobs_enabled() or community_enabled()):
                return
            dt = (sim_time - last).total_seconds() / 60.0
            if dt <= 0 or dt > 30:
                return
            pres = self.__dict__.setdefault("_presence", {})
            if jobs_enabled() and SHIFT_START <= sim_time.hour < SHIFT_END:
                for vid, emps in self.state["employees"].items():
                    for name in emps:
                        p = self._personas.get(name)
                        if p is not None and self.venue_of_address(
                                p.scratch.act_address) == vid:
                            pres[name] = pres.get(name, 0.0) + dt
                            self._staff_meal(name, p.scratch, pres[name], sim_time)
            if community_enabled() and CIVIC_START <= sim_time.hour < CIVIC_END:
                cp = self.__dict__.setdefault("_civic_presence", {})
                for name, tid in self.state["civic"].items():
                    p = self._personas.get(name)
                    if p is not None and self._at_task(tid, p.scratch.act_address):
                        cp[name] = cp.get(name, 0.0) + dt
                        self._staff_meal(name, p.scratch, cp[name], sim_time)
        except Exception as e:
            print(f"[Economy] presence tracking failed: {e}")

    # -- community fund -----------------------------------------------------
    def sales_tax_rate(self):
        """Dynamic sales tax: today's civic wage bill (signed-up workers x rate
        x expected hours), minus what the fund already collected today, spread
        over the sales still expected today (yesterday's total as the
        forecast). Never below FUND_SALES_TAX nor above FUND_SALES_TAX_MAX."""
        try:
            if not community_enabled():
                return FUND_SALES_TAX
            f = self.state["fund"]
            now = getattr(self, "_presence_t", None)
            day = now.toordinal() if now is not None else None
            civic = self.state["civic"] if self.state.get("civic_day") == day else {}
            need = sum(CIVIC_TASKS[t]["rate"] * CIVIC_EXPECTED_HOURS
                       for t in civic.values())
            got = f.get("today_in", 0.0) if f.get("today_day") == day else 0.0
            seen = sum(self.state["daily_intake"].values())
            forecast = max(sum(self.state["yesterday_intake"].values()), seen)
            left = forecast - seen
            rate = (need - got) / left if (need > got and left > 0) else 0.0
            rate = min(max(rate, FUND_SALES_TAX), FUND_SALES_TAX_MAX)
            f["rate"], f["need"] = round(rate, 4), round(need, 2)
            return rate
        except Exception:
            return FUND_SALES_TAX

    def _tax_to_fund(self, amount, rate, venue_id, why):
        """Skim `rate` of `amount` into the community fund. Returns the tax
        (0 when GA_COMMUNITY is off). The caller removes it from the payer."""
        if not community_enabled() or amount <= 0:
            return 0.0
        # whole cents only: the sub-cent remainder carries to the next sale,
        # so a 0.5% tax on a $1 item is not rounded away
        f = self._fund()
        owed = amount * rate + f.get("carry", 0.0)
        tax = round(int(owed * 100) / 100.0, 2)
        f["carry"] = round(owed - tax, 6)
        if tax <= 0:
            return 0.0
        f["balance"] = round(f["balance"] + tax, 2)
        f["in"] = round(f["in"] + tax, 2)
        now = getattr(self, "_presence_t", None)
        if now is not None:                      # tax collected today (rate control)
            d = now.toordinal()
            if f.get("today_day") != d:
                f["today_day"], f["today_in"] = d, 0.0
            f["today_in"] = round(f.get("today_in", 0.0) + tax, 2)
        self._log_pay("tax", why, tax, venue_id)
        return tax

    def _fund(self):
        """The fund dict; a new fund gets its one-off seed (GA_COMMUNITY only)."""
        f = self.state["fund"]
        if community_enabled() and not f.get("seeded"):
            f["seeded"] = True
            f["balance"] = round(f["balance"] + FUND_SEED, 2)
            f["in"] = round(f["in"] + FUND_SEED, 2)
            self._log_pay("seed", "town treasury", FUND_SEED, None)
        return f

    def _at_task(self, tid, address):
        a = (address or "").lower()
        return bool(a) and all(k in a for k in CIVIC_TASKS[tid]["keys"])

    def pick_civic_task(self, name, sim_time):
        """Sign `name` up for the open task with the most free slots (one per
        day). Returns the task's goal text, or None."""
        if not community_enabled():
            return None
        day = sim_time.toordinal()
        if self.state["civic_day"] != day:
            self.state["civic_day"] = day
            self.state["civic"] = {}
        pj = PUBLIC_JOBS.get(name)
        if pj:                                   # a town-service post, every day
            if name in self.state["civic"]:
                return None
            self.state["civic"][name] = pj
            return CIVIC_TASKS[pj]["name"]
        if name in self.state["civic"]:
            return None
        if any(name in e for e in self.state["employees"].values()) or any(
                c["owner"] == name for c in VENUES.values()):
            return None                      # owners and employees have a job
        taken = list(self.state["civic"].values())
        free = {t: c["slots"] - taken.count(t) for t, c in CIVIC_TASKS.items()
                if not c.get("public")}
        tid = max(free, key=free.get)
        if free[tid] <= 0:
            return None
        self.state["civic"][name] = tid
        return CIVIC_TASKS[tid]["name"]

    def pay_community(self, sim_time):
        """10 PM: pay today's civic workers from the fund (pro-rata when the
        fund is short; no presence -> no pay)."""
        if not community_enabled() or sim_time.hour < WAGE_HOUR:
            return
        day = sim_time.toordinal()
        if self.state.get("civic_paid_day") == day:
            return
        self.state["civic_paid_day"] = day
        cp = self.__dict__.setdefault("_civic_presence", {})
        owed = {}
        for name, tid in self.state["civic"].items():
            mins = cp.get(name, 0.0)
            if mins >= CIVIC_PRESENCE_MIN and name in (self._personas or {}):
                owed[name] = round(CIVIC_TASKS[tid]["rate"] * mins / 60.0, 2)
        cp.clear()
        f = self._fund()
        total = sum(owed.values())
        scale = min(1.0, f["balance"] / total) if total > 0 else 0.0
        for name, amt in owed.items():
            pay = round(amt * scale, 2)
            if pay <= 0:
                continue
            f["balance"] = round(f["balance"] - pay, 2)
            f["out"] = round(f["out"] + pay, 2)
            s = self._personas[name].scratch
            s.wallet = round(getattr(s, "wallet", 0.0) + pay, 2)
            self._relax_stress(s)
            self._log_pay("community", name, pay, None,
                          note=CIVIC_TASKS[self.state["civic"][name]]["name"])
            print(f"[Economy] {name} earned ${pay:.2f} community work "
                  f"(fund ${f['balance']:.0f}, wallet ${s.wallet:.0f})")
        if owed and scale < 1.0:
            print(f"[Economy] community fund short: paid {scale:.0%} of wages")
        self.save()

    def pay_dividend(self, sim_time):
        """Monday 6 AM payday: an equal share of part of the fund to everyone."""
        if not community_enabled() or not self._personas:
            return 0.0
        import os
        wd = int(os.environ.get("GA_DIVIDEND_WEEKDAY", "0"))   # Monday
        hr = int(os.environ.get("GA_DIVIDEND_HOUR", "6"))
        if sim_time.weekday() != wd or sim_time.hour < hr:
            return 0.0
        day = sim_time.toordinal()
        if self.state["last_dividend_day"] == day:
            return 0.0
        self.state["last_dividend_day"] = day
        f = self._fund()
        each = min(FUND_DIVIDEND_CAP,
                   round(f["balance"] * FUND_DIVIDEND_SHARE / len(self._personas), 2))
        if each <= 0:
            return 0.0
        for name, p in self._personas.items():
            f["balance"] = round(f["balance"] - each, 2)
            f["out"] = round(f["out"] + each, 2)
            p.scratch.wallet = round(getattr(p.scratch, "wallet", 0.0) + each, 2)
            self._relax_stress(p.scratch)
            self._log_pay("dividend", name, each, None)
        print(f"[Payday] community dividend ${each:.2f} each "
              f"(fund ${f['balance']:.0f})")
        self.save()
        return each

    def _staff_meal(self, name, scratch, on_site_min, sim_time):
        """GA_JOBS: a broke employee on shift is fed once a day (venue eats the
        cost). Without it a $1 wallet means an empty fridge and no way out."""
        if on_site_min < 30 or getattr(scratch, "wallet", 0) >= 30:
            return
        fed = self.__dict__.setdefault("_staff_fed", {})
        if fed.get(name) == sim_time.toordinal():
            return
        fed[name] = sim_time.toordinal()
        needs = getattr(scratch, "needs", None)
        if isinstance(needs, dict):
            needs["hunger"] = max(needs.get("hunger", 0), 75)
        print(f"[Economy] {name} fed a staff meal on shift "
              f"(wallet ${getattr(scratch, 'wallet', 0):.0f})")

    def pay_employees(self, sim_time):
        """10 PM daily: each employee gets 10% of the venue's daily
        intake (floor $5), from the till — owner tops up from wallet if
        the till can't cover."""
        try:
            if sim_time.hour < WAGE_HOUR:   # was never checked (see restock)
                return
            day = sim_time.toordinal()
            if self.state["last_wage_day"] == day or not self._personas:
                return
            self.state["last_wage_day"] = day
            self.pay_community(sim_time)
            for vid, emps in self.state["employees"].items():
                intake = self.state["daily_intake"].get(vid, 0.0)
                if not emps:
                    continue
                per = max(round(EMPLOYEE_WAGE_SHARE * intake, 2),
                          EMPLOYE_WAGE_FLOOR if intake > 0 else 0.0)
                if per <= 0:
                    continue
                for name in emps:
                    if name not in self._personas:
                        continue
                    if jobs_enabled():
                        _pm = self.__dict__.setdefault("_presence", {}).pop(name, 0.0)
                        if _pm < MIN_SHIFT_PRESENCE_MIN:
                            print(f"[Economy] {name} no-show at {VENUES[vid]['name']} "
                                  f"({_pm:.0f} min on site) — no wage")
                            continue
                    till = self.state["tills"][vid]
                    from_till = min(per, max(0.0, till))
                    from_owner = round(per - from_till, 2)
                    self.state["tills"][vid] = round(till - from_till, 2)
                    s = self._personas[name].scratch
                    s.wallet = round(getattr(s, "wallet", 0.0) + per, 2)
                    self._relax_stress(s)
                    if from_owner > 0:
                        os_ = self._personas[VENUES[vid]["owner"]].scratch
                        os_.wallet = round(
                            getattr(os_, "wallet", 0.0) - from_owner, 2)
                    self._log_pay("wage", name, per, vid)
                    print(f"[Economy] {name} earned ${per:.2f} shift wage "
                          f"at {VENUES[vid]['name']} "
                          f"(wallet ${s.wallet:.0f})")
            if jobs_enabled():
                self.__dict__.setdefault("_presence", {}).clear()
            self._owner_draws(sim_time)
            self.save()
        except Exception as e:
            print(f"[Economy] employee wages failed: {e}")

    def weekly_flows(self, sim_time):
        """Monday from 6 AM: student stipends. (Owner draws are daily now —
        see _owner_draws.)"""
        try:
            if sim_time.weekday() != STIPEND_DAY_HOUR[0] or sim_time.hour < STIPEND_DAY_HOUR[1]:
                return
            day = sim_time.toordinal()
            # Stipends
            if self.state["last_stipend_day"] != day and self._personas:
                self.state["last_stipend_day"] = day
                for name in STIPEND_STUDENTS:
                    if name in self._personas:
                        s = self._personas[name].scratch
                        s.wallet = round(
                            getattr(s, "wallet", 0.0) + STIPEND_AMOUNT, 2)
                        self._relax_stress(s)
                        self._log_pay("stipend", name, STIPEND_AMOUNT, None)
                        print(f"[Economy] {name} received ${STIPEND_AMOUNT:.0f} "
                              f"student stipend (wallet ${s.wallet:.0f})")
            self.save()
        except Exception as e:
            print(f"[Economy] weekly flows failed: {e}")

    def _owner_draws(self, sim_time):
        """Daily (with wages, 10 PM): each owner draws the till surplus above
        the float. Weekly draws left a successful owner broke all week —
        Isabella ended fresh25_1 with $2.60 while Hobbs Cafe held $314."""
        day = sim_time.toordinal()
        if self.state["last_draw_day"] == day or not self._personas:
            return
        self.state["last_draw_day"] = day
        for vid, cfg in VENUES.items():
            owner = cfg["owner"]
            if owner not in self._personas:
                continue
            till = self.state["tills"][vid]
            # Leave tomorrow-morning's delivery wage (Tom: 10% of today's
            # intake, paid at 6 AM from the till surplus) in the till.
            reserve = RESTOCK_WAGE_SHARE * self.state["daily_intake"].get(vid, 0.0)
            draw = round(max(0.0, till - cfg["float"] - reserve), 2)
            if draw > 0:
                self.state["tills"][vid] = round(till - draw, 2)
                draw = round(draw - self._tax_to_fund(
                    draw, FUND_DRAW_TAX, vid, "draw tax"), 2)
                s = self._personas[owner].scratch
                s.wallet = round(getattr(s, "wallet", 0.0) + draw, 2)
                self._relax_stress(s)
                self._log_pay("owner_draw", owner, draw, vid)
                print(f"[Economy] {owner} drew ${draw:.2f} daily profit from "
                      f"{cfg['name']} (wallet ${s.wallet:.0f})")

    # -- hiring market ------------------------------------------------------
    def consider_hiring(self, sim_time):
        """DAILY 7 AM: each understaffed owner's LLM decides whether to
        post a help-wanted opening. Runs EVERY day (not just Mondays) so
        a newly-broke agent can find work within a day — a weekly-only
        window starves broke agents out of the economy."""
        try:
            if sim_time.hour < HIRING_HOUR:
                return
            day = sim_time.toordinal()
            if self.state["last_hirepost_day"] == day:
                return
            self.state["last_hirepost_day"] = day
            for vid, cfg in VENUES.items():
                emps = self.state["employees"][vid]
                if len(emps) >= cfg["max_employees"]:
                    continue
                if not self._personas or cfg["owner"] not in self._personas:
                    continue
                if not self._llm:
                    continue
                owner = self._personas[cfg["owner"]]
                # Yesterday's takings (the live counter was reset at 6 AM).
                intake = self.state.get("yesterday_intake", {}).get(
                    vid, self.state["daily_intake"].get(vid, 0.0))
                prompt = (
                    f"You are {cfg['owner']}, owner of {cfg['name']} in a "
                    f"small town. Yesterday your venue took in "
                    f"${intake:.2f}. You currently have "
                    f"{len(emps)} employee(s). As this persona — "
                    f"{(getattr(owner.scratch, 'learned', '') or '')[:200]} — "
                    f"would you post a help-wanted notice to hire "
                    f"{'another' if emps else 'an'} employee? Answer "
                    f"strictly YES or NO.")
                ans = ""
                try:
                    ans = self._llm(prompt)
                except Exception as e:
                    print(f"[Economy] hire-LLM error for {cfg['owner']}: {e}")
                if "YES" in str(ans).upper()[:40]:
                    self.open_posting(vid, sim_time)
            self.save()
        except Exception as e:
            print(f"[Economy] consider_hiring failed: {e}")

    def open_posting(self, venue_id, sim_time):
        try:
            cfg = VENUES[venue_id]
            # Capacity guard: a full venue never posts.
            if len(self.state["employees"].get(venue_id, [])) >= \
                    cfg["max_employees"]:
                return
            self.state["postings"][venue_id] = {
                "posted_day": sim_time.toordinal(), "applicants": []}
            ann = (f"Help wanted: {cfg['name']} is hiring. "
                   f"Speak with {cfg['owner']} to apply.")
            if self._bulletin:
                self._bulletin(ann)
            print(f"[Economy] {cfg['owner']} posted a "
                  f"help-wanted notice for {cfg['name']}")
        except Exception as e:
            print(f"[Economy] open_posting failed: {e}")

    def applicant_pool(self):
        """Stressed agents eligible to apply for any open posting."""
        if not self._personas:
            return []
        out = []
        for name, p in self._personas.items():
            s = p.scratch
            stress = getattr(s, "financial_stress", 0.0) or 0.0
            wallet = getattr(s, "wallet", 100.0) or 0.0
            has_job = bool(getattr(s, "job", None))
            if has_job:
                continue
            is_owner = any(cfg["owner"] == name for cfg in VENUES.values())
            if is_owner:
                continue
            # Ambition (Oct 22 2026): a strong status craving also sends an
            # agent looking for work — the need's one channel into action
            # short of an affordable luxury purchase.
            status = (getattr(s, "needs", None) or {}).get("status", 100)
            if (stress >= STRESS_APPLY_THRESHOLD or wallet < BROKE_APPLY_WALLET
                    or status < 30):
                out.append(name)
        return out

    def try_hire_encounters(self, arena_paths):
        """Check for apply-encounters: an applicant co-located (same arena
        path string) with an owner who has an open posting. The owner's
        LLM decides. Returns list of hires made this tick.

        arena_paths: {persona_name: arena_path_str} for all personas.
        """
        hires = []
        try:
            if not self._personas or not self.state["postings"]:
                return hires
            pool = set(self.applicant_pool())
            for vid, posting in list(self.state["postings"].items()):
                cfg = VENUES[vid]
                if cfg["owner"] not in self._personas:
                    continue
                owner_path = arena_paths.get(cfg["owner"])
                if not owner_path:
                    continue
                already = set(self.state["employees"][vid])
                if len(already) >= cfg["max_employees"]:
                    self.state["postings"].pop(vid, None)
                    continue
                for name in pool:
                    if name in already:
                        continue
                    if name in posting.get("applicants", []):
                        continue
                    p_path = arena_paths.get(name)
                    if not p_path or p_path != owner_path:
                        continue
                    # co-located! fire the apply-encounter
                    posting.setdefault("applicants", []).append(name)
                    if self._decide_hire(vid, name):
                        self.state["employees"][vid].append(name)
                        self.state["civic"].pop(name, None)   # job beats civic work
                        s = self._personas[name].scratch
                        s.job = {"venue": vid, "employer": cfg["owner"],
                                 "wage_share": EMPLOYEE_WAGE_SHARE}
                        hires.append((vid, name))
                        ann = (f"{name} has been hired at {cfg['name']}. "
                               f"Welcome them to the team!")
                        if self._bulletin:
                            self._bulletin(ann)
                        if len(self.state["employees"][vid]) >= \
                                cfg["max_employees"]:
                            self.state["postings"].pop(vid, None)
                    self._record_hire(vid, name,
                                      name in self.state["employees"][vid])
            self.save()
        except Exception as e:
            print(f"[Economy] hire encounter failed: {e}")
        return hires

    def _decide_hire(self, venue_id, applicant_name):
        """Owner's LLM decision on one applicant. Default: decline on
        LLM failure (conservative)."""
        if not self._llm:
            return False
        try:
            cfg = VENUES[venue_id]
            owner = self._personas.get(cfg["owner"])
            app = self._personas.get(applicant_name)
            if not owner or not app:
                return False
            prompt = (
                f"You are {cfg['owner']}, owner of {cfg['name']}. "
                f"{applicant_name} approaches you and asks about the "
                f"help-wanted notice you posted. About them: "
                f"{(getattr(app.scratch, 'learned', '') or '')[:200]}. "
                f"As {cfg['owner']} — "
                f"{(getattr(owner.scratch, 'learned', '') or '')[:150]} — "
                f"would you hire {applicant_name} for a daily paid shift? "
                f"Answer strictly YES or NO.")
            ans = self._llm(prompt)
            return "YES" in str(ans).upper()[:40]
        except Exception as e:
            print(f"[Economy] hire decision error: {e}")
            return False

    def _record_hire(self, venue_id, applicant, hired):
        self.state["hire_log"].append({
            "venue": VENUES[venue_id]["name"], "owner": VENUES[venue_id]["owner"],
            "applicant": applicant, "hired": hired})

    # -- helpers ------------------------------------------------------------
    def _log_pay(self, kind, who, amount, venue_id, note=None):
        try:
            entry = {"kind": kind, "who": who, "amount": round(amount, 2)}
            _t = getattr(self, "_presence_t", None)   # sim clock, set every step
            if _t is not None:
                entry["t"] = _t.strftime("%b %d %H:%M")
            if venue_id:
                entry["venue"] = VENUES[venue_id]["name"]
            if note:
                entry["note"] = note
            self.state["pay_log"].append(entry)
            # keep the log bounded
            if len(self.state["pay_log"]) > 400:
                self.state["pay_log"] = self.state["pay_log"][-200:]
        except Exception:
            pass

    def _relax_stress(self, scratch, amount=0.15):
        """Income reduces financial stress."""
        try:
            s = getattr(scratch, "financial_stress", None)
            if s is not None and s > 0:
                scratch.financial_stress = max(0.0, s - amount)
        except Exception:
            pass

    def summary(self):
        try:
            return {
                "tills": dict(self.state["tills"]),
                "employees": {v: list(e) for v, e in
                              self.state["employees"].items()},
                "postings": {v: p for v, p in self.state["postings"].items()},
                "daily_intake": dict(self.state["daily_intake"]),
                "fund": dict(self.state["fund"]),
            }
        except Exception:
            return {}

    def __repr__(self):
        try:
            t = ", ".join(f"{v}=${x:.0f}" for v, x in self.state["tills"].items())
            return f"<EconomyManager {t}>"
        except Exception:
            return "<EconomyManager>"

def town_brief(econ=None):
    """Short Smallville primer for the daily-plan prompt: how money, the
    community fund and civic work work here. Live numbers when `econ` is set."""
    if not community_enabled():
        return ""
    f = (econ.state["fund"] if econ is not None else {}) or {}
    rate = f.get("rate") or FUND_SALES_TAX
    bal = f.get("balance")
    fund = (f" It holds ${bal:.0f} right now." if bal is not None else "")
    lines = [
        "About Smallville's economy: you earn money from a job at Hobbs Cafe, "
        "the Harvey Oak Supply Store, The Willows Market or The Rose and Crown "
        "Pub (the owner may hire you), and you spend it on food and drink. Your "
        "fridge at home runs out, so groceries at the Supply Store or Market are "
        "a regular errand; a meal at the cafe costs a few dollars.",
        f"The town keeps a Community Fund: about {rate * 100:.0f}% of every sale "
        "and of owners' daily draws is collected automatically (nobody pays "
        f"extra).{fund} The fund pays residents who do "
        "civic work between 10 AM and 4 PM (park upkeep at Johnson Park, helping "
        "at the Oak Hill College library, tutoring in its classroom; about "
        "$6-7 an hour, paid at 10 PM for at least 30 minutes on site), and "
        "pays every resident a small dividend once a week. Community tasks are "
        "posted on the community bulletin board at Johnson Park.",
    ]
    return " ".join(lines)
