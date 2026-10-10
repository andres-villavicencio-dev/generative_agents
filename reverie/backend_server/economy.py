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
        "float": 200.0,
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

RESTOCK_WORKER = "Tom Moreno"
RESTOCK_WAGE_SHARE = 0.10     # of previous day's combined sales
EMPLOYEE_WAGE_SHARE = 0.10    # of venue's daily sales, per employee
EMPLOYE_WAGE_FLOOR = 5.0      # a shift always pays at least this
WAGE_HOUR = 22                # 10 PM: employee payday within the day
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
            self.state["tills"][vid] = round(
                self.state["tills"][vid] + price, 2)
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
            day = sim_time.toordinal()
            if self.state["last_restock_wage_day"] == day:
                return
            self.state["last_restock_wage_day"] = day
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
                    from_float = max(0.0, self.state["tills"][vid]
                                     - VENUES[vid]["float"])
                    pay = round(min(share, from_float), 2)
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

    def pay_employees(self, sim_time):
        """10 PM daily: each employee gets 10% of the venue's daily
        intake (floor $5), from the till — owner tops up from wallet if
        the till can't cover."""
        try:
            day = sim_time.toordinal()
            if self.state["last_wage_day"] == day or not self._personas:
                return
            self.state["last_wage_day"] = day
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
            self.save()
        except Exception as e:
            print(f"[Economy] employee wages failed: {e}")

    def weekly_flows(self, sim_time):
        """Monday 6 AM: stipends + owner draws."""
        try:
            if sim_time.weekday() != 0 or sim_time.hour != 6:
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
            # Owner draws
            if self.state["last_draw_day"] != day:
                self.state["last_draw_day"] = day
                for vid, cfg in VENUES.items():
                    owner = cfg["owner"]
                    if not self._personas or owner not in self._personas:
                        continue
                    till = self.state["tills"][vid]
                    draw = round(max(0.0, till - cfg["float"]), 2)
                    if draw > 0:
                        self.state["tills"][vid] = round(till - draw, 2)
                        s = self._personas[owner].scratch
                        s.wallet = round(getattr(s, "wallet", 0.0) + draw, 2)
                        self._relax_stress(s)
                        self._log_pay("owner_draw", owner, draw, vid)
                        print(f"[Economy] {owner} drew ${draw:.2f} weekly "
                              f"profit from {cfg['name']} "
                              f"(wallet ${s.wallet:.0f})")
            self.save()
        except Exception as e:
            print(f"[Economy] weekly flows failed: {e}")

    # -- hiring market ------------------------------------------------------
    def consider_hiring(self, sim_time):
        """DAILY 7 AM: each understaffed owner's LLM decides whether to
        post a help-wanted opening. Runs EVERY day (not just Mondays) so
        a newly-broke agent can find work within a day — a weekly-only
        window starves broke agents out of the economy."""
        try:
            if sim_time.hour != 7:
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
                intake = self.state["daily_intake"].get(vid, 0.0)
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
            if stress >= STRESS_APPLY_THRESHOLD or wallet < BROKE_APPLY_WALLET:
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
            }
        except Exception:
            return {}

    def __repr__(self):
        try:
            t = ", ".join(f"{v}=${x:.0f}" for v, x in self.state["tills"].items())
            return f"<EconomyManager {t}>"
        except Exception:
            return "<EconomyManager>"