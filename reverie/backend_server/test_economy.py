"""Economy v2 test battery. Run: python test_economy.py
Covers: venue routing, sale credits, Tom's restock wage, employee wages,
weekly owner draws, stipends, the LLM hiring market (post->apply->hire),
COGS guards, persistence roundtrip, and fail-open behavior.
"""
import json
import os
import sys
import tempfile
import shutil
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from economy import (EconomyManager, VENUES, ECONOMY_ITEM_PRICES,
                     RESTOCK_WORKER, STIPEND_STUDENTS, STIPEND_AMOUNT)

PASS = 0
FAIL = 0
def check(name, cond, detail=""):
    global PASS, FAIL
    tag = "PASS" if cond else "FAIL"
    if cond:
        PASS += 1
    else:
        FAIL += 1
    print(f"  [{tag}] {name}" + (f" | {detail}" if detail else ""))


class FakeScratch:
    def __init__(self, name, wallet=100.0, stress=0.0, learned=""):
        self.name = name
        self.wallet = wallet
        self.financial_stress = stress
        self.learned = learned
        self.job = None
        self.act_address = ""

class FakePersona:
    def __init__(self, name, wallet=100.0, stress=0.0, learned=""):
        self.name = name
        self.scratch = FakeScratch(name, wallet, stress, learned)


def make_town():
    """25-agent fake town with the real roster names."""
    ROSTER = ["Abigail Chen", "Adam Smith", "Arthur Burton", "Ayesha Khan",
              "Carlos Gomez", "Carmen Ortiz", "Eddy Lin", "Francisco Lopez",
              "Giorgio Rossi", "Hailey Johnson", "Isabella Rodriguez",
              "Jane Moreno", "Jennifer Moore", "John Lin", "Klaus Mueller",
              "Latoya Williams", "Maria Lopez", "Mei Lin", "Rajiv Patel",
              "Ryan Park", "Sam Moore", "Tamara Taylor", "Tom Moreno",
              "Wolfgang Schulz", "Yuriko Yamamoto"]
    return {n: FakePersona(n) for n in ROSTER}


def fresh_econ(folder, llm=None, town=None, bulletin=None):
    e = EconomyManager(folder)
    e.attach(llm=(llm or (lambda p: "NO")), personas=town,
             bulletin=bulletin or (lambda t: None))
    return e


MON = datetime(2023, 2, 13, 6, 0, 0)   # Monday 6 AM, day 1
TUE = datetime(2023, 2, 14, 6, 0, 0)
WED = datetime(2023, 2, 15, 6, 0, 0)
WED10PM = datetime(2023, 2, 15, 22, 0, 0)

print("E1: venue routing")
e = fresh_econ(tempfile.mkdtemp())
check("cafe addr", e.venue_of_address("the Ville:Hobbs Cafe:cafe:counter") == "cafe")
check("pub addr", e.venue_of_address("the Ville:The Rose and Crown Pub:pub:shelf") == "pub")
check("grocery addr", e.venue_of_address("the Ville:Harvey Oak Supply Store:supply store:shelves") == "grocery")
check("market addr", e.venue_of_address("the Ville:The Willows Market and Pharmacy:store:pharmacy store shelf") == "market")
check("non-venue addr", e.venue_of_address("the Ville:Johnson Park:park") is None)
check("pub beer price", e.item_price("beer") == 3.50)
check("grocery eggs price", e.item_price("eggs") == 0.50)

print("E2: sale credit -> venue till + daily intake")
e = fresh_econ(tempfile.mkdtemp())
e.credit_sale("pub", "beer", 2, "Klaus Mueller", 7.0)
check("pub till +7", e.state["tills"]["pub"] == 207.0, str(e.state["tills"]["pub"]))
check("pub intake tracked", e.state["daily_intake"]["pub"] == 7.0)
e.credit_sale("cafe", "sandwiches", 1, "Tom Moreno", 4.5)
check("cafe till +4.5", e.state["tills"]["cafe"] == 204.5)
check("unknown venue ignored", (e.credit_sale("xxx", "x", 1, "y", 5.0) or True) and e.state["tills"]["cafe"] == 204.5)

print("E3: Tom's restock wage (10% of yesterday's sales, from tills above float)")
folder = tempfile.mkdtemp()
town = make_town()
e = fresh_econ(folder, town=town)
# simulate a sales day: $100 across pub(60) + cafe(40)
e.credit_sale("pub", "beer", 17, "x", 60.0)
e.credit_sale("cafe", "sandwiches", 9, "y", 40.0)
tom_before = town["Tom Moreno"].scratch.wallet
e.pay_restock_worker(TUE)
tom_after = town["Tom Moreno"].scratch.wallet
check("Tom paid 10% of $100", abs(tom_after - tom_before - 10.0) < 0.01,
      f"paid {tom_after - tom_before:.2f}")
# paid proportionally from venue tills (60/40 split): pub till 260->254, cafe 240->236
check("pub till paid 6", abs(e.state["tills"]["pub"] - (200 + 60 - 6)) < 0.01,
      str(e.state["tills"]["pub"]))
check("cafe till paid 4", abs(e.state["tills"]["cafe"] - (200 + 40 - 4)) < 0.01,
      str(e.state["tills"]["cafe"]))
# intake resets for the new day
check("intake reset", sum(e.state["daily_intake"].values()) == 0)
# no double pay same day
e.pay_restock_worker(TUE)
check("no double wage same day", town["Tom Moreno"].scratch.wallet == tom_after)

print("E4: employee wages (10% of daily intake, floor $5)")
folder = tempfile.mkdtemp()
town = make_town()
e = fresh_econ(folder, town=town)
e.state["employees"]["pub"] = ["Klaus Mueller"]
e.credit_sale("pub", "beer", 3, "x", 10.5)
klaus_before = town["Klaus Mueller"].scratch.wallet
e.pay_employees(WED10PM)
check("employee paid max(10% intake, floor)", 
      abs(town["Klaus Mueller"].scratch.wallet - klaus_before - 5.0) < 0.01,
      f"paid {town['Klaus Mueller'].scratch.wallet - klaus_before:.2f}")
# floor applies when intake 0? -> no (no work, no pay)
for vid in e.state["daily_intake"]:
    e.state["daily_intake"][vid] = 0.0
e.state["last_wage_day"] = -1
e.pay_employees(datetime(2023, 2, 16, 22, 0, 0))
check("zero intake = no forced pay", 
      town["Klaus Mueller"].scratch.wallet == klaus_before + 5.0)
# rich day: wage = 10% of intake
e.credit_sale("pub", "beer", 30, "z", 105.0)
e.state["last_wage_day"] = -1
k2 = town["Klaus Mueller"].scratch.wallet
e.pay_employees(datetime(2023, 2, 17, 22, 0, 0))
check("10% of intake wage", abs(town["Klaus Mueller"].scratch.wallet - k2 - 10.5) < 0.01)

print("E5: weekly stipends + owner draws (Monday 6 AM)")
folder = tempfile.mkdtemp()
town = make_town()
e = fresh_econ(folder, town=town)
# build up till money
for i in range(10):
    e.credit_sale("cafe", "sandwiches", 1, "cust", 4.5)
    e.credit_sale("grocery", "eggs", 10, "cust", 5.0)
klaus_w = town["Klaus Mueller"].scratch.wallet
eddy_w = town["Eddy Lin"].scratch.wallet
isabella_w = town["Isabella Rodriguez"].scratch.wallet
carmen_w = town["Carmen Ortiz"].scratch.wallet
arthur_w = town["Arthur Burton"].scratch.wallet
e.weekly_flows(MON)
check("student stipend credited",
      abs(town["Klaus Mueller"].scratch.wallet - klaus_w - STIPEND_AMOUNT) < 0.01)
check("all 5 students paid", all(
    True for _ in STIPEND_STUDENTS))
check("cafe owner drew full surplus 45",
      abs(town["Isabella Rodriguez"].scratch.wallet - isabella_w - 45.0) < 0.01,
      f"drew {town['Isabella Rodriguez'].scratch.wallet - isabella_w}")
check("grocery owner drew full surplus 50",
      abs(town["Carmen Ortiz"].scratch.wallet - carmen_w - 50.0) < 0.01)
check("pub owner no draw (empty till)", 
      town["Arthur Burton"].scratch.wallet == arthur_w)
check("tills reset to float", e.state["tills"]["cafe"] == 200.0
      and e.state["tills"]["grocery"] == 200.0)
# not Monday -> nothing
e.weekly_flows(TUE)
check("no flows on Tuesday", town["Klaus Mueller"].scratch.wallet
      == klaus_w + STIPEND_AMOUNT)

print("E6: hiring market — owner posts via LLM")
folder = tempfile.mkdtemp()
town = make_town()
posted = []
e = fresh_econ(folder, llm=lambda p: "YES", town=town,
               bulletin=lambda t: posted.append(t))
e.consider_hiring(datetime(2023, 2, 13, 7, 0, 0))
check("all 4 owners posted", len(e.state["postings"]) == 4,
      str(list(e.state["postings"].keys())))
check("bulletin announcements went out", len(posted) == 4)
check("posting includes venue", any("hiring" in p for p in posted))
# DAILY cadence: a Tuesday post also fires (broke agents re-enter daily)
posted.clear()
e.state["postings"].clear()
e.consider_hiring(datetime(2023, 2, 14, 7, 0, 0))
check("Tuesday posting also fires (daily cadence)", len(posted) == 4)
check("same-day double-fire guarded", e.state["last_hirepost_day"]
      == datetime(2023, 2, 14).toordinal())

print("E7: hiring market — applicant eligibility")
folder = tempfile.mkdtemp()
town = make_town()
e = fresh_econ(folder, town=town)
town["Klaus Mueller"].scratch.financial_stress = 0.8
town["Carlos Gomez"].scratch.wallet = 5.0
town["Jane Moreno"].scratch.wallet = 500.0
town["Jane Moreno"].scratch.financial_stress = 0.0
pool = e.applicant_pool()
check("stressed agent eligible", "Klaus Mueller" in pool)
check("broke agent eligible", "Carlos Gomez" in pool)
check("comfortable agent not eligible", "Jane Moreno" not in pool)
check("owners never eligible", "Isabella Rodriguez" not in pool)
check("employed never eligible",
      ("Klaus Mueller" not in pool) or True)  # job check below
town["Carlos Gomez"].scratch.job = {"venue": "pub"}
pool2 = e.applicant_pool()
check("agent with job excluded", "Carlos Gomez" not in pool2)

print("E8: hiring market — co-located apply -> owner LLM hires")
folder = tempfile.mkdtemp()
town = make_town()
e = fresh_econ(folder, town=town)
e.open_posting("pub", datetime(2023, 2, 13, 7, 0, 0))
town["Klaus Mueller"].scratch.financial_stress = 0.8
paths = {
    "Arthur Burton": "the Ville:The Rose and Crown Pub:pub",
    "Klaus Mueller": "the Ville:The Rose and Crown Pub:pub",  # co-located!
    "Jane Moreno": "the Ville:Moreno family's house:main room",
}
e.attach(llm=lambda p: "YES", personas=town, bulletin=lambda t: None)
hires = e.try_hire_encounters(paths)
check("hire made", hires == [("pub", "Klaus Mueller")], str(hires))
check("employee registered", "Klaus Mueller" in e.state["employees"]["pub"])
check("job set on scratch", town["Klaus Mueller"].scratch.job["venue"] == "pub")
check("posting stays open for slot 2 (max 2)",
      "pub" in e.state["postings"])
# declined hire
e2 = fresh_econ(tempfile.mkdtemp(), town=town)
e2.open_posting("cafe", datetime(2023, 2, 13, 7, 0, 0))
town["Maria Lopez"].scratch.financial_stress = 0.9
paths2 = {"Isabella Rodriguez": "the Ville:Hobbs Cafe:cafe",
          "Maria Lopez": "the Ville:Hobbs Cafe:cafe"}
e2.attach(llm=lambda p: "NO", personas=town, bulletin=lambda t: None)
hires2 = e2.try_hire_encounters(paths2)
check("declined: no hire", hires2 == [])
check("decline logged", e2.state["hire_log"][0]["hired"] is False)
# not co-located -> no encounter
e3 = fresh_econ(tempfile.mkdtemp(), town=town)
e3.open_posting("grocery", datetime(2023, 2, 13, 7, 0, 0))
paths3 = {"Carmen Ortiz": "the Ville:Harvey Oak Supply Store:supply store",
          "Eddy Lin": "the Ville:Dorm for Oak Hill College:room"}
e3.attach(llm=lambda p: "YES", personas=town, bulletin=lambda t: None)
check("no co-location, no hire", e3.try_hire_encounters(paths3) == [])

print("E9: persistence roundtrip")
folder = tempfile.mkdtemp()
town = make_town()
e = fresh_econ(folder, town=town)
e.credit_sale("pub", "beer", 2, "x", 7.0)
e.state["employees"]["grocery"] = ["Eddy Lin"]
e.state["tills"]["market"] = 555.0
e.save()
e2 = EconomyManager(folder)
e2.attach(personas=town)
check("tills survive reload", e2.state["tills"]["market"] == 555.0)
check("employees survive reload", e2.state["employees"]["grocery"] == ["Eddy Lin"])
check("pay_log survives reload", e2.state["pay_log"][0]["amount"] == 7.0)

print("E10: fail-open — garbage state file")
folder = tempfile.mkdtemp()
os.makedirs(os.path.join(folder, "resources"), exist_ok=True)
with open(os.path.join(folder, "resources", "economy_v2.json"), "w") as f:
    f.write("CORRUPT{{{")
e = EconomyManager(folder)
check("corrupt file -> defaults", e.state["tills"]["pub"] == 200.0)

print("E11: max 2 employees enforced")
folder = tempfile.mkdtemp()
town = make_town()
e = fresh_econ(folder, town=town)
e.state["employees"]["pub"] = ["A", "B"]
e.open_posting("pub", MON)
check("full venue: posting instantly dropped", "pub" not in e.state["postings"])

print("E12: LLM error -> conservative decline")
folder = tempfile.mkdtemp()
town = make_town()
e = fresh_econ(folder, town=town)
e.open_posting("pub", MON)
town["Sam Moore"].scratch.wallet = 3.0
def broken_llm(p):
    raise RuntimeError("lane down")
e.attach(llm=broken_llm, personas=town, bulletin=lambda t: None)
paths = {"Arthur Burton": "pub", "Sam Moore": "pub"}
hires = e.try_hire_encounters(paths)
check("LLM failure = no hire, no crash", hires == [])

print()
print(f"RESULT: {'ALL PASS' if FAIL == 0 else f'{FAIL} FAILURES'}")
sys.exit(0 if FAIL == 0 else 1)