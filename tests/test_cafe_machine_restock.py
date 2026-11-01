import datetime, os, sys, tempfile, types
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "reverie", "backend_server"))
from resource_manager import WorldResourceManager

MACHINE = "the Ville:Hobbs Cafe:cafe:coffee machine"


def test_daily_delivery_refills_coffee_machine(tmp_path):
    rm = WorldResourceManager(str(tmp_path))
    rm.world_state[MACHINE] = {"coffee_beans": 5.0, "max": 100}
    billed = []
    econ = types.SimpleNamespace(
        debit_restock=lambda vid, cost, t: billed.append((vid, round(cost, 2))))
    rm._do_daily_delivery(datetime.datetime(2023, 2, 18, 6), economy=econ)
    assert rm.world_state[MACHINE]["coffee_beans"] == 100
    assert rm.world_state[MACHINE]["max"] == 100
    assert ("cafe", 95.0) in billed      # 95 beans at $1, billed to the cafe till
