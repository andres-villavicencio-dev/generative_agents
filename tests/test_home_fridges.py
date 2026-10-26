import os, sys, types
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "reverie", "backend_server"))
from resource_manager import WorldResourceManager


def _p(area):
    return types.SimpleNamespace(scratch=types.SimpleNamespace(living_area=area))


def test_household_fridges(tmp_path, monkeypatch):
    monkeypatch.setenv("GA_HOME_FRIDGES", "1")
    rm = WorldResourceManager(str(tmp_path))
    rm.ensure_home_fridges({
        "Eddy Lin": _p("the Ville:Lin family's house:Eddy Lin's bedroom"),
        "Mei Lin": _p("the Ville:Lin family's house:Mei Lin's bedroom"),
        "Tom Moreno": _p("the Ville:Moreno family's house:Tom's bedroom"),
        "Maria Lopez": _p("the Ville:Dorm for Oak Hill College:Maria Lopez's room"),
    })
    f = rm.home_fridge_of
    assert f["Eddy Lin"] == f["Mei Lin"] != f["Tom Moreno"]       # one per household
    assert "refrigerator" in f["Eddy Lin"] and f["Eddy Lin"] in rm.world_state
    assert f["Maria Lopez"] == "Maria Lopez's apartment:kitchen:refrigerator"  # legacy kept


def test_off_by_default(tmp_path, monkeypatch):
    monkeypatch.setenv("GA_HOME_FRIDGES", "0")
    rm = WorldResourceManager(str(tmp_path))
    rm.ensure_home_fridges({"Eddy Lin": _p("the Ville:Lin family's house:x")})
    assert rm.home_fridge_of == {}


def test_legacy_cafe_counter_is_dropped(tmp_path):
    rm = WorldResourceManager(str(tmp_path))
    assert "Hobbs Cafe:front counter:prepared_food" not in rm.world_state
    assert "the Ville:Hobbs Cafe:cafe:counter" in rm.world_state
