import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "reverie", "backend_server"))
import info_lookup as IL


class Econ:
    state = {"duties": {"open": {"d1": {"venue": "cafe", "type": "cafe.clean_table", "claimed_by": None}}},
             "fund": {"balance": 250.0, "rate": 0.02}}

    def board_options(self, name):
        return [("venue", "cafe", "Hobbs Cafe: help wanted")]


def test_matching():
    assert IL.matches("looking up job openings online")
    assert IL.matches("browsing the news")
    assert not IL.matches("eating lunch")


def test_digest_reports_true_facts():
    d = IL.digest(Econ(), "Ana")
    assert "help wanted" in d and "$250" in d and "2.0%" in d and "Paid work is waiting" in d


def test_real_phrasings_and_conversation_exclusion():
    assert IL.matches("browsing on their phone (typing his query into the search bar)")
    assert IL.matches("plan a grocery trip (checking the store's online catalog)")
    assert IL.matches("apply for an opening (filling out the online application form)")
    assert not IL.matches("conversing about Tom's research on the election")
    assert not IL.matches("eating breakfast")
