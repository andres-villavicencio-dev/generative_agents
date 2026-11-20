import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "reverie", "backend_server"))
from persona.prompt_template import run_gpt_prompt as rgp

PROMPT = ("... revise the schedule from {a} to {b} ...\n"
          "The revised schedule:\n13:00 ~ 13:30 -- working on the painting\n13:30 ~")
REPLY = "14:00 -- chatting with Maria\n14:00 ~ 15:00 -- painting again"


def test_window_with_or_without_am_pm():
  """The re-plan prompt now writes 24h times without a stray PM ('13:00 PM')."""
  want = [["working on the painting", 30], ["chatting with Maria", 30], ["painting again", 60]]
  assert rgp.parse_revised_schedule(PROMPT.format(a="13:00", b="15:00"), REPLY) == want
  assert rgp.parse_revised_schedule(PROMPT.format(a="13:00 PM", b="15:00 PM"), REPLY) == want
