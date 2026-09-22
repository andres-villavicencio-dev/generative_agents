"""A/B benchmark: JEV fast paths vs legacy generation for the four new call families.

Reuses REAL persona objects loaded from the live sim's checkpointed state, so
prompts match production shape. Reads only — no sim processes touched.
"""
import json
import os
import sys
import time
import importlib

REVERIE_DIR = "/home/andus/Projects/ga-jev/reverie/backend_server"
SIM_STORE = "/home/andus/Projects/generative_agents/environment/frontend_server/storage/live3d_v3"

sys.path.insert(0, REVERIE_DIR)
os.chdir(REVERIE_DIR)  # utils.py resolution, same as backend

os.environ["GA_JEV_SCORING"] = "1"

jev_scoring = importlib.import_module("persona.prompt_template.jev_scoring")
jev_score_digit = jev_scoring.jev_score_digit
jev_score_choice = jev_scoring.jev_score_choice

gpt_structure = importlib.import_module("persona.prompt_template.gpt_structure")
generate_prompt = gpt_structure.generate_prompt
ChatGPT_safe_generate_response = gpt_structure.ChatGPT_safe_generate_response

run_gpt_prompt = importlib.import_module("persona.prompt_template.run_gpt_prompt")


def load_personas():
    """Load Persona objects from the live sim's checkpoint."""
    from persona.persona import Persona

    personas = {}
    p_dir = os.path.join(SIM_STORE, "personas")
    for name in os.listdir(p_dir):
        p_path = os.path.join(p_dir, name, "bootstrap_memory")
        if os.path.isdir(p_path):
            personas[name] = Persona(name, os.path.join(p_dir, name))
    return personas


def bench_poignancy(personas):
    """Lane A: jev_score_digit vs Lane B: legacy event_poignancy."""
    results = []
    cases = [
        ("Isabella Rodriguez", "Isabella Rodriguez is setting up the music equipment"),
        ("Maria Lopez", "Maria Lopez is studying for her physics quiz"),
        ("Klaus Mueller", "Klaus Mueller is writing a research paper"),
        ("Maria Lopez", "Maria Lopez is brushing her teeth"),
        ("Klaus Mueller", "Klaus Mueller saw Maria at the cafe and they agreed to meet tomorrow"),
    ]
    for pname, event in cases:
        persona = personas[pname]
        # Lane A: JEV digit scoring
        prompt = (
            f"Here is a brief description of {persona.scratch.name}. \n"
            f"{persona.scratch.get_str_iss()} \n\n"
            f"On the scale of 1 to 10, where 1 is purely mundane (e.g., brushing teeth, making bed) "
            f"and 10 is extremely poignant (e.g., a break up, college acceptance), "
            f"rate the likely poignancy of the following event for {persona.scratch.name}. \n\n"
            f"Event: {event}\n"
            f"Rate (return a number between 1 to 10):"
        )
        t0 = time.time()
        jev_out = jev_score_digit(prompt)
        t_jev = time.time() - t0

        # Lane B: legacy path
        t0 = time.time()
        legacy_out = run_gpt_prompt.run_gpt_prompt_event_poignancy(persona, event)[0]
        t_legacy = time.time() - t0

        results.append({
            "case": event[:40], "jev": jev_out, "jev_s": round(t_jev, 2),
            "legacy": legacy_out, "legacy_s": round(t_legacy, 2),
        })
    return results


def bench_spatial(personas):
    """Sector choice: jev_score_choice vs legacy action_sector."""
    from maze import Maze
    maze = Maze("the_ville")  # main map file
    results = []
    cases = [
        ("Isabella Rodriguez", "eating lunch", [82, 24]),
        ("Maria Lopez", "studying for her physics quiz", [110, 55]),
        ("Klaus Mueller", "writing a research paper", [58, 37]),
        ("Isabella Rodriguez", "listening to music", [82, 24]),
        ("Maria Lopez", "going to the gym", [110, 55]),
    ]
    for pname, action, tile in cases:
        persona = personas[pname]
        persona.scratch.curr_tile = tile
        act_world = f"{maze.access_tile(tile)['world']}"
        accessible = persona.s_mem.get_str_accessible_sectors(act_world)
        curr = accessible.split(", ")
        fin = [i for i in curr if ("'s house" not in i) or (persona.scratch.last_name in i)]

        # Lane A: JEV choice scoring
        prompt = (
            f"{persona.scratch.name} lives in {persona.scratch.living_area.split(':')[1]} "
            f"and is currently in {maze.access_tile(tile)['sector']}. \n"
            f"Activity: {action}\n"
            f"Which area should they go to for this activity? Stay in the current area "
            f"if the activity can be done there. Options:\n__JEV_OPTIONS__\n"
            f"Answer with the letter only:"
        )
        t0 = time.time()
        jev_out = jev_score_choice(prompt, fin)
        t_jev = time.time() - t0

        # Lane B: legacy
        t0 = time.time()
        legacy_out = run_gpt_prompt.run_gpt_prompt_action_sector(action, persona, maze)[0]
        t_legacy = time.time() - t0

        results.append({
            "case": f"{pname} {action}", "options_n": len(fin),
            "jev": jev_out, "jev_s": round(t_jev, 2),
            "legacy": legacy_out, "legacy_s": round(t_legacy, 2),
        })
    return results


def main():
    print(f"Jev scoring enabled: {jev_scoring.USE_JEV_SCORING}")
    print(f"Chat model: {gpt_structure.OLLAMA_CHAT_MODEL}")
    personas = load_personas()
    print(f"loaded {len(personas)} personas\n")

    poig = bench_poignancy(personas)
    print("=== POIGNANCY (event) ===")
    for r in poig:
        agree = r["jev"] == r["legacy"]
        print(f"  {r['case'][:38]:40s} jev={r['jev']} ({r['jev_s']}s)  legacy={r['legacy']} ({r['legacy_s']}s)  match={agree}")

    spatial = bench_spatial(personas)
    print("\n=== SPATIAL (sector) ===")
    for r in spatial:
        agree = r["jev"] == r["legacy"]
        print(f"  {r['case'][:38]:40s} jev={r['jev']} ({r['jev_s']}s)  legacy={r['legacy']} ({r['legacy_s']}s)  match={agree}")

    j = [r["jev_s"] for r in poig + spatial if r["jev"] is not None]
    l = [r["legacy_s"] for r in poig + spatial if r["legacy"] is not None]
    if j:
        print(f"\nJEV avg: {sum(j)/len(j):.2f}s (n={len(j)})   Legacy avg: {sum(l)/len(l):.2f}s (n={len(l)})")
    print("\nsaved -> bench_results_v2.json")
    with open("bench_results_v2.json", "w") as f:
        json.dump({"poignancy": poig, "spatial": [dict(r) for r in spatial]}, f, indent=1)


if __name__ == "__main__":
    main()