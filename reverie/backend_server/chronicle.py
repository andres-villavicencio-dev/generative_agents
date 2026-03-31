"""
Chronicle system for generative agents.
Records milestones (birthday, death, high-poignancy events) as JSON + markdown.
"""
import json
from pathlib import Path


def _chronicle_dir(sim_folder):
  d = Path(sim_folder) / "chronicles"
  d.mkdir(exist_ok=True)
  return d


def chronicle_milestone(persona, event_type, description, sim_folder,
                        poignancy=5):
  """Append a milestone to the agent's chronicle and regenerate markdown."""
  chronicle_dir = _chronicle_dir(sim_folder)
  json_path = chronicle_dir / f"{persona.name}.json"

  if json_path.exists():
    chronicle = json.loads(json_path.read_text())
  else:
    birth_str = (persona.scratch.birth_date.strftime("%Y-%m-%d")
                 if persona.scratch.birth_date else "unknown")
    chronicle = {
      "name": persona.name,
      "birth_date": birth_str,
      "death_date": None,
      "death_age": None,
      "milestones": []
    }

  milestone = {
    "date": persona.scratch.curr_time.strftime("%Y-%m-%d"),
    "age": persona.scratch.current_age,
    "type": event_type,
    "text": description,
    "poignancy": poignancy
  }
  chronicle["milestones"].append(milestone)

  if event_type == "death":
    chronicle["death_date"] = milestone["date"]
    chronicle["death_age"] = milestone["age"]

  json_path.write_text(json.dumps(chronicle, indent=2))
  _regenerate_markdown(chronicle_dir, chronicle)


def _regenerate_markdown(chronicle_dir, chronicle):
  """Generate human-readable markdown from chronicle data."""
  name = chronicle["name"]
  md_path = chronicle_dir / f"{name}.md"

  birth_year = chronicle["birth_date"][:4] if chronicle["birth_date"] != "unknown" else "?"
  if chronicle["death_date"]:
    death_year = chronicle["death_date"][:4]
    header = f"# {name} ({birth_year}\u2013{death_year})"
  else:
    header = f"# {name} (b. {birth_year})"

  lines = [header, ""]
  icons = {
    "birthday": "\U0001f382",
    "death": "\U0001faa6",
    "event": "\u2b50",
    "economic": "\U0001f4b0",
    "social": "\U0001f465",
  }
  for m in chronicle["milestones"]:
    icon = icons.get(m["type"], "\U0001f4cc")
    lines.append(f"- **{m['date']}** (age {m['age']}) {icon} {m['text']}")

  md_path.write_text("\n".join(lines) + "\n")
