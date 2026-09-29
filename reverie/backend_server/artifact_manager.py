"""
File: artifact_manager.py
Description: Manages artifacts created by agents in the simulation.
Artifacts are persistent world objects (books, paintings, meals, etc.) that
agents create through their actions and that other agents can discover and interact with.
"""
import json
import os
import datetime
import re


# Emoji mapping for frontend rendering
ARTIFACT_EMOJI = {
    "book": "📖",
    "painting": "🎨",
    "meal": "🍳",
    "letter": "✉️",
    "song": "🎵",
    "invention": "💡",
}

# Type properties
ARTIFACT_TYPES = {
    "book":      {"consumable": False, "portable": True,  "interaction": "read",    "need_effect": {"stimulation": 15}},
    "painting":  {"consumable": False, "portable": False, "interaction": "view",    "need_effect": {"stimulation": 15}},
    "meal":      {"consumable": True,  "portable": True,  "interaction": "eat",     "need_effect": {"hunger": 30}},
    "letter":    {"consumable": False, "portable": True,  "interaction": "read",    "need_effect": {"social": 10, "stimulation": 5}},
    "song":      {"consumable": False, "portable": False, "interaction": "listen",  "need_effect": {"stimulation": 10, "social": 5}},
    "invention": {"consumable": False, "portable": False, "interaction": "use",     "need_effect": {"stimulation": 20}},
}

# Keywords in action descriptions that trigger artifact creation
# FIX (regex word-boundary matching): the lane LLM's actual phrasing almost
# never contains the exact literal substrings ("writing", "painting"...
# appear as "write for two hours", "work on book ideas", "cook breakfast") —
# the matcher silently returned None on EVERY real description and no
# artifact was ever created. Patterns match word stems instead, ordered
# most-specific first, with negative guards so consumption phrasing
# ("reading a book", "arranging books", "composition class") doesn't create.
CREATION_ACTION_PATTERNS = [
    # (compiled regex, artifact type, duration_min, materials)
    # letter FIRST — "writing a letter" must not be swallowed by the generic
    # write→book rule below.
    (r"\b(write|writes|writing)\b.*\bletter\b|\bletter\b.*\bwrit", "letter", 20, {}),
    (r"\bwrit(e|ing|es)?\b(?!.*\bread\b)",                     "book",      60, {}),
    (r"\b(work on|outline|draft|brainstorm)(ing)?\b.*\bbook\b", "book",     60, {}),
    (r"\bbook ideas?\b",                                        "book",     60, {}),
    (r"\b(paint|painting|painted)\b(?!.*\bview\b)",             "painting", 60, {}),
    (r"\b(sketch|sketching|draw(ing)?)\b(?!.*\bblueprint\b)",   "painting", 60, {}),
    (r"\b(easel|canvas)\b",                                     "painting", 60, {}),
    (r"\bcompose(d|s|ing)?\b",                                  "song",     90, {}),
    (r"\b(song|melody|lyrics)\b.*\bwrit",                       "song",     90, {}),
    (r"\b(prep|prepare|preparing|cook(ed|ing)?|make|making)\b.*\b(breakfast|brunch|lunch|dinner|meal|dish|soup|stew)\b",
                                                                "meal",     20, {}),
    (r"\b(bak(e|ing|ed|es))\b",                                 "meal",     30, {}),
    (r"\b(prepare|preparing)\b.*\bmeal\b",                      "meal",     20, {}),
    (r"\binvent(ing|ed|s|ion)?\b",                              "invention",120, {}),
    (r"\b(craft|crafting|crafted)\b",                           "invention", 90, {}),
]
_CREATION_COMPILED = [(re.compile(p, re.IGNORECASE), t, d, m)
                      for p, t, d, m in CREATION_ACTION_PATTERNS]

# Keywords that indicate an agent is interacting with an artifact
INTERACTION_KEYWORDS = {
    "reading":      ["book", "letter"],
    "viewing":      ["painting"],
    "looking at":   ["painting"],
    "admiring":     ["painting"],
    "eating":       ["meal"],
    "listening to": ["song"],
    "using":        ["invention"],
}


class ArtifactManager:
    """
    Manages artifacts created by agents during the simulation.

    Artifacts are stored in {sim_folder}/artifacts/artifacts.json
    and placed as tile events on the maze for perception by other agents.
    """

    def __init__(self, sim_folder):
        self.sim_folder = sim_folder
        self.artifacts = {}
        self._next_id = 1
        self.load()

    def _artifact_file(self):
        return os.path.join(self.sim_folder, "artifacts", "artifacts.json")

    def load(self):
        artifact_file = self._artifact_file()
        if os.path.exists(artifact_file):
            with open(artifact_file) as f:
                self.artifacts = json.load(f)
            # Determine next ID from existing artifacts
            if self.artifacts:
                max_id = max(int(k.split("_")[1]) for k in self.artifacts.keys())
                self._next_id = max_id + 1
        else:
            self.artifacts = {}

    def save(self, sim_folder=None):
        folder = sim_folder or self.sim_folder
        artifact_dir = os.path.join(folder, "artifacts")
        os.makedirs(artifact_dir, exist_ok=True)
        with open(os.path.join(artifact_dir, "artifacts.json"), "w") as f:
            json.dump(self.artifacts, f, indent=2)

    def create_artifact(self, artifact_type, name, description,
                        content_summary, creator, created_time,
                        quality, location, materials_consumed=None):
        artifact_id = f"artifact_{self._next_id:03d}"
        self._next_id += 1

        type_info = ARTIFACT_TYPES.get(artifact_type, {})

        artifact = {
            "id": artifact_id,
            "type": artifact_type,
            "name": name,
            "description": description,
            "content_summary": content_summary,
            "creator": creator,
            "created_time": created_time,
            "quality": quality,
            "location": location,
            "materials_consumed": materials_consumed or {},
            "keywords": [artifact_type, name.lower(), creator.lower()],
            "is_consumable": type_info.get("consumable", False),
            "consumed_by": None,
            "gifted_to": None,
        }

        self.artifacts[artifact_id] = artifact
        self.save()
        return artifact

    def get_artifact(self, artifact_id):
        return self.artifacts.get(artifact_id)

    def get_artifacts_at_location(self, address):
        """Return all non-consumed artifacts at a given address."""
        results = []
        for art in self.artifacts.values():
            if art["location"] == address and not art.get("consumed_by"):
                results.append(art)
        return results

    def get_all_active_artifacts(self):
        """Return all non-consumed artifacts."""
        return [a for a in self.artifacts.values() if not a.get("consumed_by")]

    def consume_artifact(self, artifact_id, consumer_name):
        art = self.artifacts.get(artifact_id)
        if not art:
            return False
        if not art.get("is_consumable"):
            return False
        if art.get("consumed_by"):
            return False
        art["consumed_by"] = consumer_name
        self.save()
        return True

    def move_artifact(self, artifact_id, new_location):
        art = self.artifacts.get(artifact_id)
        if art:
            art["location"] = new_location
            self.save()

    def get_artifact_event_tuple(self, artifact_id):
        """Return the event tuple used to place this artifact on maze tiles."""
        art = self.artifacts.get(artifact_id)
        if not art:
            return None
        emoji = ARTIFACT_EMOJI.get(art["type"], "📦")
        return (
            f"artifact:{artifact_id}",
            "is",
            art["name"],
            f"{emoji} {art['name']} by {art['creator']} ({art['type']})"
        )

    def match_creation_action(self, act_description):
        """Check if an action description matches a creation pattern.
        Returns the mapping dict or None.
        FIX: regex word-boundary patterns instead of literal substrings —
        the literal table never matched the lane LLM's real phrasing, so
        try_create_artifact's accumulation gate never fired and zero
        artifacts were created across the entire n23–n26 runs."""
        desc_lower = act_description.lower()
        for rx, atype, duration_min, materials in _CREATION_COMPILED:
            if rx.search(desc_lower):
                return {"type": atype, "duration_min": duration_min,
                        "materials": materials}
        return None

    def match_interaction(self, act_description):
        """Check if an action description matches an artifact interaction.
        Returns list of matching artifact types or None."""
        desc_lower = act_description.lower()
        for keyword, artifact_types in INTERACTION_KEYWORDS.items():
            if keyword in desc_lower:
                return artifact_types
        return None
