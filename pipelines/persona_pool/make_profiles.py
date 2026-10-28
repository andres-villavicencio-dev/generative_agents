#!/usr/bin/env python3
"""Turn synthetic Nemotron-Personas candidates into Smallville scratch fields
(innate / learned / currently / lifestyle), tied to existing residents, with a
cloud model (glm-5.3-flash) so they read like the 25 originals. -> profiles.json"""
import json, re, sys, urllib.request

PICKS = {  # candidate slot -> (home arena "sector:arena", relationship hooks to existing residents)
 "family doctor":      ("Willow Court 1:bedroom 1", "sees John Lin at the Willows pharmacy daily; patients include Sam and Jennifer Moore"),
 "college librarian":  ("Willow Court 2:bedroom 1", "runs the Oak Hill College library; Ayesha Khan and Klaus Mueller are regulars; Mei Lin is a colleague"),
 "baker":              ("Willow Court 3:bedroom 1", "bakes pastries for Isabella Rodriguez's Hobbs Cafe before dawn; Maria Lopez drops by"),
 "market gardener":    ("Willow Court 4:bedroom 1", "grows eggs-and-produce for Carmen Ortiz's Harvey Oak Supply Store and John Lin's market"),
 "police officer":     ("Willow Court 5:bedroom 1", "patrols the town; keeps an eye on loud Carlos Gomez; friendly with Arthur Burton's pub"),
 "newspaper editor":   ("Maple Co-living:room 1", "edits the Smallville Gazette; Adam Smith and Hailey Johnson write for it; covers the mayoral election"),
 "community leader / mayor": ("Willow Court 6:bedroom 1", "the incumbent mayor seeking re-election; Yuriko Yamamoto advises on law, Mei Lin on education"),
 "retired contractor (challenger)": ("Willow Court 7:bedroom 1", "challenging the mayor in the election; old friend of Sam Moore and Tom Moreno"),
 "teen (Moreno spare room)": ("Moreno family's house:empty bedroom", "is Tom and Jane Moreno's teenage child; dual-enrolled at Oak Hill College; friends with Eddy Lin"),
 "newcomer job seeker": ("Maple Co-living:room 2", "just arrived in town, looking for work; asks Isabella Rodriguez and Arthur Burton about openings"),
}
EXAMPLE = {
 "innate": "thoughtful, reflective, intellectual",
 "learned": "Adam Smith is a philosopher who loves to explore different ideas. He is always looking for ways to challenge people’s preconceptions.",
 "currently": "Adam Smith is writing a book about the importance of creativity and how it can shape the world. Adam is also considering running for mayor.",
 "lifestyle": "Adam Smith goes to bed around 8pm, awakes up around 4am, eats dinner around 5pm.",
}
cands = {c["slot"]: c for c in json.load(open("pipelines/persona_pool/candidates.json"))}

def ask(prompt):
    body = {"model": "glm-5.3-flash:cloud", "stream": False, "think": False,
            "messages": [{"role": "user", "content": prompt}],
            "options": {"temperature": 0.6, "num_predict": 4000}}
    r = urllib.request.Request("http://127.0.0.1:11434/api/chat", json.dumps(body).encode(),
                               {"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(r))["message"]["content"]

out = {}
for slot, (home, hooks) in PICKS.items():
    c = cands[slot]
    first, last = c["name"].split(" ", 1) if " " in c["name"] else (c["name"], "")
    prompt = f"""You are writing a resident for a small simulated American town (Smallville: a cafe, pub, grocery, market and pharmacy, a college with a library, a park, a new clinic, town hall, police station and newspaper). Write them in exactly the style of this existing resident:
{json.dumps(EXAMPLE, ensure_ascii=False, indent=1)}

New resident (a synthetic persona, not a real person):
Name: {c['name']}, age {c['age']}, {c['sex']}, {c['marital_status']}, occupation: {c['occupation']}
Persona: {c['persona'][:600]}
Background: {c['cultural_background'][:300]}
Skills: {c['skills_and_expertise'][:250]}
Goals: {c['career_goals_and_ambitions'][:250]}
Role in Smallville: {slot}. Ties to existing residents: {hooks}.

Return ONLY JSON with keys innate (3 adjectives), learned (2 sentences: who they are and what they love, naming their job and where in Smallville they work), currently (2 sentences about what they are doing this week, naming at least two existing residents), lifestyle (bedtime, wake-up and dinner time, a realistic schedule for their job). Third person, use their first name and pronouns they/them only if gender is unclear, otherwise match the sex given."""
    j = None
    for attempt in range(4):
        try:
            txt = ask(prompt + "\nOutput the JSON object for the NEW resident only, as the last thing in your answer.")
            blocks = re.findall(r"\{[^{}]*\}", txt, re.S)
            for b in reversed(blocks):
                try:
                    cand = json.loads(b)
                except Exception:
                    continue
                if all(k in cand for k in EXAMPLE) and first in cand["learned"] and "Adam Smith" not in b:
                    j = cand; break
            if j: break
        except Exception as e:
            print("retry", slot, e)
    if j is None:
        print("fallback for", slot)
        j = {"innate": "friendly, reliable, curious",
             "learned": f"{c['name']} is a {slot} in Smallville.",
             "currently": f"{c['name']} is settling into town.",
             "lifestyle": f"{c['name']} goes to bed around 10pm, awakes up around 6am, eats dinner around 6pm."}
    out[c["name"]] = {"first": first, "last": last, "age": int(c["age"]), "slot": slot,
                      "living_area": f"the Ville:{home.split(':')[0]}:{home.split(':')[1]}",
                      **{k: j[k] for k in EXAMPLE}, "uuid": c["uuid"], "sex": c["sex"]}
    print(c["name"], "|", j["innate"], "|", j["learned"][:100], flush=True)
json.dump(out, open("pipelines/persona_pool/profiles.json", "w"), indent=1, ensure_ascii=False)
