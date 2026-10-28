#!/usr/bin/env python3
"""Pick 25 candidates (one per slot) from persona_pool/pool.jsonl -> candidates.json"""
import json, random, re
rows = [json.loads(l) for l in open("pipelines/persona_pool/pool.jsonl")]
# slot, tie to an existing agent, occupation codes, age lo-hi, sex (None = any)
SLOTS = [
 ("family doctor", "John Lin (pharmacist)", ["physician"], 35, 60),
 ("paramedic", "Dr. above; Ryan/others in emergencies", ["paramedic", "emergency_medical_technician"], 28, 45),
 ("police officer", "Carlos Gomez (loud, rude)", ["police_officer"], 28, 48),
 ("college lecturer", "Ayesha Khan, Klaus Mueller", ["postsecondary_teacher"], 40, 58),
 ("college librarian", "Oak Hill library; Ayesha", ["librarian_or_media_collections_specialist", "library_technician", "library_assistant_clerical"], 40, 66),
 ("teen (Moreno spare room)", "Tom and Jane Moreno", [None], 14, 17),
 ("baker", "Isabella Rodriguez (Hobbs Cafe)", ["baker"], 28, 55),
 ("barista", "Isabella Rodriguez", ["cook", "waiter_or_waitress", "cashier"], 22, 35),
 ("night cook", "Arthur Burton (pub)", ["cook", "chef_or_head_cook"], 24, 40),
 ("pub waitress", "Arthur Burton", ["waiter_or_waitress", "bartender"], 21, 30),
 ("market gardener", "Carmen Ortiz / John Lin (produce)", ["farmer_rancher_or_agricultural_manager"], 55, 72),
 ("delivery driver", "Tom Moreno (delivery)", ["courier_or_messenger", "driver_sales_worker_or_truck_driver"], 35, 55),
 ("handyman / plumber", "Moore family, Sam Moore", ["plumber_pipefitter_or_steamfitter", "maintenance_or_repair_worker_general"], 45, 62),
 ("mechanic", "Tom Moreno", ["automotive_service_technician_or_mechanic"], 38, 56),
 ("junior accountant", "Yuriko Yamamoto / Ryan Park", ["accountant_or_auditor", "bookkeeping_accounting_or_auditing_clerk"], 23, 31),
 ("newspaper editor", "Adam Smith, Hailey Johnson (writer)", ["editor", "writer_or_author"], 45, 64),
 ("community leader / mayor", "Yuriko Yamamoto (law), Mei Lin", ["social_or_community_service_manager", "chief_executive", "general_or_operations_manager"], 48, 62),
 ("retired contractor (challenger)", "Sam Moore, Tom Moreno", ["carpenter", "construction_laborer", "electrician"], 56, 68),
 ("chaplain / counsellor", "Jane Moreno, Klaus (social justice)", ["clergy", "mental_health_counselor", "counselor"], 50, 66),
 ("fitness trainer", "Wolfgang Schulz (athlete)", ["exercise_trainer_or_group_fitness_instructor"], 26, 40),
 ("piano teacher / musician", "Eddy Lin (music)", ["musician_or_singer", "music_director_or_composer"], 34, 50),
 ("retired nurse (elder)", "Jennifer Moore, Sam Moore", ["registered_nurse", None], 72, 86),
 ("retired elder", "Sam Moore", [None], 68, 80),
 ("newcomer job seeker", "hiring market (Isabella, Arthur)", ["no_occupation", "not_in_workforce"], 25, 36),
 ("hairdresser", "Jane Moreno, Mei Lin", ["hairdresser_hairstylist_or_cosmetologist", "barber"], 36, 55),
]
rnd = random.Random(11)
taken, out = set(), []
NAME = re.compile(r"^((?:Dr\.?\s)?[A-Z][\w'’\-]+(?:\s[A-Z]\.)?\s[A-Z][\w'’\-]+)")
for slot, tie, occs, lo, hi in SLOTS:
    pool = [r for r in rows if lo <= int(r["age"]) <= hi and r["uuid"] not in taken
            and any(o is None or r["occupation"] == o for o in occs)
            and (None not in occs or r["occupation"] in ("not_in_workforce", "no_occupation") or len(occs) > 1)]
    if None in occs and len(occs) == 1:
        pool = [r for r in pool if r["occupation"] in ("not_in_workforce", "no_occupation")]
    rnd.shuffle(pool)
    pool.sort(key=lambda r: abs(len(r["professional_persona"]) - 500))
    if not pool:
        print("NO MATCH", slot); continue
    r = pool[0]; taken.add(r["uuid"])
    m = NAME.match(r["persona"]) or NAME.match(r["professional_persona"])
    out.append({"slot": slot, "name": m.group(1) if m else "?", "tie": tie,
                **{k: r[k] for k in ("uuid", "age", "sex", "occupation", "marital_status", "education_level", "persona", "professional_persona", "cultural_background", "hobbies_and_interests", "skills_and_expertise", "career_goals_and_ambitions")}})
json.dump(out, open("pipelines/persona_pool/candidates.json", "w"), indent=1, ensure_ascii=False)
for c in out:
    print(f"{c['slot']:30s}| {c['name']:24s}| {c['age']:>2} {c['sex'][0]} {c['occupation'][:30]:30s}| {c['persona'][:110]}")
