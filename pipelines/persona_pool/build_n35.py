#!/usr/bin/env python3
"""Build storage/verify_n35: a copy of verify_1 (25 agents, step 17280) moved to the
200x130 map (maze_name the_ville_n50) with 10 new residents from profiles.json.
Usage: build_n35.py [--force]"""
import copy, json, os, shutil, subprocess, sys
REPO = "/home/andus/Projects/generative_agents"
ST = f"{REPO}/environment/frontend_server/storage"
SRC, DST, TEMPLATE = f"{ST}/verify_1", f"{ST}/verify_n35", f"{ST}/base_the_ville_n25/personas/Adam Smith"
sys.path.insert(0, f"{REPO}/reverie/backend_server")
import maze as M

PUBLIC = ["Oak Hill College", "The Willows Market and Pharmacy", "Harvey Oak Supply Store",
          "Johnson Park", "The Rose and Crown Pub", "Hobbs Cafe"]
CIVIC = ["Willow Clinic", "Town Hall", "Police Station", "Fire and Rescue", "Smallville Gazette"]

mz = M.Maze("the_ville_n50")
objs = {}                       # (sector, arena) -> [objects]
spawn = {}                      # (sector, arena) -> (x, y)
for y in range(mz.maze_height):
    for x in range(mz.maze_width):
        t = mz.tiles[y][x]
        if not (t["sector"] and t["arena"]):
            continue
        k = (t["sector"], t["arena"])
        if t["game_object"] and t["game_object"] not in objs.setdefault(k, []):
            objs[k].append(t["game_object"])
        objs.setdefault(k, [])
        if not t["collision"]:
            if t["spawning_location"] or k not in spawn:
                if t["spawning_location"] or not spawn.get(k):
                    spawn[k] = (x, y)

def sector_tree(sec):
    return {a: objs[(s, a)] for (s, a) in objs if s == sec}

if "--force" in sys.argv and os.path.exists(DST):
    shutil.rmtree(DST)
if not os.path.exists(DST):
    subprocess.run(["cp", "-a", SRC, DST], check=True)
profiles = json.load(open(f"{REPO}/pipelines/persona_pool/profiles.json"))

# existing residents learn the new civic buildings
for name in os.listdir(f"{DST}/personas"):
    p = f"{DST}/personas/{name}/bootstrap_memory/spatial_memory.json"
    sm = json.load(open(p))
    for sec in CIVIC:
        sm["the Ville"][sec] = sector_tree(sec)
    json.dump(sm, open(p, "w"))

env_p = f"{DST}/environment/17280.json"
mv_p = f"{DST}/movement/17279.json"
env, mv = json.load(open(env_p)), json.load(open(mv_p))
meta_p = f"{DST}/reverie/meta.json"
meta = json.load(open(meta_p))
for name, pr in profiles.items():
    home_sec, home_ar = pr["living_area"].split(":")[1:3]
    folder = f"{DST}/personas/{name}/bootstrap_memory"
    if os.path.exists(f"{DST}/personas/{name}"):
        shutil.rmtree(f"{DST}/personas/{name}")
    os.makedirs(folder)
    shutil.copytree(f"{TEMPLATE}/bootstrap_memory/associative_memory", f"{folder}/associative_memory") \
        if os.path.exists(f"{TEMPLATE}/bootstrap_memory/associative_memory") else None
    sc = json.load(open(f"{TEMPLATE}/bootstrap_memory/scratch.json"))
    sc.update({"name": name, "first_name": pr["first"], "last_name": pr["last"], "age": pr["age"],
               "innate": pr["innate"], "learned": pr["learned"], "currently": pr["currently"],
               "lifestyle": pr["lifestyle"], "living_area": pr["living_area"],
               "curr_time": None, "curr_tile": None, "daily_plan_req": "",
               "daily_req": [], "f_daily_schedule": [], "f_daily_schedule_hourly_org": [],
               "act_event": [name, None, None]})
    json.dump(sc, open(f"{folder}/scratch.json", "w"), indent=1)
    sm = {"the Ville": {}}
    for sec in PUBLIC + CIVIC + [home_sec]:
        t = sector_tree(sec)
        if t:
            sm["the Ville"][sec] = t
    json.dump(sm, open(f"{folder}/spatial_memory.json", "w"))
    x, y = spawn[(home_sec, home_ar)]
    env[name] = {"maze": "the_ville_n50", "x": x, "y": y}
    mv["persona"][name] = {"movement": [x, y], "pronunciatio": "😴",
                           "description": f"sleeping @ {pr['living_area']}:bed", "chat": None}
    if name not in meta["persona_names"]:
        meta["persona_names"].append(name)
    print(f"{name:20s} home {pr['living_area']}  spawn {(x, y)}")
for k in env:
    env[k]["maze"] = "the_ville_n50"
meta["maze_name"] = "the_ville_n50"
meta["fork_sim_code"] = "verify_1"
json.dump(env, open(env_p, "w")); json.dump(mv, open(mv_p, "w")); json.dump(meta, open(meta_p, "w"), indent=2)
print("personas now", len(meta["persona_names"]))
