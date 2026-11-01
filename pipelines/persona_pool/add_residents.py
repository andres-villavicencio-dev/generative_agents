#!/usr/bin/env python3
"""Add new residents (profiles JSON) in place to a finished sim at its last step,
so the next fork carries them. They start with an empty memory, at their bed.
Usage: add_residents.py <sim> <profiles.json> [<step>]  (step defaults to meta.json step)"""
import json, os, shutil, sys
REPO = "/home/andus/Projects/generative_agents"
ST = f"{REPO}/environment/frontend_server/storage"
sim, prof = sys.argv[1], sys.argv[2]
DST = f"{ST}/{sim}"
meta_p = f"{DST}/reverie/meta.json"
meta = json.load(open(meta_p))
step = int(sys.argv[3]) if len(sys.argv) > 3 else meta["step"]
MAZE = meta["maze_name"]
TEMPLATE = f"{ST}/base_the_ville_n25/personas/Adam Smith"
sys.path.insert(0, f"{REPO}/reverie/backend_server")
import maze as M

PUBLIC = ["Oak Hill College", "The Willows Market and Pharmacy", "Harvey Oak Supply Store",
          "Johnson Park", "The Rose and Crown Pub", "Hobbs Cafe"]
CIVIC = ["Willow Clinic", "Town Hall", "Police Station", "Fire and Rescue", "Smallville Gazette"]
mz = M.Maze(MAZE)
objs, spawn = {}, {}
for y in range(mz.maze_height):
    for x in range(mz.maze_width):
        t = mz.tiles[y][x]
        if not (t["sector"] and t["arena"]):
            continue
        k = (t["sector"], t["arena"])
        objs.setdefault(k, [])
        if t["game_object"] and t["game_object"] not in objs[k]:
            objs[k].append(t["game_object"])
        if not t["collision"] and (t["spawning_location"] or k not in spawn):
            spawn[k] = (x, y)
tree = lambda sec: {a: objs[(s, a)] for (s, a) in objs if s == sec}

profiles = json.load(open(prof))
env_p, mv_p = f"{DST}/environment/{step}.json", f"{DST}/movement/{step - 1}.json"
env, mv = json.load(open(env_p)), json.load(open(mv_p))
maze_key = next(iter(env.values()))["maze"]
for name, pr in profiles.items():
    if name in meta["persona_names"]:
        print("skip (exists)", name); continue
    hs, ha = pr["living_area"].split(":")[1:3]
    folder = f"{DST}/personas/{name}/bootstrap_memory"
    os.makedirs(folder)
    if os.path.exists(f"{TEMPLATE}/bootstrap_memory/associative_memory"):
        shutil.copytree(f"{TEMPLATE}/bootstrap_memory/associative_memory", f"{folder}/associative_memory")
    sc = json.load(open(f"{TEMPLATE}/bootstrap_memory/scratch.json"))
    sc.update({"name": name, "first_name": pr["first"], "last_name": pr["last"], "age": pr["age"],
               "innate": pr["innate"], "learned": pr["learned"], "currently": pr["currently"],
               "lifestyle": pr["lifestyle"], "living_area": pr["living_area"],
               "curr_time": None, "curr_tile": None, "daily_plan_req": "",
               "daily_req": [], "f_daily_schedule": [], "f_daily_schedule_hourly_org": [],
               "act_event": [name, None, None]})
    json.dump(sc, open(f"{folder}/scratch.json", "w"), indent=1)
    sm = {"the Ville": {}}
    for sec in PUBLIC + CIVIC + [hs]:
        if tree(sec):
            sm["the Ville"][sec] = tree(sec)
    json.dump(sm, open(f"{folder}/spatial_memory.json", "w"))
    x, y = spawn[(hs, ha)]
    env[name] = {"maze": maze_key, "x": x, "y": y}
    mv["persona"][name] = {"movement": [x, y], "pronunciatio": "😴",
                           "description": f"sleeping @ {pr['living_area']}:bed", "chat": None}
    meta["persona_names"].append(name)
    print(f"{name:20s} home {pr['living_area']}  spawn {(x, y)}")
json.dump(env, open(env_p, "w")); json.dump(mv, open(mv_p, "w"))
json.dump(meta, open(meta_p, "w"), indent=2)
print("personas now", len(meta["persona_names"]))
