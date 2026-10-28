#!/usr/bin/env python3
"""Build the_ville_n50: the existing 140x100 Smallville (coordinates unchanged)
plus a 200x130 canvas with new housing and civic buildings, made by stamping
copies of existing buildings and relabelling their sector/arena/spawn ids.
Writes to assets/the_ville_n50/ (the original map is never touched)."""
import collections, csv, json, os, shutil, sys
sys.path.insert(0, os.path.dirname(__file__))
from ville_map import Map, VIS

A = "/home/andus/Projects/generative_agents/environment/frontend_server/static_dirs/assets"
SRC, DST = f"{A}/the_ville", f"{A}/the_ville_n50"
W, H = 200, 130
FIRST_NEW = 40000            # new block gids (extra tilesets reuse blocks.png)

def read_blocks(f):
    rows = []
    for l in open(f"{SRC}/matrix/special_blocks/{f}"):
        a = [x.strip() for x in l.strip().split(",")]
        if len(a) >= 3:
            rows.append(a)
    return rows

sector_rows, arena_rows = read_blocks("sector_blocks.csv"), read_blocks("arena_blocks.csv")
object_rows, spawn_rows = read_blocks("game_object_blocks.csv"), read_blocks("spawning_location_blocks.csv")
SECT = {int(r[0]): r[2] for r in sector_rows}
ARE = {int(r[0]): (r[2], r[3]) for r in arena_rows}
SPN = {int(r[0]): (r[2], r[3], r[4]) for r in spawn_rows}

m = Map(f"{SRC}/visuals/the_ville.tmx")
grass = collections.Counter(m.layers["Bottom Ground"]).most_common(1)[0][0]
m.resize(W, H, fill=grass)
nid = [FIRST_NEW]
new_sector, new_arena, new_spawn = [], [], []

def alloc():
    nid[0] += 1
    return nid[0]

def clear(x, y, w, h):
    for n in m.order:
        if n == "Bottom Ground":
            continue
        for yy in range(h):
            for xx in range(w):
                m.put(n, x + xx, y + yy, 0)

def place(src, dst, sector, arena_names):
    """src=(x0,y0,x1,y1) source rect; dst=(dx,dy); arena_names: {old arena name: new name}."""
    x0, y0, x1, y1 = src
    w, h = x1 - x0 + 1, y1 - y0 + 1
    dx, dy = dst
    sid = alloc(); new_sector.append((sid, sector))
    amap, smap, spmap = {}, {}, {}
    for n in ("Sector Blocks", "Arena Blocks", "Spawning Blocks"):
        for yy in range(y0, y1 + 1):
            for xx in range(x0, x1 + 1):
                v = m.get(n, xx, yy)
                if not v:
                    continue
                if n == "Sector Blocks":
                    smap.setdefault(v, sid)            # every source sector id -> the new one
                elif n == "Arena Blocks":
                    if v not in ARE or ARE[v][1] not in arena_names:
                        amap.setdefault(v, 0)
                    else:
                        if v not in amap:
                            amap[v] = alloc(); new_arena.append((amap[v], sector, arena_names[ARE[v][1]]))
                else:
                    if v in SPN and SPN[v][1] in arena_names:
                        spmap[v] = alloc()
                        new_spawn.append((spmap[v], sector, arena_names[SPN[v][1]], SPN[v][2]))
                    else:
                        spmap[v] = 0
    clear(dx, dy, w, h)
    m.stamp(x0, y0, w, h, dx, dy, remap={"Sector Blocks": smap, "Arena Blocks": amap, "Spawning Blocks": spmap})
    return (dx, dy, w, h)

# source rects (tile coords in the original map)
DUPLEX = (66, 61, 83, 87)        # Moreno house: 2 bedrooms, kitchen, common room, garden
DUPLEX_AR = {"Tom and Jane Moreno's bedroom": "bedroom 1", "empty bedroom": "bedroom 2",
             "common room": "common room", "kitchen": "kitchen", "bathroom": "bathroom", "garden": "garden"}
COWORK = (11, 12, 46, 38)
COWORK_AR = {"Latoya Williams's room": "room 1", "Rajiv Patel's room": "room 2", "Abigail Chen's room": "room 3",
             "Francisco Lopez's room": "room 4", "Hailey Johnson's room": "room 5",
             "Latoya Williams's bathroom": "bathroom 1", "Rajiv Patel's bathroom": "bathroom 2",
             "Abigail Chen's bathroom": "bathroom 3", "Francisco Lopez's bathroom": "bathroom 4",
             "Hailey Johnson's bathroom": "bathroom 5", "kitchen": "kitchen", "common room": "common room"}
STUDIO = (17, 55, 26, 73)        # Adam Smith's house: one room + bathroom
CLINIC = (104, 37, 135, 64)      # the dorm: 4 rooms, common room, kitchen, garden
CLINIC_AR = {"Klaus Mueller's room": "exam room 1", "Maria Lopez's room": "exam room 2",
             "Wolfgang Schulz's room": "exam room 3", "Ayesha Khan's room": "exam room 4",
             "common room": "waiting room", "kitchen": "staff room", "garden": "healing garden",
             "man's bathroom": "restroom", "woman's bathroom": "staff restroom"}
HALL = (105, 17, 127, 36)        # the college: two big rooms + hall
HALL_AR = {"classroom": "council chamber", "library": "records office", "hallway": "lobby"}

placed = {}
for i, x in enumerate((4, 24, 44, 64, 92, 112, 132, 152)):
    placed[f"Willow Court {i+1}"] = place(DUPLEX, (x, 101), f"Willow Court {i+1}", DUPLEX_AR)
placed["Maple Co-living"] = place(COWORK, (150, 70), "Maple Co-living", COWORK_AR)
placed["Willow Clinic"] = place(CLINIC, (144, 8), "Willow Clinic", CLINIC_AR)
placed["Town Hall"] = place(HALL, (144, 46), "Town Hall", HALL_AR)
placed["Police Station"] = place(STUDIO, (178, 8), "Police Station", {"main room": "front desk", "bathroom": "restroom"})
placed["Fire and Rescue"] = place(STUDIO, (178, 28), "Fire and Rescue", {"main room": "station floor", "bathroom": "restroom"})
placed["Smallville Gazette"] = place(STUDIO, (178, 48), "Smallville Gazette", {"main room": "newsroom", "bathroom": "restroom"})

# ---- dirt roads (the original top-centre strip's tile, Desert_C 9598) -----------
DIRT = 9598
ROADS = [(122, 37, 20, 3),      # east seam through the old forest edge
         (99, 78, 4, 24),       # south seam from the lower houses
         (2, 98, 174, 2),       # Willow Court street
         (140, 6, 3, 94),       # east avenue
         (140, 6, 38, 2),       # north spur to the studios
         (176, 6, 2, 64),       # studios' lane
         (140, 44, 22, 2), (140, 67, 14, 2), (150, 96, 30, 2)]   # clinic / town hall / co-living
def road(x, y, w, h):
    for yy in range(h):
        for xx in range(w):
            for n in m.order:
                if n in ("Bottom Ground", "Exterior Ground", "Sector Blocks", "Arena Blocks",
                         "Spawning Blocks", "Object Interaction Blocks"):
                    continue
                m.put(n, x + xx, y + yy, 0)
            m.put("Exterior Ground", x + xx, y + yy, 0)
            m.put("Exterior Decoration L2", x + xx, y + yy, DIRT)
for r in ROADS:
    road(*r)

# ---- outputs --------------------------------------------------------------
if os.path.exists(DST):
    shutil.rmtree(DST)
shutil.copytree(SRC, DST, ignore=shutil.ignore_patterns("*.html", "visuals"))
os.makedirs(f"{DST}/visuals", exist_ok=True)
os.symlink("../../the_ville/visuals/map_assets", f"{DST}/visuals/map_assets")
m.save_tmx(f"{DST}/visuals/the_ville_n50.tmx")
m.save_json(f"{SRC}/visuals/the_ville_jan7.json", f"{DST}/visuals/the_ville_n50.json")
# matrices: original region verbatim from the shipped csv (they differ from the tmx in 100 hand-patched
# cells), new region from the new map
def old_grid(f):
    t = [x.strip() for x in open(f"{SRC}/matrix/maze/{f}.csv").read().replace("\n", ",").split(",") if x.strip()]
    return [t[i * 140:(i + 1) * 140] for i in range(100)]
from ville_map import SPECIAL
for layer, f in SPECIAL.items():
    g = old_grid(f)
    out = []
    for y in range(H):
        for x in range(W):
            carved = any(rx <= x < rx + rw and ry <= y < ry + rh for rx, ry, rw, rh in ROADS)
            if x < 140 and y < 100 and not carved:
                out.append(g[y][x])
            else:
                v = m.get(layer, x, y)
                out.append(str(v & 0x0FFFFFFF if layer == "Collisions" else v))
    open(f"{DST}/matrix/maze/{f}.csv", "w").write(",".join(out))
meta = json.load(open(f"{SRC}/matrix/maze_meta_info.json")); meta["maze_width"], meta["maze_height"] = W, H
meta["world_name"] = "the ville"
json.dump(meta, open(f"{DST}/matrix/maze_meta_info.json", "w"))
sb = f"{DST}/matrix/special_blocks"
def app(fn, rows):
    txt = open(f"{sb}/{fn}").read()
    with open(f"{sb}/{fn}", "a") as f:
        if txt and not txt.endswith("\n"):
            f.write("\n")
        for r in rows:
            f.write(", ".join(map(str, r)) + "\n")
app("sector_blocks.csv", [(i, "the Ville", n) for i, n in new_sector])
app("arena_blocks.csv", [(i, "the Ville", s, a) for i, s, a in new_arena])
app("spawning_location_blocks.csv", [(i, "the Ville", s, a, p) for i, s, a, p in new_spawn])
json.dump(placed, open(f"{DST}/placed.json", "w"), indent=1)
print("sectors", len(new_sector), "arenas", len(new_arena), "spawns", len(new_spawn), "->", DST)
