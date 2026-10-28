#!/usr/bin/env python3
"""Which sectors/arenas of the_ville_n50 are walkable-reachable from Hobbs Cafe."""
import sys, collections
sys.path.insert(0, "/home/andus/Projects/generative_agents/reverie/backend_server")
import maze as M
M.env_matrix = "/home/andus/Projects/generative_agents/environment/frontend_server/static_dirs/assets/the_ville_n50/matrix"
mz = M.Maze("the_ville_n50")
W, H = mz.maze_width, mz.maze_height
start = next((x, y) for y in range(H) for x in range(W)
             if mz.tiles[y][x]["sector"] == "Hobbs Cafe" and not mz.tiles[y][x]["collision"])
seen = {start}; q = [start]
while q:
    x, y = q.pop()
    for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        nx, ny = x + dx, y + dy
        if 0 <= nx < W and 0 <= ny < H and (nx, ny) not in seen and not mz.tiles[ny][nx]["collision"]:
            seen.add((nx, ny)); q.append((nx, ny))
tot, ok = collections.Counter(), collections.Counter()
for y in range(H):
    for x in range(W):
        t = mz.tiles[y][x]
        if t["arena"]:
            tot[t["sector"]] += 1
            ok[t["sector"]] += (x, y) in seen
bad = {s: (ok[s], tot[s]) for s in tot if ok[s] < tot[s] * 0.95}
print("sectors", len(tot), "not fully reachable:", bad)
if len(sys.argv) > 1:
    x0, y0, x1, y1 = map(int, sys.argv[1:5])
    for y in range(y0, y1):
        print(f"{y:3d} " + "".join("o" if (x, y) in seen else ("#" if mz.tiles[y][x]["collision"] else ".") for x in range(x0, x1)))
