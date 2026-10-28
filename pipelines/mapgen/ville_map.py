"""Load / edit / save the Smallville Tiled map (csv layers) and export the
backend matrices + Phaser JSON. Layers are dicts name -> flat list of gids."""
import json, os, re, csv

VIS = ["Bottom Ground", "Exterior Ground", "Exterior Decoration L1", "Exterior Decoration L2",
       "Interior Ground", "Wall", "Interior Furniture L1", "Interior Furniture L2",
       "Foreground L1", "Foreground L2"]
SPECIAL = {"Collisions": "collision_maze", "Object Interaction Blocks": "game_object_maze",
           "Arena Blocks": "arena_maze", "Sector Blocks": "sector_maze",
           "Spawning Blocks": "spawning_location_maze"}
DATA = re.compile(r'(<layer id="\d+" name="([^"]*)"[^>]*>\s*<data encoding="csv">)(.*?)(</data>)', re.S)


class Map:
    def __init__(self, tmx):
        self.path = tmx
        self.text = open(tmx, encoding="utf-8").read()
        m = re.search(r'<map [^>]*width="(\d+)" height="(\d+)"', self.text)
        self.w, self.h = int(m.group(1)), int(m.group(2))
        self.layers = {}
        self.order = []
        for m in DATA.finditer(self.text):
            name = m.group(2).strip()
            g = [int(x) for x in re.split(r"[,\s]+", m.group(3).strip()) if x]
            assert len(g) == self.w * self.h, (name, len(g))
            self.layers[name] = g
            self.order.append(name)

    # -- geometry ----------------------------------------------------------
    def resize(self, nw, nh, fill=None):
        """Grow the canvas (top-left anchored). Bottom Ground is filled with `fill`."""
        for n, g in self.layers.items():
            ng = [0] * (nw * nh)
            for y in range(self.h):
                ng[y * nw:y * nw + self.w] = g[y * self.w:(y + 1) * self.w]
            if n == "Bottom Ground" and fill:
                for y in range(nh):
                    for x in range(nw):
                        if x >= self.w or y >= self.h:
                            ng[y * nw + x] = fill
            self.layers[n] = ng
        self.w, self.h = nw, nh

    def get(self, layer, x, y):
        return self.layers[layer][y * self.w + x]

    def put(self, layer, x, y, v):
        self.layers[layer][y * self.w + x] = v

    def stamp(self, sx, sy, sw, sh, dx, dy, remap=None, layers=None, skip_zero=True):
        """Copy a rectangle (all layers) to (dx,dy). remap: {layer: {gid: gid}} for special ids."""
        remap = remap or {}
        for n in (layers or self.order):
            rm = remap.get(n, {})
            for yy in range(sh):
                for xx in range(sw):
                    v = self.get(n, sx + xx, sy + yy)
                    if skip_zero and v == 0 and n != "Bottom Ground":
                        continue
                    self.put(n, dx + xx, dy + yy, rm.get(v, v))

    # -- output ----------------------------------------------------------
    EXTRA_SETS = [("blocks_n50_a", 40001), ("blocks_n50_b", 40081)]   # new special-block gids

    def save_tmx(self, out):
        t = self.text
        extra = "".join(
            f' <tileset firstgid="{fg}" name="{nm}" tilewidth="32" tileheight="32" tilecount="80" columns="10">\n'
            f'  <image source="map_assets/blocks/blocks_1.png" trans="ff00ff" width="320" height="256"/>\n </tileset>\n'
            for nm, fg in self.EXTRA_SETS)
        t = t.replace(" <layer ", extra + " <layer ", 1)
        t = re.sub(r'(<map [^>]*?)width="\d+" height="\d+"', rf'\1width="{self.w}" height="{self.h}"', t, 1)
        t = re.sub(r'(<layer id="\d+" name="[^"]*" )width="\d+" height="\d+"', rf'\1width="{self.w}" height="{self.h}"', t)

        def rep(m):
            name = m.group(2).strip()
            rows = []
            g = self.layers[name]
            for y in range(self.h):
                rows.append(",".join(map(str, g[y * self.w:(y + 1) * self.w])))
            return m.group(1) + "\n" + ",\n".join(rows) + "\n" + m.group(4)
        t = DATA.sub(rep, t)
        open(out, "w", encoding="utf-8").write(t)

    def save_json(self, ref_json, out):
        d = json.load(open(ref_json))
        d["width"], d["height"] = self.w, self.h
        for nm, fg in self.EXTRA_SETS:
            d["tilesets"].append({"columns": 10, "firstgid": fg, "image": "map_assets/blocks/blocks_1.png",
                                  "imageheight": 256, "imagewidth": 320, "margin": 0, "name": nm,
                                  "spacing": 0, "tilecount": 80, "tileheight": 32, "tilewidth": 32,
                                  "transparentcolor": "#ff00ff"})
        for ly in d["layers"]:
            n = ly["name"].strip()
            if n in self.layers:
                ly["data"] = self.layers[n]
                ly["width"], ly["height"] = self.w, self.h
        json.dump(d, open(out, "w"))

    def save_matrices(self, folder):
        os.makedirs(folder, exist_ok=True)
        for layer, fname in SPECIAL.items():
            open(os.path.join(folder, fname + ".csv"), "w").write(",".join(map(str, self.layers[layer])))
