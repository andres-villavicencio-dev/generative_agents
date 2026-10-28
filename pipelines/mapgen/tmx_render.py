#!/usr/bin/env python3
"""Render a Tiled TMX (csv layers) to PNG with PIL. Usage: tmx_render.py in.tmx out.png [scale] [layer,...]"""
import re, sys, os
import xml.etree.ElementTree as ET
from PIL import Image

def load(tmx):
    root = ET.parse(tmx).getroot()
    W, H = int(root.get("width")), int(root.get("height"))
    base = os.path.dirname(tmx)
    sets = []
    for ts in root.findall("tileset"):
        im = ts.find("image")
        sets.append((int(ts.get("firstgid")), int(ts.get("columns")), int(ts.get("tilecount")),
                     os.path.join(base, im.get("source"))))
    layers = {}
    for ly in root.findall("layer"):
        data = ly.find("data").text
        g = [int(x) for x in re.split(r"[,\s]+", data.strip()) if x]
        layers[ly.get("name").strip()] = g
    return W, H, sets, layers

def render(tmx, out, scale=1.0, names=None):
    W, H, sets, layers = load(tmx)
    sets.sort()
    cache = {}
    def tile(gid):
        for fg, cols, cnt, path in reversed(sets):
            if gid >= fg:
                if path not in cache:
                    cache[path] = Image.open(path).convert("RGBA")
                i = gid - fg
                x, y = (i % cols) * 32, (i // cols) * 32
                return cache[path].crop((x, y, x + 32, y + 32))
    img = Image.new("RGBA", (W * 32, H * 32), (30, 60, 30, 255))
    order = names or ["Bottom Ground", "Exterior Ground", "Exterior Decoration L1", "Exterior Decoration L2",
                      "Interior Ground", "Wall", "Interior Furniture L1", "Interior Furniture L2",
                      "Foreground L1", "Foreground L2"]
    for n in order:
        g = layers.get(n)
        if not g:
            continue
        for i, gid in enumerate(g):
            if gid:
                t = tile(gid & 0x0FFFFFFF)
                if t:
                    img.alpha_composite(t, ((i % W) * 32, (i // W) * 32))
    if scale != 1.0:
        img = img.resize((int(img.width * scale), int(img.height * scale)), Image.LANCZOS)
    img.convert("RGB").save(out)

if __name__ == "__main__":
    render(sys.argv[1], sys.argv[2], float(sys.argv[3]) if len(sys.argv) > 3 else 0.25,
           sys.argv[4].split(",") if len(sys.argv) > 4 else None)
