#!/usr/bin/env python3
"""Sprites for the new residents: recolour (clothes/hair hues; skin hues kept)
an existing sprite of similar age/sex so no two residents look identical."""
import colorsys, json, os
from PIL import Image
A = "/home/andus/Projects/generative_agents/environment/frontend_server/static_dirs/assets/characters"
BASE = {"Chad Kibler": ("Ryan_Park", 150), "Andrea Lovetro": ("Jane_Moreno", 200), "Sharon Antle": ("Mei_Lin", 300),
        "Amilcar Rosales": ("Sam_Moore", 90), "Hua Lai": ("Carlos_Gomez", 220), "Lesbia Martinez": ("Jennifer_Moore", 120),
        "Carl Demissie": ("Tom_Moreno", 180), "Ethel Langley": ("Carmen_Ortiz", 260),
        "Michael Poirier": ("Eddy_Lin", 330), "Maria Urban": ("Tamara_Taylor", 40)}

BASE2 = {"Tony Stewart": ("Ryan_Park", 240), "Cledith Hardin": ("Adam_Smith", 100), "Eugene Johnson": ("Klaus_Mueller", 260),
         "David Puskar": ("Giorgio_Rossi", 300), "Ana Abad": ("Abigail_Chen", 210), "Dalton Robinson": ("John_Lin", 80),
         "Job Amely": ("Arthur_Burton", 160), "Jeanne Love": ("Jane_Moreno", 120), "Eugene Middlebrooks": ("Rajiv_Patel", 20),
         "Jason O'Brien": ("Sam_Moore", 200), "Jenny Rivera": ("Maria_Lopez", 150), "Nair Viggiani": ("Francisco_Lopez", 270),
         "Victor Rosales": ("Sam_Moore", 320), "Amber Williamson": ("Jennifer_Moore", 250), "Diane Bosse": ("Hailey_Johnson", 90)}
import sys as _sys
if "--group2" in _sys.argv:
    BASE = BASE2

def recolor(im, deg):
    px = im.load()
    for y in range(im.height):
        for x in range(im.width):
            r, g, b, a = px[x, y]
            if a == 0:
                continue
            h, s, v = colorsys.rgb_to_hsv(r / 255, g / 255, b / 255)
            if s < 0.18 or (0.0 <= h * 360 <= 55):      # greys/blacks and skin-to-orange stay
                continue
            h = (h + deg / 360) % 1
            r, g, b = colorsys.hsv_to_rgb(h, s, v)
            px[x, y] = (int(r * 255), int(g * 255), int(b * 255), a)
    return im

for name, (src, deg) in BASE.items():
    dst = name.replace(" ", "_")
    for sub in ("", "profile/"):
        im = Image.open(f"{A}/{sub}{src}.png").convert("RGBA")
        recolor(im, deg).save(f"{A}/{sub}{dst}.png")
    print(name, "<-", src, deg)
