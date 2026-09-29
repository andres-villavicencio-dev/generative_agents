#!/usr/bin/env python3
"""
char3d_pipeline.py — 3D -> 2D character sprite pipeline for the GA sim.

Replaces the single shared "misa" sprite with a unique low-poly 3D
character per persona, rendered to 4-direction walk-cycle sprite sheets.

Stages:
  1. source  - procedural low-poly humanoid (deterministic palette/height/
               hair from persona-name hash), OR a Meshy export dropped at
               pipelines/meshy_imports/<Persona_Name>.glb (overrides the
               procedural body for that persona; imports keep their own
               materials/textures, animated with body bob until rigged
               clips are wired).
  2. pose    - walk cycle: thigh/shin/arm swing + body bob, 4 keyframes.
  3. render  - pyrender EGL, orthographic camera sized per character,
               flat-shaded procedural colors (or lit textured imports),
               transparent background.
  4. pack    - home-route sheet: assets/characters_3d/<Name>.png with
               32x64 frames (5 cols x 4 rows: 4 walk frames + idle per
               direction) + shared Hash-format atlas.json with frame names
               <dir>-walk.000..003 and <dir> idles.

Directions (char authored facing -Z, camera fixed due-south):
  front yaw 0 | back 180 | left -90 | right +90

Usage:
  venv/bin/python pipelines/char3d_pipeline.py --all
  venv/bin/python pipelines/char3d_pipeline.py --personas "Klaus Mueller,Maria Lopez"
"""
import argparse
import colorsys
import json
import math
import os
import random
import zlib

import numpy as np
import trimesh

os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
import pyrender  # noqa: E402
from PIL import Image, ImageEnhance  # noqa: E402

BASE = os.path.expanduser("~/Projects/generative_agents")
FRONTEND = os.path.join(BASE, "environment/frontend_server")
OUT_DIR = os.path.join(FRONTEND, "static_dirs/assets/characters_3d")
MESHY_DIR = os.path.join(BASE, "pipelines/meshy_imports")
CURR_SIM = os.path.join(FRONTEND, "temp_storage/curr_sim_code.json")

# Feet sit ~3px above the 64px frame bottom (camera headroom). Paste rows
# shifted down so soles land on the frame bottom edge = tile ground line.
PASTE_DY = 3

FRAME_W, FRAME_H = 32, 64
RENDER_W, RENDER_H = FRAME_W * 4, FRAME_H * 4
ROWS = ["front", "back", "left", "right"]
YAWS = {"front": 0.0, "back": 180.0, "left": -90.0, "right": 90.0}


def hsl(h, s, l):
    r, g, b = colorsys.hls_to_rgb(h / 360.0, l / 360.0, s / 360.0)
    return (r, g, b)


def style_for(name):
    """Deterministic per-persona look from the name hash."""
    rng = random.Random(zlib.crc32(name.encode()))
    shirt_h = rng.uniform(0, 360)
    shirt = hsl(shirt_h, rng.uniform(150, 220), rng.uniform(38, 58))
    pants_h = (shirt_h + rng.uniform(120, 240)) % 360
    pants = hsl(pants_h, rng.uniform(80, 180), rng.uniform(18, 40))
    skins = [(38, 140, 80), (32, 120, 66), (27, 110, 50),
             (20, 95, 38), (42, 90, 86)]
    skin = hsl(*rng.choice(skins))
    hair = hsl(rng.uniform(15, 55), rng.uniform(10, 55), rng.uniform(8, 30))
    shoe = hsl(25, 30, 12)
    return {
        "H": 1.7 * rng.uniform(0.94, 1.06),
        "shirt": shirt, "pants": pants, "skin": skin,
        "hair": hair, "shoe": shoe,
        "hair_style": rng.choice(["crop", "bob", "long", "bun"]),
    }


def rot_x_deg(deg):
    return trimesh.transformations.rotation_matrix(math.radians(deg), [1, 0, 0])


def rot_y_deg(deg):
    return trimesh.transformations.rotation_matrix(math.radians(deg), [0, 1, 0])


def rx_vec(deg, v):
    return (rot_x_deg(deg)[:3, :3] @ np.array(v)).tolist()


def part(extents, color, pivot, offset, rx=0.0, ry=0.0):
    """One rigged body part: box rotated about a pivot, then offset."""
    return {"extents": extents, "color": color, "pivot": pivot,
            "offset": offset, "rx": rx, "ry": ry}


def build_parts(style, phase=None):
    """Pose the procedural rig. phase=None -> standing idle."""
    H = style["H"]
    hip_y = 0.46 * H
    thigh_len, shin_len = 0.23 * H, 0.23 * H
    torso_len = 0.34 * H
    arm_len, fore_len = 0.16 * H, 0.15 * H

    if phase is None:
        bob = 0.0
        th_l = th_r = 2.0
        kn_l = kn_r = 4.0
        ua_l, ua_r = 4.0, -4.0
        head_sway = 0.0
    else:
        p = phase
        bob = 0.028 * H * (0.5 - 0.5 * math.cos(2 * p))
        th_l, th_r = 30 * math.sin(p), -30 * math.sin(p)
        kn_l = 42 * max(0.0, math.sin(p + 1.2))
        kn_r = 42 * max(0.0, math.sin(p + 1.2 + math.pi))
        ua_l, ua_r = -28 * math.sin(p), 28 * math.sin(p)
        head_sway = 3 * math.sin(p + math.pi / 2)

    parts = []
    for s, th, kn in ((1, th_l, kn_l), (-1, th_r, kn_r)):
        pv = [0.055 * H * s, hip_y + bob, 0]
        parts.append(part([0.075 * H, thigh_len, 0.085 * H], style["pants"],
                          pv, [0, -thigh_len / 2, 0], rx=th))
        knee = [pv[i] + rx_vec(th, [0, -thigh_len, 0])[i] for i in range(3)]
        parts.append(part([0.065 * H, shin_len, 0.075 * H], style["pants"],
                          knee, [0, -shin_len / 2, 0], rx=th - kn))
        ankle = [knee[i] + rx_vec(th - kn, [0, -shin_len, 0])[i]
                 for i in range(3)]
        parts.append(part([0.08 * H, 0.035 * H, 0.13 * H], style["shoe"],
                          ankle, [0, -0.017 * H, -0.02 * H], rx=th - kn))
    for s, ua in ((1, ua_l), (-1, ua_r)):
        pv = [0.115 * H * s, 0.80 * H + bob, 0]
        parts.append(part([0.05 * H, arm_len, 0.055 * H], style["shirt"],
                          pv, [0, -arm_len / 2, 0], rx=ua))
        elbow = [pv[i] + rx_vec(ua, [0, -arm_len, 0])[i] for i in range(3)]
        parts.append(part([0.045 * H, fore_len, 0.05 * H], style["skin"],
                          elbow, [0, -fore_len / 2, 0], rx=ua + 18))
        wrist = [elbow[i] + rx_vec(ua + 18, [0, -fore_len, 0])[i]
                 for i in range(3)]
        parts.append(part([0.05 * H, 0.05 * H, 0.05 * H], style["skin"],
                          wrist, [0, -0.025 * H, 0], rx=ua + 18))
    # torso, neck, head, hair
    parts.append(part([0.26 * H, torso_len, 0.13 * H], style["shirt"],
                      [0, hip_y + bob, 0], [0, torso_len / 2, 0]))
    parts.append(part([0.055 * H, 0.055 * H, 0.055 * H], style["skin"],
                      [0, 0.815 * H + bob, 0], [0, 0, 0]))
    head_pv = [0, 0.90 * H + bob, 0]
    parts.append(part([0.17 * H, 0.16 * H, 0.16 * H], style["skin"],
                      head_pv, [0, 0, 0], ry=head_sway))
    hs = style["hair_style"]
    hair = style["hair"]
    parts.append(part([0.18 * H, 0.055 * H, 0.17 * H], hair,
                      head_pv, [0, 0.09 * H, 0], ry=head_sway))
    if hs in ("bob", "long"):
        back_h = 0.12 * H if hs == "bob" else 0.30 * H
        parts.append(part([0.17 * H, back_h, 0.035 * H], hair,
                          head_pv, [0, 0.06 * H - back_h / 2, 0.085 * H],
                          ry=head_sway))
    if hs == "bun":
        parts.append(part([0.08 * H, 0.08 * H, 0.08 * H], hair,
                          head_pv, [0, 0.09 * H, 0.09 * H], ry=head_sway))
    return parts


def load_meshy(name, target_h):
    """Load a Meshy GLB export for this persona, normalized: feet at y=0,
    centered in x/z, scaled to target_h. Returns [(mesh, 4x4 pose)] or None."""
    path = os.path.join(MESHY_DIR, f"{name.replace(' ', '_')}.glb")
    if not os.path.exists(path):
        return None
    scene = trimesh.load(path, force="scene", process=False)
    geoms = []
    for node_name in scene.graph.nodes_geometry:
        xf, _ = scene.graph[node_name]
        m = scene.geometry[scene.graph[node_name][1]]
        geoms.append((m, np.array(xf)))
    if not geoms:
        return None
    corners = [trimesh.transformations.transform_points(m.bounds, xf)
               for m, xf in geoms]
    lo = np.min(np.vstack(corners), axis=0)
    hi = np.max(np.vstack(corners), axis=0)
    s = target_h / max(hi[1] - lo[1], 1e-6)
    cx, cz = (lo[0] + hi[0]) / 2, (lo[2] + hi[2]) / 2
    fix = trimesh.transformations.scale_and_translate(
        scale=s, translate=[-cx * s, -lo[1] * s, -cz * s])
    return [(m, fix @ xf) for m, xf in geoms]


def look_at(eye, target):
    eye, target = np.array(eye, float), np.array(target, float)
    z = eye - target
    z /= np.linalg.norm(z)
    x = np.cross([0, 1, 0], z)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    pose = np.eye(4)
    pose[:3, :3] = np.column_stack([x, y, z])
    pose[:3, 3] = eye
    return pose


def render_pose(renderer, entries, yaw_deg, cam_h, flat=True):
    """entries: [(trimesh geom, 4x4 pose, rgb color or None)] -> RGBA image."""
    scene = pyrender.Scene(bg_color=[0, 0, 0, 0],
                           ambient_light=[1.0, 1.0, 1.0] if flat
                           else [0.55, 0.55, 0.55])
    for m, pose, rgb in entries:
        mat = None
        if rgb is not None:
            mat = pyrender.MetallicRoughnessMaterial(
                baseColorFactor=[rgb[0], rgb[1], rgb[2], 1.0],
                metallicFactor=0.0, roughnessFactor=0.9)
        mesh = pyrender.Mesh.from_trimesh(m, smooth=False, material=mat)
        scene.add(mesh, pose=pose)
    if not flat:
        light = pyrender.DirectionalLight(intensity=2.4)
        scene.add(light, pose=look_at([1.5, 3.5, -3.0], [0, cam_h * 0.6, 0]))
    # Ortho frame: char ~cam_h tall, feet near frame bottom, head margin.
    cam = pyrender.OrthographicCamera(xmag=0.30 * cam_h, ymag=0.60 * cam_h,
                                      znear=0.05, zfar=50)
    scene.add(cam, pose=look_at([0, 0.52 * cam_h, -5.0],
                                [0, 0.52 * cam_h, 0.0]))
    if yaw_deg:
        for node in scene.get_nodes():
            if node.mesh is not None and node.camera is None \
                    and node.light is None:
                node.matrix = rot_y_deg(yaw_deg) @ node.matrix
    flags = pyrender.RenderFlags.RGBA
    if flat:
        flags |= pyrender.RenderFlags.FLAT
    color, _ = renderer.render(scene, flags=flags)
    img = Image.fromarray(color).resize((FRAME_W, FRAME_H), Image.BOX)
    img = ImageEnhance.Color(img).enhance(1.3)
    img = ImageEnhance.Contrast(img).enhance(1.1)
    return img


def parts_to_entries(parts):
    entries = []
    for p in parts:
        m = trimesh.creation.box(extents=p["extents"])
        T = (trimesh.transformations.translation_matrix(p["pivot"])
             @ rot_y_deg(p["ry"]) @ rot_x_deg(p["rx"])
             @ trimesh.transformations.translation_matrix(p["offset"]))
        entries.append((m, T, p["color"]))
    return entries


def build_atlas_json():
    frames = {}
    for row, d in enumerate(ROWS):
        for col in range(5):
            fname = d if col == 4 else f"{d}-walk.{col:03d}"
            frames[fname] = {
                "frame": {"x": col * FRAME_W, "y": row * FRAME_H,
                          "w": FRAME_W, "h": FRAME_H},
                "rotated": False, "trimmed": False,
                "spriteSourceSize": {"x": 0, "y": 0,
                                     "w": FRAME_W, "h": FRAME_H},
                "sourceSize": {"w": FRAME_W, "h": FRAME_H}}
    return {"frames": frames}


def generate(name, renderer):
    style = style_for(name)
    key = name.replace(" ", "_")
    meshy = load_meshy(name, style["H"])
    H = style["H"]

    sheet = Image.new("RGBA", (5 * FRAME_W, 4 * FRAME_H), (0, 0, 0, 0))
    for row, d in enumerate(ROWS):
        idle_img = None
        for fi in range(4):
            if meshy:
                bob = 0.028 * H * (0.5 - 0.5 * math.cos(fi * math.pi))
                entries = [(m, trimesh.transformations.translation_matrix(
                                [0, bob, 0]) @ xf, None) for m, xf in meshy]
                img = render_pose(renderer, entries, YAWS[d], H, flat=False)
                if fi == 0:
                    idle_img = img
            else:
                img = render_pose(renderer,
                                 parts_to_entries(build_parts(style, fi * math.pi / 2)),
                                 YAWS[d], H, flat=True)
                if idle_img is None:
                    idle_img = render_pose(
                        renderer, parts_to_entries(build_parts(style, None)),
                        YAWS[d], H, flat=True)
            sheet.paste(img, (fi * FRAME_W, row * FRAME_H + PASTE_DY))
        sheet.paste(idle_img, (4 * FRAME_W, row * FRAME_H + PASTE_DY))
    out_png = os.path.join(OUT_DIR, f"{key}.png")
    sheet.save(out_png)
    return {"name": name, "key": key, "png": out_png,
            "source": "meshy" if meshy else "procedural"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--personas", default="")
    args = ap.parse_args()

    if args.all:
        with open(CURR_SIM) as f:
            sim = json.load(f)["sim_code"]
        pdir = os.path.join(FRONTEND, "storage", sim, "personas")
        names = sorted(n for n in os.listdir(pdir)
                       if not n.startswith(".") and os.path.isdir(
                           os.path.join(pdir, n)))
    else:
        names = [n.strip() for n in args.personas.split(",") if n.strip()]
    if not names:
        ap.error("no personas: use --all or --personas")

    os.makedirs(OUT_DIR, exist_ok=True)
    with open(os.path.join(OUT_DIR, "atlas.json"), "w") as f:
        json.dump(build_atlas_json(), f, indent=1)

    renderer = pyrender.OffscreenRenderer(RENDER_W, RENDER_H)
    results = []
    try:
        for n in names:
            r = generate(n, renderer)
            results.append(r)
            print(f"  built {r['key']}.png ({r['source']})")
    finally:
        renderer.delete()
    print(f"done: {len(results)} personas -> {OUT_DIR}")


if __name__ == "__main__":
    main()