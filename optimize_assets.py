#!/usr/bin/env python3
"""
Asset optimization for generative_agents frontend.
- Converts large tileset PNGs to WebP (with PNG originals kept as fallback)
- Removes Interiors_32x32_full.png (not referenced in tilemap)
- Compresses the tilemap JSON
- Reports before/after sizes
"""

import os
import json
import shutil
from pathlib import Path

try:
    from PIL import Image
except ImportError:
    print("ERROR: Pillow not installed. Run: pip install Pillow")
    exit(1)

BASE = Path(__file__).parent / "environment/frontend_server/static_dirs"
ASSETS = BASE / "assets/the_ville/visuals/map_assets"
TILEMAP = BASE / "assets/the_ville/visuals/the_ville_jan7.json"

# Tilesets to convert to WebP (confirmed referenced in tilemap)
TILESETS_V1 = [
    ASSETS / "v1/Room_Builder_32x32.png",
    ASSETS / "v1/interiors_pt1.png",
    ASSETS / "v1/interiors_pt2.png",
    ASSETS / "v1/interiors_pt3.png",
    ASSETS / "v1/interiors_pt4.png",
    ASSETS / "v1/interiors_pt5.png",
]

# Dead file — not referenced in tilemap JSON, safe to remove
DEAD_FILES = [
    ASSETS / "v1/Interiors_32x32_full.png",  # 2.5MB combined tileset, unused
]

total_saved = 0


def fmt(n):
    return f"{n/1024:.0f}KB" if n < 1024*1024 else f"{n/1024/1024:.1f}MB"


def convert_to_webp(png_path):
    global total_saved
    webp_path = png_path.with_suffix(".webp")
    if webp_path.exists():
        print(f"  SKIP (already exists): {webp_path.name}")
        return

    img = Image.open(png_path)
    before = png_path.stat().st_size

    # Preserve transparency — RGBA → WebP lossless for tileset accuracy
    img.save(webp_path, "WEBP", lossless=True, quality=90, method=6)
    after = webp_path.stat().st_size
    saved = before - after
    total_saved += saved
    print(f"  {png_path.name} → {webp_path.name}: {fmt(before)} → {fmt(after)} (saved {fmt(saved)})")


def remove_dead(path):
    global total_saved
    if path.exists():
        size = path.stat().st_size
        total_saved += size
        path.unlink()
        print(f"  REMOVED {path.name}: freed {fmt(size)}")
    else:
        print(f"  SKIP (not found): {path.name}")


def compress_tilemap():
    global total_saved
    if not TILEMAP.exists():
        print(f"  Tilemap not found: {TILEMAP}")
        return

    before = TILEMAP.stat().st_size
    with open(TILEMAP) as f:
        data = json.load(f)

    # Re-serialize with minimal whitespace
    compressed = json.dumps(data, separators=(',', ':'))
    with open(TILEMAP, 'w') as f:
        f.write(compressed)

    after = TILEMAP.stat().st_size
    saved = before - after
    total_saved += saved
    print(f"  tilemap JSON: {fmt(before)} → {fmt(after)} (saved {fmt(saved)})")


if __name__ == "__main__":
    print("=== Asset Optimization ===\n")

    print("1. Converting tilesets to WebP (lossless, with PNG kept as fallback):")
    for p in TILESETS_V1:
        if p.exists():
            convert_to_webp(p)
        else:
            print(f"  SKIP (not found): {p.name}")

    print("\n2. Removing unused files:")
    for p in DEAD_FILES:
        remove_dead(p)

    print("\n3. Compressing tilemap JSON:")
    compress_tilemap()

    print(f"\n=== Total saved: {fmt(total_saved)} ===")
    print("\nNote: Update demo/main_script.html to load .webp files instead of .png")
    print("      PNG originals kept — WebP loads faster on modern mobile browsers.")
