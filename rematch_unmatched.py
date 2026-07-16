#!/usr/bin/env python3
"""Re-run matching on images that were previously unmatched, with relaxed thresholds.
This time, allow over-writing existing matches if a better candidate image is found."""
import json
from collections import defaultdict
from pathlib import Path
from typing import List

import cv2

import match_stickers as m

BASE = Path(__file__).parent
UNMATCHED_DIR = BASE / "output" / "sin_asociar"
CATALOG_PATH = BASE / "panini-wc-2026-catalog-matched.json"
RESULT_PATH = BASE / "panini-wc-2026-catalog-matched.json"

RELAXED_NAME_THRESHOLD = 0.75
RELAXED_TEAM_THRESHOLD = 0.75


def main():
    with open(CATALOG_PATH, encoding="utf-8") as f:
        catalog = json.load(f)
    stickers = catalog["stickers"]

    norm_names = [
        "" if (s["team"] == "Host Countries and Cities") else m.normalize(s["name"])
        for s in stickers
    ]

    image_files = sorted(UNMATCHED_DIR.glob("*.png"))
    print(f"Found {len(image_files)} unmatched images. Running OCR...")

    image_lines = {}
    for i, path in enumerate(image_files, 1):
        image_lines[path.name] = m.ocr_lines(path)
        if i % 10 == 0:
            print(f"  progress: {i}/{len(image_files)}", flush=True)

    # Stage 1: fuzzy-match with relaxed threshold, allowing overwrites
    candidates = []
    for img_name, lines in image_lines.items():
        for idx, score in m.all_line_matches(lines, norm_names, RELAXED_NAME_THRESHOLD).items():
            candidates.append((score, img_name, idx))
    candidates.sort(key=lambda c: -c[0])

    used_images = set()
    used_catalog = set()
    for score, img_name, idx in candidates:
        if img_name in used_images:
            continue
        # Allow re-matching, no restrictions on overwriting
        stickers[idx]["img"] = img_name
        used_images.add(img_name)
        used_catalog.add(idx)

    print(f"Stage 1 (player names, relaxed): matched {len(used_images)}/{len(image_files)}")

    # Stage 2: team-based matching for remaining (emblems/team photos)
    by_team_name = defaultdict(list)
    for idx, s in enumerate(stickers):
        by_team_name[(s["team"], m.normalize(s["name"]))].append(idx)

    all_teams = sorted({s["team"] for s in stickers})
    norm_teams = [m.normalize(t) for t in all_teams]

    remaining = [p.name for p in image_files if p.name not in used_images]
    print(f"Stage 2 (Emblem/Team Photo, relaxed): matching {len(remaining)} remaining...")

    stage2_matched = 0
    for img_name in remaining:
        lines = image_lines[img_name]
        team_matches = m.all_line_matches(lines, norm_teams, RELAXED_TEAM_THRESHOLD)
        ranked_teams = sorted(team_matches.items(), key=lambda t: -t[1])
        if not ranked_teams:
            continue

        faces = m.count_faces(UNMATCHED_DIR / img_name)
        guess_name = "TEAM PHOTO" if faces >= m.TEAM_PHOTO_MIN_FACES else "EMBLEM"
        other_name = "EMBLEM" if guess_name == "TEAM PHOTO" else "TEAM PHOTO"

        for team_idx, _score in ranked_teams:
            team = all_teams[team_idx]
            pool = [i for i in by_team_name.get((team, guess_name), []) if i not in used_catalog]
            if not pool:
                pool = [i for i in by_team_name.get((team, other_name), []) if i not in used_catalog]
            if pool:
                stickers[pool[0]]["img"] = img_name
                used_catalog.add(pool[0])
                used_images.add(img_name)
                stage2_matched += 1
                break

    print(f"Stage 2: matched {stage2_matched}/{len(remaining)}")
    print(f"Total newly matched from sin_asociar: {len(used_images)}/{len(image_files)}")

    with open(RESULT_PATH, "w", encoding="utf-8") as f:
        json.dump(catalog, f, indent=2, ensure_ascii=False)
    print(f"Saved to {RESULT_PATH}")

    # Summary
    total_matched = sum(1 for s in stickers if "img" in s)
    total_unmatched = sum(1 for s in stickers if "img" not in s)
    print(f"\nFinal catalog status:")
    print(f"  Total matched: {total_matched}")
    print(f"  Total unmatched: {total_unmatched}")
    print(f"  Match rate: {100*total_matched/(total_matched+total_unmatched):.1f}%")


if __name__ == "__main__":
    main()
