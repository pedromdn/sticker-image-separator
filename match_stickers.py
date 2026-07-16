#!/usr/bin/env python3
"""Match the 864 extracted sticker PNGs to entries in the Panini catalog JSON.

The album turns out to use at least three different card layouts (a
standard portrait with a horizontal name bar, "special" action shots with
the name rotated vertically along an edge, and "Extra Sticker" inserts with
the name bar sitting right at the bottom edge with nothing below it). A
single fixed crop region can't read all of them, so instead of guessing
where the name is, every image is OCR'd whole, at 0/90/270 degrees, and
every resulting text line becomes a candidate. Whichever candidate line
best fuzzy-matches a real catalog entry wins -- birthdate/club/watermark
lines never score well against actual names, so the genuine name line rises
to the top on its own.

Approach
========
1. OCR every image at three orientations and collect every line.
2. Stage 1: fuzzy-match each image's best candidate line against every
   catalog sticker NAME, then assign greedily (highest similarity first).
   Catalog names that repeat (a player's regular + special/shiny printing)
   always share the same team, so which specific duplicate a card lands on
   doesn't matter.
3. Stage 2: images still unmatched are almost always a team's "Emblem" or
   "Team Photo" card (no player name to read at all). Their candidate lines
   are instead fuzzy-matched against catalog TEAM names, restricted to that
   team's unused Emblem/Team Photo slot. A classic Haar-cascade face count
   picks which of the two it is (a team photo has many faces, a crest ~0).

Not everything resolves this way -- some Emblem crests carry no Latin-script
country name at all (e.g. an Arabic federation wordmark), so there's
nothing for OCR to read. Those are left without an "img" field and reported
at the end rather than guessed at.
"""
import json
import re
import unicodedata
from collections import defaultdict
from difflib import SequenceMatcher
from pathlib import Path
from typing import List

import cv2
import numpy as np
import pytesseract
from PIL import Image

BASE = Path(__file__).parent
OUTPUT_DIR = BASE / "output"
CATALOG_PATH = BASE / "panini-wc-2026-catalog.json"
RESULT_PATH = BASE / "panini-wc-2026-catalog-matched.json"

NAME_MATCH_THRESHOLD = 0.8
TEAM_MATCH_THRESHOLD = 0.8
MIN_LINE_LENGTH = 5
FACE_CASCADE = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
TEAM_PHOTO_MIN_FACES = 4
ROTATIONS = (None, cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_90_COUNTERCLOCKWISE)


def normalize(text: str) -> str:
    """Strip accents/punctuation and uppercase, so OCR noise and catalog
    mojibake (some accented letters were saved as U+FFFD) compare fairly."""
    text = unicodedata.normalize("NFKD", text)
    text = "".join(c for c in text if not unicodedata.combining(c))
    text = re.sub(r"[^A-Za-z ]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text.upper()


def _whiten_text(img: np.ndarray) -> np.ndarray:
    """Isolate bold white card text as black-on-white, since Tesseract
    expects dark text on a light background and these cards print the name
    in white over a solid color (or black) bar."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
    return cv2.bitwise_not(mask)


def ocr_lines(path: Path) -> List[str]:
    """Every plausible text line found in the image, tried on both the raw
    and white-text-isolated versions, each at 0/90/270 degrees (the name is
    sometimes printed sideways along an edge)."""
    img = cv2.imread(str(path))
    variants = [img, _whiten_text(img)]
    lines = []
    for variant in variants:
        for rotation in ROTATIONS:
            rotated = cv2.rotate(variant, rotation) if rotation is not None else variant
            text = pytesseract.image_to_string(Image.fromarray(rotated), config="--psm 11", lang="eng")
            for line in text.splitlines():
                normalized = normalize(line)
                if len(normalized) >= MIN_LINE_LENGTH:
                    lines.append(normalized)
    return lines


def count_faces(path: Path) -> int:
    img = cv2.imread(str(path))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = FACE_CASCADE.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=6, minSize=(30, 30))
    return len(faces)


def similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, a, b).ratio()


def all_line_matches(lines: List[str], targets: List[str], threshold: float) -> dict:
    """Every target index that clears ``threshold`` against some line, each
    mapped to its best score. A card can carry text that coincidentally
    matches an unrelated catalog entry (e.g. a Mexican player's card OCR'ing
    the word "Mexico" itself, which also happens to be a real, unique
    catalog entry) -- keeping every candidate, not just the top one, lets
    the global greedy assignment fall through to a card's real name if its
    best guess turns out to be claimed by something else."""
    best_per_idx: dict = {}
    for line in lines:
        for i, target in enumerate(targets):
            score = similarity(line, target)
            if score >= threshold and score > best_per_idx.get(i, 0.0):
                best_per_idx[i] = score
    return best_per_idx


def main():
    with open(CATALOG_PATH, encoding="utf-8") as f:
        catalog = json.load(f)
    stickers = catalog["stickers"]
    for s in stickers:
        s.pop("img", None)

    # A card's own OCR text often includes its player's country, so a bare
    # country name (the three host-nation entries) is too collision-prone to
    # match as if it were a player name -- e.g. any Mexican player's card can
    # OCR the word "Mexico", which would otherwise outscore the player's own
    # (harder to read) name. These are excluded from Stage 1 by blanking
    # them out; Stage 2 doesn't need them either since they're not an
    # Emblem/Team Photo for a real squad.
    norm_names = [
        "" if (s["team"] == "Host Countries and Cities") else normalize(s["name"])
        for s in stickers
    ]

    image_files = sorted(OUTPUT_DIR.glob("*.png"))
    print(f"Found {len(image_files)} images. Running OCR (0/90/270 degrees each)...")

    image_lines = {}
    for i, path in enumerate(image_files, 1):
        image_lines[path.name] = ocr_lines(path)
        if i % 50 == 0:
            print(f"  progress: {i}/{len(image_files)}", flush=True)

    # --- Stage 1: fuzzy-match every card's lines against every catalog name -----
    candidates = []
    for img_name, lines in image_lines.items():
        for idx, score in all_line_matches(lines, norm_names, NAME_MATCH_THRESHOLD).items():
            candidates.append((score, img_name, idx))
    candidates.sort(key=lambda c: -c[0])

    used_images: set = set()
    used_catalog: set = set()
    for score, img_name, idx in candidates:
        if img_name in used_images or idx in used_catalog:
            continue
        stickers[idx]["img"] = img_name
        used_images.add(img_name)
        used_catalog.add(idx)

    print(f"Stage 1 (player names): matched {len(used_images)}/{len(image_files)}")

    # --- Stage 2: Emblem / Team Photo cards, matched by team name instead -------
    by_team_name: dict = defaultdict(list)
    for idx, s in enumerate(stickers):
        by_team_name[(s["team"], normalize(s["name"]))].append(idx)

    all_teams = sorted({s["team"] for s in stickers})
    norm_teams = [normalize(t) for t in all_teams]

    remaining = [p.name for p in image_files if p.name not in used_images]
    print(f"Stage 2 (Emblem/Team Photo): matching {len(remaining)} remaining cards by team...")

    stage2_matched = 0
    unresolved = []
    for img_name in remaining:
        lines = image_lines[img_name]
        team_matches = all_line_matches(lines, norm_teams, TEAM_MATCH_THRESHOLD)
        ranked_teams = sorted(team_matches.items(), key=lambda t: -t[1])
        if not ranked_teams:
            unresolved.append((img_name, lines[:3], "no team name recognized"))
            continue

        faces = count_faces(OUTPUT_DIR / img_name)
        guess_name = "TEAM PHOTO" if faces >= TEAM_PHOTO_MIN_FACES else "EMBLEM"
        other_name = "EMBLEM" if guess_name == "TEAM PHOTO" else "TEAM PHOTO"

        pool = None
        for team_idx, _score in ranked_teams:
            team = all_teams[team_idx]
            pool = [i for i in by_team_name.get((team, guess_name), []) if i not in used_catalog]
            if not pool:
                pool = [i for i in by_team_name.get((team, other_name), []) if i not in used_catalog]
            if pool:
                break

        if pool:
            stickers[pool[0]]["img"] = img_name
            used_catalog.add(pool[0])
            used_images.add(img_name)
            stage2_matched += 1
        else:
            best_team = all_teams[ranked_teams[0][0]]
            unresolved.append((img_name, lines[:3], f"team '{best_team}' recognized but no free slot ({faces} faces)"))

    print(f"Stage 2: matched {stage2_matched}/{len(remaining)}")
    print(f"Total matched: {len(used_images)}/{len(image_files)}")

    with open(RESULT_PATH, "w", encoding="utf-8") as f:
        json.dump(catalog, f, indent=2, ensure_ascii=False)
    print(f"Saved to {RESULT_PATH}")

    if unresolved:
        print(f"\nUnresolved images ({len(unresolved)}):")
        for img_name, lines, reason in unresolved:
            print(f"  {img_name}: lines={lines} -> {reason}")

    missing = [s for s in stickers if "img" not in s]
    print(f"\nCatalog entries with no image: {len(missing)}")


if __name__ == "__main__":
    main()
