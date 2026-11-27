"""
check_own_photos.py
Checks our own market photos BEFORE they are used for the v2 species model.

Expected layout (one folder per species, one number per FISH):
    data/raw/own_photos/bangus/bangus_fish01_01.jpg     whole fish (species model)
    data/raw/own_photos/bangus/bangus_fish01_02.jpg     same fish, another photo
    data/raw/own_eyes/bangus/bangus_fish01_eye01.jpg    eye close-up of the SAME fish
    ...  folders: bangus, tilapia, hito, dalag, puyo, pangasius

It prints, per species: whole-fish photos, different fish, photos per fish, eye
close-ups, and problems:
  - wrong file names (the fish number is needed to split by fish)
  - files that cannot be opened (for example iPhone HEIC files)
  - exact duplicates (the same file copied twice, also between the two folders)
  - eye close-ups of a fish that has no whole-fish photos (number typo?)
  - species with too few fish to put some in the test set

Run from the project folder:  python scripts/check_own_photos.py
"""

import hashlib
import sys
from collections import defaultdict
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))  # so "from src..." works

from PIL import Image

from src.species_v2 import OWN_FOLDERS, parse_eye_name, parse_own_name

OWN_DIR = PROJECT_DIR / "data" / "raw" / "own_photos"
EYES_DIR = PROJECT_DIR / "data" / "raw" / "own_eyes"
MIN_FISH = 6   # fewer fish than this -> too few to keep some fish for testing


def make_folders():
    """Creates the empty species folders (nothing is overwritten)."""
    for folder in OWN_FOLDERS:
        (OWN_DIR / folder).mkdir(parents=True, exist_ok=True)
        (EYES_DIR / folder).mkdir(parents=True, exist_ok=True)


def check_folder(folder_path, parse, example, seen):
    """Returns (fish -> photo count, list of problems) for one species folder."""
    fish_photos = defaultdict(int)
    problems = []
    for path in sorted(p for p in folder_path.iterdir() if p.is_file()):
        parsed = parse(path)
        if parsed is None:
            problems.append(f"bad name: {path.name}  (expected {example})")
            continue
        try:
            with Image.open(path) as image:
                image.verify()
        except Exception:
            problems.append(f"cannot open: {path.name}  (HEIC? save as JPG)")
            continue
        digest = hashlib.md5(path.read_bytes()).hexdigest()
        if digest in seen:
            problems.append(f"duplicate: {path.name} = {seen[digest]}")
            continue
        seen[digest] = path.name
        fish_photos[parsed[1]] += 1
    return fish_photos, problems


def main():
    make_folders()
    print(f"Whole-fish photos: {OWN_DIR}")
    print(f"Eye close-ups:     {EYES_DIR}\n")
    print(f"{'species':<11}{'photos':>7}{'fish':>6}{'per fish':>10}{'eyes':>6}")
    all_problems = []
    seen = {}   # MD5 -> file name, shared so copies between the two folders are found too
    for folder in OWN_FOLDERS:
        fish_photos, problems = check_folder(OWN_DIR / folder, parse_own_name,
                                             f"{folder}_fish01_01.jpg", seen)
        eye_photos, eye_problems = check_folder(EYES_DIR / folder, parse_eye_name,
                                                f"{folder}_fish01_eye01.jpg", seen)
        for fish in sorted(set(eye_photos) - set(fish_photos)):
            eye_problems.append(f"eye photo of {fish.split(':')[-1]} but no whole-fish photo of it")
        counts = sorted(fish_photos.values())
        spread = f"{counts[0]}-{counts[-1]}" if counts else "-"
        warning = "   <-- fewer than 6 fish" if len(counts) < MIN_FISH else ""
        print(f"{folder:<11}{sum(counts):>7}{len(counts):>6}{spread:>10}"
              f"{sum(eye_photos.values()):>6}{warning}")
        all_problems += [f"{folder}: {p}" for p in problems]
        all_problems += [f"{folder} (eyes): {p}" for p in eye_problems]

    if all_problems:
        print(f"\nProblems ({len(all_problems)}):")
        for problem in all_problems:
            print(f"  - {problem}")
    else:
        print("\nNo problems found.")


main()
