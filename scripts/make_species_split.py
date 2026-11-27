"""
make_species_split.py
Makes the train / validation / test split for the v2 species model and saves it to
data/species_split.csv. Run it ONCE, after our own market photos are in.

Images come from two sources:
  - "bangladesh": raw photos from the Bangladesh dataset (kept rows of
    data/bd_species_manifest.csv, made by scripts/inspect_bd_species.py)
  - "own": our photos in data/raw/own_photos/<species>/<species>_fishNN_MM.jpg

Rules:
  1. Grouped: all photos of one of OUR fish (own:bangus:fish01) stay in one split;
     Bangladesh photos of one scene (taken seconds apart) stay in one split.
  2. Stratified by species AND source, so every class has our own fish in the
     validation and test sets too (the model must not learn "our photos = bangus").
  3. Capped: each Bangladesh class keeps at most BD_CAP images, spread over as
     many groups as possible, so the 5 Bangladesh classes do not drown bangus.
     Class weights in training balance what is left.

Run from the project folder:  python scripts/make_species_split.py
"""

import os
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))  # so "from src..." works

import numpy as np
import pandas as pd

from src.species_v2 import BD_FOLDERS, OWN_FOLDERS, SPECIES_V2, parse_own_name

RAW_DIR = PROJECT_DIR / "data" / "raw"   # split paths are relative to this folder
OWN_DIR = Path(os.environ.get("FISHSCAN_OWN_DIR", RAW_DIR / "own_photos"))
BD_MANIFEST = PROJECT_DIR / "data" / "bd_species_manifest.csv"
SPLIT_CSV = Path(os.environ.get("FISHSCAN_SPECIES_SPLIT", PROJECT_DIR / "data" / "species_split.csv"))

BD_CAP = 400     # most Bangladesh images per class (about 4x our ~100 bangus photos)
MIN_FISH = 6     # each species needs at least this many of OUR fish
FRACTION = 0.15  # share of each species + source for test, and again for validation
SEED = 42


def ask_yes_no(question):
    """Asks a y/n question and returns True for yes."""
    while True:
        try:
            answer = input(f"{question} (y/n): ").strip().lower()
            if answer not in ("y", "n"):
                raise ValueError
            break
        except ValueError:
            print("Please type y or n.")
            continue
    return answer == "y"


def load_bangladesh():
    """Kept Bangladesh rows: path (relative to data/raw), species, source, group."""
    table = pd.read_csv(BD_MANIFEST)
    table = table[(table["decision"] == "keep") & table["folder"].isin(BD_FOLDERS)]
    return pd.DataFrame({
        "path": "bd_fish/" + table["path"],
        "species": table["folder"].map(BD_FOLDERS),
        "source": "bangladesh",
        "group": table["group"],
    })


def load_own():
    """Our photos with a valid name: path, species, source, group (= the fish)."""
    rows = []
    for folder, species in OWN_FOLDERS.items():
        for path in sorted((OWN_DIR / folder).glob("*")):
            parsed = parse_own_name(path)
            if parsed:
                rows.append({"path": path.relative_to(RAW_DIR).as_posix() if path.is_relative_to(RAW_DIR)
                             else path.as_posix(),
                             "species": species, "source": "own", "group": parsed[1]})
    return pd.DataFrame(rows, columns=["path", "species", "source", "group"])


def cap_bangladesh(table):
    """Keeps at most BD_CAP images per Bangladesh class, one group at a time.

    Round-robin over groups (first photo of every scene, then the second, ...),
    so the cap removes repeated shots of the same scene first.
    """
    rng = np.random.default_rng(SEED)
    keep = []
    for _, part in table.groupby("species"):
        part = part.iloc[rng.permutation(len(part))]
        rank = part.groupby("group").cumcount()   # 0 = first photo of its group
        keep.append(part.assign(_rank=rank).sort_values("_rank", kind="stable").head(BD_CAP))
    return pd.concat(keep).drop(columns="_rank")


def split(table):
    """About 70/15/15, grouped by fish/scene, done separately for every species + source.

    For each stratum (e.g. "Hito | own") the groups are shuffled and whole groups
    go to test until it has ~15% of the images, then to validation, the rest to
    train. Every stratum gets at least one group in test and in validation.
    (scikit-learn's StratifiedGroupKFold could leave a class with very few big
    groups, like Hito, without any test images.)
    """
    rng = np.random.default_rng(SEED)
    table = table.copy()
    table["split"] = "train"
    for _, part in table.groupby(["species", "source"]):
        sizes = part.groupby("group").size()
        order = sizes.index[rng.permutation(len(sizes))]
        total, used = len(part), 0
        for split_name in ("test", "val"):
            taken = 0
            for group in order[used:]:
                if taken > 0 and taken + sizes[group] / 2 > FRACTION * total:
                    break
                table.loc[part.index[part["group"] == group], "split"] = split_name
                taken += sizes[group]
                used += 1
            if used >= len(order) - 1:
                break   # always leave at least one group for training
    return table


def print_split(table):
    """Images per species + source in each split, and the leakage check."""
    counts = pd.crosstab([table["species"], table["source"]], table["split"])
    counts = counts.reindex(columns=["train", "val", "test"], fill_value=0)
    print("\nImages per species and source:")
    print(counts.to_string())
    fish = table[table["source"] == "own"].groupby(["species", "split"])["group"].nunique()
    print("\nOur fish per split:")
    print(fish.unstack(fill_value=0).reindex(columns=["train", "val", "test"], fill_value=0).to_string())
    missing = counts[(counts["val"] == 0) | (counts["test"] == 0)].index.tolist()
    if missing:
        print(f"  WARNING: no validation or test images for {missing}")
    leaks = int((table.groupby("group")["split"].nunique() > 1).sum())
    print(f"\nLeakage check: {'OK - no fish or scene in two splits' if leaks == 0 else f'PROBLEM - {leaks} groups in two splits'}")


def main():
    if not BD_MANIFEST.exists():
        print("Run scripts/inspect_bd_species.py first.")
        return
    own = load_own()
    fish_per_species = own.groupby("species")["group"].nunique().reindex(SPECIES_V2, fill_value=0)
    print("Our fish per species:\n" + fish_per_species.to_string())
    too_few = fish_per_species[fish_per_species < MIN_FISH].index.tolist()
    if too_few:
        print(f"\nNot enough of our own fish yet (need {MIN_FISH}+): {too_few}")
        print("Add the market photos and run scripts/check_own_photos.py first. Nothing saved.")
        return
    if SPLIT_CSV.exists():
        print(f"A split already exists: {SPLIT_CSV}")
        print("Making a new one changes which images are in the test set.")
        if not ask_yes_no("Replace it?"):
            print("Kept the existing split. Nothing changed.")
            return

    bd = cap_bangladesh(load_bangladesh())
    table = split(pd.concat([bd, own], ignore_index=True))
    print_split(table)
    SPLIT_CSV.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(SPLIT_CSV, index=False)
    print(f"\nSaved the split to {SPLIT_CSV}")


main()
