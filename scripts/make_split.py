"""
make_split.py
Uses DatasetLoader to scan FFE, make the group-aware 70/15/15 split,
check for leakage, and save the split to data/ffe_split.csv.
Training (local or Colab) then loads this SAME file, so the test set
never changes between runs.

Run from the project folder:  python scripts/make_split.py
"""

import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))  # so "from src..." works

import pandas as pd

from src.dataset_loader import DatasetLoader

DATA_DIR = PROJECT_DIR / "data" / "raw" / "ffe"
SPLIT_CSV = PROJECT_DIR / "data" / "ffe_split.csv"


def ask_overwrite():
    """Asks before replacing an existing split file. Returns True to continue."""
    if not SPLIT_CSV.exists():
        return True
    print(f"A split already exists: {SPLIT_CSV}")
    print("Making a new one changes which images are in the test set.")
    while True:
        try:
            answer = input("Replace it? (y/n): ").strip().lower()
            if answer not in ("y", "n"):
                raise ValueError
            break
        except ValueError:
            print("Please type y or n.")
            continue
    return answer == "y"


def print_split_table(table):
    """Prints image counts per species and per freshness level in each split."""
    for task in ("species", "freshness"):
        counts = pd.crosstab(table[task], table["split"])[["train", "val", "test"]]
        counts["total"] = counts.sum(axis=1)
        print(f"\nImages per {task} class:")
        print(counts.to_string())
        small = counts[counts["test"] < 30].index.tolist()
        if small:
            print(f"  Note: few test images for {small}, so their test metrics will be noisy.")


def print_group_check(loader):
    """Prints how many source photos there are and confirms no leakage."""
    table = loader.table
    print(f"\nSource photos (groups): {table['source_photo'].nunique()} "
          f"for {len(table)} images")
    photos_per_split = table.groupby("split")["source_photo"].nunique()
    print("Source photos per split: " + ", ".join(
        f"{s}={photos_per_split[s]}" for s in ("train", "val", "test")))

    leaks = loader.check_no_leakage()
    if leaks == 0:
        print("Leakage check: OK - no source photo appears in two splits.")
    else:
        print(f"Leakage check: PROBLEM - {leaks} source photos appear in more than one split!")


def print_class_weights(loader):
    """Prints the class weights the training script will use."""
    for task in ("species", "freshness"):
        weights = loader.class_weights(task)
        names = loader.class_names(task)
        print(f"\nClass weights ({task}):")
        for index, weight in weights.items():
            print(f"  {names[index]:<22} {weight:.2f}")


def print_batch_check(loader):
    """Loads one training batch to prove the TensorFlow pipeline works."""
    images, labels = next(iter(loader.make_dataset("train", "species")))
    print(f"\nOne training batch: images {tuple(images.shape)}, labels {tuple(labels.shape)}, "
          f"pixel range {float(images.numpy().min()):.0f}-{float(images.numpy().max()):.0f}")


def main():
    if not DATA_DIR.is_dir():
        print(f"FFE folder not found: {DATA_DIR}")
        print("Unzip the dataset there first (see data/DATASETS.md).")
        return
    if not ask_overwrite():
        print("Kept the existing split. Nothing changed.")
        return

    loader = DatasetLoader(DATA_DIR)
    result = loader.scan()
    print(f"Scanned {result['images']} images ({result['skipped']} skipped).")

    sizes = loader.split()
    total = sum(sizes.values())
    print("Split sizes: " + ", ".join(
        f"{s}={sizes[s]} ({sizes[s] / total:.0%})" for s in ("train", "val", "test")))

    print_group_check(loader)
    print_split_table(loader.table)
    print_class_weights(loader)
    print_batch_check(loader)

    loader.save_split(SPLIT_CSV)
    print(f"\nSaved the split to {SPLIT_CSV}")


main()
