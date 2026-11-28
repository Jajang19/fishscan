"""
eye_check_tradeoff.py
Shows what would happen if the fish-eye check ("Not recognized") were made less strict.
It does NOT change the threshold; it only measures and explains.

Cut-offs are chosen on FFE VALIDATION eye photos only (never on our market photos):
  - 95% kept: the current rule (threshold saved in models/eye_check.npz)
  - 99% kept: 99% of real validation eyes pass
  - warning zone: between the 95% and 99% cut-offs -> show the result with a note
Optional cost side: photos that are NOT fish-eye close-ups (FISHSCAN_NEGATIVES_DIR,
e.g. the Philippines Fish Market Dataset, which v1 already used for this) - how many
would still be caught at each cut-off.

Saves <reports>/eye_check_tradeoff.csv and eye_check_tradeoff.json (the cut-offs, read
by scripts/real_world_test.py).

Run in Colab:  %run scripts/eye_check_tradeoff.py   (see notebooks/train_colab.ipynb)
"""

import json
import os
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))  # so "from src..." works

import numpy as np
import pandas as pd
from PIL import Image

from src.dataset_loader import DatasetLoader
from src.fish_predictor import FishPredictor

DATA_DIR = Path(os.environ.get("FISHSCAN_DATA_DIR", PROJECT_DIR / "data" / "raw" / "ffe"))
NEGATIVES_DIR = os.environ.get("FISHSCAN_NEGATIVES_DIR", "")
REPORTS_DIR = Path(os.environ.get("FISHSCAN_REPORTS_DIR", PROJECT_DIR / "reports"))
SPLIT_CSV = PROJECT_DIR / "data" / "ffe_split.csv"
MODELS_DIR = PROJECT_DIR / "models"

BATCH_SIZE = 32
KEEP_LEVELS = [0.95, 0.99]   # share of real validation eyes that must pass
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def distances_for(predictor, paths, label):
    """Eye-check distance of every photo (prepared exactly like in the app)."""
    print(f"Measuring {len(paths)} {label} ...")
    result = []
    for start in range(0, len(paths), BATCH_SIZE):
        batch = []
        for path in paths[start:start + BATCH_SIZE]:
            with Image.open(path) as image:
                batch.append(predictor.prepare_image(image))
        features, _ = predictor.species_features(np.concatenate(batch))
        result.append(predictor.eye_checker.distances(features))
    return np.concatenate(result)


def find_negatives():
    """All images in FISHSCAN_NEGATIVES_DIR (any folders), or [] if not set."""
    if not NEGATIVES_DIR or not Path(NEGATIVES_DIR).is_dir():
        return []
    return sorted(p for p in Path(NEGATIVES_DIR).rglob("*")
                  if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)


def explain(val, negatives, current, cutoffs):
    """Prints the trade-off in plain words."""
    t95, t99 = cutoffs[0.95], cutoffs[0.99]
    n = len(val)
    print("\n================ Eye-check trade-off (FFE validation eyes) ================")
    print(f"Saved threshold (current app): {current:.4f}")
    print(f"95% cut-off measured again:    {t95:.4f}")
    print(f"99% cut-off:                   {t99:.4f}")
    print(f"Largest validation distance:   {val.max():.4f}\n")
    for name, t in (("current (95%)", current), ("99%", t99)):
        rejected = int((val > t).sum())
        print(f"With the {name} cut-off, {rejected} of {n} real validation eyes "
              f"({rejected / n:.1%}) would be called 'Not recognized'.")
    zone = int(((val > current) & (val <= t99)).sum())
    print(f"A warning zone between {current:.3f} and {t99:.3f} would hold {zone} validation "
          f"eyes ({zone / n:.1%}): they would get results plus a 'photo looks unusual' note.")

    if len(negatives):
        print(f"\nNon-eye photos ({len(negatives)}): how many are still caught?")
        for name, t in (("current (95%)", current), ("99%", t99)):
            caught = int((negatives > t).sum())
            print(f"  {name}: {caught} of {len(negatives)} ({caught / len(negatives):.1%}) caught")
        in_zone = int(((negatives > current) & (negatives <= t99)).sum())
        print(f"  In the warning zone: {in_zone} ({in_zone / len(negatives):.1%}) - "
              "these would show results WITH the warning note.")
    else:
        print("\n(No non-eye photos given, so the cost side is not measured. "
              "Set FISHSCAN_NEGATIVES_DIR to measure it.)")

    print("\nIn plain words: a higher cut-off lets more real eye photos through (fewer "
          "false 'Not recognized'), but also lets more strange photos through, which then "
          "get a confident-looking species. A warning zone keeps the strict rule for "
          "'clearly not an eye' and only softens the borderline cases.")


def main():
    if not SPLIT_CSV.exists() or not DATA_DIR.is_dir():
        print("Missing data/ffe_split.csv or the FFE images (set FISHSCAN_DATA_DIR).")
        return
    predictor = FishPredictor(MODELS_DIR)
    if predictor.eye_checker is None:
        print("models/eye_check.npz is missing.")
        return

    loader = DatasetLoader(DATA_DIR)
    loader.load_split(SPLIT_CSV)
    val_paths = [DATA_DIR / p for p in loader.table.query("split == 'val'")["path"]]
    val = distances_for(predictor, val_paths, "FFE validation eyes")
    negative_paths = find_negatives()
    negatives = distances_for(predictor, negative_paths, "non-eye photos") if negative_paths else np.array([])

    current = predictor.eye_checker.threshold
    cutoffs = {keep: float(np.quantile(val, keep)) for keep in KEEP_LEVELS}
    explain(val, negatives, current, cutoffs)

    rows = []
    for name, t in (("current", current), ("keep_95", cutoffs[0.95]), ("keep_99", cutoffs[0.99])):
        rows.append({"cutoff": name, "threshold": t,
                     "val_eyes_rejected": float((val > t).mean()),
                     "non_eyes_caught": float((negatives > t).mean()) if len(negatives) else np.nan})
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(REPORTS_DIR / "eye_check_tradeoff.csv", index=False)
    with open(REPORTS_DIR / "eye_check_tradeoff.json", "w", encoding="utf-8") as f:
        json.dump({"current": current, "keep_95": cutoffs[0.95], "keep_99": cutoffs[0.99],
                   "validation_images": len(val), "non_eye_images": len(negatives)}, f, indent=2)
    print(f"\nSaved eye_check_tradeoff.csv and .json to {REPORTS_DIR}")
    print("Nothing was changed: the app still uses the saved threshold.")


main()
