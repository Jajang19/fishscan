"""
build_eye_check.py
Builds the fish-eye check file models/eye_check.npz that the app uses to say
"Not recognized" for photos that do not look like FFE fish-eye close-ups.

  1. The species model turns every TRAINING image into 1,280 feature numbers.
     These are the reference set.
  2. The same is done for every VALIDATION image.
  3. EyeChecker sets the cut-off (threshold) so that 95% of the validation
     eye photos pass. The test set is NOT used here.
  4. The reference features + threshold + k are saved to models/eye_check.npz.

Images are prepared with FishPredictor.prepare_image (centre square, 224 x 224),
exactly like a photo uploaded in the app.

Needs: models/species_model.keras, data/ffe_split.csv and the FFE images.
No training: the species model is only used to read features.

Run from the project folder:  python scripts/build_eye_check.py

(Rebuilt on 2026-11-23 from src/eye_checker.py and reports/eye_check_results.json
after the original script was lost.)
"""

import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))  # so "from src..." works

import numpy as np
from PIL import Image

from src.dataset_loader import DatasetLoader
from src.eye_checker import EyeChecker, K_NEIGHBOURS, KEEP_FRACTION
from src.fish_predictor import FishPredictor, EYE_CHECK_FILE

DATA_DIR = PROJECT_DIR / "data" / "raw" / "ffe"
SPLIT_CSV = PROJECT_DIR / "data" / "ffe_split.csv"
MODELS_DIR = PROJECT_DIR / "models"
OUT_PATH = MODELS_DIR / EYE_CHECK_FILE

BATCH_SIZE = 32


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


def split_features(predictor, loader, split_name):
    """Returns the species-model features (n, 1280) of every image in one split."""
    paths = [DATA_DIR / p for p in loader.table.query("split == @split_name")["path"]]
    print(f"Reading features of {len(paths)} {split_name} images ...")
    parts = []
    for start in range(0, len(paths), BATCH_SIZE):
        batch = []
        for path in paths[start:start + BATCH_SIZE]:
            with Image.open(path) as image:
                batch.append(predictor.prepare_image(image))
        features, _ = predictor.species_features(np.concatenate(batch))
        parts.append(features)
        done = min(start + BATCH_SIZE, len(paths))
        if done % (BATCH_SIZE * 20) == 0 or done == len(paths):
            print(f"  {done}/{len(paths)}")
    return np.concatenate(parts)


def print_existing():
    """Prints the settings of the eye_check.npz that is already there (if any)."""
    if not OUT_PATH.exists():
        return
    old = np.load(OUT_PATH)
    print(f"Existing {OUT_PATH.name}: {old['reference'].shape[0]} reference photos, "
          f"threshold {float(old['threshold']):.4f}, k={int(old['k'])}")


def main():
    if not SPLIT_CSV.exists() or not DATA_DIR.is_dir():
        print("Missing data/ffe_split.csv or data/raw/ffe.")
        return
    try:
        predictor = FishPredictor(MODELS_DIR)
    except FileNotFoundError as error:
        print(f"Cannot load the models: {error}")
        return

    loader = DatasetLoader(DATA_DIR)
    loader.load_split(SPLIT_CSV)

    train_features = split_features(predictor, loader, "train")
    val_features = split_features(predictor, loader, "val")

    checker = EyeChecker(train_features, k=K_NEIGHBOURS)
    threshold = checker.fit_threshold(val_features, keep=KEEP_FRACTION)
    passed = checker.is_fish_eye(val_features).mean()

    print(f"\nReference set: {len(train_features)} training photos, k={K_NEIGHBOURS}")
    print(f"Threshold (distance {KEEP_FRACTION:.0%} of validation eye photos stay under): "
          f"{threshold:.4f}")
    print(f"Validation eye photos that pass: {passed:.1%}")
    print_existing()

    if OUT_PATH.exists() and not ask_yes_no(f"\nReplace {OUT_PATH}?"):
        print("Kept the existing file. Nothing changed.")
        return
    checker.save(OUT_PATH)
    print(f"Saved {OUT_PATH}")


main()
