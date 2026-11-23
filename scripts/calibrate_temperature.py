"""
calibrate_temperature.py
Temperature scaling for the freshness model, using the VALIDATION set only
(the test set is never touched here).

Why: E2_deep_strong_aug was +6 accuracy points better than the baseline on
validation, but its calibration error (ECE 0.086) was too high for our
decision rule. Temperature scaling (src/fish_model.py) can fix the
confidence WITHOUT changing any prediction, so the accuracy stays the same.

To keep the check honest, the 613 validation images are cut in two halves
(grouped by source photo, stratified by species + freshness):
  half 1 -> used to FIT the temperature T
  half 2 -> used to CHECK the result (never seen while fitting)
The table for half 2 is saved to reports/experiments/temperature_scaling_half2.csv.

Options:
  1. The E2 experiment (models/experiments/E2_deep_strong_aug), compared with
     the baseline in models/ (this is how the final model was chosen).
  2. The current final freshness model in models/ (re-check only; its saved
     temperature is ignored while measuring "unscaled").

Nothing is saved without asking. Saving T only rewrites the small
<task>_meta.joblib file; the .keras model itself is never changed.

Run from the project folder:  python scripts/calibrate_temperature.py

(Rebuilt on 2026-11-23 from src/fish_model.py and
reports/experiments/temperature_scaling_half2.csv after the original was lost.)
"""

import os
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))  # so "from src..." works

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, log_loss
from sklearn.model_selection import StratifiedGroupKFold

from src.dataset_loader import DatasetLoader, FRESHNESS_LEVELS
from src.fish_model import FishModel, apply_temperature, fit_temperature
from src.model_evaluator import ModelEvaluator

DATA_DIR = Path(os.environ.get("FISHSCAN_DATA_DIR", PROJECT_DIR / "data" / "raw" / "ffe"))
SPLIT_CSV = PROJECT_DIR / "data" / "ffe_split.csv"
MODELS_DIR = PROJECT_DIR / "models"
E2_DIR = MODELS_DIR / "experiments" / "E2_deep_strong_aug"
OUT_CSV = PROJECT_DIR / "reports" / "experiments" / "temperature_scaling_half2.csv"

THRESHOLD = 0.60  # the app's "not sure" threshold
SEED = 42


def ask_number(question, options):
    """Shows numbered options and returns the chosen number."""
    print(question)
    for number, text in options.items():
        print(f"  {number}. {text}")
    while True:
        try:
            choice = int(input("Your choice: "))
            if choice not in options:
                raise ValueError
            break
        except ValueError:
            print(f"Please type one of: {', '.join(str(n) for n in options)}")
            continue
    return choice


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


def split_validation(loader):
    """Cuts the validation images in two halves (positions 0..n-1 in split order).

    Grouped: all crops of one source photo stay in the same half.
    Stratified: both halves get about the same species + freshness mix.
    """
    val = loader.table[loader.table["split"] == "val"].reset_index(drop=True)
    strata = val["species"] + " | " + val["freshness"]
    halves = StratifiedGroupKFold(n_splits=2, shuffle=True, random_state=SEED)
    fit_idx, check_idx = next(halves.split(val, strata, val["source_photo"]))
    return fit_idx, check_idx


def predict_val(loader, model):
    """Freshness probabilities (n, 3) for all validation images, WITHOUT temperature."""
    model.set_temperature(1.0)
    parts = [model.predict_probabilities(images.numpy())
             for images, _ in loader.make_dataset("val", "freshness")]
    return np.concatenate(parts)


def score(y_true, probabilities):
    """All comparison numbers for one row of the table."""
    y_pred = probabilities.argmax(axis=1)
    confidence = probabilities.max(axis=1)
    confident = confidence >= THRESHOLD
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "log_loss": log_loss(y_true, probabilities, labels=range(len(FRESHNESS_LEVELS))),
        "ece": ModelEvaluator.expected_calibration_error(y_true, probabilities),
        "avg_confidence": float(confidence.mean()),
        "share_above_60": float(confident.mean()),
        "accuracy_above_60": accuracy_score(y_true[confident], y_pred[confident])
                             if confident.any() else float("nan"),
    }


def print_table(table, n_images):
    """Prints the half-2 table with readable formatting."""
    shown = table.copy()
    for column in ("accuracy", "avg_confidence", "share_above_60", "accuracy_above_60"):
        shown[column] = (shown[column] * 100).map("{:.1f}%".format)
    for column in ("log_loss", "ece"):
        shown[column] = shown[column].map("{:.3f}".format)
    print(f"\nFreshness on validation HALF 2 ({n_images} images, not used for fitting):")
    print(shown.to_string())


def main():
    if not SPLIT_CSV.exists() or not DATA_DIR.is_dir():
        print("Missing data/ffe_split.csv or data/raw/ffe.")
        return

    choice = ask_number("Which freshness model do you want to calibrate?", {
        1: f"E2 experiment ({E2_DIR.relative_to(PROJECT_DIR)}), compared with the baseline",
        2: "Current final freshness model (models/), re-check only"})
    folder, name = (E2_DIR, "E2") if choice == 1 else (MODELS_DIR, "Final")
    try:
        model = FishModel.load(folder, "freshness")
    except FileNotFoundError as error:
        print(f"Cannot calibrate: {error}")
        return
    saved_temperature = model.temperature
    print(f"Temperature currently saved with this model: {saved_temperature:.4f}")

    loader = DatasetLoader(DATA_DIR)
    loader.load_split(SPLIT_CSV)
    y_true = loader.labels("val", "freshness")
    fit_idx, check_idx = split_validation(loader)
    print(f"Validation halves: {len(fit_idx)} images to fit T, {len(check_idx)} to check it")

    rows = {}
    if choice == 1:
        try:
            baseline = FishModel.load(MODELS_DIR, "freshness")
            rows["Baseline"] = score(y_true[check_idx], predict_val(loader, baseline)[check_idx])
        except FileNotFoundError:
            print("(No baseline model in models/, so the Baseline row is skipped.)")

    print(f"Predicting the validation set with {name} ...")
    probabilities = predict_val(loader, model)
    temperature = fit_temperature(probabilities[fit_idx], y_true[fit_idx])
    scaled = apply_temperature(probabilities, temperature)
    rows[f"{name} unscaled"] = score(y_true[check_idx], probabilities[check_idx])
    rows[f"{name} scaled (T={temperature:.2f})"] = score(y_true[check_idx], scaled[check_idx])

    table = pd.DataFrame(rows).transpose()
    print(f"\nFitted temperature on half 1: T = {temperature:.4f}"
          f"  ({'softer' if temperature > 1 else 'sharper'} probabilities)")
    print_table(table, len(check_idx))
    print("Accuracy is the same with and without T: scaling never changes the top class.")

    if not OUT_CSV.exists() or ask_yes_no(f"\nReplace {OUT_CSV.relative_to(PROJECT_DIR)}?"):
        OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
        table.to_csv(OUT_CSV)
        print(f"Saved the table to {OUT_CSV}")

    if ask_yes_no(f"\nSave T={temperature:.4f} in {folder.relative_to(PROJECT_DIR)}/freshness_meta.joblib "
                  f"(was {saved_temperature:.4f})?"):
        model.set_temperature(temperature)
        model.save_metadata(folder)
        print("Saved. The app applies this temperature to every freshness prediction.")
    else:
        print("Temperature not saved. Nothing changed.")


main()
