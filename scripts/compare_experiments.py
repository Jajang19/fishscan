"""
compare_experiments.py
Compares the baseline freshness model with the freshness experiments
using the VALIDATION set ONLY. The test set is NOT touched here; it is used
once, later, for the model we pick.

For the species-aware experiments (21 classes), freshness probabilities are
made by adding up the species probabilities of each level.

Decision rule (agreed before running the experiments):
  an experiment replaces the baseline only if its validation freshness
  accuracy is at least 3 points higher AND its calibration error (ECE)
  is not worse (we allow +0.01 for noise).

Run locally:   python scripts/compare_experiments.py
Run in Colab:  %run scripts/compare_experiments.py
"""

import os
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, log_loss

from src.dataset_loader import DatasetLoader, freshness_from_combined, FRESHNESS_LEVELS
from src.fish_model import FishModel
from src.model_evaluator import ModelEvaluator

DATA_DIR = Path(os.environ.get("FISHSCAN_DATA_DIR", PROJECT_DIR / "data" / "raw" / "ffe"))
SPLIT_CSV = PROJECT_DIR / "data" / "ffe_split.csv"
MODELS_DIR = PROJECT_DIR / "models"
OUT_CSV = PROJECT_DIR / "reports" / "experiments" / "comparison_val.csv"

MIN_GAIN = 0.03        # +3 accuracy points needed to replace the baseline
ECE_TOLERANCE = 0.01   # calibration may be at most this much worse
THRESHOLD = 0.60       # the app's "not sure" threshold

# (name shown in the table, folder, task the model was trained on)
CANDIDATES = [
    ("Baseline", MODELS_DIR, "freshness"),
    ("E1_species_aware", MODELS_DIR / "experiments" / "E1_species_aware", "species_freshness"),
    ("E2_deep_strong_aug", MODELS_DIR / "experiments" / "E2_deep_strong_aug", "freshness"),
    ("E3_species_aware_deep_strong",
     MODELS_DIR / "experiments" / "E3_species_aware_deep_strong", "species_freshness"),
]


def predict_val_freshness(loader, model, task):
    """Returns freshness probabilities (n, 3) for the validation images."""
    parts = [model.predict_probabilities(images.numpy())
             for images, _ in loader.make_dataset("val", task)]
    probabilities = np.concatenate(parts)
    if task == "species_freshness":
        probabilities = freshness_from_combined(probabilities, model.class_names)
    return probabilities


def score(y_true, probabilities):
    """All comparison numbers for one model (validation set)."""
    y_pred = probabilities.argmax(axis=1)
    confident = probabilities.max(axis=1) >= THRESHOLD
    return {
        "val_accuracy": accuracy_score(y_true, y_pred),
        "val_macro_f1": f1_score(y_true, y_pred, average="macro"),
        "val_log_loss": log_loss(y_true, probabilities, labels=range(len(FRESHNESS_LEVELS))),
        "val_ece": ModelEvaluator.expected_calibration_error(y_true, probabilities),
        "share_above_60": confident.mean(),
        "accuracy_above_60": accuracy_score(y_true[confident], y_pred[confident])
                             if confident.any() else float("nan"),
    }


def print_table(table):
    """Prints the comparison table with readable formatting."""
    shown = table.copy()
    for column in ("val_accuracy", "val_macro_f1", "share_above_60", "accuracy_above_60"):
        shown[column] = (shown[column] * 100).map("{:.1f}%".format)
    for column in ("val_log_loss", "val_ece"):
        shown[column] = shown[column].map("{:.3f}".format)
    print("\nFreshness on the VALIDATION set (613 images):")
    print(shown.to_string())


def print_decision(table):
    """Applies the agreed decision rule and says whether to run E3."""
    base = table.loc["Baseline"]
    winners = []
    print(f"\nDecision rule: at least +{MIN_GAIN:.0%} accuracy and ECE not worse "
          f"than baseline + {ECE_TOLERANCE}")
    for name, row in table.drop(index="Baseline").iterrows():
        gain = row["val_accuracy"] - base["val_accuracy"]
        beats = gain >= MIN_GAIN and row["val_ece"] <= base["val_ece"] + ECE_TOLERANCE
        print(f"  {name}: {gain * 100:+.1f} points, ECE {row['val_ece']:.3f} -> "
              f"{'BEATS the baseline' if beats else 'does not beat the baseline'}")
        if beats:
            winners.append(name)

    tried = set(table.index)
    if "E3_species_aware_deep_strong" not in tried and {"E1_species_aware", "E2_deep_strong_aug"} <= tried:
        both = {"E1_species_aware", "E2_deep_strong_aug"} <= set(winners)
        print("\nE3: " + ("RUN IT - both E1 and E2 beat the baseline."
                          if both else "skip it - E1 and E2 did not both beat the baseline."))
    if winners:
        best = table.loc[winners, "val_accuracy"].idxmax()
        print(f"Best candidate so far: {best}")
    else:
        print("Best candidate so far: Baseline (keep the current freshness model).")


def main():
    loader = DatasetLoader(DATA_DIR)
    loader.load_split(SPLIT_CSV)
    y_true = loader.labels("val", "freshness")

    rows = {}
    for name, folder, task in CANDIDATES:
        try:
            model = FishModel.load(folder, task)
        except FileNotFoundError:
            continue  # this experiment has not been trained yet
        print(f"Scoring {name} ...")
        rows[name] = score(y_true, predict_val_freshness(loader, model, task))

    if "Baseline" not in rows:
        print("The baseline model (models/freshness_model.keras) is missing.")
        return
    if len(rows) == 1:
        print("No experiments found in models/experiments/. Train E1/E2 first.")
        return

    table = pd.DataFrame(rows).transpose()
    print_table(table)
    print_decision(table)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(OUT_CSV)
    print(f"\nSaved the table to {OUT_CSV}")


main()
