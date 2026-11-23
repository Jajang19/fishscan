"""
evaluate.py
Tests trained models on the TEST split (images the models never saw)
and saves all charts and tables to reports/ for the Model Performance page.

Options:
  1-3. the normal species / freshness models  -> reports/<task>_...
  4.   ONE freshness experiment (the winner of compare_experiments.py)
       -> reports/experiments/freshness_<experiment>_...
       Do this only ONCE, for the experiment you picked on the validation set.

Needs: the model files, data/ffe_split.csv and the FFE images in data/raw/ffe.
Optional: the history CSV from Colab for the training curves.

Run from the project folder:  python scripts/evaluate.py
"""

import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))

import numpy as np
import pandas as pd

from src.dataset_loader import DatasetLoader, freshness_from_combined, FRESHNESS_LEVELS
from src.fish_model import FishModel
from src.model_evaluator import ModelEvaluator

DATA_DIR = PROJECT_DIR / "data" / "raw" / "ffe"
SPLIT_CSV = PROJECT_DIR / "data" / "ffe_split.csv"
MODELS_DIR = PROJECT_DIR / "models"
REPORTS_DIR = PROJECT_DIR / "reports"

# experiment name -> task it was trained on (same names as in train.py)
EXPERIMENTS = {
    "E1_species_aware": "species_freshness",
    "E2_deep_strong_aug": "freshness",
    "E3_species_aware_deep_strong": "species_freshness",
}


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


def choose_jobs():
    """Returns a list of jobs: (report_name, model_folder, model_task, reports_folder, history_csv)."""
    choice = ask_number("Which model do you want to evaluate on the TEST set?", {
        1: "Species", 2: "Freshness (baseline)", 3: "Both",
        4: "A freshness experiment (final pick only, run once)"})
    if choice in (1, 2, 3):
        tasks = {1: ["species"], 2: ["freshness"], 3: ["species", "freshness"]}[choice]
        return [(task, MODELS_DIR, task, REPORTS_DIR, REPORTS_DIR / f"{task}_history.csv")
                for task in tasks]

    names = list(EXPERIMENTS)
    pick = ask_number("\nWhich experiment did you pick on the validation set?",
                      {i + 1: name for i, name in enumerate(names)})
    name = names[pick - 1]
    print("\nThis looks at the TEST set. Only do this once, for the model you picked.")
    if not ask_yes_no("Continue?"):
        return []
    exp_reports = REPORTS_DIR / "experiments"
    return [(f"freshness_{name}", MODELS_DIR / "experiments" / name, EXPERIMENTS[name],
             exp_reports, exp_reports / f"{name}_history.csv")]


def predict_test_set(loader, model, model_task):
    """Runs the model on every test image. Returns (true labels, probabilities).

    A 21-class species-aware model is turned into 3 freshness probabilities
    by adding up the species probabilities of each level.
    """
    parts = [model.predict_probabilities(images.numpy())
             for images, _ in loader.make_dataset("test", model_task)]
    probabilities = np.concatenate(parts)
    if model_task == "species_freshness":
        return loader.labels("test", "freshness"), freshness_from_combined(probabilities, model.class_names)
    return loader.labels("test", model_task), probabilities


def print_summary(summary, report_csv):
    """Prints accuracy, confidence, calibration and the per-class table."""
    print(f"\nTest images:        {summary['test_images']}")
    print(f"Test accuracy:      {summary['accuracy']:.1%}")
    print(f"Macro F1:           {summary['macro_f1']:.3f}")
    print(f"Average confidence: {summary['average_confidence']:.1%}")
    print(f"Calibration (ECE):  {summary['ece']:.3f}  -> {summary['calibration_verdict']}")
    print(f"  {summary['calibration_explanation']}")
    if summary["small_test_classes"]:
        print(f"  Note: few test images for {summary['small_test_classes']}, "
              "so their metrics are noisy.")
    print("\nPer-class results (test set):")
    print(pd.read_csv(report_csv, index_col=0).round(3).to_string())


def evaluate_one(loader, report_name, model_dir, model_task, reports_dir, history_csv):
    """Loads one model, evaluates it on the test set, and saves all reports."""
    print(f"\n========== Evaluating: {report_name} ==========")
    try:
        model = FishModel.load(model_dir, model_task)
    except FileNotFoundError as error:
        print(f"Cannot evaluate: {error}")
        return

    true_labels, probabilities = predict_test_set(loader, model, model_task)
    class_names = FRESHNESS_LEVELS if model_task == "species_freshness" else model.class_names
    evaluator = ModelEvaluator(report_name, class_names, reports_dir)
    summary = evaluator.evaluate(true_labels, probabilities)
    print_summary(summary, reports_dir / f"{report_name}_classification_report.csv")

    test_paths = [DATA_DIR / p for p in loader.table.query("split == 'test'")["path"]]
    evaluator.save_examples(test_paths, true_labels, probabilities)

    if evaluator.save_training_curves(history_csv):
        print("\nSaved the training curves.")
    else:
        print(f"\nNo {history_csv.name} found - download it from Google Drive to get the curves.")
    print(f"All results are saved in {reports_dir}")


def main():
    if not SPLIT_CSV.exists() or not DATA_DIR.is_dir():
        print("Missing data/ffe_split.csv or data/raw/ffe. See data/DATASETS.md.")
        return
    jobs = choose_jobs()
    loader = DatasetLoader(DATA_DIR)
    loader.load_split(SPLIT_CSV)
    for job in jobs:
        evaluate_one(loader, *job)


main()
