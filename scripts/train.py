"""
train.py
Trains the species model, the freshness model, or a freshness EXPERIMENT
on FFE eye photos.

  Stage 1: train only the new softmax head (MobileNetV2 frozen)
  Stage 2: fine-tune the last MobileNetV2 layers with a tiny learning rate

Uses the saved split in data/ffe_split.csv (made by scripts/make_split.py),
so the train/validation/test images are always the same.

Saves (normal models):
  models/<task>_model.keras, models/<task>_meta.joblib, reports/<task>_history.csv
Saves (experiments, never overwrite the normal models):
  models/experiments/<experiment>/..., reports/experiments/<experiment>_history.csv

Run locally:   python scripts/train.py
Run in Colab:  %run scripts/train.py   (see notebooks/train_colab.ipynb)
"""

import os
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))

import tensorflow as tf

from src.dataset_loader import DatasetLoader
from src.fish_model import FishModel

# In Colab the images are copied to a fast local folder; this env variable
# tells the script where they are. Locally it falls back to data/raw/ffe.
DATA_DIR = Path(os.environ.get("FISHSCAN_DATA_DIR", PROJECT_DIR / "data" / "raw" / "ffe"))
SPLIT_CSV = PROJECT_DIR / "data" / "ffe_split.csv"
MODELS_DIR = PROJECT_DIR / "models"
REPORTS_DIR = PROJECT_DIR / "reports"

IMAGE_SIZE = 224
BATCH_SIZE = 32

# Settings of the normal (baseline) models
BASELINE = {"augmentation": "standard", "head_epochs": 10, "fine_tune_epochs": 10,
            "unfreeze_last": 30, "patience": 3}

# Freshness experiments. Each one is compared with the baseline freshness
# model on the VALIDATION set (scripts/compare_experiments.py).
EXPERIMENTS = {
    "E1_species_aware": {
        "description": "21 classes (species x level), freshness = sum of probabilities; "
                       "same training settings as the baseline",
        "task": "species_freshness", **BASELINE,
    },
    "E2_deep_strong_aug": {
        "description": "3 classes; unfreeze 60 layers, up to 20 fine-tuning epochs, "
                       "patience 4, strong lighting/colour augmentation",
        "task": "freshness", "augmentation": "strong", "head_epochs": 10,
        "fine_tune_epochs": 20, "unfreeze_last": 60, "patience": 4,
    },
    "E3_species_aware_deep_strong": {
        "description": "E1's 21 classes + E2's settings (run ONLY if E1 and E2 both "
                       "beat the baseline)",
        "task": "species_freshness", "augmentation": "strong", "head_epochs": 10,
        "fine_tune_epochs": 20, "unfreeze_last": 60, "patience": 4,
    },
}


def ask_choice(question, options):
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


def print_setup():
    """Prints whether a GPU is available and where the data is."""
    gpus = tf.config.list_physical_devices("GPU")
    print(f"TensorFlow {tf.__version__} | GPU: {'YES - ' + gpus[0].name if gpus else 'NO (CPU only, slow)'}")
    print(f"Images folder: {DATA_DIR}")
    print(f"Split file:    {SPLIT_CSV}")


def choose_runs():
    """Asks what to train. Returns a list of (run_name, task, settings, models_dir, reports_dir)."""
    what = ask_choice("\nWhat do you want to train?", {
        1: "Species model",
        2: "Freshness model (baseline)",
        3: "Both normal models",
        4: "Freshness experiment(s)",
    })
    if what in (1, 2, 3):
        tasks = {1: ["species"], 2: ["freshness"], 3: ["species", "freshness"]}[what]
        return [(task, task, BASELINE, MODELS_DIR, REPORTS_DIR) for task in tasks]

    names = list(EXPERIMENTS)
    print("\nExperiments:")
    for name in names:
        print(f"  - {name}: {EXPERIMENTS[name]['description']}")
    pick = ask_choice("\nWhich experiment(s)?", {
        1: "E1 then E2 (back to back)",
        2: "E1 only",
        3: "E2 only",
        4: "E3 only",
    })
    chosen = {1: names[:2], 2: [names[0]], 3: [names[1]], 4: [names[2]]}[pick]
    return [(name, EXPERIMENTS[name]["task"], EXPERIMENTS[name],
             MODELS_DIR / "experiments" / name, REPORTS_DIR / "experiments")
            for name in chosen]


def train_one(run_name, task, settings, models_dir, reports_dir, quick):
    """Trains, saves and reports one model."""
    print(f"\n========== Training: {run_name} (task: {task}) ==========")
    if quick:
        # Quick test: a few batches and 1 epoch per stage, saved somewhere else
        models_dir = reports_dir = MODELS_DIR / "quick_test" / run_name
    elif (models_dir / f"{task}_model.keras").exists():
        if not ask_yes_no(f"{models_dir / (task + '_model.keras')} already exists. Overwrite it?"):
            print("Skipped.")
            return

    tf.keras.utils.set_random_seed(42)  # same random choices every run
    loader = DatasetLoader(DATA_DIR, IMAGE_SIZE, BATCH_SIZE,
                           augmentation=settings["augmentation"])
    loader.load_split(SPLIT_CSV)
    train_ds = loader.make_dataset("train", task)
    val_ds = loader.make_dataset("val", task)
    head_epochs, tune_epochs = settings["head_epochs"], settings["fine_tune_epochs"]
    if quick:
        train_ds, val_ds = train_ds.take(3), val_ds.take(2)
        head_epochs = tune_epochs = 1

    model = FishModel(task, loader.class_names(task), IMAGE_SIZE)
    model.build()
    weights = loader.class_weights(task)
    checkpoint = models_dir / "checkpoints" / f"{task}_best.keras"

    print(f"\n--- Stage 1: training the new head (augmentation: {settings['augmentation']}) ---")
    model.train(train_ds, val_ds, head_epochs, weights, checkpoint, patience=settings["patience"])

    print(f"\n--- Stage 2: fine-tuning the last {settings['unfreeze_last']} MobileNetV2 layers ---")
    model.fine_tune(train_ds, val_ds, tune_epochs, unfreeze_last=settings["unfreeze_last"],
                    class_weight=weights, checkpoint_path=checkpoint,
                    patience=settings["patience"])

    result = model.evaluate(val_ds)
    print(f"\nValidation accuracy ({task}, {len(model.class_names)} classes): "
          f"{result['accuracy']:.1%}  (loss {result['loss']:.3f})")
    if task == "species_freshness":
        print("(For freshness accuracy, run scripts/compare_experiments.py.)")

    model.save(models_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)
    history_name = f"{task}_history.csv" if models_dir == MODELS_DIR else f"{run_name}_history.csv"
    model.history_table().to_csv(reports_dir / history_name, index=False)
    print(f"Saved the model to {models_dir}")
    print(f"Saved {history_name} to {reports_dir}")


def main():
    print_setup()
    if not SPLIT_CSV.exists():
        print("No split file found. Run scripts/make_split.py first.")
        return
    if not DATA_DIR.is_dir():
        print(f"Images folder not found: {DATA_DIR}")
        return

    runs = choose_runs()
    mode = ask_choice("\nFull training or a quick test run?",
                      {1: "Full training", 2: "Quick test (checks that everything works)"})
    for run_name, task, settings, models_dir, reports_dir in runs:
        train_one(run_name, task, settings, models_dir, reports_dir, quick=(mode == 2))
    print("\nDone.")


main()
