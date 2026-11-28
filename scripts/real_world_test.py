"""
real_world_test.py
Runs FishScan v1 on our own market photos (eye crops) as a REAL-WORLD test.

Input: a folder of eye crops named <Species>_<NN>_eye.jpg (made from our market photos,
kept OUT of GitHub). Bangus and Tilapia are species the model knows; every other species
is "unknown" - the right answer for those is "Not recognized" (or at least a low
confidence), not a confident wrong species.

For each photo it saves: the top species and its probability, the top freshness level,
the eye-check distance and its zone (normal / warning / not_recognized), using the
cut-offs saved in models/eye_check.npz (chosen on FFE validation only). Nothing is
tuned on these photos.

Saves <reports>/real_world_test.csv and real_world_test.png (contact sheet).

Run in Colab:  %run scripts/real_world_test.py   (see notebooks/train_colab.ipynb)
"""

import os
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))  # so "from src..." works

import pandas as pd
from PIL import Image, ImageDraw

from src.fish_predictor import FishPredictor

CROPS_DIR = Path(os.environ.get("FISHSCAN_CROPS_DIR",
                                PROJECT_DIR / "data" / "raw" / "real_world" / "fish_eye_crops"))
REPORTS_DIR = Path(os.environ.get("FISHSCAN_REPORTS_DIR", PROJECT_DIR / "reports"))
MODELS_DIR = PROJECT_DIR / "models"

# File-name species -> the class name the model uses (others are unknown species)
KNOWN = {"Bangus": "Bangus (Milkfish)", "Tilapia": "Tilapia"}
ZONE_TEXT = {"normal": "normal", "warning": "WARNING note", "not_recognized": "NOT recognized"}


def test_photo(predictor, path):
    """One row of results for one eye crop."""
    species_name = path.stem.split("_")[0]
    true_class = KNOWN.get(species_name, "")
    with Image.open(path) as image:
        result = predictor.predict(image)
    species, freshness, eye = result["species"], result["freshness"], result["eye_check"]
    return {
        "file": path.name,
        "species_in_photo": species_name,
        "known_species": bool(true_class),
        "predicted_species": species["top_class"],
        "species_probability": round(species["top_probability"], 4),
        "species_correct": (species["top_class"] == true_class) if true_class else None,
        "predicted_freshness": freshness["top_class"],
        "freshness_probability": round(freshness["top_probability"], 4),
        "eye_distance": round(eye.get("distance", float("nan")), 4),
        "eye_zone": eye["zone"],
        "results_shown": eye["recognized"],   # normal or warning zone
    }


def contact_sheet(table, out_path):
    """Grid of the crops with the prediction under each one."""
    cols, size = 6, 200
    rows = (len(table) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * size, rows * (size + 58) + 30), "white")
    draw = ImageDraw.Draw(sheet)
    draw.text((6, 8), "Real-world test: our market photos (eye crops), FishScan v1", fill="black")
    for k, row in enumerate(table.itertuples()):
        x, y = (k % cols) * size, 30 + (k // cols) * (size + 58)
        with Image.open(CROPS_DIR / row.file) as image:
            thumb = image.convert("RGB")
            thumb.thumbnail((size - 6, size - 6))
        sheet.paste(thumb, (x + 3, y + 3))
        if row.known_species:
            color = "green" if row.species_correct else "red"
        else:
            # any species name is wrong for an unknown fish: red if the app would show it,
            # grey if the eye check hides it behind "Not recognized"
            color = "red" if row.results_shown else "gray"
        draw.text((x + 3, y + size), f"{row.species_in_photo}", fill="black")
        draw.text((x + 3, y + size + 12),
                  f"{row.predicted_species[:16]} {row.species_probability:.0%}", fill=color)
        draw.text((x + 3, y + size + 24),
                  f"{row.predicted_freshness} {row.freshness_probability:.0%}", fill="black")
        draw.text((x + 3, y + size + 36),
                  f"eye {row.eye_distance:.3f} {ZONE_TEXT[row.eye_zone]}", fill="black")
    sheet.save(out_path)


def print_summary(table):
    """Plain-words summary for the defense."""
    known = table[table["known_species"]]
    unknown = table[~table["known_species"]]
    print(f"\nKnown species (Bangus, Tilapia): {int(known['species_correct'].sum())} of "
          f"{len(known)} species correct; {int(known['results_shown'].sum())} of "
          f"{len(known)} get results (normal or warning zone).")
    print(f"Unknown species: {int((~unknown['results_shown']).sum())} of {len(unknown)} "
          "correctly 'Not recognized'.")
    confident_wrong = unknown[unknown["results_shown"] & (unknown["species_probability"] >= 0.6)]
    if len(confident_wrong):
        print(f"  WARNING: {len(confident_wrong)} unknown-species photos get results WITH a "
              f"confident species: {', '.join(confident_wrong['file'])}")
    print("\nEye-check zones (cut-offs from FFE validation only):")
    print(table.groupby(["known_species", "eye_zone"]).size().to_string())
    print("\nAll photos:")
    columns = ["file", "predicted_species", "species_probability", "predicted_freshness",
               "freshness_probability", "eye_distance", "eye_zone"]
    print(table[columns].to_string(index=False))


def main():
    paths = sorted(CROPS_DIR.glob("*_eye.jpg"))
    if not paths:
        print(f"No *_eye.jpg crops in {CROPS_DIR} (set FISHSCAN_CROPS_DIR).")
        return
    predictor = FishPredictor(MODELS_DIR)
    checker = predictor.eye_checker
    if checker is not None:
        warning = checker.warning_threshold
        print(f"Eye-check cut-offs: normal up to {checker.threshold:.4f}, warning up to "
              f"{'(none)' if warning is None else f'{warning:.4f}'}")
    print(f"Testing {len(paths)} eye crops from {CROPS_DIR}")
    table = pd.DataFrame([test_photo(predictor, path) for path in paths])
    print_summary(table)

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    table.to_csv(REPORTS_DIR / "real_world_test.csv", index=False)
    contact_sheet(table, REPORTS_DIR / "real_world_test.png")
    print(f"\nSaved real_world_test.csv and real_world_test.png to {REPORTS_DIR}")


main()
