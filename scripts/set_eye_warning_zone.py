"""
set_eye_warning_zone.py
Adds the warning-zone cut-off to the fish-eye check file models/eye_check.npz.

The cut-off is the distance that 99% of the FFE VALIDATION eye photos stay under.
It was measured by scripts/eye_check_tradeoff.py (in Colab); it is NOT chosen on our
market photos. The value is read from reports/eye_check_tradeoff.json if that file
is here, otherwise you type it in.

Everything else in the file (the 3,113 reference features, the 95% threshold, k)
is copied exactly as it is - only "warning_threshold" is added or replaced.

Run from the project folder:  python scripts/set_eye_warning_zone.py
"""

import json
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))  # so "from src..." works

import numpy as np

from src.eye_checker import EyeChecker

EYE_CHECK_FILE = PROJECT_DIR / "models" / "eye_check.npz"
TRADEOFF_JSON = PROJECT_DIR / "reports" / "eye_check_tradeoff.json"


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


def ask_cutoff(threshold):
    """Asks for the 99% cut-off (a number between the threshold and 1)."""
    while True:
        try:
            value = float(input("99% cut-off from eye_check_tradeoff.py (e.g. 0.2838): "))
            if not threshold <= value < 1:
                raise ValueError
            break
        except ValueError:
            print(f"Please type a number from {threshold:.4f} up to 1.")
            continue
    return value


def main():
    if not EYE_CHECK_FILE.exists():
        print(f"Missing {EYE_CHECK_FILE}")
        return
    data = dict(np.load(EYE_CHECK_FILE))   # all arrays, exactly as stored
    threshold = float(data["threshold"])
    old = float(data["warning_threshold"]) if "warning_threshold" in data else None
    print(f"Normal threshold (95% of validation eyes): {threshold:.4f}")
    print(f"Current warning cut-off: {'none' if old is None else f'{old:.4f}'}")

    if TRADEOFF_JSON.exists():
        with open(TRADEOFF_JSON, encoding="utf-8") as f:
            cutoff = float(json.load(f)["keep_99"])
        print(f"99% cut-off from {TRADEOFF_JSON.name}: {cutoff:.4f}")
    else:
        cutoff = ask_cutoff(threshold)

    print(f"\nZones: up to {threshold:.4f} normal | up to {cutoff:.4f} results with a warning "
          f"| above that 'Not recognized'")
    if not ask_yes_no(f"Save this warning cut-off in {EYE_CHECK_FILE.name}?"):
        print("Nothing changed.")
        return
    data["warning_threshold"] = np.float32(cutoff)
    np.savez_compressed(EYE_CHECK_FILE, **data)

    checker = EyeChecker.load(EYE_CHECK_FILE)   # read back to make sure it works
    print(f"Saved. Loaded again: threshold {checker.threshold:.4f}, "
          f"warning cut-off {checker.warning_threshold:.4f}, k={checker.k}")


main()
