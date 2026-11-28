"""
fish_predictor.py
FishPredictor takes ONE fish-eye photo and returns, for both models
(species and freshness):
  - the probability of every class (from the softmax output),
  - the top-1 class and its probability,
  - a low-confidence flag (top probability below 60%).
It also gives a rough freshness guide that is based ONLY on how the FFE
dataset defines its labels (storage days), not on invented shelf-life numbers.

This file only DEFINES things. Nothing runs when it is imported.
"""

from pathlib import Path

import numpy as np
import tensorflow as tf
from PIL import Image, ImageOps

from src.dataset_loader import SPECIES_NAMES
from src.eye_checker import EyeChecker
from src.fish_model import FishModel

CONFIDENCE_THRESHOLD = 0.60
EYE_CHECK_FILE = "eye_check.npz"  # made by scripts/build_eye_check.py

# Freshness guide, taken from the FFE dataset's label definitions.
# The dataset does not document how the fish were stored, so we do NOT turn
# these days into hours or a shelf life; we only say what the photo looks like.
FRESHNESS_GUIDE = {
    "Highly Fresh": {
        "dataset_days": "storage day 1-2",
        "color": "#2e7d32",  # green
        "icon": "🟢",
        "advice": "Looks like the freshest fish in the dataset. Keep it cold and cook it soon.",
    },
    "Fresh": {
        "dataset_days": "storage day 3-4",
        "color": "#f9a825",  # amber
        "icon": "🟡",
        "advice": "Looks like fish a few days into storage. Cook it soon; do not keep it much longer.",
    },
    "Not Fresh": {
        "dataset_days": "storage day 5-6",
        "color": "#c62828",  # red
        "icon": "🔴",
        "advice": "Looks like the oldest fish in the dataset. Check the smell, gills and "
                  "firmness yourself. If in doubt, do not eat it.",
    },
}


def scientific_names_by_species():
    """{"Tilapia": ["Oreochromis Mossambicus", "Oreochromis Niloticus"], ...}"""
    names = {}
    for scientific, common in SPECIES_NAMES.items():
        names.setdefault(common, []).append(scientific)
    return names


class FishPredictor:
    """Loads the two trained models once and predicts from a photo."""

    def __init__(self, models_dir, threshold=CONFIDENCE_THRESHOLD):
        # FishModel.load raises FileNotFoundError if a model file is missing;
        # the app catches it and shows a friendly message.
        self._species_model = FishModel.load(models_dir, "species")
        self._freshness_model = FishModel.load(models_dir, "freshness")
        self._threshold = threshold
        # The fish-eye check is optional: if its file is missing, scans still work
        # and the result says the check was not available.
        try:
            self._eye_checker = EyeChecker.load(Path(models_dir) / EYE_CHECK_FILE)
        except FileNotFoundError:
            self._eye_checker = None

    @property
    def threshold(self):
        return self._threshold

    @staticmethod
    def crop_square(image, zoom=1.0, center_x=0.5, center_y=0.5):
        """Cuts the square the user chose on the Scan page (the eye).

        zoom 1 = the biggest square that fits; zoom 4 = a square 4x smaller.
        center_x / center_y = where the square's centre is, from 0 (left/top)
        to 1 (right/bottom). The square always stays inside the photo.
        With the defaults this is exactly the old centre-square crop.
        """
        image = ImageOps.exif_transpose(image).convert("RGB")
        width, height = image.size
        side = max(1, round(min(width, height) / max(zoom, 1.0)))
        left = min(max(round(center_x * width - side / 2), 0), width - side)
        top = min(max(round(center_y * height - side / 2), 0), height - side)
        return image.crop((left, top, left + side, top + side))

    def prepare_image(self, image):
        """PIL image -> array (1, 224, 224, 3) with pixels 0-255, like in training.

        1. Rotate the photo the way the phone meant it (EXIF orientation).
        2. Convert to RGB (removes transparency, handles grayscale).
        3. Cut out the centre square (the training images are square eye
           crops, so the eye should be in the middle of the photo).
        4. Resize to 224 x 224 with the same function used in training.
        """
        image = ImageOps.exif_transpose(image).convert("RGB")
        width, height = image.size
        side = min(width, height)
        left, top = (width - side) // 2, (height - side) // 2
        image = image.crop((left, top, left + side, top + side))

        pixels = np.asarray(image)
        size = self._species_model.image_size
        resized = tf.image.resize(pixels, (size, size))
        resized = tf.cast(resized, tf.uint8)  # same rounding as the training pipeline
        return np.asarray(resized, dtype=np.float32)[np.newaxis, ...]

    @property
    def eye_checker(self):
        return self._eye_checker

    @property
    def species_class_names(self):
        return self._species_model.class_names

    def species_features(self, batch):
        """Features + species probabilities for a batch made by prepare_image
        (used by the scripts that build and test the fish-eye check)."""
        return self._species_model.predict_features_and_probabilities(batch)

    def predict(self, image):
        """Returns {"eye_check": {...}, "species": {...}, "freshness": {...}} for one PIL image."""
        batch = self.prepare_image(image)
        features, species_probabilities = self.species_features(batch)
        species = self._describe(self._species_model.class_names, species_probabilities[0])
        freshness = self._describe(self._freshness_model.class_names,
                                   self._freshness_model.predict_probabilities(batch)[0])
        freshness["guide"] = FRESHNESS_GUIDE[freshness["top_class"]]
        return {"eye_check": self._check_eye(features),
                "species": species, "freshness": freshness}

    def _check_eye(self, features):
        """Runs the fish-eye check (if its file was found)."""
        if self._eye_checker is None:
            return {"available": False, "recognized": True}
        distance = float(self._eye_checker.distances(features)[0])
        return {"available": True,
                "recognized": distance <= self._eye_checker.threshold,
                "distance": distance,
                "threshold": self._eye_checker.threshold}

    def _describe(self, class_names, probabilities):
        """Turns one model's softmax output into an easy-to-show dictionary."""
        ranked = sorted(zip(class_names, probabilities.tolist()),
                        key=lambda pair: pair[1], reverse=True)
        top_class, top_probability = ranked[0]
        return {
            "probabilities": dict(ranked),        # every class, highest first
            "top_class": top_class,
            "top_probability": top_probability,
            "low_confidence": top_probability < self._threshold,
        }
