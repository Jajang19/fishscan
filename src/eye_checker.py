"""
eye_checker.py
EyeChecker answers ONE question before FishScan shows any result:
"Does this photo look like the fish-eye close-ups the model learned from?"

Why we need it: softmax always splits 100% among the classes it knows, so a
photo of a hand can still come out as "Tilapia 99%". The probability only
says "more tilapia-like than the other 6 species", not "this is a fish eye".

How it works (no retraining):
1. The species model turns every photo into 1,280 numbers ("features").
2. We saved the features of all TRAINING eye photos (the reference set).
3. For a new photo we find its k most similar training photos (cosine
   similarity) and turn that into a distance:
       distance = 1 - average similarity of the k closest training photos
   Small distance = looks like our eye photos. Large = looks like something else.
4. Two cut-offs, both set on VALIDATION eye photos only (never on other photos):
     threshold          = the distance 95% of real validation eyes stay under
     warning_threshold  = the distance 99% of real validation eyes stay under
   That gives three zones:
     distance <= threshold                      -> "normal": show the results
     threshold < distance <= warning_threshold  -> "warning": show the results
                                                    with an "unusual photo" note
     distance > warning_threshold               -> "not_recognized"

This file only DEFINES things. Nothing runs when it is imported.
"""

from pathlib import Path

import numpy as np

K_NEIGHBOURS = 5              # how many closest training photos we average over
KEEP_FRACTION = 0.95          # share of real validation eyes with NO warning
WARNING_KEEP_FRACTION = 0.99  # share of real validation eyes that get results at all

ZONE_NORMAL = "normal"
ZONE_WARNING = "warning"
ZONE_NOT_RECOGNIZED = "not_recognized"


def normalize(features):
    """Scales every feature vector to length 1, so a dot product = cosine similarity."""
    features = np.asarray(features, dtype=np.float32)
    lengths = np.linalg.norm(features, axis=1, keepdims=True)
    return features / np.maximum(lengths, 1e-12)


class EyeChecker:
    """Decides if photos are similar enough to the training fish-eye photos."""

    def __init__(self, reference_features, threshold=None, k=K_NEIGHBOURS,
                 warning_threshold=None):
        # Private attributes (encapsulation)
        self._reference = normalize(reference_features)  # (number of training photos, 1280)
        self._threshold = threshold
        self._warning_threshold = warning_threshold      # None = no warning zone
        self._k = k

    @property
    def threshold(self):
        return self._threshold

    @property
    def warning_threshold(self):
        """Upper end of the warning zone, or None if this file has no warning zone."""
        return self._warning_threshold

    @property
    def k(self):
        return self._k

    def distances(self, features):
        """Distance of each photo to its k closest training photos (0 = identical)."""
        similarity = normalize(features) @ self._reference.T  # (photos, training photos)
        closest = np.sort(similarity, axis=1)[:, -self._k:]   # the k highest similarities
        return 1.0 - closest.mean(axis=1)

    def fit_threshold(self, validation_features, keep=KEEP_FRACTION,
                      warning_keep=WARNING_KEEP_FRACTION):
        """Sets both cut-offs from the validation eye photos.

        threshold: `keep` (95%) of the validation eyes are at or below it.
        warning_threshold: `warning_keep` (99%) of them are at or below it.
        Uses ONLY validation photos; the test set and any other photos are
        never used to choose the cut-offs.
        """
        distances = self.distances(validation_features)
        self._threshold = float(np.quantile(distances, keep))
        self._warning_threshold = float(np.quantile(distances, warning_keep))
        return self._threshold

    def set_warning_threshold(self, value):
        """Stores a warning cut-off measured on validation eyes (must be >= threshold)."""
        if self._threshold is None or value < self._threshold:
            raise ValueError("The warning cut-off must be at least the normal threshold.")
        self._warning_threshold = float(value)

    def zone(self, distance):
        """'normal', 'warning' or 'not_recognized' for one distance."""
        if self._threshold is None:
            raise ValueError("The threshold has not been set yet (call fit_threshold).")
        if distance <= self._threshold:
            return ZONE_NORMAL
        if self._warning_threshold is not None and distance <= self._warning_threshold:
            return ZONE_WARNING
        return ZONE_NOT_RECOGNIZED

    def is_fish_eye(self, features):
        """True/False for each photo: results are shown (normal or warning zone)."""
        return np.array([self.zone(d) != ZONE_NOT_RECOGNIZED for d in self.distances(features)])

    # ------------------------------------------------------------------
    # Saving and loading (one small .npz file in models/)
    # ------------------------------------------------------------------
    def save(self, path):
        """Saves the reference features (as float16 to keep the file small) and settings."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        extra = {}
        if self._warning_threshold is not None:
            extra["warning_threshold"] = np.float32(self._warning_threshold)
        np.savez_compressed(path, reference=self._reference.astype(np.float16),
                            threshold=np.float32(self._threshold), k=np.int32(self._k), **extra)

    @classmethod
    def load(cls, path):
        """Loads a saved checker. Raises FileNotFoundError if the file is missing.
        Older files have no warning_threshold: then there is no warning zone."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Missing {path.name}")
        data = np.load(path)
        warning = float(data["warning_threshold"]) if "warning_threshold" in data.files else None
        return cls(data["reference"].astype(np.float32),
                   float(data["threshold"]), int(data["k"]), warning)
