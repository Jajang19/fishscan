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
4. The cut-off (threshold) was set on VALIDATION eye photos only: the distance
   that 95% of real validation eye photos stay under. Anything farther than
   that is "Not recognized".

This file only DEFINES things. Nothing runs when it is imported.
"""

from pathlib import Path

import numpy as np

K_NEIGHBOURS = 5      # how many closest training photos we average over
KEEP_FRACTION = 0.95  # share of real validation eye photos that must pass


def normalize(features):
    """Scales every feature vector to length 1, so a dot product = cosine similarity."""
    features = np.asarray(features, dtype=np.float32)
    lengths = np.linalg.norm(features, axis=1, keepdims=True)
    return features / np.maximum(lengths, 1e-12)


class EyeChecker:
    """Decides if photos are similar enough to the training fish-eye photos."""

    def __init__(self, reference_features, threshold=None, k=K_NEIGHBOURS):
        # Private attributes (encapsulation)
        self._reference = normalize(reference_features)  # (number of training photos, 1280)
        self._threshold = threshold
        self._k = k

    @property
    def threshold(self):
        return self._threshold

    @property
    def k(self):
        return self._k

    def distances(self, features):
        """Distance of each photo to its k closest training photos (0 = identical)."""
        similarity = normalize(features) @ self._reference.T  # (photos, training photos)
        closest = np.sort(similarity, axis=1)[:, -self._k:]   # the k highest similarities
        return 1.0 - closest.mean(axis=1)

    def fit_threshold(self, validation_features, keep=KEEP_FRACTION):
        """Sets the cut-off so that `keep` (95%) of the validation eye photos pass.

        Uses ONLY validation photos; the test set and any other photos are
        never used to choose the cut-off.
        """
        self._threshold = float(np.quantile(self.distances(validation_features), keep))
        return self._threshold

    def is_fish_eye(self, features):
        """True/False for each photo: distance at or below the cut-off."""
        if self._threshold is None:
            raise ValueError("The threshold has not been set yet (call fit_threshold).")
        return self.distances(features) <= self._threshold

    # ------------------------------------------------------------------
    # Saving and loading (one small .npz file in models/)
    # ------------------------------------------------------------------
    def save(self, path):
        """Saves the reference features (as float16 to keep the file small) and settings."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, reference=self._reference.astype(np.float16),
                            threshold=np.float32(self._threshold), k=np.int32(self._k))

    @classmethod
    def load(cls, path):
        """Loads a saved checker. Raises FileNotFoundError if the file is missing."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Missing {path.name}")
        data = np.load(path)
        return cls(data["reference"].astype(np.float32),
                   float(data["threshold"]), int(data["k"]))
