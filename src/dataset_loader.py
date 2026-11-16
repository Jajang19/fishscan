"""
dataset_loader.py
DatasetLoader reads the FFE (Freshness of the Fish Eyes) folders, makes the
group-aware train/validation/test split, and turns each split into a
TensorFlow dataset that the models can train on.

This file only DEFINES things. Nothing runs when it is imported.
"""

import re
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from PIL import Image
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.utils.class_weight import compute_class_weight

# Folder species name -> name shown to the user.
# The two tilapia species are merged into one "Tilapia" class, because they
# cannot be told apart reliably from an eye photo.
SPECIES_NAMES = {
    "Chanos Chanos": "Bangus (Milkfish)",
    "Oreochromis Mossambicus": "Tilapia",
    "Oreochromis Niloticus": "Tilapia",
    "Rastrelliger Faughni": "Island Mackerel",
    "Upeneus Moluccensis": "Goldband Goatfish",
    "Johnius Trachycephalus": "Leaftail Croaker",
    "Nibea Albiflora": "Yellow Drum",
    "Eleutheronema Tetradactylum": "Fourfinger Threadfin",
}

# Freshness levels in a logical order (best to worst), from the FFE documentation.
FRESHNESS_LEVELS = ["Highly Fresh", "Fresh", "Not Fresh"]

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def get_source_photo(file_name):
    """Returns the ID of the original phone photo a crop came from.

    FFE file names look like IMG_20191110_055936_3.jpg, where "_3" means
    "crop number 3 from the photo taken at 05:59:36". All crops of one photo
    get the same ID, so they always land in the same split.
    """
    match = re.match(r"(IMG_\d{8}_\d{6})", file_name, re.IGNORECASE)
    if match:
        return match.group(1).upper()
    return Path(file_name).stem  # fallback: the file itself is its own group


def freshness_from_combined(probabilities, combined_class_names):
    """Turns 21 'Species | Level' probabilities into 3 freshness probabilities.

    Example: P(Fresh) = P(Bangus | Fresh) + P(Tilapia | Fresh) + ... (all species).
    This is allowed because the 21 classes never overlap, so their
    probabilities simply add up.
    probabilities: array (n, 21). Returns array (n, 3) in FRESHNESS_LEVELS order.
    """
    probabilities = np.asarray(probabilities)
    result = np.zeros((probabilities.shape[0], len(FRESHNESS_LEVELS)), dtype=np.float32)
    for column, name in enumerate(combined_class_names):
        level = name.split(" | ")[1]
        result[:, FRESHNESS_LEVELS.index(level)] += probabilities[:, column]
    return result


class DatasetLoader:
    """Loads FFE images, splits them by source photo, and builds tf.data datasets."""

    def __init__(self, data_dir, image_size=224, batch_size=32, seed=42,
                 augmentation="standard"):
        # Private attributes (encapsulation): other code uses the methods below.
        self._data_dir = Path(data_dir)
        self._image_size = image_size
        self._batch_size = batch_size
        self._seed = seed
        self._augmentation = augmentation  # "standard" or "strong"
        self._table = None  # pandas DataFrame, one row per image

    # ------------------------------------------------------------------
    # Read-only information other classes may need
    # ------------------------------------------------------------------
    @property
    def table(self):
        return self._table

    @property
    def species_classes(self):
        """Sorted list of species names (the order of the model's outputs)."""
        return sorted(set(SPECIES_NAMES.values()))

    @property
    def freshness_classes(self):
        return list(FRESHNESS_LEVELS)

    @property
    def combined_classes(self):
        """21 'Species | Level' names, used by the species-aware freshness model."""
        return [f"{s} | {level}" for s in self.species_classes for level in FRESHNESS_LEVELS]

    def class_names(self, task):
        """task is 'species', 'freshness' or 'species_freshness' (21 classes)."""
        if task == "species":
            return self.species_classes
        if task == "species_freshness":
            return self.combined_classes
        return self.freshness_classes

    # ------------------------------------------------------------------
    # Step 1: scan the folders
    # ------------------------------------------------------------------
    def scan(self):
        """Reads every class folder and builds the image table.

        Folder names look like "Chanos Chanos - Fresh". We split them into
        species and freshness. Files that are not images, or are corrupted,
        are skipped and counted.
        """
        rows = []
        skipped = 0
        for folder in sorted(p for p in self._data_dir.rglob("*") if p.is_dir()):
            if " - " not in folder.name:
                continue  # not a class folder (for example the top folder)

            # "Nibea Albiflora -  Highly Fresh" has a double space, so we
            # clean up the spaces before using the names.
            species_part, level_part = folder.name.split(" - ", 1)
            species = " ".join(species_part.split())
            level = " ".join(level_part.split())
            if species not in SPECIES_NAMES or level not in FRESHNESS_LEVELS:
                print(f"Skipping unknown folder: {folder.name}")
                continue

            for path in sorted(folder.iterdir()):
                if path.suffix.lower() not in IMAGE_EXTENSIONS or not self._is_valid_image(path):
                    skipped += 1
                    continue
                rows.append({
                    # Path relative to data_dir with "/" so it also works in Colab
                    "path": path.relative_to(self._data_dir).as_posix(),
                    "species_scientific": species,
                    "species": SPECIES_NAMES[species],
                    "freshness": level,
                    "source_photo": get_source_photo(path.name),
                })

        self._table = pd.DataFrame(rows)
        self._table["split"] = ""
        return {"images": len(self._table), "skipped": skipped}

    @staticmethod
    def _is_valid_image(path):
        """True if Pillow can open the file as an image."""
        try:
            with Image.open(path) as img:
                img.verify()
            return True
        except Exception:
            return False

    # ------------------------------------------------------------------
    # Step 2: group-aware, stratified split (about 70 / 15 / 15)
    # ------------------------------------------------------------------
    def split(self):
        """Gives every image a split: 'train', 'val' or 'test'.

        - Grouped: all crops of one source photo stay in the same split.
        - Stratified: each split gets about the same mix of
          species + freshness combinations as the whole dataset.

        StratifiedGroupKFold cuts the data into k equal, grouped, stratified
        parts ("folds"). We use one fold out of 7 (about 14%) as the test
        set, then one fold out of 6 of the rest (about 14% of everything)
        as the validation set. The remaining ~72% is the training set.
        """
        table = self._table
        strata = table["species"] + " | " + table["freshness"]
        groups = table["source_photo"]

        # Test set: 1 fold out of 7
        test_cv = StratifiedGroupKFold(n_splits=7, shuffle=True, random_state=self._seed)
        rest_idx, test_idx = next(test_cv.split(table, strata, groups))

        # Validation set: 1 fold out of 6 from what is left
        rest = table.iloc[rest_idx]
        val_cv = StratifiedGroupKFold(n_splits=6, shuffle=True, random_state=self._seed)
        train_pos, val_pos = next(val_cv.split(rest, strata.iloc[rest_idx], groups.iloc[rest_idx]))

        table["split"] = "train"
        table.loc[table.index[test_idx], "split"] = "test"
        table.loc[rest.index[val_pos], "split"] = "val"
        return table["split"].value_counts().to_dict()

    def check_no_leakage(self):
        """Returns how many source photos appear in more than one split (should be 0)."""
        splits_per_photo = self._table.groupby("source_photo")["split"].nunique()
        return int((splits_per_photo > 1).sum())

    # ------------------------------------------------------------------
    # Step 3: save / load the split so Colab and the laptop use the same one
    # ------------------------------------------------------------------
    def save_split(self, csv_path):
        Path(csv_path).parent.mkdir(parents=True, exist_ok=True)
        self._table.to_csv(csv_path, index=False)

    def load_split(self, csv_path):
        """Loads a split made earlier (so training never re-shuffles the data)."""
        self._table = pd.read_csv(csv_path)
        return self._table["split"].value_counts().to_dict()

    # ------------------------------------------------------------------
    # Step 4: class weights and TensorFlow datasets
    # ------------------------------------------------------------------
    def class_weights(self, task):
        """Bigger weight for classes with fewer training images.

        Returns {class_index: weight}, the format Keras model.fit() expects.
        """
        names = self.class_names(task)
        train_labels = self._labels(self._table[self._table["split"] == "train"], task)
        weights = compute_class_weight("balanced", classes=np.arange(len(names)), y=train_labels)
        return {i: float(w) for i, w in enumerate(weights)}

    def make_dataset(self, split_name, task):
        """Builds a tf.data.Dataset of (image, label) batches for one split.

        - Images are resized to image_size x image_size, values stay 0-255.
          (The model itself rescales them for MobileNetV2.)
        - Labels are class numbers (0, 1, 2, ...) for the chosen task.
        - Augmentation (flip, small rotation, zoom, brightness) is applied
          ONLY to the training split, and freshly every epoch.
        """
        part = self._table[self._table["split"] == split_name]
        paths = [str(self._data_dir / p) for p in part["path"]]
        labels = self._labels(part, task)
        training = split_name == "train"

        ds = tf.data.Dataset.from_tensor_slices((paths, labels))
        ds = ds.map(self._load_image, num_parallel_calls=tf.data.AUTOTUNE)
        ds = ds.cache()  # keep resized images in memory after the first epoch
        if training:
            ds = ds.shuffle(len(paths), seed=self._seed)
        ds = ds.batch(self._batch_size)
        ds = ds.map(lambda x, y: (tf.cast(x, tf.float32), y))  # Keras layers want floats
        if training:
            augment = self._augmenter()
            ds = ds.map(lambda x, y: (augment(x, training=True), y),
                        num_parallel_calls=tf.data.AUTOTUNE)
        return ds.prefetch(tf.data.AUTOTUNE)

    def _labels(self, part, task):
        """Turns class names into class numbers, e.g. 'Fresh' -> 1."""
        names = self.class_names(task)
        if task == "species_freshness":
            values = part["species"] + " | " + part["freshness"]  # e.g. "Tilapia | Fresh"
        else:
            values = part[task]
        return np.array([names.index(value) for value in values], dtype=np.int32)

    def labels(self, split_name, task):
        """True class numbers of one split, in the same order as make_dataset()."""
        return self._labels(self._table[self._table["split"] == split_name], task)

    def _load_image(self, path, label):
        """Reads one image file and resizes it (runs inside TensorFlow)."""
        data = tf.io.read_file(path)
        image = tf.io.decode_image(data, channels=3, expand_animations=False)
        image = tf.image.resize(image, (self._image_size, self._image_size))
        image = tf.cast(image, tf.uint8)  # uint8 uses 4x less memory in the cache
        return image, label

    def _augmenter(self):
        """Random changes so the model sees slightly different photos.

        "standard": flip, small rotation, zoom, small brightness change.
        "strong":   the same, plus bigger brightness and contrast changes and
                    small colour (saturation, hue) shifts. Each freshness level
                    was photographed on different days, so lighting could become
                    a shortcut; strong lighting changes make that shortcut useless.
        """
        layers = [
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomRotation(0.05),        # about +/- 18 degrees
            tf.keras.layers.RandomZoom(0.1),
        ]
        if self._augmentation == "strong":
            layers += [
                tf.keras.layers.RandomBrightness(0.3, value_range=(0, 255)),
                tf.keras.layers.RandomContrast(0.3),
                tf.keras.layers.RandomSaturation((0.45, 0.55)),  # 0.5 = no change
                tf.keras.layers.RandomHue(0.03),                  # tiny colour shift
            ]
        else:
            layers.append(tf.keras.layers.RandomBrightness(0.1, value_range=(0, 255)))
        model = tf.keras.Sequential(layers)
        # Keep pixel values inside 0-255 after the colour changes
        return lambda x, training: tf.clip_by_value(model(x, training=training), 0.0, 255.0)
