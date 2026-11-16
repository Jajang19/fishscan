"""
fish_model.py
FishModel wraps one transfer-learning CNN (MobileNetV2 + a small softmax head).
We make two FishModel objects from the same class:
  - species model   (7 classes)
  - freshness model (3 classes)

This file only DEFINES things. Nothing runs when it is imported.
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from scipy.optimize import minimize_scalar


def apply_temperature(probabilities, temperature):
    """Temperature scaling: softens (T > 1) or sharpens (T < 1) probabilities.

    The model's last layer is softmax(scores). Taking the log of the
    probabilities gives back the scores (plus a constant that softmax ignores).
    We divide the scores by T and apply softmax again:
        new_probabilities = softmax(log(p) / T)
    The ORDER of the classes never changes, so the top prediction (and the
    accuracy) stays exactly the same. Only the confidence changes.
    """
    scores = np.log(np.clip(probabilities, 1e-12, 1.0)) / temperature
    scores -= scores.max(axis=1, keepdims=True)  # for numerical safety
    exp = np.exp(scores)
    return exp / exp.sum(axis=1, keepdims=True)


def fit_temperature(probabilities, true_labels):
    """Finds the T that makes the probabilities most honest on held-out images.

    "Most honest" = lowest log loss (the average of -log(probability given to
    the TRUE class)). We simply try values of T between 0.05 and 10 and keep
    the best one.
    """
    true_labels = np.asarray(true_labels)

    def log_loss_for(temperature):
        scaled = apply_temperature(probabilities, temperature)
        return -np.mean(np.log(np.clip(scaled[np.arange(len(true_labels)), true_labels], 1e-12, 1)))

    result = minimize_scalar(log_loss_for, bounds=(0.05, 10.0), method="bounded")
    return float(result.x)


class FishModel:
    """A MobileNetV2-based image classifier that outputs class probabilities."""

    def __init__(self, name, class_names, image_size=224, temperature=1.0):
        # Private attributes (encapsulation)
        self._name = name                    # "species" or "freshness"
        self._class_names = list(class_names)
        self._image_size = image_size
        self._temperature = temperature      # 1.0 = no temperature scaling
        self._model = None                   # the Keras model
        self._base = None                    # the MobileNetV2 part
        self._history = []                   # one row per epoch (for the curves)
        self._feature_model = None           # built on first use (fish-eye check)

    @property
    def name(self):
        return self._name

    @property
    def temperature(self):
        return self._temperature

    def set_temperature(self, temperature):
        """Stores a fitted temperature (saved in the metadata file by save())."""
        if temperature <= 0:
            raise ValueError("Temperature must be positive.")
        self._temperature = float(temperature)

    @property
    def class_names(self):
        return list(self._class_names)

    @property
    def image_size(self):
        return self._image_size

    # ------------------------------------------------------------------
    # Building the model
    # ------------------------------------------------------------------
    def build(self):
        """Builds: image -> rescale -> MobileNetV2 (frozen) -> pooling -> softmax.

        Transfer learning: MobileNetV2 already learned general shapes,
        edges and textures from 1.4 million ImageNet photos. We keep that
        knowledge (frozen) and only train a small new "head" that turns those
        features into OUR classes.
        """
        size = self._image_size
        self._base = tf.keras.applications.MobileNetV2(
            input_shape=(size, size, 3), include_top=False, weights="imagenet")
        self._base.trainable = False  # stage 1: do not change the pretrained part

        inputs = tf.keras.Input(shape=(size, size, 3))         # pixels 0-255
        x = tf.keras.layers.Rescaling(1 / 127.5, offset=-1)(inputs)  # -> -1..1 (what MobileNetV2 expects)
        x = self._base(x, training=False)                      # keep BatchNorm in inference mode
        x = tf.keras.layers.GlobalAveragePooling2D()(x)        # 7x7x1280 -> 1280 numbers
        x = tf.keras.layers.Dropout(0.2)(x)                    # helps against overfitting
        # Softmax turns the scores into probabilities that add up to 1 (100%)
        outputs = tf.keras.layers.Dense(len(self._class_names), activation="softmax")(x)

        self._model = tf.keras.Model(inputs, outputs, name=f"{self._name}_model")
        self._compile(learning_rate=1e-3)

    def _compile(self, learning_rate):
        self._model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate),
            loss="sparse_categorical_crossentropy",  # labels are class numbers
            metrics=["accuracy"],
        )

    # ------------------------------------------------------------------
    # Training (stage 1: head only, stage 2: fine-tuning)
    # ------------------------------------------------------------------
    def train(self, train_ds, val_ds, epochs, class_weight=None, checkpoint_path=None,
              patience=3):
        """Stage 1: trains only the new head (MobileNetV2 stays frozen)."""
        return self._fit(train_ds, val_ds, epochs, class_weight, checkpoint_path,
                         stage=1, patience=patience)

    def fine_tune(self, train_ds, val_ds, epochs, unfreeze_last=30,
                  class_weight=None, checkpoint_path=None, patience=3, learning_rate=1e-5):
        """Stage 2: unfreezes the last layers of MobileNetV2 and trains them
        with a very small learning rate, so the model adapts to fish eyes
        without forgetting what it learned from ImageNet.
        """
        self._base.trainable = True
        for layer in self._base.layers[:-unfreeze_last]:
            layer.trainable = False
        # BatchNorm layers stay frozen. The base is also called with
        # training=False in build(), so their saved averages never change.
        # Updating them with our small batches would damage what MobileNetV2 learned.
        for layer in self._base.layers:
            if isinstance(layer, tf.keras.layers.BatchNormalization):
                layer.trainable = False
        self._compile(learning_rate=learning_rate)  # recompile so the change takes effect
        return self._fit(train_ds, val_ds, epochs, class_weight, checkpoint_path,
                         stage=2, patience=patience)

    def _fit(self, train_ds, val_ds, epochs, class_weight, checkpoint_path, stage, patience):
        callbacks = [
            # Stop when validation loss has not improved for `patience` epochs,
            # and go back to the best epoch
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=patience, restore_best_weights=True),
        ]
        if checkpoint_path:
            # Saves the best model so far (useful if Colab disconnects)
            Path(checkpoint_path).parent.mkdir(parents=True, exist_ok=True)
            callbacks.append(tf.keras.callbacks.ModelCheckpoint(
                checkpoint_path, monitor="val_loss", save_best_only=True))

        history = self._model.fit(
            train_ds, validation_data=val_ds, epochs=epochs,
            class_weight=class_weight, callbacks=callbacks, verbose=2)

        # Remember every epoch for the training curves
        start = len(self._history)
        for i in range(len(history.history["loss"])):
            row = {key: float(values[i]) for key, values in history.history.items()}
            row.update({"epoch": start + i + 1, "stage": stage})
            self._history.append(row)
        return history.history

    def history_table(self):
        """Training history as a table (epoch, stage, loss, accuracy, ...)."""
        return pd.DataFrame(self._history)

    # ------------------------------------------------------------------
    # Predicting
    # ------------------------------------------------------------------
    def predict_probabilities(self, images):
        """images: array (n, size, size, 3) with pixels 0-255.
        Returns an array (n, number_of_classes); each row adds up to 1.
        If the model has a fitted temperature, it is applied here.
        """
        probabilities = self._model.predict(np.asarray(images, dtype=np.float32), verbose=0)
        if self._temperature != 1.0:
            probabilities = apply_temperature(probabilities, self._temperature)
        return probabilities

    def predict_features_and_probabilities(self, images):
        """Like predict_probabilities, but ALSO returns the 1,280 'feature' numbers
        that MobileNetV2 produced for each image (the output of the pooling layer,
        just before the softmax layer). The fish-eye check compares these numbers
        with the training photos.

        Returns (features (n, 1280), probabilities (n, classes)).
        """
        if self._feature_model is None:
            # A second view of the SAME model with two outputs (no new weights)
            pooling = next(layer for layer in self._model.layers
                           if isinstance(layer, tf.keras.layers.GlobalAveragePooling2D))
            self._feature_model = tf.keras.Model(self._model.input,
                                                 [pooling.output, self._model.output])
        features, probabilities = self._feature_model.predict(
            np.asarray(images, dtype=np.float32), verbose=0)
        if self._temperature != 1.0:
            probabilities = apply_temperature(probabilities, self._temperature)
        return features, probabilities

    def evaluate(self, dataset):
        """Returns {'loss': ..., 'accuracy': ...} on a dataset (without temperature;
        used during training only)."""
        loss, accuracy = self._model.evaluate(dataset, verbose=0)
        return {"loss": float(loss), "accuracy": float(accuracy)}

    # ------------------------------------------------------------------
    # Saving and loading
    # ------------------------------------------------------------------
    def save(self, models_dir):
        """Saves <name>_model.keras and <name>_meta.joblib
        (class names, image size, temperature)."""
        models_dir = Path(models_dir)
        models_dir.mkdir(parents=True, exist_ok=True)
        self._model.save(models_dir / f"{self._name}_model.keras")
        self.save_metadata(models_dir)

    def save_metadata(self, models_dir):
        """Saves only <name>_meta.joblib (for example after fitting a temperature)."""
        joblib.dump(
            {"name": self._name, "class_names": self._class_names,
             "image_size": self._image_size, "temperature": self._temperature},
            Path(models_dir) / f"{self._name}_meta.joblib")

    @classmethod
    def load(cls, models_dir, name):
        """Loads a saved model. Raises FileNotFoundError if the files are missing."""
        models_dir = Path(models_dir)
        model_path = models_dir / f"{name}_model.keras"
        meta_path = models_dir / f"{name}_meta.joblib"
        if not model_path.exists() or not meta_path.exists():
            raise FileNotFoundError(f"Missing {model_path.name} or {meta_path.name} in {models_dir}")

        meta = joblib.load(meta_path)
        # Older metadata files have no temperature: 1.0 means "no scaling"
        fish_model = cls(meta["name"], meta["class_names"], meta["image_size"],
                         meta.get("temperature", 1.0))
        fish_model._model = tf.keras.models.load_model(model_path)
        return fish_model
