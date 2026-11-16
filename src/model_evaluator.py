"""
model_evaluator.py
ModelEvaluator measures how good a trained model is on the TEST set and saves
every result to reports/ so the Streamlit app can show them:

  <task>_summary.json               accuracy, average confidence, calibration verdict
  <task>_classification_report.csv  precision / recall / F1 per class
  <task>_confusion_matrix.png       which classes get mixed up
  <task>_training_curves.png        accuracy and loss per epoch
  <task>_calibration.csv / .png     confidence vs. real accuracy (reliability diagram)
  <task>_examples.png / .csv        correct and wrong test predictions with probabilities

This file only DEFINES things. Nothing runs when it is imported.
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # draw charts to files (no window needed)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from PIL import Image
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# A class with fewer test images than this gets a "noisy metrics" note
SMALL_TEST_CLASS = 40


class ModelEvaluator:
    """Evaluates one model's predicted probabilities against the true labels."""

    def __init__(self, task, class_names, reports_dir):
        self._task = task
        self._class_names = list(class_names)
        self._reports_dir = Path(reports_dir)
        self._reports_dir.mkdir(parents=True, exist_ok=True)

    def _out(self, suffix):
        return self._reports_dir / f"{self._task}_{suffix}"

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------
    def evaluate(self, y_true, probabilities):
        """y_true: true class numbers. probabilities: softmax outputs (n, classes).

        Saves the report files and returns the summary dictionary.
        """
        y_true = np.asarray(y_true)
        y_pred = probabilities.argmax(axis=1)          # top-1 class
        confidence = probabilities.max(axis=1)         # its probability

        report = self._save_classification_report(y_true, y_pred)
        self._save_confusion_matrix(y_true, y_pred)
        calibration = self._save_calibration(y_true, y_pred, confidence)

        counts = np.bincount(y_true, minlength=len(self._class_names))
        summary = {
            "task": self._task,
            "test_images": int(len(y_true)),
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "macro_f1": float(report.loc["macro avg", "f1-score"]),
            "average_confidence": float(confidence.mean()),
            **calibration,
            "small_test_classes": [self._class_names[i] for i, n in enumerate(counts)
                                   if n < SMALL_TEST_CLASS],
        }
        with open(self._out("summary.json"), "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        return summary

    # ------------------------------------------------------------------
    # Per-class metrics and confusion matrix
    # ------------------------------------------------------------------
    def _save_classification_report(self, y_true, y_pred):
        """Precision, recall and F1 for every class, saved as a CSV table."""
        report = classification_report(
            y_true, y_pred, labels=range(len(self._class_names)),
            target_names=self._class_names, output_dict=True, zero_division=0)
        table = pd.DataFrame(report).transpose()
        table.loc["accuracy", "support"] = len(y_true)  # sklearn leaves this blank
        table["support"] = table["support"].astype(int)
        table.to_csv(self._out("classification_report.csv"))
        return table

    def _save_confusion_matrix(self, y_true, y_pred):
        """Rows = true class, columns = predicted class. The diagonal is correct."""
        matrix = confusion_matrix(y_true, y_pred, labels=range(len(self._class_names)))
        size = 2 + 0.9 * len(self._class_names)
        fig, ax = plt.subplots(figsize=(size, size * 0.85))
        sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues", cbar=False,
                    xticklabels=self._class_names, yticklabels=self._class_names, ax=ax)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title(f"Confusion matrix - {self._task} (test set)")
        plt.setp(ax.get_xticklabels(), rotation=35, ha="right")
        fig.tight_layout()
        fig.savefig(self._out("confusion_matrix.png"), dpi=150)
        plt.close(fig)

    # ------------------------------------------------------------------
    # Calibration check
    # ------------------------------------------------------------------
    def _save_calibration(self, y_true, y_pred, confidence, n_bins=10):
        """Is the model's confidence honest?

        We put the test predictions into bins by confidence (0-10%, ..., 90-100%).
        In each bin we compare the AVERAGE CONFIDENCE with the ACTUAL ACCURACY.
        A perfectly calibrated model that says "80% sure" is right 80% of the time.
          confidence > accuracy  -> overconfident
          confidence < accuracy  -> underconfident
        ECE (Expected Calibration Error) = the average gap, weighted by how
        many images are in each bin. Smaller is better.
        """
        correct = (y_pred == y_true).astype(float)
        edges = np.linspace(0, 1, n_bins + 1)
        rows = []
        for low, high in zip(edges[:-1], edges[1:]):
            in_bin = (confidence > low) & (confidence <= high)
            if in_bin.sum() == 0:
                continue
            rows.append({
                "bin": f"{low:.0%}-{high:.0%}",
                "images": int(in_bin.sum()),
                "avg_confidence": float(confidence[in_bin].mean()),
                "accuracy": float(correct[in_bin].mean()),
            })
        table = pd.DataFrame(rows)
        table["gap"] = table["avg_confidence"] - table["accuracy"]
        table.to_csv(self._out("calibration.csv"), index=False)

        ece = float((table["images"] * table["gap"].abs()).sum() / len(y_true))
        overall_gap = float(confidence.mean() - correct.mean())
        if overall_gap > 0.02:
            verdict = "overconfident"
            explanation = (f"On average the model is {overall_gap:.1%} more confident than it "
                           "is accurate, so its percentages are a bit too high.")
        elif overall_gap < -0.02:
            verdict = "underconfident"
            explanation = (f"On average the model is {-overall_gap:.1%} less confident than it "
                           "is accurate, so it is right more often than its percentages say.")
        else:
            verdict = "well calibrated"
            explanation = ("The model's average confidence is within 2% of its real accuracy, "
                           "so its percentages can be read roughly at face value.")

        self._plot_calibration(table)
        return {"ece": ece, "confidence_minus_accuracy": overall_gap,
                "calibration_verdict": verdict, "calibration_explanation": explanation}

    def _plot_calibration(self, table):
        """Reliability diagram: bars = real accuracy, diagonal = perfect calibration."""
        fig, (ax, ax_count) = plt.subplots(
            2, 1, figsize=(6, 6.5), sharex=True, gridspec_kw={"height_ratios": [3, 1]})
        ax.plot([0, 1], [0, 1], "--", color="gray", label="Perfect calibration")
        ax.bar(table["avg_confidence"], table["accuracy"], width=0.07,
               color="#1f8a8a", alpha=0.85, label="Real accuracy in bin")
        ax.set_ylabel("Accuracy")
        ax.set_ylim(0, 1)
        ax.set_title(f"Calibration - {self._task} (test set)")
        ax.legend(loc="upper left")
        ax_count.bar(table["avg_confidence"], table["images"], width=0.07, color="#7a8b99")
        ax_count.set_ylabel("Images")
        ax_count.set_xlabel("Model confidence (top probability)")
        ax_count.set_xlim(0, 1)
        fig.tight_layout()
        fig.savefig(self._out("calibration.png"), dpi=150)
        plt.close(fig)

    # ------------------------------------------------------------------
    # Reading saved results back (used by the app's Model Performance page)
    # ------------------------------------------------------------------
    @staticmethod
    def load_saved_report(reports_dir, task, threshold=0.60):
        """Reads the files evaluate() saved for one task.

        Returns a dictionary with the summary numbers, the per-class table,
        the paths of the charts that exist, and how the model does on
        predictions at or above the app's confidence threshold.
        Returns None if the summary file is missing.
        """
        reports_dir = Path(reports_dir)
        summary_path = reports_dir / f"{task}_summary.json"
        if not summary_path.exists():
            return None
        with open(summary_path, encoding="utf-8") as f:
            summary = json.load(f)

        table_path = reports_dir / f"{task}_classification_report.csv"
        table = pd.read_csv(table_path, index_col=0) if table_path.exists() else None

        # Calibration bins are 10% wide ("60%-70%", ...), so "60% or more"
        # is simply the bins that start at 60% or higher.
        above = None
        calibration_path = reports_dir / f"{task}_calibration.csv"
        if calibration_path.exists():
            bins = pd.read_csv(calibration_path)
            low_edges = bins["bin"].str.split("-").str[0].str.rstrip("%").astype(float) / 100
            confident = bins[low_edges >= threshold - 1e-9]
            if confident["images"].sum() > 0:
                above = {
                    "share": float(confident["images"].sum() / bins["images"].sum()),
                    "accuracy": float((confident["accuracy"] * confident["images"]).sum()
                                      / confident["images"].sum()),
                }

        charts = {}
        for name in ("confusion_matrix", "training_curves", "calibration", "examples"):
            path = reports_dir / f"{task}_{name}.png"
            if path.exists():
                charts[name] = path
        return {"summary": summary, "table": table, "charts": charts, "above_threshold": above}

    @staticmethod
    def expected_calibration_error(y_true, probabilities, n_bins=10):
        """ECE in one number (same bins as the calibration chart). Smaller is better."""
        y_true = np.asarray(y_true)
        confidence = probabilities.max(axis=1)
        correct = (probabilities.argmax(axis=1) == y_true).astype(float)
        edges = np.linspace(0, 1, n_bins + 1)
        ece = 0.0
        for low, high in zip(edges[:-1], edges[1:]):
            in_bin = (confidence > low) & (confidence <= high)
            if in_bin.any():
                ece += in_bin.sum() * abs(confidence[in_bin].mean() - correct[in_bin].mean())
        return float(ece / len(y_true))

    # ------------------------------------------------------------------
    # Training curves
    # ------------------------------------------------------------------
    def save_training_curves(self, history_csv):
        """Plots accuracy and loss per epoch. Returns False if the file is missing."""
        history_csv = Path(history_csv)
        if not history_csv.exists():
            return False
        history = pd.read_csv(history_csv)
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        for ax, metric in zip(axes, ["accuracy", "loss"]):
            ax.plot(history["epoch"], history[metric], "o-", label=f"Training {metric}")
            ax.plot(history["epoch"], history[f"val_{metric}"], "o-", label=f"Validation {metric}")
            stage2 = history[history["stage"] == 2]
            if len(stage2):
                ax.axvline(stage2["epoch"].min() - 0.5, color="gray", linestyle=":",
                           label="Fine-tuning starts")
            ax.set_xlabel("Epoch")
            ax.set_title(metric.capitalize())
            ax.legend()
        fig.suptitle(f"Training curves - {self._task}")
        fig.tight_layout()
        fig.savefig(self._out("training_curves.png"), dpi=150)
        plt.close(fig)
        return True

    # ------------------------------------------------------------------
    # Example predictions
    # ------------------------------------------------------------------
    def save_examples(self, image_paths, y_true, probabilities, n_each=4, seed=42):
        """Saves a picture grid of correct and wrong test predictions.

        For wrong ones we pick the MOST confident mistakes, because those are
        the most interesting to explain in the defense.
        """
        y_true = np.asarray(y_true)
        y_pred = probabilities.argmax(axis=1)
        confidence = probabilities.max(axis=1)
        rng = np.random.default_rng(seed)

        correct_idx = np.flatnonzero(y_pred == y_true)
        wrong_idx = np.flatnonzero(y_pred != y_true)
        chosen_correct = rng.choice(correct_idx, min(n_each, len(correct_idx)), replace=False)
        chosen_wrong = wrong_idx[np.argsort(-confidence[wrong_idx])][:n_each]
        chosen = list(chosen_correct) + list(chosen_wrong)

        rows = []
        for i in chosen:
            top3 = np.argsort(-probabilities[i])[:3]
            rows.append({
                "path": str(image_paths[i]),
                "true": self._class_names[y_true[i]],
                "predicted": self._class_names[y_pred[i]],
                "correct": bool(y_pred[i] == y_true[i]),
                "top3": ", ".join(f"{self._class_names[k]} {probabilities[i][k]:.0%}" for k in top3),
            })
        table = pd.DataFrame(rows)
        table.to_csv(self._out("examples.csv"), index=False)

        fig, axes = plt.subplots(2, n_each, figsize=(3.2 * n_each, 9),
                                 gridspec_kw={"hspace": 0.55})
        for ax in axes.flat:
            ax.axis("off")
        for ax, row in zip(axes.flat, rows):
            with Image.open(row["path"]) as img:
                ax.imshow(img.convert("RGB"))
            mark = "CORRECT" if row["correct"] else "WRONG"
            ax.set_title(f"{mark}\nTrue: {row['true']}\n" + row["top3"].replace(", ", "\n"),
                         fontsize=8, color="green" if row["correct"] else "red")
        fig.suptitle(f"Example test predictions - {self._task} (top: correct, bottom: wrong)")
        fig.subplots_adjust(top=0.84, bottom=0.02, left=0.02, right=0.98)
        fig.savefig(self._out("examples.png"), dpi=130)
        plt.close(fig)
        return table
