"""
app.py
FishScan - Streamlit app (mobile-first).

  Scan page: take or upload a close-up photo of a fish's EYE, then see the
  species and freshness probabilities, a low-confidence warning, a rough
  freshness guide and a safety note.
  Model Performance page: test-set results, charts and a plain-language
  explanation of how the probabilities are made (read from reports/).

The app only LOADS the saved models (models/*.keras); it never trains.
The photo is analysed in memory and is never saved to disk.

Run:  streamlit run app.py
Phone on the same Wi-Fi: open the "Network URL" that Streamlit prints.
"""

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, UnidentifiedImageError

from src.fish_predictor import FishPredictor, scientific_names_by_species
from src.model_evaluator import ModelEvaluator

PROJECT_DIR = Path(__file__).resolve().parent
MODELS_DIR = PROJECT_DIR / "models"
REPORTS_DIR = PROJECT_DIR / "reports"


@st.cache_resource(show_spinner="Loading the FishScan models...")
def load_predictor():
    """Loads both models ONCE and keeps them in memory for every click and user."""
    return FishPredictor(MODELS_DIR)


@st.cache_data
def load_report(task):
    """Reads the saved evaluation files for one model (cached after the first read)."""
    return ModelEvaluator.load_saved_report(REPORTS_DIR, task)


@st.cache_data
def load_csv(name):
    """Reads a small table from reports/experiments/, or None if it is missing."""
    path = REPORTS_DIR / "experiments" / name
    return pd.read_csv(path, index_col=0) if path.exists() else None


def softmax(scores, temperature=1.0):
    """The same formula the model's last layer uses (for the explanation tab)."""
    scaled = np.asarray(scores, dtype=float) / temperature
    exp = np.exp(scaled - scaled.max())
    return exp / exp.sum()


def format_scientific(name):
    """'Chanos Chanos' -> 'Chanos chanos' (genus capitalised, species lowercase)."""
    genus, species = name.split(" ", 1)
    return f"{genus} {species.lower()}"


class FishScanApp:
    """The Streamlit user interface. It only calls FishPredictor (scanning) and
    ModelEvaluator (saved results); no model logic here."""

    def __init__(self):
        st.set_page_config(page_title="FishScan", page_icon="🐟", layout="centered")
        self._scientific = scientific_names_by_species()
        # Session state survives Streamlit's reruns (every click reruns the script)
        st.session_state.setdefault("input_key", 0)     # changing it clears the inputs
        st.session_state.setdefault("result", None)
        st.session_state.setdefault("image_hash", None)

    # ------------------------------------------------------------------
    # Page layout
    # ------------------------------------------------------------------
    def run(self):
        # Menu at the top of the page (a sidebar is hidden behind a button on phones)
        page = st.radio("Menu", ["Scan", "Model Performance", "About"],
                        horizontal=True, label_visibility="collapsed", key="page")
        if page == "Scan":
            self.show_scan_page()
        elif page == "Model Performance":
            self.show_performance_page()
        else:
            st.title(page)
            st.info("This page is coming soon.")

    def show_scan_page(self):
        st.title("🐟 FishScan")
        st.markdown("**Check a fish's species and freshness from one photo of its eye.**")

        try:
            predictor = load_predictor()
        except FileNotFoundError as error:
            st.error("The trained model files are missing, so FishScan cannot scan yet.")
            st.caption(f"Details: {error}. Put the .keras and .joblib files in the models/ folder.")
            return

        self._show_photo_tips()
        image, image_hash = self._get_image()
        if image is None:
            return

        # A new photo makes the old result invalid
        if image_hash != st.session_state.image_hash:
            st.session_state.result = None
            st.session_state.image_hash = image_hash

        st.image(image, caption="Your photo", width=320)
        if st.button("🔍 Analyze", type="primary", width="stretch"):
            with st.spinner("Looking at the fish's eye..."):
                st.session_state.result = predictor.predict(image)

        if st.session_state.result is not None:
            self._show_results(st.session_state.result, predictor.threshold)
            if st.button("📷 Scan another fish", width="stretch"):
                st.session_state.input_key += 1
                st.session_state.result = None
                st.session_state.image_hash = None
                st.rerun()

    # ------------------------------------------------------------------
    # Input
    # ------------------------------------------------------------------
    def _show_photo_tips(self):
        with st.expander("📸 How to take a good photo", expanded=False):
            st.markdown(
                "- Take a **close-up of ONE eye**, filling most of the photo, with the eye in the **centre**.\n"
                "- Use **good, even light** (daylight is best). Avoid flash glare and dark shadows.\n"
                "- Hold the phone **steady** so the eye is sharp, not blurry.\n"
                "- FishScan was trained on **bangus, tilapia, island mackerel, goldband goatfish, "
                "yellow drum, leaftail croaker and fourfinger threadfin** only.")

    def _get_image(self):
        """Returns (PIL image, hash) from the camera or the uploader, or (None, None)."""
        key = st.session_state.input_key
        # Upload is the default: on a phone it also offers "Take photo", and it works
        # even where the browser blocks the live camera (plain http over Wi-Fi).
        method = st.radio("Photo source", ["🖼️ Upload / phone camera", "📷 Live camera"],
                          horizontal=True, key=f"method_{key}")
        if method.startswith("📷"):
            photo = st.camera_input("Point the camera at the fish's eye", key=f"camera_{key}")
            st.caption("If the live camera does not open, use **Upload / phone camera** instead - "
                       "on a phone it lets you take a picture too.")
        else:
            photo = st.file_uploader("Choose or take a JPG or PNG photo", type=["jpg", "jpeg", "png"],
                                     key=f"upload_{key}")

        if photo is None:
            st.info("Take or upload a photo of a fish's eye to start.")
            return None, None

        data = photo.getvalue()
        try:
            image = Image.open(photo)
            image.load()  # forces reading the whole file, so broken files fail here
        except (UnidentifiedImageError, OSError):
            st.error("This file could not be opened as an image. Please try another JPG or PNG photo.")
            return None, None
        st.caption("🔒 Your photo is analysed in memory and is not saved.")
        return image, hashlib.md5(data).hexdigest()

    # ------------------------------------------------------------------
    # Results
    # ------------------------------------------------------------------
    def _show_results(self, result, threshold):
        st.divider()
        self._show_species(result["species"], threshold)
        st.divider()
        self._show_freshness(result["freshness"], threshold)
        st.divider()
        st.info("**Safety note:** fish that smells sour or ammonia-like, has slimy or dull skin, "
                "grey or brown gills, or soft flesh that does not spring back should not be eaten, "
                "whatever this app says. Keep fish cold and cook it thoroughly.")
        st.caption("⚠️ FishScan is a student machine-learning project, NOT a food safety test. "
                   "A photo cannot detect bacteria, chemicals or toxins.")

    def _low_confidence_box(self, what, probability, threshold):
        """The 'Low confidence' label with a tip (probabilities are still shown below it)."""
        st.warning(f"**⚠️ Low confidence** - the top {what} is only {probability:.0%} "
                   f"(below {threshold:.0%}). Treat this result as a guess.\n\n"
                   "**Tip:** retake a sharper, well-lit close-up of the eye, centred in the photo.")

    def _show_bars(self, probabilities, limit=None):
        """One labelled bar per class (text + percentage, so colour is never needed)."""
        for name, probability in list(probabilities.items())[:limit]:
            st.progress(min(max(probability, 0.0), 1.0), text=f"{name}: {probability:.1%}")

    def _show_species(self, species, threshold):
        st.subheader("Species")
        name = species["top_class"]
        scientific = ", ".join(format_scientific(s) for s in self._scientific.get(name, []))
        if species["low_confidence"]:
            self._low_confidence_box("species", species["top_probability"], threshold)
            st.markdown(f"Best guess: **{name}** ({species['top_probability']:.0%})")
        else:
            st.markdown(f"## {name} - {species['top_probability']:.0%}")
        st.caption(f"Scientific name: *{scientific}*")
        st.markdown("**Top 3 species**")
        self._show_bars(species["probabilities"], limit=3)

    def _show_freshness(self, freshness, threshold):
        st.subheader("Freshness")
        level = freshness["top_class"]
        guide = freshness["guide"]
        if freshness["low_confidence"]:
            self._low_confidence_box("freshness level", freshness["top_probability"], threshold)
            st.markdown(f"Best guess: {guide['icon']} **{level}** ({freshness['top_probability']:.0%})")
        else:
            # Coloured label, but the text says the level too (colour is never the only cue)
            st.markdown(
                f"<div style='background:{guide['color']};color:white;padding:0.6rem 1rem;"
                f"border-radius:0.6rem;font-size:1.4rem;font-weight:700;'>"
                f"{guide['icon']} {level} - {freshness['top_probability']:.0%}</div>",
                unsafe_allow_html=True)
        st.markdown("**All freshness levels**")
        self._show_bars(freshness["probabilities"])

        prefix = "Only if this guess is right: " if freshness["low_confidence"] else ""
        st.markdown(f"**Rough guide (estimate):** {prefix}the eye looks like fish on "
                    f"**{guide['dataset_days']}** in the FFE dataset. {guide['advice']}")
        st.caption("This is only a visual estimate based on how the dataset labels its photos. "
                   "The dataset does not say how its fish were stored, so FishScan cannot tell "
                   "you an exact number of hours or days left.")

    # ------------------------------------------------------------------
    # Model Performance page
    # ------------------------------------------------------------------
    def show_performance_page(self):
        st.title("📊 Model Performance")
        st.markdown("How well do FishScan's two models work? All numbers below come from the "
                    "**test set: 664 eye photos the models never saw during training** "
                    "(about 15% of the dataset, split so that crops of the same photo never "
                    "appear in both training and testing).")

        species_tab, freshness_tab, explain_tab = st.tabs(
            ["🐟 Species", "🧊 Freshness", "💡 How to read this"])
        with species_tab:
            self._show_model_report(
                "species", "Species model (7 classes)",
                "Predicts which of the 7 species the eye belongs to.")
        with freshness_tab:
            self._show_model_report(
                "freshness", "Freshness model (3 levels)",
                "Predicts Highly Fresh (storage day 1-2), Fresh (day 3-4) or Not Fresh "
                "(day 5-6), as defined by the FFE dataset.")
            self._show_freshness_experiments()
        with explain_tab:
            self._show_explanations()

    def _show_model_report(self, task, title, description):
        report = load_report(task)
        st.subheader(title)
        st.caption(description)
        if report is None:
            st.warning(f"No saved results for the {task} model yet. "
                       "Run `python scripts/evaluate.py` to create them in reports/.")
            return
        summary = report["summary"]

        # Key numbers, two per row so they also fit on a phone
        row1 = st.columns(2)
        row1[0].metric("Accuracy", f"{summary['accuracy']:.1%}",
                       help="Share of test photos where the top prediction was correct.")
        row1[1].metric("Macro F1", f"{summary['macro_f1']:.2f}",
                       help="Average F1 score of all classes, each class counting equally.")
        row2 = st.columns(2)
        row2[0].metric("Average confidence", f"{summary['average_confidence']:.1%}",
                       help="Average top probability the model gave on the test photos.")
        row2[1].metric("Calibration error (ECE)", f"{summary['ece']:.3f}",
                       help="Average gap between confidence and real accuracy. Smaller is better.")

        st.markdown(f"**Calibration verdict: {summary['calibration_verdict']}.** "
                    f"{summary['calibration_explanation']}")
        above = report["above_threshold"]
        if above:
            st.info(f"When this model is **at least 60% sure** ({above['share']:.0%} of the "
                    f"test photos), it is right **{above['accuracy']:.0%}** of the time. "
                    "Below 60%, the Scan page shows the ⚠️ Low confidence label.")
        if summary["small_test_classes"]:
            st.caption("Note: " + ", ".join(summary["small_test_classes"]) +
                       " have fewer than 40 test photos, so their numbers are less reliable.")

        if report["table"] is not None:
            st.markdown("**Results per class**")
            self._show_class_table(report["table"])

        charts = report["charts"]
        explanations = {
            "confusion_matrix": ("Confusion matrix",
                                 "Rows are the TRUE class, columns are what the model PREDICTED. "
                                 "The diagonal (top-left to bottom-right) counts correct answers; "
                                 "every other cell shows which classes get mixed up."),
            "training_curves": ("Training curves",
                                "Accuracy and loss after each training round (epoch) for the "
                                "training and validation photos. The dotted line is where "
                                "fine-tuning started. When validation stops improving while "
                                "training keeps improving, the model starts memorising "
                                "(overfitting), so training was stopped at the best epoch."),
            "calibration": ("Calibration check (reliability diagram)",
                            "Test photos are grouped by how confident the model was. Each bar is "
                            "the real accuracy in that group. Bars on the dashed line = honest "
                            "percentages; bars below it = overconfident; above it = "
                            "underconfident. The small chart shows how many photos are in "
                            "each group."),
            "examples": ("Example predictions",
                         "Top row: correct predictions. Bottom row: the most confident "
                         "MISTAKES, with the top 3 probabilities. Many mistakes are dark or "
                         "blurry photos, which is why the Scan page asks for a sharp, "
                         "well-lit close-up."),
        }
        for key, (heading, text) in explanations.items():
            if key in charts:
                st.markdown(f"**{heading}**")
                st.image(str(charts[key]), width="stretch")
                st.caption(text)

    def _show_class_table(self, table):
        """Per-class precision / recall / F1 as percentages (class rows only)."""
        rows = table.drop(index=["accuracy", "macro avg", "weighted avg"], errors="ignore")
        shown = pd.DataFrame({
            "Precision": rows["precision"].map("{:.0%}".format),
            "Recall": rows["recall"].map("{:.0%}".format),
            "F1": rows["f1-score"].map("{:.2f}".format),
            "Test photos": rows["support"].astype(int),
        })
        st.dataframe(shown, width="stretch")
        st.caption("**Precision**: when the model says this class, how often it is right. "
                   "**Recall**: of the photos that really are this class, how many it found. "
                   "**F1**: one number that balances both (1.00 = perfect).")

    def _show_freshness_experiments(self):
        """How the final freshness model was chosen (validation data only)."""
        comparison = load_csv("comparison_val.csv")
        scaling = load_csv("temperature_scaling_half2.csv")
        if comparison is None and scaling is None:
            return
        # Friendlier row names for the tables
        names = {"Baseline": "First model", "E1_species_aware": "E1 (21 classes)",
                 "E2_deep_strong_aug": "E2 (deeper + strong augmentation)"}
        comparison = comparison.rename(index=names) if comparison is not None else None
        scaling = scaling.rename(index=names) if scaling is not None else None
        st.divider()
        st.subheader("How we improved the freshness model")
        st.markdown(
            "The first freshness model reached 62.8% test accuracy. We tried two "
            "improvements and compared them **only on the validation photos**, so the test "
            "set stayed untouched until the final check:\n"
            "- **E1:** one model with 21 classes (species x freshness), adding up the "
            "probabilities per freshness level.\n"
            "- **E2:** deeper fine-tuning plus strong lighting and colour augmentation, so the "
            "model relies less on the lighting of each photo session.")
        if comparison is not None:
            shown = pd.DataFrame({
                "Validation accuracy": comparison["val_accuracy"].map("{:.1%}".format),
                "Log loss": comparison["val_log_loss"].map("{:.3f}".format),
                "Calibration error": comparison["val_ece"].map("{:.3f}".format),
            })
            st.dataframe(shown, width="stretch")
            st.caption("Lower log loss and calibration error are better.")
        st.markdown(
            "E2 was the most accurate but **overconfident**. We fixed that with "
            "**temperature scaling**: the model's scores are divided by one number T before "
            "softmax. T was fitted on one half of the validation photos and checked on the "
            "other half.")
        if scaling is not None:
            shown = pd.DataFrame({
                "Accuracy": scaling["accuracy"].map("{:.1%}".format),
                "Avg confidence": scaling["avg_confidence"].map("{:.1%}".format),
                "Calibration error": scaling["ece"].map("{:.3f}".format),
            })
            st.dataframe(shown, width="stretch")
        st.markdown(
            "A rule we set **before** seeing these results (scaled E2 must be at least as "
            "well calibrated as the old model, within 0.01) chose **E2 + temperature** as the "
            "final freshness model. On the test set it scored 65.5% (old model: 62.8%).")

    def _show_explanations(self):
        st.subheader("How FishScan makes its percentages")
        st.markdown(
            "1. The photo is resized to 224 x 224 pixels and passed through **MobileNetV2**, a "
            "network pre-trained on 1.4 million everyday photos (transfer learning). It turns "
            "the eye into 1,280 numbers that describe shapes, colours and textures.\n"
            "2. A small final layer gives every class a **score**.\n"
            "3. **Softmax** turns the scores into probabilities that add up to 100%. "
            "Bigger gaps between scores give more extreme percentages.")

        scores = [4.0, 1.0, 0.5]
        plain, softened = softmax(scores), softmax(scores, temperature=1.38)
        example = pd.DataFrame({
            "Score": scores,
            "Softmax": [f"{p:.1%}" for p in plain],
            "Softmax with T = 1.38": [f"{p:.1%}" for p in softened],
        }, index=["Highly Fresh", "Fresh", "Not Fresh"])
        st.markdown("**Example** (made-up scores):")
        st.dataframe(example, width="stretch")
        st.caption("Temperature scaling divides the scores by T before softmax. The order "
                   "never changes (so the answer and the accuracy stay the same); only the "
                   "confidence becomes more honest.")

        st.subheader("What does \"87% Fresh\" mean?")
        st.markdown(
            "It is the model's **confidence**, not a lab measurement. A well-calibrated model "
            "that says 87% is right about 87 out of 100 times it says that. The calibration "
            "charts on the other tabs check exactly this.")

        st.subheader("Where the percentages can be wrong")
        st.markdown(
            "- **Unknown things:** softmax always picks one of the classes it knows. A photo "
            "of something that is not a fish, or a species it was not trained on, can still "
            "get a high percentage. In our tests a plain colour gradient was called Yellow "
            "Drum with 99%.\n"
            "- **Photo quality:** dark, blurry or off-centre photos cause most mistakes.\n"
            "- **Freshness is visual only:** it cannot detect bacteria, chemicals or toxins.\n"
            "- **Dataset limits:** each freshness level was photographed on different days, "
            "so lighting may have helped the model; the same fish appear on several days.")


def main():
    app = FishScanApp()
    app.run()


main()
