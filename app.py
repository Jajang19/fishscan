"""
app.py
FishScan - Streamlit app (mobile-first, responsive).

  Scan:        take or choose a close-up photo of a fish's EYE and get species and
               freshness probabilities, with Low-confidence and Not-recognized states.
  Performance: test-set results, charts and how the probabilities work.
  About:       what FishScan does, photo tips, limitations, credits.

The app only LOADS the saved models (models/*.keras); it never trains.
The photo is analysed in memory and is never saved to disk.
All custom styling lives in assets/style.css (loaded once below).

Run:  streamlit run app.py
"""

import base64
import hashlib
import html
import json
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from PIL import Image, UnidentifiedImageError

from src.eye_checker import ZONE_NOT_RECOGNIZED, ZONE_WARNING, EyeChecker
from src.fish_predictor import (EYE_CHECK_FILE, FRESHNESS_GUIDE, FishPredictor,
                                scientific_names_by_species)
from src.history_manager import HistoryManager
from src.model_evaluator import ModelEvaluator

PROJECT_DIR = Path(__file__).resolve().parent
MODELS_DIR = PROJECT_DIR / "models"
REPORTS_DIR = PROJECT_DIR / "reports"
ASSETS_DIR = PROJECT_DIR / "assets"

# Shown on the About page
DEVELOPERS = ["Chavez, Jaylord E.", "Dalumpines, Rozel G."]

# Text colour on each freshness badge: dark text on the light amber badge, white on
# green/red, so the label always has enough contrast to read
BADGE_TEXT = {"Highly Fresh": "#ffffff", "Fresh": "#1f2937", "Not Fresh": "#ffffff"}

PAGES = {"Scan": ":material/photo_camera: Scan",
         "History": ":material/history: History",
         "Performance": ":material/insights: Performance",
         "About": ":material/info: About"}

PHOTO_TIPS = (
    "- Take a **close-up of ONE eye**. Before **Analyze**, use **Zoom** and **Move** until the "
    "eye fills most of the square.\n"
    "- Use **good, even light** (daylight is best). Avoid flash glare and dark shadows.\n"
    "- Hold the phone **steady** so the eye is sharp, not blurry.\n"
    "- FishScan knows **bangus, tilapia, island mackerel, goldband goatfish, yellow drum, "
    "leaftail croaker and fourfinger threadfin** only.")


# ----------------------------------------------------------------------
# Cached loaders (run once, then reused on every click)
# ----------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading the FishScan models...")
def load_predictor():
    """Loads both models ONCE and keeps them in memory for every click and user."""
    return FishPredictor(MODELS_DIR)


@st.cache_data
def load_report(task):
    """Reads the saved evaluation files for one model (cached after the first read)."""
    return ModelEvaluator.load_saved_report(REPORTS_DIR, task)


@st.cache_data
def load_eye_check_results():
    """Reads reports/eye_check_results.json (made by scripts/test_eye_check.py), or None."""
    path = REPORTS_DIR / "eye_check_results.json"
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


@st.cache_data
def load_eye_zones():
    """The eye-check cut-offs (from models/eye_check.npz) and how the FFE validation eyes
    fell into the three zones (reports/eye_check_zones.json). Either part may be None."""
    try:
        checker = EyeChecker.load(MODELS_DIR / EYE_CHECK_FILE)
        cutoffs = {"threshold": checker.threshold, "warning": checker.warning_threshold}
    except FileNotFoundError:
        cutoffs = None
    path = REPORTS_DIR / "eye_check_zones.json"
    counts = None
    if path.exists():
        with open(path, encoding="utf-8") as f:
            counts = json.load(f)
    return cutoffs, counts


@st.cache_data
def load_csv(name):
    """Reads a small table from reports/experiments/, or None if it is missing."""
    path = REPORTS_DIR / "experiments" / name
    return pd.read_csv(path, index_col=0) if path.exists() else None


@st.cache_data
def image_as_data_uri(filename):
    """Small image from assets/ as text, so it can sit inside our HTML."""
    data = (ASSETS_DIR / filename).read_bytes()
    kind = "png" if filename.endswith(".png") else "jpeg"
    return f"data:image/{kind};base64,{base64.b64encode(data).decode()}"


# ----------------------------------------------------------------------
# Small helpers that build HTML pieces (styled by assets/style.css)
# ----------------------------------------------------------------------
def softmax(scores, temperature=1.0):
    """The same formula the model's last layer uses (for the explanation tab)."""
    scaled = np.asarray(scores, dtype=float) / temperature
    exp = np.exp(scaled - scaled.max())
    return exp / exp.sum()


def format_scientific(name):
    """'Chanos Chanos' -> 'Chanos chanos' (genus capitalised, species lowercase)."""
    genus, species = name.split(" ", 1)
    return f"{genus} {species.lower()}"


def ring_html(probability):
    """Round confidence meter with the percentage written in the middle."""
    percent = round(probability * 100)
    return f'<div class="fs-ring" style="--p:{percent}"><span>{percent}%</span></div>'


def bars_html(probabilities, limit=None, colors=None):
    """One labelled bar per class: the name and % are text, the bar is extra."""
    rows = []
    for name, probability in list(probabilities.items())[:limit]:
        color = f"background:{colors[name]};" if colors else ""
        rows.append(
            f'<div class="fs-bar"><div class="fs-bar-top"><span>{html.escape(name)}</span>'
            f'<b>{probability:.1%}</b></div><div class="fs-bar-track">'
            f'<div class="fs-bar-fill" style="width:{probability * 100:.1f}%;{color}"></div>'
            f'</div></div>')
    return "".join(rows)


def note_html(text, kind=""):
    """Small info card. kind: '' (info), 'warn' or 'bad'."""
    return f'<div class="fs-note {kind}">{text}</div>'


class FishScanApp:
    """The Streamlit user interface. It only calls FishPredictor (scanning) and
    ModelEvaluator (saved results); no model logic here."""

    def __init__(self):
        st.set_page_config(page_title="FishScan - fish eye scanner",
                           page_icon=str(ASSETS_DIR / "favicon.png"),
                           layout="wide", initial_sidebar_state="collapsed")
        # Our stylesheet: one file, injected once per page load
        css = (ASSETS_DIR / "style.css").read_text(encoding="utf-8")
        st.html(f"<style>{css}</style>")
        self._scientific = scientific_names_by_species()
        self._history = HistoryManager(css)   # past scans, kept in the user's browser
        # Session state survives Streamlit's reruns (every click reruns the script)
        st.session_state.setdefault("input_key", 0)     # changing it clears the inputs
        st.session_state.setdefault("result", None)
        st.session_state.setdefault("image_hash", None)
        st.session_state.setdefault("last_page", "Scan")
        st.session_state.setdefault("history_entry", None)  # the latest scan, to save in the browser

    # ------------------------------------------------------------------
    # Header and navigation
    # ------------------------------------------------------------------
    def run(self):
        self._show_header()
        page = self._show_nav()
        if page == "Scan":
            self.show_scan_page()
        elif page == "History":
            self.show_history_page()
        elif page == "Performance":
            self.show_performance_page()
        else:
            self.show_about_page()

    def _show_header(self):
        st.html(f'<div class="fs-header"><img src="{image_as_data_uri("logo.png")}" alt="">'
                f'<div><div class="fs-title">FishScan</div>'
                f'<div class="fs-tagline">Fish species &amp; freshness from one eye photo</div>'
                f'</div></div>')

    def _show_nav(self):
        """App-style tab bar. A segmented control can be 'unselected', so we keep
        the last page in that case."""
        page = st.segmented_control("Menu", list(PAGES), format_func=PAGES.get,
                                    default="Scan", key="nav", label_visibility="collapsed",
                                    width="stretch")
        if page is None:
            page = st.session_state.last_page
        st.session_state.last_page = page
        return page

    # ------------------------------------------------------------------
    # Scan page
    # ------------------------------------------------------------------
    def show_scan_page(self):
        try:
            predictor = load_predictor()
        except FileNotFoundError as error:
            st.html(note_html("<b>The trained model files are missing</b>, so FishScan "
                              f"cannot scan yet. ({html.escape(str(error))})", "bad"))
            return

        # Two columns on tablets/laptops; Streamlit stacks them on phones
        input_col, result_col = st.columns([5, 6], gap="large")
        with input_col:
            image, image_hash = self._show_input_card()
        if image is not None and image_hash != st.session_state.image_hash:
            st.session_state.result = None              # a new photo makes the old result invalid
            st.session_state.image_hash = image_hash

        with result_col:
            if image is not None and st.session_state.get("analyze_clicked"):
                with st.spinner("Looking at the fish's eye..."):
                    st.session_state.result = predictor.predict(image)
                # Remember this scan on the user's device (thumbnail + results only)
                st.session_state.history_entry = self._history.make_entry(
                    # id = the photo's fingerprint, so analyzing the same photo twice
                    # does not create two history entries
                    st.session_state.image_hash, image, st.session_state.result,
                    self._card_html(st.session_state.result, predictor.threshold))
            if st.session_state.result is not None:
                self._show_results(st.session_state.result, predictor.threshold)
                self._save_to_history()
            elif image is not None:
                with st.container(key="card_ready"):
                    st.html('<div class="fs-label">Ready</div><b>Photo loaded.</b> '
                            '<span class="fs-muted">Tap <b>Analyze</b> to see the species and '
                            'freshness.</span>')
            else:
                self._show_empty_state()

    def _show_input_card(self):
        """Photo source switch, the big photo button, the preview and the buttons.
        Returns (PIL image, hash) or (None, None)."""
        key = st.session_state.input_key
        with st.container(key="card_input"):
            st.html('<div class="fs-label">Your photo</div>')
            # Upload is the default: on a phone it also offers the camera app, and it
            # works even where the browser blocks the live camera.
            source = st.segmented_control(
                "Photo source", ["upload", "camera"], default="upload", key=f"source_{key}",
                format_func={"upload": ":material/add_photo_alternate: Photo",
                             "camera": ":material/videocam: Live camera"}.get,
                label_visibility="collapsed", width="stretch")
            if source == "camera":
                photo = st.camera_input("Point the camera at the fish's eye",
                                        key=f"camera_{key}", label_visibility="collapsed")
                st.caption("If the live camera does not open, choose **Photo** - on a phone "
                           "it lets you use the camera app too.")
            else:
                photo = st.file_uploader("Take or choose a photo", type=["jpg", "jpeg", "png"],
                                         key=f"upload_{key}", label_visibility="collapsed")

            image, image_hash = None, None
            if photo is not None:
                try:
                    image = Image.open(photo)
                    image.load()  # forces reading the whole file, so broken files fail here
                    image_hash = hashlib.md5(photo.getvalue()).hexdigest()
                except (UnidentifiedImageError, OSError):
                    image = None
                    st.html(note_html("<b>This file could not be opened as an image.</b> "
                                      "Please try another JPG or PNG photo.", "bad"))

            if image is not None:
                image, image_hash = self._show_crop_step(image, image_hash, key)
                st.session_state.analyze_clicked = st.button(
                    "Analyze", icon=":material/search:", type="primary", width="stretch")
                if st.session_state.result is not None or st.session_state.analyze_clicked:
                    if st.button("Scan another fish", icon=":material/restart_alt:",
                                 width="stretch"):
                        st.session_state.input_key += 1
                        st.session_state.result = None
                        st.session_state.image_hash = None
                        st.rerun()
            else:
                st.session_state.analyze_clicked = False
            st.caption(":material/lock: Your photo is analysed in memory, not stored on our server. "
                       "A small thumbnail and the result are kept in History on this device only.")
        return image, image_hash

    def _show_crop_step(self, image, image_hash, key):
        """Lets the user zoom in and move the square onto the eye BEFORE Analyze.

        The preview shows exactly the square the model will look at. Sliders are
        used because they work with a thumb on a phone and need no extra library.
        Returns the cropped image and a hash that also includes the crop, so
        changing the crop clears an old result.
        """
        with st.expander(":material/crop: Center the eye (zoom and move)", expanded=True):
            zoom = st.slider("Zoom", 1.0, 8.0, 1.0, 0.1, key=f"zoom_{key}", format="%.1fx")
            center_x = st.slider("Move left ↔ right", 0, 100, 50, key=f"move_x_{key}",
                                 format="%d%%")
            center_y = st.slider("Move up ↕ down", 0, 100, 50, key=f"move_y_{key}",
                                 format="%d%%")
        cropped = FishPredictor.crop_square(image, zoom, center_x / 100, center_y / 100)
        preview = cropped.copy()
        preview.thumbnail((640, 640))   # small copy for the screen; the model gets the full crop
        st.image(preview, width="stretch")
        st.caption("FishScan analyses only this square. Zoom in until the eye fills most of it.")
        crop_hash = hashlib.md5(f"{image_hash}|{zoom}|{center_x}|{center_y}".encode()).hexdigest()
        return cropped, crop_hash

    def _show_empty_state(self):
        """What a good photo looks like, before anything is chosen."""
        with st.container(key="card_empty"):
            st.html('<div class="fs-label">How to get a good result</div>')
            left, right = st.columns([2, 3], vertical_alignment="center")
            left.image(str(ASSETS_DIR / "sample_eye.jpg"), width="stretch")
            right.markdown("**A good photo looks like this:** one sharp, well-lit eye that "
                           "fills most of the picture.")
            st.markdown(PHOTO_TIPS)
            st.caption("Example photo: FFE dataset (Prasetyo et al., CC BY 4.0).")

    # ------------------------------------------------------------------
    # Results
    # ------------------------------------------------------------------
    def _show_results(self, result, threshold):
        eye_check = result["eye_check"]
        if not eye_check["recognized"]:
            # Not similar to any fish-eye photo the model learned from:
            # do NOT show a confident-looking species/freshness answer.
            with st.container(key="card_unrecognized"):
                st.html('<div class="fs-label">Result</div>'
                        '<div class="fs-big">🚫 Not recognized</div>'
                        '<p class="fs-muted">This photo does not look like the fish-eye '
                        'close-ups FishScan was trained on, so species and freshness '
                        'percentages would not mean anything.</p>'
                        + note_html("<b>Try again:</b> a sharp, well-lit close-up of ONE fish "
                                    "eye, centred and filling most of the photo.", "warn"))
                with st.expander("Show the model's guesses anyway (for learning only)"):
                    st.caption("Softmax always splits 100% among the classes it knows, so even "
                               "a photo of a hand gets a species. These are NOT real results.")
                    st.html(self._species_html(result["species"], threshold)
                            + self._freshness_html(result["freshness"], threshold))
        else:
            if eye_check.get("zone") == ZONE_WARNING:
                # Borderline: show the results, but say clearly they may be less reliable
                with st.container(key="card_unusual"):
                    st.html(self._unusual_photo_html())
            with st.container(key="card_species"):
                st.html(self._species_html(result["species"], threshold))
            with st.container(key="card_freshness"):
                st.html(self._freshness_html(result["freshness"], threshold))
        if not eye_check["available"]:
            st.caption("(The fish-eye check file was not found, so this photo was not checked.)")
        with st.container(key="card_safety"):
            st.html(note_html("<b>Safety note:</b> fish that smells sour or ammonia-like, has "
                              "slimy or dull skin, grey or brown gills, or soft flesh that does "
                              "not spring back should not be eaten, whatever this app says. "
                              "Keep fish cold and cook it thoroughly.")
                    + '<div class="fs-muted">⚠️ FishScan is a student machine-learning project, '
                      'NOT a food safety test. A photo cannot detect bacteria, chemicals or '
                      'toxins.</div>')

    @staticmethod
    def _unusual_photo_html():
        """Warning-zone note: the photo is close to, but not quite like, the training eyes."""
        return note_html("<b>⚠️ This photo looks unusual - results may be less reliable.</b> "
                         "It is less similar to the fish-eye photos FishScan learned from than "
                         "most real eye photos (blur, glare, angle, crop, or a fish FishScan does "
                         "not know). Treat the percentages below with extra care. "
                         "<b>Tip:</b> retake a sharp, well-lit close-up and zoom in on the eye.",
                         "warn")

    def _low_confidence_html(self, what, probability, threshold):
        return note_html(f"<b>⚠️ Low confidence</b> - the top {what} is only "
                         f"{probability:.0%} (below {threshold:.0%}), so treat it as a guess. "
                         "<b>Tip:</b> retake a sharper, well-lit close-up of the eye.", "warn")

    def _species_html(self, species, threshold):
        name = species["top_class"]
        scientific = ", ".join(format_scientific(s) for s in self._scientific.get(name, []))
        heading = "Best guess" if species["low_confidence"] else "Species"
        parts = [f'<div class="fs-label">{heading}</div>',
                 f'<div class="fs-result-top"><div><div class="fs-big">{html.escape(name)}</div>'
                 f'<div class="fs-sci">{html.escape(scientific)}</div></div>'
                 f'{ring_html(species["top_probability"])}</div>']
        if species["low_confidence"]:
            parts.append(self._low_confidence_html("species", species["top_probability"], threshold))
        parts.append('<div class="fs-label" style="margin-top:.8rem">Top 3 species</div>')
        parts.append(bars_html(species["probabilities"], limit=3))
        return "".join(parts)

    def _freshness_html(self, freshness, threshold):
        level, guide = freshness["top_class"], freshness["guide"]
        heading = "Freshness - best guess" if freshness["low_confidence"] else "Freshness"
        colors = {name: info["color"] for name, info in FRESHNESS_GUIDE.items()}
        prefix = "Only if this guess is right: " if freshness["low_confidence"] else ""
        parts = [f'<div class="fs-label">{heading}</div>',
                 # Badge = colour + icon + TEXT, so it never relies on colour alone
                 f'<span class="fs-badge" style="background:{guide["color"]};color:{BADGE_TEXT[level]}">{guide["icon"]} '
                 f'{html.escape(level)} · {freshness["top_probability"]:.0%}</span>']
        if freshness["low_confidence"]:
            parts.append(self._low_confidence_html("freshness level",
                                                   freshness["top_probability"], threshold))
        parts.append('<div class="fs-label" style="margin-top:.8rem">All levels</div>')
        parts.append(bars_html(freshness["probabilities"], colors=colors))
        parts.append(note_html(f"<b>Rough guide (estimate):</b> {prefix}the eye looks like fish "
                               f"on <b>{guide['dataset_days']}</b> in the FFE dataset. "
                               f"{guide['advice']}<br><span class='fs-muted'>The dataset does "
                               "not say how its fish were stored, so this is not an exact "
                               "number of hours or days left.</span>"))
        return "".join(parts)

    # ------------------------------------------------------------------
    # History (stored in the user's browser by HistoryManager)
    # ------------------------------------------------------------------
    def _card_html(self, result, threshold):
        """The result card saved with each scan, so History shows the same card."""
        if result["eye_check"]["recognized"]:
            unusual = (self._unusual_photo_html()
                       if result["eye_check"].get("zone") == ZONE_WARNING else "")
            return (unusual + self._species_html(result["species"], threshold)
                    + self._freshness_html(result["freshness"], threshold))
        return ('<div class="fs-big">🚫 Not recognized</div>'
                '<p class="fs-muted">This photo did not look like a fish-eye close-up. '
                "The model's guesses below are NOT real results.</p>"
                + self._species_html(result["species"], threshold)
                + self._freshness_html(result["freshness"], threshold))

    def _save_to_history(self):
        """Sends the latest scan to the browser. Invisible (height 0); safe to
        repeat on every rerun because the browser ignores an id it already has."""
        entry = st.session_state.history_entry
        if entry is not None:
            with st.container(key="history_saver"):
                components.html(self._history.save_html(entry), height=0)

    def show_history_page(self):
        st.html('<div class="fs-page-title">History</div>'
                '<div class="fs-muted">Your last 20 scans on this device, newest first. '
                'Tap a scan to see its full result.</div>')
        # The list is drawn by the browser itself from its own storage
        components.html(self._history.page_html(), height=720, scrolling=True)

    # ------------------------------------------------------------------
    # Performance page
    # ------------------------------------------------------------------
    def show_performance_page(self):
        st.html('<div class="fs-page-title">Model performance</div>'
                '<div class="fs-muted">All numbers come from the <b>test set: 664 eye photos the '
                'models never saw during training</b>, split so that crops of the same photo '
                'never appear in both training and testing.</div>')
        species_tab, freshness_tab, check_tab, explain_tab = st.tabs(
            ["Species", "Freshness", "Eye check", "How it works"])
        with species_tab:
            self._show_model_report("species", "Species model · 7 classes",
                                    "Which of the 7 species the eye belongs to.")
        with freshness_tab:
            self._show_model_report(
                "freshness", "Freshness model · 3 levels",
                "Highly Fresh (storage day 1-2), Fresh (day 3-4) or Not Fresh (day 5-6), "
                "as defined by the FFE dataset.")
            self._show_freshness_experiments()
        with check_tab:
            self._show_eye_check_report()
        with explain_tab:
            self._show_explanations()

    def _show_model_report(self, task, title, description):
        report = load_report(task)
        st.html(f'<div class="fs-page-title" style="font-size:1.25rem">{title}</div>'
                f'<div class="fs-muted">{description}</div>')
        if report is None:
            st.html(note_html(f"No saved results for the {task} model yet. "
                              "Run <code>python scripts/evaluate.py</code>.", "warn"))
            return
        summary = report["summary"]
        stats = [("Accuracy", f"{summary['accuracy']:.1%}", "top prediction correct"),
                 ("Macro F1", f"{summary['macro_f1']:.2f}", "all classes count equally"),
                 ("Avg confidence", f"{summary['average_confidence']:.1%}", "average top probability"),
                 ("Calibration error", f"{summary['ece']:.3f}", "smaller is better")]
        st.html('<div class="fs-stats">' + "".join(
            f'<div class="fs-stat"><small>{label}</small><b>{value}</b><small>{hint}</small></div>'
            for label, value, hint in stats) + "</div>")

        text = (f"<b>Calibration: {summary['calibration_verdict']}.</b> "
                f"{summary['calibration_explanation']}")
        above = report["above_threshold"]
        if above:
            text += (f"<br>When at least 60% sure ({above['share']:.0%} of test photos), it is "
                     f"right <b>{above['accuracy']:.0%}</b> of the time. Below 60%, the Scan "
                     "page shows ⚠️ Low confidence.")
        if summary["small_test_classes"]:
            text += ("<br><span class='fs-muted'>" + ", ".join(summary["small_test_classes"]) +
                     " have fewer than 40 test photos, so their numbers are less reliable.</span>")
        st.html(note_html(text))

        if report["table"] is not None:
            with st.container(key=f"card_table_{task}"):
                st.html('<div class="fs-label">Results per class</div>')
                self._show_class_table(report["table"])

        charts = report["charts"]
        explanations = {
            "confusion_matrix": ("Confusion matrix",
                                 "Rows = TRUE class, columns = PREDICTED class. The diagonal "
                                 "counts correct answers; other cells show mix-ups."),
            "calibration": ("Calibration check",
                            "Photos grouped by confidence; each bar is the real accuracy in that "
                            "group. On the dashed line = honest percentages; below = "
                            "overconfident; above = underconfident."),
            "training_curves": ("Training curves",
                                "Accuracy and loss per epoch. Dotted line = fine-tuning starts. "
                                "Training stopped at the best validation epoch to avoid "
                                "overfitting."),
            "examples": ("Example predictions",
                         "Top: correct. Bottom: the most confident mistakes - often dark or "
                         "blurry photos."),
        }
        available = [(k, v) for k, v in explanations.items() if k in charts]
        # Two charts per row on wide screens; Streamlit stacks them on phones
        for start in range(0, len(available), 2):
            columns = st.columns(2, gap="medium")
            for column, (key, (heading, text)) in zip(columns, available[start:start + 2]):
                with column, st.container(key=f"card_chart_{task}_{key}"):
                    st.html(f'<div class="fs-label">{heading}</div>')
                    st.image(str(charts[key]), width="stretch")
                    st.caption(text)

    def _show_class_table(self, table):
        """Per-class precision / recall / F1 as percentages (class rows only)."""
        rows = table.drop(index=["accuracy", "macro avg", "weighted avg"], errors="ignore")
        shown = pd.DataFrame({
            "Precision": rows["precision"].map("{:.0%}".format),
            "Recall": rows["recall"].map("{:.0%}".format),
            "F1": rows["f1-score"].map("{:.2f}".format),
            "Photos": rows["support"].astype(int),
        })
        st.dataframe(shown, width="stretch")
        st.caption("**Precision**: when it says this class, how often it is right. "
                   "**Recall**: of the photos really in this class, how many it found. "
                   "**F1**: balances both (1.00 = perfect).")

    def _show_freshness_experiments(self):
        """How the final freshness model was chosen (validation data only)."""
        comparison = load_csv("comparison_val.csv")
        scaling = load_csv("temperature_scaling_half2.csv")
        if comparison is None and scaling is None:
            return
        names = {"Baseline": "First model", "E1_species_aware": "E1 (21 classes)",
                 "E2_deep_strong_aug": "E2 (deeper + strong augmentation)"}
        with st.container(key="card_experiments"):
            st.html('<div class="fs-label">How we improved the freshness model</div>')
            st.markdown(
                "The first freshness model reached 62.8% test accuracy. We tried two "
                "improvements and compared them **only on validation photos**:\n"
                "- **E1:** 21 classes (species x freshness), adding up probabilities per level.\n"
                "- **E2:** deeper fine-tuning + strong lighting/colour augmentation.")
            if comparison is not None:
                comparison = comparison.rename(index=names)
                st.dataframe(pd.DataFrame({
                    "Val. accuracy": comparison["val_accuracy"].map("{:.1%}".format),
                    "Log loss": comparison["val_log_loss"].map("{:.3f}".format),
                    "Calibration err.": comparison["val_ece"].map("{:.3f}".format),
                }), width="stretch")
            st.markdown("E2 was the most accurate but **overconfident**, so we added "
                        "**temperature scaling** (divide the scores by T before softmax), "
                        "fitted on one half of the validation photos and checked on the other.")
            if scaling is not None:
                scaling = scaling.rename(index=names)
                st.dataframe(pd.DataFrame({
                    "Accuracy": scaling["accuracy"].map("{:.1%}".format),
                    "Avg confidence": scaling["avg_confidence"].map("{:.1%}".format),
                    "Calibration err.": scaling["ece"].map("{:.3f}".format),
                }), width="stretch")
            st.markdown("A rule set **before** seeing these results chose **E2 + temperature** "
                        "as the final model. Test accuracy: 65.5% (first model: 62.8%).")

    def _show_eye_check_report(self):
        """How the eye check and its three zones work, and how well it did."""
        cutoffs, counts = load_eye_zones()
        with st.container(key="card_eyecheck"):
            st.html('<div class="fs-label">Is it really a fish eye?</div>')
            st.markdown(
                "Softmax always splits 100% among the 7 known species, so a photo of a hand "
                "can still come out as \"Tilapia 99%\". FishScan first checks whether the photo "
                "**looks like the eye photos it learned from**:\n"
                "1. The species model turns the photo into 1,280 feature numbers.\n"
                "2. They are compared with all 3,113 **training** eye photos. The **distance** "
                "is 1 minus the average similarity of the **5 most similar** ones "
                "(0 = identical).\n"
                "3. The distance puts the photo in one of **three zones**.")
            if cutoffs and cutoffs["warning"] is not None:
                t, w = cutoffs["threshold"], cutoffs["warning"]
                zones = pd.DataFrame({
                    "Zone": ["✅ Normal", "⚠️ Unusual photo", "🚫 Not recognized"],
                    "Distance": [f"up to {t:.3f}", f"{t:.3f} - {w:.3f}", f"above {w:.3f}"],
                    "What you see": ["species and freshness",
                                     "results + 'may be less reliable' note",
                                     "no results (guesses hidden)"],
                })
                if counts:
                    n, m = counts["validation_images"], counts["non_eye_images"]
                    zones["Real FFE validation eyes"] = [
                        f"{counts['validation_normal'] / n:.1%}",
                        f"{counts['validation_warning'] / n:.1%}",
                        f"{counts['validation_not_recognized'] / n:.1%}"]
                    zones["Non-eye market photos"] = [
                        f"{(m - counts['non_eye_warning'] - counts['non_eye_not_recognized']) / m:.1%}",
                        f"{counts['non_eye_warning'] / m:.1%}",
                        f"{counts['non_eye_not_recognized'] / m:.1%}"]
                st.dataframe(zones, hide_index=True, width="stretch")
                caption = ("Both cut-offs were chosen on **validation eye photos only**: 95% of "
                           "them are below the first, 99% below the second.")
                if counts:
                    caption += (f" Measured on {counts['validation_images']} validation eyes and "
                                f"{counts['non_eye_images']:,} market photos that are not eye "
                                "close-ups.")
                st.caption(caption + " Why a middle zone: without it, 5% of real eye photos "
                           "were rejected; with it, only about 1% are, while every non-eye photo "
                           "is still caught. Borderline photos get results, but with a clear "
                           "warning instead of looking fully trustworthy.")
            results = load_eye_check_results()
            if results is None:
                st.info("No test results yet. Run `python scripts/test_eye_check.py`.")
                return
            st.html('<div class="fs-label" style="margin-top:.8rem">Earlier test (strict rule '
                    'only, before the warning zone)</div>')
            rows = []
            for group in results["groups"]:
                should_pass = group["wrongly_rejected"] is not None
                rate = group["wrongly_rejected"] if should_pass else group["caught"]
                rows.append({"Photos": group["group"].replace("Extra folder: ", ""),
                             "Should be": "accepted" if should_pass else "caught",
                             "Count": group["photos"],
                             "Result": (f"{rate:.1%} wrongly rejected" if should_pass
                                        else f"{rate:.1%} caught")})
            st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
            st.caption("These results used only the first (95%) cut-off, so photos now in the "
                       "warning zone counted as rejected here. The groups were used only for "
                       "reporting, never to choose a cut-off.")

    def _show_explanations(self):
        with st.container(key="card_how"):
            st.html('<div class="fs-label">How FishScan makes its percentages</div>')
            st.markdown(
                "1. The photo is resized to 224 x 224 pixels and passed through **MobileNetV2**, "
                "pre-trained on 1.4 million everyday photos (transfer learning). It turns the "
                "eye into 1,280 numbers describing shapes, colours and textures.\n"
                "2. A small final layer gives every class a **score**.\n"
                "3. **Softmax** turns the scores into probabilities that add up to 100%.")
            scores = [4.0, 1.0, 0.5]
            plain, softened = softmax(scores), softmax(scores, temperature=1.38)
            st.dataframe(pd.DataFrame({
                "Score": scores,
                "Softmax": [f"{p:.1%}" for p in plain],
                "With T = 1.38": [f"{p:.1%}" for p in softened],
            }, index=["Highly Fresh", "Fresh", "Not Fresh"]), width="stretch")
            st.caption("Example with made-up scores. Temperature scaling divides the scores by T: "
                       "the order (and the answer) never changes, only the confidence.")
        with st.container(key="card_meaning"):
            st.html('<div class="fs-label">What does "87% Fresh" mean?</div>')
            st.markdown("It is the model's **confidence**, not a lab measurement. A well-calibrated "
                        "model that says 87% is right about 87 out of 100 times it says that.")
        with st.container(key="card_wrong"):
            st.html('<div class="fs-label">Where the percentages can be wrong</div>')
            st.markdown(
                "- **Unknown things:** softmax always picks a known class; the fish-eye check "
                "catches many such photos (\"Not recognized\") or flags them (\"Unusual "
                "photo\"), but not all. An eye of a fish FishScan does not know can still look "
                "like a normal eye.\n"
                "- **Photo quality:** dark, blurry or off-centre photos cause most mistakes.\n"
                "- **Freshness is visual only:** no bacteria, chemicals or toxins.\n"
                "- **Dataset limits:** each freshness level was shot on different days, and the "
                "same fish appear on several days.")

    # ------------------------------------------------------------------
    # About page
    # ------------------------------------------------------------------
    def show_about_page(self):
        st.html('<div class="fs-page-title">About FishScan</div>')
        left, right = st.columns(2, gap="large")
        with left:
            with st.container(key="card_what"):
                st.html('<div class="fs-label">What it does</div>')
                st.markdown(
                    "FishScan looks at **one close-up photo of a fish's eye** and estimates:\n"
                    "1. **Which species** it is (probability for each of 7 species).\n"
                    "2. **How fresh** it looks (probability for each of 3 levels).")
                st.html(note_html("<b>Student project, NOT a food safety test.</b> Always check "
                                  "the smell, gills, skin and firmness yourself.", "warn"))
            with st.container(key="card_tips"):
                st.html('<div class="fs-label">How to take a good photo</div>')
                st.markdown(PHOTO_TIPS)
            with st.container(key="card_zones"):
                st.html('<div class="fs-label">What the result screens mean</div>')
                st.markdown(
                    "Before showing a result, FishScan checks how much the photo looks like the "
                    "fish-eye photos it learned from:\n"
                    "- **✅ Normal** - looks like a typical training eye: species and freshness "
                    "are shown.\n"
                    "- **⚠️ Unusual photo** - it is an eye-like photo, but less typical "
                    "(blur, glare, angle, crop, or a fish FishScan does not know). Results are "
                    "shown **with a note that they may be less reliable**.\n"
                    "- **🚫 Not recognized** - it does not look like a fish-eye close-up, so no "
                    "results are shown.\n\n"
                    "Separately, **Low confidence** means the top answer is below 60%.")
                cutoffs, _ = load_eye_zones()
                if cutoffs and cutoffs["warning"] is not None:
                    st.caption(f"Cut-offs: distance up to {cutoffs['threshold']:.3f} normal, up "
                               f"to {cutoffs['warning']:.3f} unusual, above that not recognized. "
                               "Chosen so that 95% / 99% of real validation eye photos pass; "
                               "see Performance → Eye check.")
            with st.container(key="card_species_list"):
                # A list (not a table) so long scientific names wrap nicely on small phones
                st.html('<div class="fs-label">Species FishScan knows</div>' + "".join(
                    f'<div class="fs-list-row"><b>{html.escape(common)}</b>'
                    f'<span class="fs-sci">{html.escape(", ".join(format_scientific(s) for s in sci))}'
                    f'</span></div>' for common, sci in sorted(self._scientific.items())))
                st.caption("Names follow FishBase. Mozambique and Nile tilapia are combined into "
                           "one \"Tilapia\" class (hard to tell apart from an eye photo).")
            with st.container(key="card_levels"):
                st.html('<div class="fs-label">Freshness levels</div>' + "".join(
                    f'<div style="margin:.35rem 0"><span class="fs-badge" '
                    f'style="background:{g["color"]};color:{BADGE_TEXT[lvl]};font-size:.95rem">{g["icon"]} {lvl}</span> '
                    f'<span class="fs-muted">photographed on {g["dataset_days"]}</span></div>'
                    for lvl, g in FRESHNESS_GUIDE.items()))
                st.caption("From the dataset's documentation; not shelf-life numbers.")
        with right:
            with st.container(key="card_limits"):
                st.html('<div class="fs-label">Known limitations</div>')
                st.markdown(
                    "- **Only what it was trained on:** 7 species, 3 freshness levels, eye "
                    "close-ups. Galunggong, tulingan, dalagang bukid and lapu-lapu are not "
                    "included.\n"
                    "- **It always picks an answer:** softmax forces a choice among known "
                    "classes. The fish-eye check (\"Not recognized\" / \"Unusual photo\") and the "
                    "Low confidence warning catch many such cases, but not every one.\n"
                    "- **Visual estimate only:** no bacteria, chemicals or toxins; not a lab test.\n"
                    "- **Photo quality matters:** dark, blurry or glare photos cause most mistakes.\n"
                    "- **Dataset limits:** freshness levels were shot on different days; the same "
                    "fish appear on several days; freshness accuracy is moderate (about 65%).")
            with st.container(key="card_privacy"):
                st.html('<div class="fs-label">Privacy</div>'
                        'Your photo is analysed in memory and is <b>not stored on the server</b>. '
                        'A small thumbnail and the result are kept in <b>History on your own device '
                        'only</b> (last 20 scans); you can delete them there at any time.')
            with st.container(key="card_credits"):
                st.html('<div class="fs-label">Dataset and credits</div>')
                st.markdown(
                    "- **Dataset:** *The Freshness of the Fish Eyes (FFE) Dataset*, Mendeley Data, "
                    "[doi:10.17632/xzyx7pbr3w.1](https://doi.org/10.17632/xzyx7pbr3w.1), CC BY 4.0. "
                    "Prasetyo, E., Purbaningtyas, R., Adityo, R.D., Suciati, N., & Fatichah, C. "
                    "(2022). *Combining MobileNetV1 and Depthwise Separable Convolution Bottleneck "
                    "with Expansion for Classifying the Freshness of Fish Eyes.* Information "
                    "Processing in Agriculture. "
                    "[doi:10.1016/j.inpa.2022.01.002](https://doi.org/10.1016/j.inpa.2022.01.002)\n"
                    "- **Model:** MobileNetV2 (ImageNet) with transfer learning, TensorFlow / Keras.\n"
                    "- **Species names:** FishBase. **App:** Streamlit.")
            with st.container(key="card_devs"):
                st.html('<div class="fs-label">Developers</div>'
                        + "".join(f"<div>{html.escape(n)}</div>" for n in DEVELOPERS)
                        + '<div class="fs-muted" style="margin-top:.4rem">IPT 102 - Integrative '
                          'Programming and Technologies 2 · Predicting Probabilities · 2026</div>')


def main():
    app = FishScanApp()
    app.run()


main()
