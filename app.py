"""
app.py
FishScan - Streamlit app (mobile-first).

  Scan page: take or upload a close-up photo of a fish's EYE, then see the
  species and freshness probabilities, a low-confidence warning, a rough
  freshness guide and a safety note.

The app only LOADS the saved models (models/*.keras); it never trains.
The photo is analysed in memory and is never saved to disk.

Run:  streamlit run app.py
Phone on the same Wi-Fi: open the "Network URL" that Streamlit prints.
"""

import hashlib
from pathlib import Path

import streamlit as st
from PIL import Image, UnidentifiedImageError

from src.fish_predictor import FishPredictor, scientific_names_by_species

PROJECT_DIR = Path(__file__).resolve().parent
MODELS_DIR = PROJECT_DIR / "models"


@st.cache_resource(show_spinner="Loading the FishScan models...")
def load_predictor():
    """Loads both models ONCE and keeps them in memory for every click and user."""
    return FishPredictor(MODELS_DIR)


def format_scientific(name):
    """'Chanos Chanos' -> 'Chanos chanos' (genus capitalised, species lowercase)."""
    genus, species = name.split(" ", 1)
    return f"{genus} {species.lower()}"


class FishScanApp:
    """The Streamlit user interface. It only calls FishPredictor; no model logic here."""

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
        page = st.sidebar.radio("Menu", ["Scan", "Model Performance", "About"])
        st.sidebar.caption("FishScan - IPT 102 final project")
        if page == "Scan":
            self.show_scan_page()
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


def main():
    app = FishScanApp()
    app.run()


main()
