# FishScan 🐟

Fish species and freshness probability prediction from one close-up photo of a
fish's **eye**, using computer vision and deep learning (MobileNetV2 transfer
learning, TensorFlow/Keras) with a Streamlit web app.

IPT 102 (Integrative Programming and Technologies 2) final project -
theme: *Predicting Probabilities*.

Developers: Chavez, Jaylord E. and Dalumpines, Rozel G.

## What it does
- **Species** probabilities for 7 classes: Bangus (Milkfish), Tilapia, Island Mackerel,
  Goldband Goatfish, Yellow Drum, Leaftail Croaker, Fourfinger Threadfin.
- **Freshness** probabilities for 3 levels: Highly Fresh, Fresh, Not Fresh.
- A **Low confidence** warning when the top probability is below 60%.
- A rough freshness guide based on the dataset's label definitions, a safety note,
  and a disclaimer.

**FishScan is a student project, NOT a food safety test.** A photo cannot detect
bacteria, chemicals or toxins.

## Run locally
Requires Python 3.13.

```
py -3.13 -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

## Dataset and credits
Freshness of the Fish Eyes (FFE) dataset, Mendeley Data, CC BY 4.0,
DOI 10.17632/xzyx7pbr3w.1. Please cite:
Prasetyo, E., Purbaningtyas, R., Adityo, R.D., Suciati, N., & Fatichah, C. (2022).
Combining MobileNetV1 and Depthwise Separable Convolution Bottleneck with Expansion
for Classifying the Freshness of Fish Eyes. *Information Processing in Agriculture*.
https://doi.org/10.1016/j.inpa.2022.01.002

The dataset images are not included in this repository.
