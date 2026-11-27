# FishScan – project notes for Claude

## Goal
IPT 102 (Integrative Programming and Technologies 2) final project, theme
*Predicting Probabilities*. Developers: Chavez, Jaylord E. and Dalumpines, Rozel G.

FishScan predicts, from ONE close-up photo of a fish's **eye**:
- **species** probabilities (7 classes), and
- **freshness** probabilities (Highly Fresh / Fresh / Not Fresh),
with a "Low confidence" warning below 60% and a "Not recognized" check for photos
that are not fish-eye close-ups. Streamlit app, MobileNetV2 transfer learning.
It is a student project, NOT a food-safety test.

**Defense: December 5, 2026** (moved).

## Layout
```
app.py                 Streamlit app (Scan, Model Performance, About, History pages)
src/                   classes only, nothing runs on import
  dataset_loader.py    DatasetLoader: scan FFE, grouped/stratified split, tf.data
  fish_model.py        FishModel: MobileNetV2 + softmax head, temperature scaling
  fish_predictor.py    FishPredictor: one photo -> species + freshness + eye check
  eye_checker.py       EyeChecker: k-NN "is this a fish eye?" check
  model_evaluator.py   ModelEvaluator: test metrics, calibration, charts -> reports/
  history_manager.py   HistoryManager: last 20 scans in the browser (localStorage)
scripts/               interactive command-line scripts (run from the project folder)
  inspect_dataset.py   look at a dataset before training (counts, duplicates, sizes)
  make_split.py        make data/ffe_split.csv (asks before replacing it)
  train.py             train species / freshness / experiments (run in Colab)
  compare_experiments.py  baseline vs E1/E2/E3 on VALIDATION only
  calibrate_temperature.py fit temperature T on validation half 1, check on half 2
  build_eye_check.py   build models/eye_check.npz from train + val features
  evaluate.py          final TEST-set reports for the Model Performance page
notebooks/train_colab.ipynb   Colab steps (T4 GPU, Drive at MyDrive/FishScan)
data/ffe_split.csv     the fixed split (committed); data/raw/ffe/ is NOT committed
models/                final models only (species, freshness, eye_check.npz)
reports/               files the Model Performance page reads
```

## Setup (laptop)
- Python **3.13** (TensorFlow 2.21 has no 3.14 wheels). `py -3.13 -m venv .venv`,
  `.venv\Scripts\pip install -r requirements.txt`, `.venv\Scripts\streamlit run app.py`.
- FFE images go in `data\raw\ffe\xzyx7pbr3w-1\<Species - Level>\*.jpg`. Keep the
  `xzyx7pbr3w-1` folder: every path in `ffe_split.csv` starts with it, and Colab
  unzips the same way.
- **Training stays on Google Colab.** The laptop has an RTX 3050, but we do NOT set up
  local GPU/CUDA or WSL2. The laptop only runs the app, evaluation and small scripts (CPU).
- requirements.txt pins the SAME versions for Colab and the laptop so .keras files load.

## Coding conventions
- Scripts end with a plain `main()` call at the bottom. **No `if __name__ == "__main__":` guard.**
- User input: `while True:` + `try:` / `raise ValueError` / `except ValueError:` + `continue`,
  `break` on valid input (see `ask_yes_no` / `ask_number` in the scripts).
- Scripts start with a docstring (what it does, how to run), then
  `PROJECT_DIR = Path(__file__).resolve().parent.parent` and `sys.path.insert(0, ...)`.
- `src/` files only DEFINE classes/functions; OOP with private `_attributes` + properties.
- Simple English comments that explain *why*; student-readable code.
- Never overwrite a model or the split without asking (y/n prompt).

## Decisions (and why)
- **Dataset: FFE** (Freshness of the Fish Eyes, Mendeley, CC BY 4.0, DOI 10.17632/xzyx7pbr3w.1),
  4,390 eye crops, 8 species folders x 3 freshness levels.
- **Two tilapia species merged into "Tilapia"** (not separable from an eye photo) -> 7 species.
- **Split by source photo** (`IMG_yyyymmdd_hhmmss`): all crops of one photo stay in one split
  (no leakage); stratified by species + freshness; seed 42. train 3,113 / val 613 / test 664.
- **Two separate models** (species 7 classes, freshness 3 classes), same FishModel class:
  MobileNetV2 frozen head training (stage 1), then fine-tune last layers with BatchNorm frozen.
  Class weights for imbalance.
- **Freshness experiments** compared on VALIDATION only, rule agreed beforehand:
  replace baseline only if +3 accuracy points AND ECE not worse than baseline + 0.01.
  - Baseline 57.3% val acc, ECE 0.053
  - E1 species-aware 21 classes: 59.7%, ECE 0.119 -> no
  - E2 deeper fine-tune (60 layers) + strong lighting/colour augmentation: 63.3%, ECE 0.086
    -> +6.0 points but failed on ECE; E3 skipped.
  - **Temperature scaling** fixed E2's calibration: T fitted on val half 1 = **1.3818**,
    checked on val half 2 (311 images): ECE 0.084 -> 0.065 (baseline 0.084), accuracy
    unchanged at 62.1% vs baseline 53.7%. **Final freshness model = E2 + T=1.3818**
    (T stored in `models/freshness_meta.joblib`). Species model has no temperature.
- **Fish-eye check** (softmax always sums to 100%, even for a photo of a hand):
  k-NN (k=5, cosine) on the species model's 1,280 pooled features vs. all 3,113 training
  photos; threshold = distance that 95% of validation eye photos stay under (0.2371).
- **Confidence threshold 60%** -> "Low confidence" warning.
- **Freshness guide** uses only the dataset's storage-day labels (no invented shelf life).
- **History** is stored in the user's browser (public app; server files are lost on restart).

## Results (TEST set, 664 images, evaluated once)
| | Accuracy | Macro F1 | ECE | Calibration |
|---|---|---|---|---|
| Species | 96.5% | 0.958 | 0.018 | well calibrated |
| Freshness | 65.5% | 0.649 | 0.034 | slightly underconfident (-2.9%) |

- Species: weakest class Yellow Drum (recall 0.84). Few test images for Fourfinger
  Threadfin (31) and Leaftail Croaker (38) -> noisy metrics.
- Freshness: Highly Fresh F1 0.74, Fresh F1 0.55 (hardest, the middle level), Not Fresh F1 0.65.
- Eye check: FFE test 619/664 pass (6.8% wrongly rejected); 40/40 generated images and
  1,813/1,813 Philippines fish-market photos (not eye close-ups) caught.

## History of this repo
- 2026-11-23: old laptop broke. Restored from GitHub + Google Drive. `make_split.py` and
  `inspect_dataset.py` were found in the Drive backup (restored as-is).
  `calibrate_temperature.py` and `build_eye_check.py` were lost and REBUILT from `src/` and
  the reports (see the note at the top of each). Both were checked against the saved files
  without saving anything: build_eye_check gives the same 3,113 references and threshold
  0.2371; calibrate_temperature (option 2) gives the same T=1.3818, the same 302/311
  validation halves and the same half-2 table. Scripts, notebook, split and this file
  are now pushed to GitHub.

## To-do
- [x] Separate species dataset: only a **SUGGESTION** from our teacher, **not a requirement**.
      Checked Philippine datasets on Nov 24, 2026 (see data/DATASETS.md): none clean
      enough (stock/web photos, FFE images inside, near-duplicate leakage, no galunggong).
      Keep v1 (FFE only, tag `v1-ffe-only`) and present it as future work.
- [x] Branch `v2-species`: Philippines Fish Market Dataset rejected (too few real photos).
- [ ] Branch `v2-species` (Option A, decided Nov 27): species = Bangladesh Mendeley dataset
      (raw photos; Tilapia, Hito, Dalag, Puyo, Pangasius) + our own photos (bangus + all
      5 others), freshness stays FFE. Evaluation, split rules, cap 400, and the source
      check rule are in data/DATASETS.md. **Don't train until our photos are in**; then
      scripts/check_own_photos.py -> scripts/make_species_split.py. Species photos are
      WHOLE fish (Bangladesh style), so the app will need a second photo for species.
- [ ] Prepare for the defense on **December 5, 2026**.
- [ ] `scripts/test_eye_check.py` (wrote `reports/eye_check_results.json`; app.py mentions it)
      is also lost: rebuild if the eye-check results need re-running.
- [ ] Training history CSVs (`reports/*_history.csv`) and experiment models
      (`models/experiments/`) exist only on Google Drive.
