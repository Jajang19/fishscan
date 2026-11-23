# Datasets

## FFE – Freshness of the Fish Eyes (used by FishScan v1)

| | |
|---|---|
| Source | Mendeley Data: https://data.mendeley.com/datasets/xzyx7pbr3w/1 |
| DOI | 10.17632/xzyx7pbr3w.1 |
| License | CC BY 4.0 (free to use, credit the authors) |
| Content | 4,390 eye crops, 8 species folders x 3 freshness levels (Highly Fresh / Fresh / Not Fresh) |
| File | `xzyx7pbr3w-1.zip` (about 188 MB) |

**Please cite:**
Prasetyo, E., Purbaningtyas, R., Adityo, R.D., Suciati, N., & Fatichah, C. (2022).
Combining MobileNetV1 and Depthwise Separable Convolution Bottleneck with Expansion
for Classifying the Freshness of Fish Eyes. *Information Processing in Agriculture*.
https://doi.org/10.1016/j.inpa.2022.01.002

### Where to put it
The images are NOT in GitHub (too big). After downloading the zip:

```
Laptop:  unzip into data\raw\ffe\  so you get  data\raw\ffe\xzyx7pbr3w-1\<Species - Level>\*.jpg
Colab:   notebooks/train_colab.ipynb Step 4 unzips it to /content/ffe/xzyx7pbr3w-1/...
```

Keep the `xzyx7pbr3w-1` folder. Every path in `data/ffe_split.csv` starts with it.
Check: `python scripts/inspect_dataset.py` should count 4,390 images.

### Split (`data/ffe_split.csv`, committed)
- Made once by `scripts/make_split.py` (seed 42). Training, evaluation and the
  eye check all load this same file, so the test set never changes.
- Grouped by source photo (`IMG_yyyymmdd_hhmmss`): all crops of one phone photo stay
  in one split, so no near-copies leak from train into test.
- Stratified by species + freshness: train 3,113 / val 613 / test 664.

## Why we use FFE only (v1)
- **One photo, both labels.** Every FFE image has a species AND a freshness label, so
  both models learn from the same kind of photo the app receives: a close-up of one eye.
- **Same conditions for both models.** One dataset means the same camera style,
  cropping and background, so the species model's features also work for the
  fish-eye check (`models/eye_check.npz`).
- **Clean, documented source.** Published on Mendeley with a DOI, a peer-reviewed paper,
  and an open license. Freshness levels come from the dataset's own storage-day labels.
- **Leakage-safe split.** File names show which crops come from the same photo, so
  we could split by source photo.
- **Results are good enough for the project.** Species 96.5%, freshness 65.5% test
  accuracy, both with a low calibration error (ECE 0.018 and 0.034, see `reports/`).

Limits: the 7 species are FFE's species (no galunggong or tulingan), and only eye
photos are supported. Two tilapia species are merged into one "Tilapia" class.

Our teacher **suggested** (not required) a separate dataset for the species model.
v1 is tagged `v1-ffe-only` on GitHub. Philippine fish datasets are being checked;
one will only be used if it is clean and reliable.

## Future work: DaFiF – Dataset for Fish's Freshness problems

| | |
|---|---|
| Source | Mendeley Data: https://data.mendeley.com/datasets/vx4ptwk3pb/1 |
| Paper | Prasetyo, E. et al. (2024). DaFiF: A complete dataset for fish's freshness problems. *Data in Brief*. https://doi.org/10.1016/j.dib.2024.111016 |
| Content | Mackerel, tilapia and tuna (21 fish each): about 2,500 images, gas-sensor readings (MQ-135, TGS 2602) and sensory freshness scores |
| License | Check the Mendeley page before using it |

Why it is interesting: it comes from the same research group as FFE, and it covers
tuna and mackerel, which are common in Philippine markets. It also measures freshness
with sensors and sensory scores, not only storage days. Why we did not use it now:
only 3 species, so it would need new
training and a new eye check, and its photos have not been checked yet.
