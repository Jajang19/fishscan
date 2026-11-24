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
v1 is tagged `v1-ffe-only` on GitHub. We checked Philippine fish datasets (below);
none was clean enough, so this stays future work.

## Checked: Philippine fish datasets for a separate species model (Nov 24, 2026)

Our teacher suggested a separate species dataset with Philippine fish (bangus, tilapia,
galunggong, tulingan). We searched Roboflow Universe and Kaggle. Most hits were
grocery/food detection projects; these were the best fish-only candidates:

| | 5 Classes Fish Casptone | Katambak (BISU) | Fish Dataset (Marinig Port) |
|---|---|---|---|
| Where | [Roboflow](https://universe.roboflow.com/fish-befuh/5-classes-fish-casptone-rq4yk) (PUP capstone) | [Roboflow](https://universe.roboflow.com/bisu-lpet1/katambak) | [Kaggle](https://www.kaggle.com/datasets/markdaniellampa/fish-dataset) |
| Task | object detection | object detection | classification |
| Images | 10,488 (v2: 17,810 with 2x rotation) | 7,827, no augmentation | 13,304 (own counts don't add up) |
| Classes | Bangus, Island Mackerel, Lapu-Lapu, Tilapia, Tulingan | 11 species x Day 1-7 = 78 (+ a stray "\\" class); has bangus, tilapia, tulingan | 31 mostly freshwater species; has bangus, tilapia |
| Galunggong | no | no | no |
| Photo sources (sample of 50) | 13 stock/web/Facebook/screenshots, **9 FFE images** (our own test data!), 17 phone photos (11 burst near-copies), 11 from other sets; one file uploaded twice | all real phone photos, Oct 5-21 2023 | renamed `Bangus 1.jpg ...`, many 4-10 kB files (web thumbnails); source not documented |
| Duplicates | `.rf.` re-exports, burst shots, a repeated file | same-second pairs (`102806` / `102806_2`) split across train/valid/test | unknown |
| License | CC BY 4.0 | CC BY 4.0 | CDLA-Sharing-1.0 |

**Verdict: none is clean enough to rely on.**
- 5 Classes Fish Casptone: mixed internet/stock photos and it **contains FFE images**,
  so testing on FFE would leak. A sibling project, "Fish Classification" (DA Project,
  4,672 images), uses the same source photos and has the same problems.
- Katambak: the most honest photos, but each class is a few fish shot within minutes,
  and near-identical shots are randomly split, so its 99.4% mAP is leakage. Whole-fish
  detection, not eye close-ups, and no galunggong.
- Marinig Fish Dataset: undocumented sources (likely web images), inconsistent stats,
  no galunggong or tulingan.

So FishScan stays **v1 (FFE only)**. A separate Philippine species dataset is
**future work**: it should be photographed by us (real market fish, several fish per
species, eye close-ups, split by fish, not by photo).

## v2-species: Philippines Fish Market Dataset (evaluated, not used for training)

| | |
|---|---|
| Source | GitHub: https://github.com/CloneIsL1fe/Fish_Identifier_Philippines (by CloneIsL1fe); images from the Google Drive link in its README (`Philippines_Fish_Market_Dataset.zip`, 1.07 GB, uploaded Dec 14, 2024) |
| License | **Unclear.** The repo has no LICENSE file and the README gives no image license. Many images are copied from the internet (Shutterstock, Adobe Stock, online shops, Wikimedia Commons), so they keep their original owners' rights. Fine for a class experiment with credit; NOT safe to redistribute. |
| Content | 1,814 files in 5 folders: Bangus 538, Dalagang bukid 179, Hiwas 157, Tilapia 800, Tulingan 140 |
| Where | `data\raw\ph_market\Philippines Fish Market Dataset\` (not in GitHub) |

Checked with `scripts/clean_species_dataset.py` (nothing deleted; decisions are in
`data/species_manifest.csv`, contact sheets in `reports/species_cleaning/`, kept local):

- 922 Roboflow `.rf.` copies, grouped by base name so copies stay in one split.
- 160 exact duplicates (MD5) and 206 near-duplicates (pHash, at most 6 of 64 bits
  different, same species only) removed.
- Removed 16 stock / online-shop pictures (Shutterstock watermarks, product photos).
- Label errors removed: 9 "Hiwas" screenshots that show tuna, 1 "Tilapia" that is a
  Pelvicachromis cichlid.
- Most Bangus images (287) come from a lab spoilage experiment of about 20 tagged fish
  (T1-T20), not a market. The same files also appear in Roboflow projects.
- Most Tilapia images are internet pictures (white backgrounds, aquariums, cooked fish,
  watermarks).

Independent photos left per species (copies counted once):

| Species | Files | Kept | Real photos | Lab | Internet |
|---|---|---|---|---|---|
| Bangus | 538 | 366 | 56 | 162 | 148 |
| Dalagang bukid | 179 | 95 | 85 | 0 | 10 |
| Hiwas | 157 | 80 | 62 | 0 | 18 |
| Tilapia | 800 | 338 | 23 | 0 | 330 |
| Tulingan | 140 | 94 | 89 | 0 | 5 |

"Real" = phone photos, market video frames, Wikimedia market photos. Even these are
few fish: the phone photos of Dalagang bukid, Hiwas and Tulingan were each taken
within about 2 minutes on Dec 12, 2024 (a few fish on one plate), and the numbered
PNGs are frames from a few market videos.

**Verdict: too small and not reliable.** No species reaches 100 real photos, and the
real ones show only a few individual fish. A model would mostly learn the plate,
the video, or the internet photo style, and its test accuracy would look better than
it is. Not used for training. (This is also the same dataset FishScan v1 used as
"not a fish eye" photos for the eye check.)

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
