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

**Verdict: too small and not reliable; decided Nov 27 not to train on it.** No species reaches 100 real photos, and the
real ones show only a few individual fish. A model would mostly learn the plate,
the video, or the internet photo style, and its test accuracy would look better than
it is. Not used for training. (This is also the same dataset FishScan v1 used as
"not a fish eye" photos for the eye check.)

## v2-species: Bangladesh fish dataset (chosen for species, Option A)

| | |
|---|---|
| Source | Mendeley Data: https://data.mendeley.com/datasets/8r24222wcc/3 |
| DOI | 10.17632/8r24222wcc.3 |
| License | CC BY 4.0 |
| Authors | Sunny, S., Prodhan, S., Shakib, N., Rashid, M. R. A., & Mansoor, N. (2025). Comprehensive Smartphone Image Dataset for Fish Species Identification in Bangladesh's Freshwater Ecosystems. Mendeley Data, V3. |
| File used | `Original Image.zip` (4.09 GB, RAW photos only). Version 3 splits the data into this zip and `Augmented.zip` (5.48 GB), which we do not use. |
| Where | `data\raw\bd_fish\Original Image\<Folder>\*.jpg` (only the 8 folders below are unzipped) |

Folders used: Telapiya -> Tilapia, Magur -> Hito, Shol -> Dalag, Koi -> Puyo,
Pangas -> Pangasius. Taki, Tarabaim and Foli are also read, only to look for label
conflicts with the snakeheads. Checked with `scripts/inspect_bd_species.py`
(results in `data/bd_species_manifest.csv`, contact sheets in `reports/bd_check/`, kept local).

| Species | Raw | Kept | Groups (2-min scenes) | Shooting days |
|---|---|---|---|---|
| Telapiya (Tilapia) | 2,058 | 1,578 | 33 | 4 |
| Shol (Dalag) | 1,424 | 1,190 | 34 | 5 |
| Koi (Puyo) | 842 | 487 | 11 | 2 (Feb 19 excluded) |
| Pangas (Pangasius) | 934 | 620 | 22 | 4 |
| Magur (Hito) | 574 | 248 | 9 | 1 |

- **Duplicates:** no exact duplicates, but 2,147 near-duplicates (burst shots of the same
  fish in the same pose) across the 8 folders, removed. Magur lost 57%.
- **Labels:** contact sheets look right for all 5 species. No file appears in two folders;
  the 14 cross-folder look-alikes are different fish on similar backgrounds.
  "Foli" is described as "Spotted Snakehead" on Mendeley, but its photos show a flat,
  knife-shaped featherback; Taki is the real spotted snakehead. We use neither.
  **Koi, Feb 19, 2022 (273 raw photos, both phones):** pale/golden, slimmer fish on a dark
  speckled tray, while every other Koi day shows the usual dark climbing perch. We could
  not confirm they are climbing perch, so that whole session is excluded
  (`EXCLUDED_SESSIONS` in the script; files are not deleted).
- **Fish / scenes:** the photos were taken about once per second, 1-5 days per species,
  on 3-4 phones (Feb 19 - Apr 1, 2022). Individual fish cannot be counted from the files,
  so the split groups photos by phone + 2-minute block. **Magur is one 6-minute session
  on one sack background**, so its test accuracy will look better than it is.
- **Photo type:** whole fish on trays, sacks and tarps, NOT eye close-ups. The species
  model therefore needs a whole-fish photo; the freshness model (FFE) still needs an
  eye close-up.

**Verdict: conditional GO** for Tilapia, Dalag, Puyo and Pangasius (real market photos,
correct labels, open license, ~500-1,600 clean images each). **Hito is weak** (248 images from
one session): usable only because our own hito photos give an honest test. The real
check is how the model does on OUR photos (source check below), not on the Bangladesh
test set.

## v2-species: our own market photos

Folder: `data\raw\own_photos\<species>\` with species `bangus`, `tilapia`, `hito`,
`dalag`, `puyo`, `pangasius`. File name = species, fish number, photo number:
`bangus_fish01_01.jpg`, `bangus_fish01_02.jpg`, ... `bangus_fish02_01.jpg`.
The fish number is the GROUP: all photos of one fish stay in the same split.

At the market (decided Nov 27):
- Every fish: 3-5 **whole-fish** photos for species, like the Bangladesh photos (fish fills
  most of the photo, tray or table behind it), plus 1-2 **eye close-ups** for freshness.
- Eye close-ups go in `data\raw\own_eyes\<species>\`, named with the SAME fish number:
  `bangus_fish01_eye01.jpg`. They never mix into the species model (the split only reads
  `own_photos`).
- More fish beats more photos: 3-5 photos per fish. Bangus 15-20 fish; the other five
  species at least 6 fish each, better 8-10 (each test set gets only about 1-2 of our
  fish per species).
- Vary the background (tray, table, scale, hand) for EVERY species, not only bangus.
- JPG files (iPhone: Settings > Camera > Formats > Most Compatible).
- Afterwards: `python scripts\check_own_photos.py` (names, unreadable files, duplicates,
  fish counts, eye photos without a matching fish).

## v2-species: split, balance and source check (plan; nothing trained yet)

`scripts/make_species_split.py` (run once our photos are in) -> `data/species_split.csv`:
- groups = our fish (`own:bangus:fish01`) and Bangladesh 2-minute scenes; the leakage
  check confirms no group is in two splits;
- every species + source gets ~70/15/15 by groups, so each species has our own fish in
  validation and test (tested with fake photos: OK);
- **cap:** at most 400 Bangladesh images per class (one photo per scene first, so repeated
  shots go first); training then uses balanced class weights for what is left
  (~400 per Bangladesh class vs ~90 bangus).

Source check (to run on the TEST set once, after training; rule agreed now):
1. Accuracy and macro F1 separately for Bangladesh photos and for our photos.
2. "Predicted Bangus" rate on our NON-bangus photos (does "our photo style" = bangus?),
   and on Bangladesh photos.
3. Per-fish results for our photos (few fish -> report them one by one, not only %).

Pass rule (agreed Nov 27, before training): the model is acceptable only if (a) accuracy on our photos is no more than 10 points
below the Bangladesh accuracy and (b) at most 10% of our non-bangus photos are predicted
as Bangus. Otherwise we report it honestly and keep v1 for the defense.

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
