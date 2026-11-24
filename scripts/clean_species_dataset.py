"""
clean_species_dataset.py
Checks the Philippines Fish Market Dataset (github.com/CloneIsL1fe/Fish_Identifier_Philippines)
BEFORE we use it for a separate species model. Nothing is deleted: every image gets
one row in data/species_manifest.csv with a decision ("keep" / "remove") and a reason.

  1. Exact duplicates      same file content (MD5)      -> keep one copy
  2. Near-duplicates       perceptual hash (pHash), Hamming distance <= NEAR_DUP_BITS
                           -> keep one copy (the biggest image)
  3. Groups                .rf. copies with the same base name, burst copies "(0)", "(1)",
                           and near-duplicates share one group id, so a split by group
                           keeps them together (no leakage)
  4. Source of each photo  guessed from the file name: phone photo, market video frame,
                           Wikimedia, lab experiment, stock, online shop, internet, ...
                           Stock and shop photos and known label errors are marked "remove";
                           each image also gets a trust level: real / lab / internet
  5. Contact sheets        reports/species_cleaning/<source>.jpg so a person can look
                           at what would be removed before anything is decided

Run from the project folder:  python scripts/clean_species_dataset.py
"""

import hashlib
import re
import sys
from collections import defaultdict
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))  # so "from src..." works

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageOps
from scipy.fft import dctn

DATA_DIR = PROJECT_DIR / "data" / "raw" / "ph_market" / "Philippines Fish Market Dataset"
MANIFEST_CSV = PROJECT_DIR / "data" / "species_manifest.csv"
SHEETS_DIR = PROJECT_DIR / "reports" / "species_cleaning"

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif"}
NEAR_DUP_BITS = 6   # pHash bits that may differ for "almost the same picture" (out of 64)
MIN_REAL_PER_SPECIES = 100

# File-name patterns -> where the photo probably came from (checked top to bottom)
SOURCE_PATTERNS = [
    ("stock", r"shutterstock|stock-photo|adobestock|^\d+_f_\d+|istock|dreamstime|depositphotos|123rf|alamy|"
              r"freepik|vecteezy|isolated|white-background|-\d{8,9}\.webp$"),
    ("online_shop", r"\d+x\.webp$|@2x|_\d+x\d+|approx|cleaningonly|-\d+w\.jpg$|product|shopee|lazada"),
    ("wikimedia", r"photos_taken|^\d+px-|wiki|commons"),
    ("screenshot", r"screenshot"),
    ("bangus_spoilage_set", r"^(fresh|autolysis|bloated|putrefaction)-\d+"),
    ("phone_photo", r"^(img_)?20[12]\d[01]\d[0-3]\d_[0-2]\d{5}"),  # phone date_time names
    ("numbered_png", r"^[a-z]\d+\.png$"),
    ("web_misc", r"^images|thumbnail|^picture\d+|download|^\d+_\d+\.\d+\.\d+\.\d+|"
                 r"^[0-9a-f]{8}-[0-9a-f]{4}-|^th[ (_]"),
]

# Sources that are removed: stock photos and online-shop product pictures
REMOVE_SOURCES = {"stock", "online_shop"}

# How much we trust each source as a REAL photo of a fish (not from the internet)
TRUST = {
    "phone_photo": "real",          # 20241212_174000.jpg / IMG_20241125_...: phone camera
    "numbered_png": "real",         # frames from market videos (b19.png, u9.png, ...)
    "screenshot": "real",           # screenshots of market videos
    "wikimedia": "real",            # Bulacan market photos on Wikimedia Commons
    "bangus_spoilage_set": "lab",   # ~20 tagged bangus (T1-T20) in a spoilage experiment
    "web_misc": "internet",
    "unknown": "internet",
    "stock": "internet",
    "online_shop": "internet",
}

# Wrong labels found by looking at the contact sheets (file name start -> why)
LABEL_ERRORS = {
    "Hiwas/Screenshot-2024-10-20-": "label error: tuna in a green tub, not hiwas (moonfish)",
    "Tilapia/pelvicachromis_taeniata": "label error: Pelvicachromis cichlid, not tilapia",
}


def ask_yes_no(question):
    """Asks a y/n question and returns True for yes."""
    while True:
        try:
            answer = input(f"{question} (y/n): ").strip().lower()
            if answer not in ("y", "n"):
                raise ValueError
            break
        except ValueError:
            print("Please type y or n.")
            continue
    return answer == "y"


def get_base_name(file_name):
    """Strips Roboflow and burst endings so copies of one photo share one base name.

    Examples:
      BLOATED-14-_JPG.rf.897815d9....jpg  -> bloated-14-
      20241212_174032(1).jpg              -> 20241212_174032
      t6_jpg.rf.22afc2a2....jpg           -> t6
    """
    stem = Path(file_name).stem.lower()
    stem = re.sub(r"\.rf\.[0-9a-f]{32}$", "", stem)            # Roboflow export hash
    stem = re.sub(r"(_(jpe?g|png|webp))+$", "", stem)          # "_jpg", "_jpeg_jpg", "_png"
    stem = re.sub(r"\(\d+\)$", "", stem)                       # burst copies (0), (1)
    return stem.strip(" _-") or stem


def guess_source(file_name):
    """Returns the most likely source of a photo, from its file name.

    EXIF camera data is NOT used: many internet pictures (museum specimens,
    underwater photos) still carry the camera make of whoever took them.
    """
    name = file_name.lower()
    for source, pattern in SOURCE_PATTERNS:
        if re.search(pattern, name):
            return source
    return "unknown"


def perceptual_hash(image):
    """64-bit pHash: similar-looking pictures get hashes that differ in few bits."""
    small = np.asarray(image.convert("L").resize((32, 32), Image.LANCZOS), dtype=np.float32)
    low = dctn(small, norm="ortho")[:8, :8].flatten()[1:]   # 63 low frequencies, no DC
    return np.append(low > np.median(low), False)            # 64 booleans


def read_images():
    """One row per image file: species, size, camera, MD5, pHash, source, base name."""
    rows, hashes = [], []
    paths = sorted(p for p in DATA_DIR.rglob("*") if p.is_file())
    print(f"Reading {len(paths)} files, please wait...")
    for path in paths:
        row = {"path": path.relative_to(DATA_DIR).as_posix(), "species": path.parent.name,
               "file_name": path.name, "ok": False, "width": 0, "height": 0,
               "camera_make": "", "md5": "", "base_name": get_base_name(path.name)}
        hash_bits = np.zeros(64, dtype=bool)
        if path.suffix.lower() in IMAGE_EXTENSIONS:
            try:
                with Image.open(path) as image:
                    image.load()
                    row["width"], row["height"] = image.size
                    row["camera_make"] = str(image.getexif().get(271, "")).strip("\x00 ")
                    hash_bits = perceptual_hash(ImageOps.exif_transpose(image).convert("RGB"))
                row["ok"] = True
                row["md5"] = hashlib.md5(path.read_bytes()).hexdigest()
            except Exception:
                pass  # unreadable image: stays ok=False
        row["source"] = guess_source(path.name)
        row["trust"] = TRUST[row["source"]]
        rows.append(row)
        hashes.append(hash_bits)
    return pd.DataFrame(rows), np.array(hashes)


class Groups:
    """Union-find: puts images that belong together into one group."""

    def __init__(self, n):
        self._parent = list(range(n))

    def find(self, i):
        while self._parent[i] != i:
            self._parent[i] = self._parent[self._parent[i]]
            i = self._parent[i]
        return i

    def join(self, i, j):
        self._parent[self.find(i)] = self.find(j)


def decide(table, hashes):
    """Adds group ids and the keep/remove decision with a reason."""
    table["decision"], table["reason"] = "keep", ""
    table.loc[~table["ok"], ["decision", "reason"]] = ["remove", "unreadable or not an image"]
    for start, why in LABEL_ERRORS.items():
        wrong = table["path"].str.startswith(start)
        table.loc[wrong, ["decision", "reason"]] = ["remove", why]
    ok = table.index[table["ok"]].to_numpy()

    groups = Groups(len(table))
    # .rf. copies and burst copies: same species + same base name
    for _, members in table[table["ok"]].groupby(["species", "base_name"]).groups.items():
        members = list(members)
        for other in members[1:]:
            groups.join(members[0], other)

    # exact duplicates (any species): keep the first copy
    for _, members in table[table["ok"]].groupby("md5").groups.items():
        members = sorted(members)
        for other in members[1:]:
            groups.join(members[0], other)
            if table.at[other, "decision"] == "keep":
                table.loc[other, ["decision", "reason"]] = [
                    "remove", f"exact duplicate of {table.at[members[0], 'path']}"]

    # near-duplicates: pairs of the SAME species whose pHash differs in few bits.
    # (Across species, two different fish cut-outs on a white background can get
    # similar hashes, so only exact duplicates are matched across species.)
    bits = hashes[ok].astype(np.uint8)
    distance = (bits[:, None, :] != bits[None, :, :]).sum(axis=2)
    species = table.loc[ok, "species"].to_numpy()
    same_species = species[:, None] == species[None, :]
    close = np.triu((distance <= NEAR_DUP_BITS) & same_species, k=1)
    near_pairs = [(ok[a], ok[b]) for a, b in zip(*np.nonzero(close))]
    near = Groups(len(table))
    for a, b in near_pairs:
        groups.join(a, b)
        near.join(a, b)
    clusters = defaultdict(list)
    for a, b in near_pairs:
        clusters[near.find(a)] += [a, b]
    for members in clusters.values():
        members = sorted(set(members), key=lambda i: -table.at[i, "width"] * table.at[i, "height"])
        for other in members[1:]:
            if table.at[other, "decision"] == "keep":
                table.loc[other, ["decision", "reason"]] = [
                    "remove", f"near duplicate of {table.at[members[0], 'path']}"]

    table["group"] = [groups.find(i) for i in range(len(table))]
    table["cross_species_group"] = table.groupby("group")["species"].transform("nunique") > 1

    # untrusted sources
    untrusted = table["source"].isin(REMOVE_SOURCES) & (table["decision"] == "keep")
    table.loc[untrusted, "decision"] = "remove"
    table.loc[untrusted, "reason"] = "source: " + table.loc[untrusted, "source"]
    return table, len(near_pairs)


def contact_sheet(table, title, out_path, n=40, seed=42):
    """Saves a grid of up to n thumbnails with their file names."""
    if table.empty:
        return
    sample = table.sample(min(n, len(table)), random_state=seed)
    cols, size = 8, 150
    rows = (len(sample) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * size, rows * (size + 28) + 30), "white")
    draw = ImageDraw.Draw(sheet)
    draw.text((6, 8), f"{title}  ({len(table)} images, showing {len(sample)})", fill="black")
    for k, (_, row) in enumerate(sample.iterrows()):
        x, y = (k % cols) * size, 30 + (k // cols) * (size + 28)
        with Image.open(DATA_DIR / row["path"]) as image:
            thumb = ImageOps.exif_transpose(image).convert("RGB")
            thumb.thumbnail((size - 6, size - 6))
        sheet.paste(thumb, (x + 3, y + 3))
        draw.text((x + 3, y + size), f"{row['species'][:9]}: {row['file_name'][:20]}", fill="black")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out_path, quality=85)


def print_report(table, near_pairs):
    """Prints what was found and the final counts per species."""
    ok = table[table["ok"]]
    print(f"\nImages: {len(table)} files, {len(ok)} readable")
    print(f".rf. copies: {table['file_name'].str.contains('.rf.', regex=False).sum()}")
    extra = len(ok) - ok["md5"].nunique()
    print(f"Exact duplicates: {extra} extra copies in "
          f"{(ok.groupby('md5').size() > 1).sum()} groups")
    print(f"Near-duplicate pairs (pHash <= {NEAR_DUP_BITS} bits): {near_pairs}")
    crossing = table.loc[table["cross_species_group"], "group"].nunique()
    print(f"Groups that contain more than one species (label conflict!): {crossing}")

    print("\nSource of each image (from file name / EXIF):")
    print(pd.crosstab(table["species"], table["source"], margins=True).to_string())

    print("\nWhy images are removed:")
    reasons = table.loc[table["decision"] == "remove", "reason"].str.replace(r" of .*", "", regex=True)
    print(reasons.value_counts().to_string())

    kept = table[table["decision"] == "keep"]
    summary = pd.DataFrame({
        "files": table.groupby("species").size(),
        "kept_images": kept.groupby("species").size(),
        "kept_groups": kept.groupby("species")["group"].nunique(),
        **{f"{trust}_groups": kept[kept["trust"] == trust].groupby("species")["group"].nunique()
           for trust in ("real", "lab", "internet")},
    }).fillna(0).astype(int)
    print("\nPer species (groups = copies and near-duplicates counted once):")
    print(summary.to_string())
    low = summary.index[summary["real_groups"] < MIN_REAL_PER_SPECIES].tolist()
    if low:
        print(f"\nBelow {MIN_REAL_PER_SPECIES} real (non-internet) photos: {low}")
    return summary


def main():
    if not DATA_DIR.is_dir():
        print(f"Dataset folder not found: {DATA_DIR}")
        return
    if MANIFEST_CSV.exists() and not ask_yes_no(f"Replace {MANIFEST_CSV.name}?"):
        print("Kept the existing manifest. Nothing changed.")
        return

    table, hashes = read_images()
    table, near_pairs = decide(table, hashes)
    print_report(table, near_pairs)

    table.drop(columns=["md5"]).to_csv(MANIFEST_CSV, index=False)
    print(f"\nSaved {MANIFEST_CSV}")
    for source, part in table[table["ok"]].groupby("source"):
        contact_sheet(part, f"source: {source}", SHEETS_DIR / f"source_{source}.jpg")
    near = table[table["reason"].str.startswith("near duplicate")]
    contact_sheet(near, "removed as near duplicate", SHEETS_DIR / "near_duplicates.jpg")
    removed = table[table["reason"].str.startswith(("source:", "label error"))]
    contact_sheet(removed, "removed: stock / online shop / label errors",
                  SHEETS_DIR / "removed_stock_shop_labels.jpg", n=48)
    internet = table[(table["decision"] == "keep") & (table["trust"] == "internet")]
    contact_sheet(internet, "kept for now but from the internet (candidates to remove)",
                  SHEETS_DIR / "internet_candidates.jpg", n=48)
    print(f"Saved contact sheets to {SHEETS_DIR}")


main()
