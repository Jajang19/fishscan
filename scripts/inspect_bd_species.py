"""
inspect_bd_species.py
Checks the RAW photos of the Bangladesh fish dataset (Mendeley, DOI 10.17632/8r24222wcc.3,
"Original Image.zip") for the 5 species we want for the v2 species model:
    Telapiya -> Tilapia, Magur -> Hito, Shol -> Dalag, Koi -> Puyo, Pangas -> Pangasius

For a fair label check it also reads the snakehead look-alikes Taki, Tarabaim and Foli
(the dataset describes BOTH Foli and Taki as "Spotted Snakehead").

  1. Exact duplicates (MD5) and near-duplicates (pHash, same species)
  2. The same picture in two species folders -> label conflict
  3. Scenes: photos from the same phone in the same 2-minute block (EXIF time) are one
     scene. Scenes are the GROUPS for the split, so burst shots of one fish never end
     up in both train and test. (Individual fish cannot be told apart: the photos were
     taken about once per second, all day, so any chaining merges whole sessions.)
  4. Visits: photos more than VISIT_GAP minutes apart start a new market visit
  5. Contact sheets per species in reports/bd_check/ to look at the labels

Nothing is deleted. Results: data/bd_species_manifest.csv (one row per image).

Run from the project folder:  python scripts/inspect_bd_species.py
"""

import hashlib
import sys
from datetime import datetime
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))  # so "from src..." works

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageOps

from src.image_checks import Groups, hamming_matrix, perceptual_hash
from src.species_v2 import BD_FOLDERS

BD_DIR = PROJECT_DIR / "data" / "raw" / "bd_fish"
MANIFEST_CSV = PROJECT_DIR / "data" / "bd_species_manifest.csv"
SHEETS_DIR = PROJECT_DIR / "reports" / "bd_check"

LOOKALIKES = ["Taki", "Tarabaim", "Foli"]   # read only for the label check
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
NEAR_DUP_BITS = 6      # pHash bits that may differ for "almost the same picture" -> removed
SCENE_BLOCK = 120      # seconds: photos from one phone in the same 2-minute block = one group
                       # (they shot about 1 photo per second, so chaining by gaps or by
                       #  look-alike pictures merges a whole session into one group)
VISIT_GAP = 30         # minutes: a bigger gap = a new market visit
MIN_SCENES = 100       # what we want per species, at least

# Photos we leave out after looking at them: (folder, shooting date) -> why.
# Koi on 2022-02-19 (both phones) shows pale/golden, slimmer fish on a dark speckled
# tray. All other Koi days show the usual dark climbing perch. We could not confirm
# that the golden fish are climbing perch, so they are excluded.
EXCLUDED_SESSIONS = {
    ("Koi", "2022-02-19"): "excluded: pale/golden fish, not confirmed as climbing perch",
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


def find_species_folders():
    """{folder name: path} for the folders we need, wherever they are after unzipping."""
    wanted = set(BD_FOLDERS) | set(LOOKALIKES)
    found = {}
    for folder in sorted(p for p in BD_DIR.rglob("*") if p.is_dir()):
        if folder.name in wanted and "_download" not in folder.parts:
            found.setdefault(folder.name, folder)
    return found


def photo_time(exif):
    """EXIF DateTimeOriginal (or DateTime) as a datetime, or None."""
    raw = exif.get_ifd(0x8769).get(36867) or exif.get(306)
    try:
        return datetime.strptime(str(raw).strip("\x00 "), "%Y:%m:%d %H:%M:%S")
    except ValueError:
        return None


def read_images(folders):
    """One row per image: folder, size, phone model, time, MD5, plus the pHash."""
    rows, hashes = [], []
    for name, folder in folders.items():
        paths = sorted(p for p in folder.rglob("*")
                       if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)
        print(f"  {name}: {len(paths)} images")
        for path in paths:
            row = {"path": path.relative_to(BD_DIR).as_posix(), "folder": name,
                   "species": BD_FOLDERS.get(name, ""), "file_name": path.name,
                   "ok": False, "width": 0, "height": 0, "camera": "", "time": None, "md5": ""}
            hash_bits = np.zeros(64, dtype=bool)
            try:
                with Image.open(path) as image:
                    row["width"], row["height"] = image.size
                    exif = image.getexif()
                    row["camera"] = str(exif.get(272, "")).strip("\x00 ")   # phone model
                    row["time"] = photo_time(exif)
                    image.draft("RGB", (128, 128))   # fast JPEG decode at low size
                    hash_bits = perceptual_hash(ImageOps.exif_transpose(image).convert("RGB"))
                row["md5"] = hashlib.md5(path.read_bytes()).hexdigest()
                row["ok"] = True
            except Exception:
                pass  # unreadable: stays ok=False
            rows.append(row)
            hashes.append(hash_bits)
    return pd.DataFrame(rows), np.array(hashes)


def add_scenes(table):
    """Scene and visit numbers from the EXIF time (per folder and phone)."""
    table["scene"], table["visit"] = "", ""
    timed = table[table["time"].notna()].sort_values(["folder", "camera", "time"])
    for (folder, camera), part in timed.groupby(["folder", "camera"]):
        gaps = part["time"].diff().dt.total_seconds().fillna(np.inf)
        scene_no = part["time"].map(lambda t: int(t.timestamp() // SCENE_BLOCK))
        visit_no = (gaps > VISIT_GAP * 60).cumsum()
        table.loc[part.index, "scene"] = [f"{folder}|{camera}|s{n}" for n in scene_no]
        table.loc[part.index, "visit"] = [f"{folder}|{camera}|v{n}" for n in visit_no]
    no_time = table["time"].isna()
    table.loc[no_time, "scene"] = "notime|" + table.loc[no_time, "path"]   # own scene each
    table.loc[no_time, "visit"] = "notime"
    return table


def find_duplicates(table, hashes):
    """Marks exact and near duplicates; joins them into the same group as their scene."""
    table["decision"], table["reason"] = "keep", ""
    table.loc[~table["ok"], ["decision", "reason"]] = ["remove", "unreadable"]
    day = table["time"].map(lambda t: t.strftime("%Y-%m-%d") if t else "")
    for (folder, date), why in EXCLUDED_SESSIONS.items():
        table.loc[(table["folder"] == folder) & (day == date), ["decision", "reason"]] = ["remove", why]
    groups = Groups(len(table))
    for _, members in table.groupby("scene").groups.items():
        members = list(members)
        for other in members[1:]:
            groups.join(members[0], other)

    conflicts = []
    for _, members in table[table["ok"]].groupby("md5").groups.items():
        members = sorted(members)
        if table.loc[members, "folder"].nunique() > 1:
            conflicts.append(" = ".join(table.loc[members, "path"]))
        for other in members[1:]:
            groups.join(members[0], other)
            if table.at[other, "decision"] == "keep":
                table.loc[other, ["decision", "reason"]] = ["remove", "exact duplicate"]

    ok = table.index[table["ok"]].to_numpy()
    distance = hamming_matrix(hashes[ok])
    folder = table.loc[ok, "folder"].to_numpy()
    same = folder[:, None] == folder[None, :]
    close = np.triu(distance <= NEAR_DUP_BITS, k=1)
    near_same = list(zip(*np.nonzero(close & same)))
    near_cross = list(zip(*np.nonzero(close & ~same)))
    for a, b in near_same:
        b = ok[b]
        if table.at[b, "decision"] == "keep":
            table.loc[b, ["decision", "reason"]] = ["remove", "near duplicate"]
    # (near-duplicates are removed, not joined: joining look-alike photos chains
    #  every fish on the same tray into one group)
    cross_pairs = [(table.at[ok[a], "path"], table.at[ok[b], "path"]) for a, b in near_cross]

    table["group"] = [f"bd{groups.find(i)}" for i in range(len(table))]
    return table, conflicts, cross_pairs


def contact_sheet(table, title, out_path, n=48, seed=42):
    """Saves a grid of up to n thumbnails with their file names."""
    if table.empty:
        return
    sample = table.sample(min(n, len(table)), random_state=seed)
    cols, size = 8, 150
    rows = (len(sample) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * size, rows * (size + 18) + 30), "white")
    draw = ImageDraw.Draw(sheet)
    draw.text((6, 8), f"{title}  ({len(table)} images, showing {len(sample)})", fill="black")
    for k, (_, row) in enumerate(sample.iterrows()):
        x, y = (k % cols) * size, 30 + (k // cols) * (size + 18)
        with Image.open(BD_DIR / row["path"]) as image:
            image.draft("RGB", (size * 2, size * 2))
            thumb = ImageOps.exif_transpose(image).convert("RGB")
            thumb.thumbnail((size - 6, size - 6))
        sheet.paste(thumb, (x + 3, y + 3))
        draw.text((x + 3, y + size), row["file_name"][:22], fill="black")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out_path, quality=85)


def print_report(table, conflicts, cross_pairs):
    """Prints duplicates, label conflicts and clean counts per species."""
    print(f"\nExact duplicates removed: {(table['reason'] == 'exact duplicate').sum()}")
    print(f"Near-duplicates removed (pHash <= {NEAR_DUP_BITS} bits): "
          f"{(table['reason'] == 'near duplicate').sum()}")
    for why, count in table.loc[table["reason"].str.startswith("excluded"), "reason"].value_counts().items():
        print(f"Excluded after looking at them: {count} ({why})")
    print(f"Same file in two folders (label conflict): {len(conflicts)}")
    for line in conflicts[:10]:
        print(f"  - {line}")
    print(f"Near-identical pictures in two folders: {len(cross_pairs)}")
    for a, b in cross_pairs[:10]:
        print(f"  - {a}  ~  {b}")

    print("\nPhones and EXIF time:")
    print(table.groupby("folder").agg(
        phones=("camera", "nunique"), with_time=("time", lambda s: int(s.notna().sum())),
        first=("time", "min"), last=("time", "max")).to_string())

    kept = table[table["decision"] == "keep"]
    summary = pd.DataFrame({
        "raw_images": table.groupby("folder").size(),
        "kept": kept.groupby("folder").size(),
        "groups": kept.groupby("folder")["group"].nunique(),
        "visits": kept[kept["visit"] != "notime"].groupby("folder")["visit"].nunique(),
    }).fillna(0).astype(int)
    summary["kept_per_group"] = (summary["kept"] / summary["groups"].clip(lower=1)).round(1)
    print("\nPer folder (groups = scenes + duplicates joined; what the split will use):")
    print(summary.to_string())
    return summary


def main():
    folders = find_species_folders()
    missing = [name for name in list(BD_FOLDERS) + LOOKALIKES if name not in folders]
    if missing:
        print(f"Missing folders in {BD_DIR}: {missing}")
        print("Unzip 'Original Image.zip' there first (see data/DATASETS.md).")
        return
    if MANIFEST_CSV.exists() and not ask_yes_no(f"Replace {MANIFEST_CSV.name}?"):
        print("Kept the existing manifest. Nothing changed.")
        return

    print("Reading images (this takes a while):")
    table, hashes = read_images(folders)
    table = add_scenes(table)
    table, conflicts, cross_pairs = find_duplicates(table, hashes)
    print_report(table, conflicts, cross_pairs)

    table.drop(columns=["md5"]).to_csv(MANIFEST_CSV, index=False)
    print(f"\nSaved {MANIFEST_CSV}")
    for name, part in table[table["decision"] == "keep"].groupby("folder"):
        contact_sheet(part, f"{name} -> {BD_FOLDERS.get(name, 'look-alike')}",
                      SHEETS_DIR / f"{name}.jpg")
    print(f"Saved contact sheets to {SHEETS_DIR}")


main()
