"""
inspect_dataset.py
Looks inside a dataset folder BEFORE we train anything and prints:
  1. the folder structure (so we can see how the dataset is organized)
  2. how many images are in each folder (class counts)
  3. non-image files (for example CSV/Excel score sheets) and corrupted images
  4. image sizes (resolution)
  5. exact duplicate images (same file content, found with an MD5 hash)
  6. "base name" groups (augmented copies of the same original photo)
  7. a few sample file names per folder (to spot fish IDs or day numbers)

It also saves one row per file to reports/inspection_<name>.csv.

Run from the project folder:  python scripts/inspect_dataset.py
Only needs Pillow (plus the Python standard library).
"""

import csv
import hashlib
import re
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image

PROJECT_DIR = Path(__file__).resolve().parent.parent
REPORTS_DIR = PROJECT_DIR / "reports"

# File types we treat as images
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# Datasets we know about (name -> folder)
KNOWN_DATASETS = {
    1: ("ffe", PROJECT_DIR / "data" / "raw" / "ffe"),
}


def get_base_name(file_name):
    """Strips common 'copy' endings so augmented copies share one base name.

    Examples:
      fish01_jpg.rf.3a9f...c1.jpg  -> fish01   (Roboflow export)
      fish01_aug3.jpg               -> fish01
      fish01 (2).jpg                -> fish01
      IMG_20191110_055936_3.jpg     -> img_20191110_055936  (crop 3 of one photo, FFE)
    """
    stem = Path(file_name).stem.lower()
    stem = re.sub(r"^(img_\d{8}_\d{6})_\d+$", r"\1", stem)       # FFE crops
    stem = re.sub(r"_(jpg|jpeg|png)\.rf\.[0-9a-f]+$", "", stem)  # Roboflow
    stem = re.sub(r"[_-]aug(mented)?[_-]?\d*$", "", stem)        # _aug3
    stem = re.sub(r"\s*\(\d+\)$", "", stem)                       # (2)
    return stem


def file_md5(path):
    """Returns the MD5 hash of a file (same hash = exact same bytes)."""
    md5 = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            md5.update(chunk)
    return md5.hexdigest()


def choose_dataset():
    """Asks which dataset folder to inspect and returns (name, folder)."""
    print("Which dataset do you want to inspect?")
    for number, (name, folder) in KNOWN_DATASETS.items():
        print(f"  {number}. {name}  ({folder})")
    print("  2. Other folder (type a path)")

    while True:
        try:
            choice = int(input("Enter 1 or 2: "))
            if choice not in (1, 2):
                raise ValueError
            break
        except ValueError:
            print("Please type 1 or 2.")
            continue

    if choice == 2:
        while True:
            try:
                folder = Path(input("Folder path: ").strip().strip('"'))
                if not folder.is_dir():
                    raise ValueError
                break
            except ValueError:
                print("That folder does not exist. Try again.")
                continue
        return folder.name, folder

    name, folder = KNOWN_DATASETS[choice]
    return name, folder


def print_tree(folder, max_depth=3):
    """Prints the folder structure (folders only) with a file count each."""
    print("\n=== 1. Folder structure ===")
    for sub in sorted([folder] + [p for p in folder.rglob("*") if p.is_dir()]):
        depth = len(sub.relative_to(folder).parts)
        if depth > max_depth:
            continue
        n_files = sum(1 for p in sub.iterdir() if p.is_file())
        indent = "  " * depth
        label = sub.name if depth else str(folder)
        print(f"{indent}{label}/  ({n_files} files)")


def scan_files(folder):
    """Checks every file and returns one dictionary (row) per file."""
    rows = []
    all_files = sorted(p for p in folder.rglob("*") if p.is_file())
    print(f"\nScanning {len(all_files)} files, please wait...")

    for path in all_files:
        row = {
            "path": str(path.relative_to(folder)),
            "folder": str(path.parent.relative_to(folder)),
            "file_name": path.name,
            "extension": path.suffix.lower(),
            "is_image": path.suffix.lower() in IMAGE_EXTENSIONS,
            "ok": False,
            "width": "",
            "height": "",
            "md5": "",
            "base_name": get_base_name(path.name),
        }
        if row["is_image"]:
            try:
                with Image.open(path) as img:
                    img.verify()  # checks the file is a real, readable image
                with Image.open(path) as img:
                    row["width"], row["height"] = img.size
                row["ok"] = True
                row["md5"] = file_md5(path)
            except Exception:
                row["ok"] = False  # corrupted or not really an image
        rows.append(row)
    return rows


def print_class_counts(rows):
    """Prints how many good images each folder (class) has."""
    print("\n=== 2. Images per folder ===")
    counts = Counter(r["folder"] for r in rows if r["ok"])
    for folder_name, count in sorted(counts.items()):
        warning = "   <-- fewer than 100 images" if count < 100 else ""
        print(f"  {folder_name:<50} {count:>6}{warning}")
    print(f"  {'TOTAL':<50} {sum(counts.values()):>6}")


def print_problem_files(rows):
    """Prints non-image files and images that could not be opened."""
    print("\n=== 3. Non-image files and corrupted images ===")
    non_images = [r for r in rows if not r["is_image"]]
    broken = [r for r in rows if r["is_image"] and not r["ok"]]

    print(f"  Non-image files: {len(non_images)}")
    for ext, n in Counter(r["extension"] or "(none)" for r in non_images).items():
        print(f"    {ext}: {n}")
    for r in non_images[:15]:
        print(f"    - {r['path']}")

    print(f"  Corrupted / unreadable images: {len(broken)}")
    for r in broken[:15]:
        print(f"    - {r['path']}")


def print_image_sizes(rows):
    """Prints the most common image resolutions."""
    print("\n=== 4. Image sizes (width x height) ===")
    sizes = Counter((r["width"], r["height"]) for r in rows if r["ok"])
    for (w, h), n in sizes.most_common(8):
        print(f"  {w} x {h}: {n} images")
    if len(sizes) > 8:
        print(f"  ... and {len(sizes) - 8} other sizes")


def print_duplicates(rows):
    """Prints exact duplicates (same MD5) and how many cross folders."""
    print("\n=== 5. Exact duplicate images (same file content) ===")
    by_hash = defaultdict(list)
    for r in rows:
        if r["ok"]:
            by_hash[r["md5"]].append(r)
    groups = [g for g in by_hash.values() if len(g) > 1]
    cross = [g for g in groups if len({r["folder"] for r in g}) > 1]

    print(f"  Duplicate groups: {len(groups)} "
          f"({sum(len(g) - 1 for g in groups)} extra copies)")
    print(f"  Groups that appear in DIFFERENT folders (label conflict!): {len(cross)}")
    for g in cross[:5]:
        print("    - " + " | ".join(r["path"] for r in g))


def print_base_name_groups(rows):
    """Prints how many files share a base name (likely augmented copies)."""
    print("\n=== 6. Base-name groups (possible augmented copies) ===")
    groups = Counter((r["folder"], r["base_name"]) for r in rows if r["ok"])
    multi = {k: v for k, v in groups.items() if v > 1}
    print(f"  Unique base names: {len(groups)}")
    print(f"  Base names with more than one file: {len(multi)}")
    for (folder_name, base), n in list(multi.items())[:5]:
        print(f"    - {folder_name}/{base}: {n} files")


def print_sample_names(rows, per_folder=4):
    """Prints a few file names per folder so we can spot fish IDs or days."""
    print("\n=== 7. Sample file names per folder ===")
    by_folder = defaultdict(list)
    for r in rows:
        if r["ok"]:
            by_folder[r["folder"]].append(r["file_name"])
    for folder_name, names in sorted(by_folder.items())[:30]:
        print(f"  {folder_name}: {', '.join(names[:per_folder])}")


def save_report(rows, dataset_name):
    """Saves one row per file to reports/inspection_<name>.csv."""
    REPORTS_DIR.mkdir(exist_ok=True)
    out_path = REPORTS_DIR / f"inspection_{dataset_name}.csv"
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nSaved the full file list to {out_path}")


def main():
    dataset_name, folder = choose_dataset()
    if not folder.is_dir():
        print(f"\nFolder not found: {folder}")
        print("Unzip the dataset there first (see data/DATASETS.md).")
        return

    print_tree(folder)
    rows = scan_files(folder)
    if not rows:
        print("The folder is empty.")
        return

    print_class_counts(rows)
    print_problem_files(rows)
    print_image_sizes(rows)
    print_duplicates(rows)
    print_base_name_groups(rows)
    print_sample_names(rows)
    save_report(rows, dataset_name)


main()
