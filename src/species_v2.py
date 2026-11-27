"""
species_v2.py
Settings for the v2 species model (branch v2-species):
  - 6 classes: our own bangus photos + 5 species from the Bangladesh dataset
  - which Bangladesh folder becomes which class
  - how our own photos must be named, so every photo knows which FISH it shows

Freshness stays on FFE (v1); this is only for the species model.

This file only DEFINES things. Nothing runs when it is imported.
"""

import re
from pathlib import Path

# Class names shown in the app (Filipino market names)
SPECIES_V2 = ["Bangus", "Dalag", "Hito", "Pangasius", "Puyo", "Tilapia"]

# Bangladesh dataset folder -> our class (DOI 10.17632/8r24222wcc.3)
BD_FOLDERS = {
    "Telapiya": "Tilapia",    # Oreochromis niloticus
    "Magur": "Hito",          # Clarias batrachus (walking catfish)
    "Shol": "Dalag",          # Channa striata (snakehead murrel)
    "Koi": "Puyo",            # Anabas testudineus (climbing perch)
    "Pangas": "Pangasius",    # Pangasius pangasius
}

# Our own photo folders: data/raw/own_photos/<folder>/<folder>_fish<NN>_<MM>.jpg
OWN_FOLDERS = {
    "bangus": "Bangus",
    "tilapia": "Tilapia",
    "hito": "Hito",
    "dalag": "Dalag",
    "puyo": "Puyo",
    "pangasius": "Pangasius",
}

# e.g. bangus_fish01_01.jpg  ->  species "bangus", fish "01", photo "01"
OWN_NAME = re.compile(r"^(?P<species>[a-z]+)_fish(?P<fish>\d{2,3})_(?P<photo>\d{2,3})\.(jpe?g|png)$",
                      re.IGNORECASE)


def parse_own_name(path):
    """Returns (folder species, fish id) for one of our photos, or None if the name is wrong.

    The fish id ("own:bangus:fish01") is the GROUP: all photos of one fish
    must stay in the same split, otherwise the test set has fish the model
    has already seen.
    """
    path = Path(path)
    match = OWN_NAME.match(path.name)
    if not match or match["species"].lower() != path.parent.name.lower():
        return None
    species = match["species"].lower()
    return species, f"own:{species}:fish{int(match['fish']):02d}"
