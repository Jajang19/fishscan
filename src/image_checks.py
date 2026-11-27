"""
image_checks.py
Small tools for checking an image dataset BEFORE training:
  - get_base_name     strips Roboflow / burst / copy endings from a file name
  - perceptual_hash   64-bit pHash, similar pictures get similar hashes
  - hamming_matrix    how many bits differ between every pair of hashes
  - Groups            union-find: puts images that belong together in one group

This file only DEFINES things. Nothing runs when it is imported.
"""

import re
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.fft import dctn


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


def perceptual_hash(image):
    """64-bit pHash of a PIL image, as 64 True/False values.

    The picture is shrunk to 32x32 grey pixels, then a DCT keeps only the
    coarse patterns (top-left 8x8). Each bit says "above or below the median".
    Two photos that look almost the same differ in only a few bits.
    """
    small = np.asarray(image.convert("L").resize((32, 32), Image.LANCZOS), dtype=np.float32)
    low = dctn(small, norm="ortho")[:8, :8].flatten()[1:]   # 63 low frequencies, no DC
    return np.append(low > np.median(low), False)            # 64 booleans


def hamming_matrix(hashes):
    """(n, 64) booleans -> (n, n) number of different bits (uint8).

    Works in blocks of rows so thousands of images still fit in memory.
    """
    packed = np.packbits(np.asarray(hashes, dtype=bool), axis=1)   # (n, 8) bytes
    bit_counts = np.unpackbits(np.arange(256, dtype=np.uint8)[:, None], axis=1).sum(axis=1)
    result = np.zeros((len(packed), len(packed)), dtype=np.uint8)
    for start in range(0, len(packed), 500):
        xor = packed[start:start + 500, None, :] ^ packed[None, :, :]   # (block, n, 8)
        result[start:start + 500] = bit_counts[xor].sum(axis=2)
    return result


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
