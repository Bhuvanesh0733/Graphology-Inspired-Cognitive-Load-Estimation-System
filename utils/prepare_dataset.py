"""
Dataset Preparation — IAM Word Dataset
=======================================
Exact IAM folder structure (as shown in your screenshots):

  iam_words/
    words/
      a01/              ← writer/form group  (a01, a02, ... b01, b02 ...)
        a01-000u/       ← form subfolder
          a01-000u-00-00.png   ← word image
          a01-000u-00-01.png
          a01-000u-01-00.png
          ...

Filename pattern:  {form}-{line:02d}-{word:02d}.png
  e.g. a01-000u-07-03.png  → form=a01-000u, line=07, word=03

COGNITIVE LOAD LABEL STRATEGY
──────────────────────────────
Since IAM is a handwriting recognition dataset (not a cognitive load
dataset), we assign labels using a line-position proxy that is
consistent with motor-fatigue / cognitive-load literature:

  Lines 00-02  →  LOW     (writer is fresh, copying first lines)
  Lines 03-05  →  MEDIUM  (sustained effort, attention maintained)
  Lines 06+    →  HIGH    (fatigue accumulates, working memory taxed)

This is the same approach used in Rosenblum et al. (2010) where
later lines in a dictation task showed measurable handwriting changes.

You can OVERRIDE any label by adding rows to:
  data/labels/manual_labels.csv  (format: filename,label)
"""

import os
import re
import sys
import csv
import pandas as pd
import numpy as np
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.feature_extractor import extract_all_features, FEATURE_NAMES


# ── CONFIG ────────────────────────────────────────────────────────────────────
IAM_ROOT          = "iam_words"          # must contain a words/ subfolder
OUTPUT_CSV        = "data/dataset.csv"
MANUAL_LABELS_CSV = "data/labels/manual_labels.csv"

# Label thresholds (line index within a form)
LOW_MAX    = 2   # lines 0-2   → LOW
MEDIUM_MAX = 5   # lines 3-5   → MEDIUM
                 # lines 6+    → HIGH
# ─────────────────────────────────────────────────────────────────────────────


def parse_line_number(filename):
    """
    Extract the line index from an IAM word filename.
    Pattern: <form>-<LINE>-<word>.png
    e.g. a01-000u-07-03.png → 7
    """
    name = os.path.splitext(os.path.basename(filename))[0]
    parts = name.split('-')
    # IAM word filenames have 4 dash-separated parts:
    # a01 - 000u - 07 - 03
    if len(parts) >= 4:
        try:
            return int(parts[-2])
        except ValueError:
            pass
    # fallback: find any two-digit number near the end
    matches = re.findall(r'-(\d{2})-\d{2}$', name)
    if matches:
        return int(matches[0])
    return 0


def assign_label(filename):
    """Assign LOW / MEDIUM / HIGH based on line position."""
    line = parse_line_number(filename)
    if line <= LOW_MAX:
        return "LOW"
    elif line <= MEDIUM_MAX:
        return "MEDIUM"
    else:
        return "HIGH"


def load_manual_labels(path):
    if not os.path.exists(path):
        return {}
    manual = {}
    with open(path, newline='') as f:
        for row in csv.DictReader(f):
            manual[row['filename'].strip()] = row['label'].strip().upper()
    print(f"  Loaded {len(manual)} manual label overrides from {path}")
    return manual


def collect_images(root):
    """
    Walk iam_words/words/<group>/<form>/*.png and collect all images.
    Returns list of absolute paths.
    """
    words_dir = os.path.join(root, "words")
    if not os.path.isdir(words_dir):
        # Maybe user put the folder directly as iam_words/a01/...
        words_dir = root
        print(f"  Note: 'words' subfolder not found — scanning {root} directly.")

    paths = []
    for dirpath, _, files in os.walk(words_dir):
        for fname in sorted(files):
            if fname.lower().endswith('.png'):
                paths.append(os.path.join(dirpath, fname))
    return paths


def build_dataset():
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    os.makedirs(os.path.dirname(MANUAL_LABELS_CSV), exist_ok=True)

    # Create manual labels template if absent
    if not os.path.exists(MANUAL_LABELS_CSV):
        with open(MANUAL_LABELS_CSV, 'w', newline='') as f:
            csv.DictWriter(f, fieldnames=['filename','label']).writeheader()
        print(f"  Created empty manual labels file: {MANUAL_LABELS_CSV}")

    print(f"\n[1/4] Scanning: {IAM_ROOT}")
    paths = collect_images(IAM_ROOT)
    print(f"      Found {len(paths):,} PNG images")

    if not paths:
        print("\nERROR: No images found!")
        print(f"  Expected structure: {IAM_ROOT}/words/a01/a01-000u/a01-000u-00-00.png")
        sys.exit(1)

    print("[2/4] Loading manual labels...")
    manual = load_manual_labels(MANUAL_LABELS_CSV)

    print("[3/4] Extracting graphological features...")
    rows = []
    errors = []

    for img_path in tqdm(paths, desc="Extracting", unit="img"):
        fname = os.path.basename(img_path)
        try:
            feat_dict, feat_vec = extract_all_features(img_path)
            label = manual.get(fname, assign_label(fname))
            row = {
                "filename":  fname,
                "path":      img_path,
                "label":     label,
                "line_num":  parse_line_number(fname),
            }
            for name, val in zip(FEATURE_NAMES, feat_vec):
                row[name] = round(float(val), 6)
            rows.append(row)
        except Exception as e:
            errors.append((img_path, str(e)))

    print(f"\n[4/4] Saving → {OUTPUT_CSV}")
    df = pd.DataFrame(rows)

    # Make sure we have all three classes; warn if any is missing
    present = set(df['label'].unique())
    for lbl in ['LOW','MEDIUM','HIGH']:
        if lbl not in present:
            print(f"  WARNING: label '{lbl}' has 0 samples! "
                  f"Consider adjusting LOW_MAX / MEDIUM_MAX thresholds "
                  f"or adding manual labels.")

    df.to_csv(OUTPUT_CSV, index=False)

    print("\n✅ Dataset complete!")
    print(f"   Samples  : {len(df):,}")
    print(f"   Errors   : {len(errors)}")
    print(f"   Labels   :\n{df['label'].value_counts().to_string()}")
    print(f"\n   Line distribution (first 10 line numbers found):")
    print(f"   {sorted(df['line_num'].unique())[:10]}")

    if errors:
        print(f"\n   First 3 errors:")
        for p, e in errors[:3]:
            print(f"   {os.path.basename(p)}: {e}")

    return df


if __name__ == "__main__":
    build_dataset()
