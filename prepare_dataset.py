"""
prepare_dataset.py
==================
Prepares the HAM10000 skin dataset for training.

Steps:
  1. Reads HAM10000_metadata.csv
  2. Splits into train/test (80/20, stratified)
  3. Copies images into dataset/train/<class>/ and dataset/test/<class>/
  4. Applies class balancing via oversampling for minority classes
  5. Creates dataset/invalid_for_retinal/ and dataset/invalid_for_skin/ dirs
     (filled later by generate_invalid_dataset.py)

Usage:
  python prepare_dataset.py

Expected inputs:
  HAM10000_metadata.csv       — from Kaggle HAM10000
  HAM10000_images/            — folder of all ISIC_*.jpg images

Outputs:
  dataset/train/<class>/
  dataset/test/<class>/
  dataset/invalid_for_retinal/  (created, filled by generate_invalid_dataset.py)
  dataset/invalid_for_skin/     (created, filled by generate_invalid_dataset.py)
"""

import os
import pandas as pd
import shutil
import random
from sklearn.model_selection import train_test_split
from collections import Counter

random.seed(42)

IMAGE_DIR  = "HAM10000_images"
CSV_PATH   = "HAM10000_metadata.csv"
OUTPUT_DIR = "dataset"

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
# Oversample minority classes to at least this many training samples.
# Set to 0 to disable oversampling.
MIN_TRAIN_SAMPLES_PER_CLASS = 600

# ─────────────────────────────────────────────
# LOAD + SPLIT
# ─────────────────────────────────────────────
print("Loading HAM10000_metadata.csv...")
df = pd.read_csv(CSV_PATH)
df = df[["image_id", "dx"]].copy()

print(f"Total entries: {len(df)}")
print("Class distribution:")
for cls, cnt in sorted(df["dx"].value_counts().items()):
    print(f"  {cls:8s}: {cnt}")

# Only keep rows where the image file actually exists
print("\nChecking image files...")
available = []
missing   = 0
for _, row in df.iterrows():
    found = False
    for ext in [".jpg", ".jpeg", ".png"]:
        if os.path.exists(os.path.join(IMAGE_DIR, row["image_id"] + ext)):
            found = True
            break
    if found:
        available.append(row)
    else:
        missing += 1

if missing:
    print(f"  ⚠️  {missing} images not found in {IMAGE_DIR}/ — skipped")

df = pd.DataFrame(available)
print(f"  {len(df)} images available for training")

# ─────────────────────────────────────────────
# TRAIN / TEST SPLIT (stratified)
# ─────────────────────────────────────────────
train_df, test_df = train_test_split(
    df, test_size=0.2, stratify=df["dx"], random_state=42
)

print(f"\nSplit: {len(train_df)} train / {len(test_df)} test")

# ─────────────────────────────────────────────
# CREATE OUTPUT DIRS
# ─────────────────────────────────────────────
classes = df["dx"].unique()

for split in ["train", "test"]:
    for label in classes:
        os.makedirs(os.path.join(OUTPUT_DIR, split, label), exist_ok=True)

# Create invalid dirs (will be filled by generate_invalid_dataset.py)
os.makedirs(os.path.join(OUTPUT_DIR, "invalid_for_retinal"), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "invalid_for_skin"),    exist_ok=True)

# ─────────────────────────────────────────────
# COPY IMAGES
# ─────────────────────────────────────────────
def get_image_path(image_id):
    for ext in [".jpg", ".jpeg", ".png"]:
        p = os.path.join(IMAGE_DIR, image_id + ext)
        if os.path.exists(p):
            return p
    return None

def copy_split(data, split_name):
    copied = 0
    for _, row in data.iterrows():
        src = get_image_path(row["image_id"])
        if src is None:
            continue
        ext = os.path.splitext(src)[1]
        dst = os.path.join(OUTPUT_DIR, split_name, row["dx"], row["image_id"] + ext)
        if not os.path.exists(dst):
            shutil.copy(src, dst)
        copied += 1
    return copied

print("\nCopying training images...")
n_train = copy_split(train_df, "train")
print(f"  Copied {n_train} training images")

print("Copying test images...")
n_test = copy_split(test_df, "test")
print(f"  Copied {n_test} test images")

# ─────────────────────────────────────────────
# OVERSAMPLING (minority class balancing)
# ─────────────────────────────────────────────
if MIN_TRAIN_SAMPLES_PER_CLASS > 0:
    print(f"\nOversampling minority classes to {MIN_TRAIN_SAMPLES_PER_CLASS} samples...")

    for label in classes:
        class_dir = os.path.join(OUTPUT_DIR, "train", label)
        existing  = [f for f in os.listdir(class_dir) if f.lower().endswith(('.jpg','.jpeg','.png'))]
        n_existing = len(existing)

        if n_existing >= MIN_TRAIN_SAMPLES_PER_CLASS:
            print(f"  {label:8s}: {n_existing} samples — OK (no oversampling needed)")
            continue

        n_needed = MIN_TRAIN_SAMPLES_PER_CLASS - n_existing
        print(f"  {label:8s}: {n_existing} → {MIN_TRAIN_SAMPLES_PER_CLASS} (+{n_needed} copies)")

        for i in range(n_needed):
            src_file = random.choice(existing)
            src_path = os.path.join(class_dir, src_file)
            name, ext = os.path.splitext(src_file)
            dst_path = os.path.join(class_dir, f"{name}_aug{i:04d}{ext}")
            if not os.path.exists(dst_path):
                shutil.copy(src_path, dst_path)

# ─────────────────────────────────────────────
# SUMMARY
# ─────────────────────────────────────────────
print("\n" + "=" * 50)
print("FINAL DATASET SUMMARY")
print("=" * 50)

for split in ["train", "test"]:
    print(f"\n{split.upper()}:")
    total = 0
    for label in sorted(classes):
        d = os.path.join(OUTPUT_DIR, split, label)
        n = len(os.listdir(d)) if os.path.exists(d) else 0
        print(f"  {label:8s}: {n}")
        total += n
    print(f"  {'TOTAL':8s}: {total}")

print("\n✅ Dataset ready!")
print("\nNext steps:")
print("  1. python generate_invalid_dataset.py   — generate rejection training data")
print("  2. python train.py                      — train retinal model")
print("  3. python train_skin_model.py           — train skin model")