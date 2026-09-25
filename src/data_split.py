"""
src/data_split.py

Shared GTSRB preprocessing + train/val/test split for the SE4050 group project.
Every member's notebook should import from this file (or load the .npz files
it produces) so all 4 models are trained/evaluated on IDENTICAL data — this is
a rubric requirement ("fair and comparable experimental conditions").

Expected folder layout (standard Kaggle GTSRB download):
    data/raw/
        Train/
            0/   *.png   (class 0 images)
            1/   *.png
            ...
            42/  *.png
        Test/
            *.png
        Test.csv        (official test labels: Path, ClassId, ...)
        Meta/
            0.png ... 42.png   (one representative sign per class)

Usage:
    python src/data_split.py
This will:
    1. Run EDA (class distribution, sample image checks) and save a plot.
    2. Build a stratified train/val split from the Train/ folder.
    3. Load the official Test/ set separately (this is your untouched holdout).
    4. Resize + normalize all images.
    5. Save everything as compressed .npz arrays under data/processed/
       so every notebook just does: np.load("data/processed/gtsrb_data.npz")
"""

import os
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# ---------------------------------------------------------------------------
# Config — keep these identical across the whole group
# ---------------------------------------------------------------------------
RANDOM_SEED = 42
IMG_SIZE = 64          # 64x64 works well for transfer-learning models (ResNet50 etc.
                        # need >=32x32 minimum; 64 gives more detail without being slow)
VAL_SIZE = 0.15         # fraction of Train/ held out for validation
NUM_CLASSES = 43

RAW_DATA_DIR = os.path.join("data", "raw")
PROCESSED_DIR = os.path.join("data", "processed")

np.random.seed(RANDOM_SEED)


# ---------------------------------------------------------------------------
# Helper — list only the real class folders, ignoring macOS junk like
# .DS_Store and AppleDouble files (._0, ._1, ...) that show up when the
# dataset was copied from/through a Mac.
# ---------------------------------------------------------------------------
def list_class_dirs(train_dir):
    return sorted(
        (
            d for d in os.listdir(train_dir)
            if d.isdigit() and os.path.isdir(os.path.join(train_dir, d))
        ),
        key=lambda x: int(x),
    )


# ---------------------------------------------------------------------------
# 1. EDA — class distribution (GTSRB is known to be heavily imbalanced,
#    some classes have 2000+ images, others under 200 — worth showing this
#    in your report's EDA section, and it justifies any class-weighting /
#    augmentation decisions later)
# ---------------------------------------------------------------------------
def run_eda(train_dir):
    class_counts = {}
    for class_id in list_class_dirs(train_dir):
        class_path = os.path.join(train_dir, class_id)
        class_counts[int(class_id)] = len(
            [f for f in os.listdir(class_path) if not f.startswith(".")]
        )

    counts_df = pd.DataFrame(
        list(class_counts.items()), columns=["ClassId", "ImageCount"]
    ).sort_values("ClassId")

    print("Total training images:", counts_df["ImageCount"].sum())
    print("Min class size:", counts_df["ImageCount"].min(),
          "| Max class size:", counts_df["ImageCount"].max())
    print("Imbalance ratio (max/min):",
          round(counts_df["ImageCount"].max() / counts_df["ImageCount"].min(), 1))

    plt.figure(figsize=(14, 5))
    plt.bar(counts_df["ClassId"], counts_df["ImageCount"])
    plt.xlabel("Class ID")
    plt.ylabel("Number of images")
    plt.title("GTSRB — Class Distribution (Train set)")
    plt.tight_layout()
    os.makedirs("results", exist_ok=True)
    plt.savefig(os.path.join("results", "class_distribution.png"), dpi=150)
    plt.close()
    print("Saved class distribution plot to results/class_distribution.png")

    return counts_df


# ---------------------------------------------------------------------------
# 2. Load + resize + normalize images from the Train/ folder
# ---------------------------------------------------------------------------
def load_train_images(train_dir, img_size=IMG_SIZE):
    images, labels = [], []
    class_ids = list_class_dirs(train_dir)

    for class_id in class_ids:
        class_path = os.path.join(train_dir, class_id)
        for fname in os.listdir(class_path):
            if not fname.lower().endswith((".png", ".jpg", ".ppm")):
                continue
            img_path = os.path.join(class_path, fname)
            img = cv2.imread(img_path)
            if img is None:
                continue
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (img_size, img_size))
            images.append(img)
            labels.append(int(class_id))

    X = np.array(images, dtype=np.float32) / 255.0   # normalize to [0, 1]
    y = np.array(labels, dtype=np.int64)
    return X, y


# ---------------------------------------------------------------------------
# 3. Load the OFFICIAL test set (via Test.csv) — this stays untouched until
#    final evaluation, per the assignment rules.
# ---------------------------------------------------------------------------
def load_official_test(raw_dir, img_size=IMG_SIZE):
    test_csv = pd.read_csv(os.path.join(raw_dir, "Test.csv"))
    images, labels = [], []

    for _, row in test_csv.iterrows():
        img_path = os.path.join(raw_dir, row["Path"])
        img = cv2.imread(img_path)
        if img is None:
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (img_size, img_size))
        images.append(img)
        labels.append(int(row["ClassId"]))

    X_test = np.array(images, dtype=np.float32) / 255.0
    y_test = np.array(labels, dtype=np.int64)
    return X_test, y_test


# ---------------------------------------------------------------------------
# 4. Main — build the split, save everything to disk
# ---------------------------------------------------------------------------
def main():
    train_dir = os.path.join(RAW_DATA_DIR, "Train")

    print("Running EDA...")
    run_eda(train_dir)

    print("\nLoading and preprocessing training images (this can take a few minutes)...")
    X, y = load_train_images(train_dir)
    print(f"Loaded {X.shape[0]} images, shape={X.shape[1:]}")

    print("\nCreating stratified train/val split...")
    X_train, X_val, y_train, y_val = train_test_split(
        X, y,
        test_size=VAL_SIZE,
        random_state=RANDOM_SEED,
        stratify=y,          # stratify = each class keeps its proportion in both splits
    )
    print(f"Train: {X_train.shape[0]} | Val: {X_val.shape[0]}")

    print("\nLoading official test set...")
    X_test, y_test = load_official_test(RAW_DATA_DIR)
    print(f"Test: {X_test.shape[0]}")

    os.makedirs(PROCESSED_DIR, exist_ok=True)
    out_path = os.path.join(PROCESSED_DIR, "gtsrb_data.npz")
    np.savez_compressed(
        out_path,
        X_train=X_train, y_train=y_train,
        X_val=X_val, y_val=y_val,
        X_test=X_test, y_test=y_test,
    )
    print(f"\nSaved processed data to {out_path}")
    print("Every notebook should load this file instead of reprocessing raw images:")
    print("    data = np.load('data/processed/gtsrb_data.npz')")
    print("    X_train, y_train = data['X_train'], data['y_train']")


if __name__ == "__main__":
    main()