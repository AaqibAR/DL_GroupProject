"""
src/eda.py

Standalone EDA script for the SE4050 group report's "Dataset Description and
Exploratory Data Analysis" section. Covers:
    1. Class balance — how many images per class, imbalance ratio
    2. Image size distribution — width/height spread across the dataset
       (GTSRB images are NOT all the same size — this matters for justifying
       your resize choice in preprocessing)

Does NOT touch data/processed/ or the shared split — this is pure EDA on the
raw dataset, independent of src/data_split.py, so anyone can run it without
needing to regenerate the whole preprocessed .npz file.

Usage:
    python src/eda.py

Outputs (all saved under results/):
    - eda_class_balance.png          bar chart of images per class
    - eda_image_size_distribution.png  histograms of image width & height
    - eda_summary.csv                per-class image counts + summary stats printed to console
"""

import os
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt

RAW_DATA_DIR = os.path.join("data", "raw")
TRAIN_DIR = os.path.join(RAW_DATA_DIR, "Train")
RESULTS_DIR = "results"

# If you want faster results while iterating, sample a subset per class
# instead of reading all ~50K images. Set to None to use every image.
SAMPLE_PER_CLASS = None   # e.g. set to 200 for a quick draft run


def collect_stats(train_dir, sample_per_class=None):
    class_counts = {}
    widths, heights = [], []

    class_ids = sorted(os.listdir(train_dir), key=lambda x: int(x))

    for class_id in class_ids:
        class_path = os.path.join(train_dir, class_id)
        if not os.path.isdir(class_path):
            continue

        files = [f for f in os.listdir(class_path)
                 if f.lower().endswith((".png", ".jpg", ".ppm"))]
        class_counts[int(class_id)] = len(files)

        if sample_per_class is not None:
            files = files[:sample_per_class]

        for fname in files:
            img_path = os.path.join(class_path, fname)
            img = cv2.imread(img_path)
            if img is None:
                continue
            h, w = img.shape[:2]
            widths.append(w)
            heights.append(h)

    counts_df = pd.DataFrame(
        list(class_counts.items()), columns=["ClassId", "ImageCount"]
    ).sort_values("ClassId")

    sizes_df = pd.DataFrame({"width": widths, "height": heights})

    return counts_df, sizes_df


def plot_class_balance(counts_df, out_path):
    plt.figure(figsize=(14, 5))
    plt.bar(counts_df["ClassId"], counts_df["ImageCount"], color="#4C72B0")
    plt.xlabel("Class ID")
    plt.ylabel("Number of images")
    plt.title("GTSRB — Class Balance (Train set)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_image_size_distribution(sizes_df, out_path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    axes[0].hist(sizes_df["width"], bins=40, color="#55A868")
    axes[0].set_title("Image Width Distribution")
    axes[0].set_xlabel("Width (px)")
    axes[0].set_ylabel("Frequency")

    axes[1].hist(sizes_df["height"], bins=40, color="#C44E52")
    axes[1].set_title("Image Height Distribution")
    axes[1].set_xlabel("Height (px)")
    axes[1].set_ylabel("Frequency")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def main():
    if not os.path.isdir(TRAIN_DIR):
        raise FileNotFoundError(
            f"Couldn't find {TRAIN_DIR}. Make sure the GTSRB Train/ folder "
            f"is placed at data/raw/Train/ before running this script."
        )

    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("Scanning dataset (this can take a minute or two)...")
    counts_df, sizes_df = collect_stats(TRAIN_DIR, sample_per_class=SAMPLE_PER_CLASS)

    # ---- Class balance ----
    total_images = counts_df["ImageCount"].sum()
    min_class = counts_df.loc[counts_df["ImageCount"].idxmin()]
    max_class = counts_df.loc[counts_df["ImageCount"].idxmax()]
    imbalance_ratio = max_class["ImageCount"] / min_class["ImageCount"]

    print("\n--- Class Balance ---")
    print(f"Total classes: {len(counts_df)}")
    print(f"Total training images: {total_images}")
    print(f"Smallest class: ClassId {int(min_class['ClassId'])} "
          f"with {int(min_class['ImageCount'])} images")
    print(f"Largest class:  ClassId {int(max_class['ClassId'])} "
          f"with {int(max_class['ImageCount'])} images")
    print(f"Imbalance ratio (max/min): {imbalance_ratio:.1f}x")

    class_balance_path = os.path.join(RESULTS_DIR, "eda_class_balance.png")
    plot_class_balance(counts_df, class_balance_path)
    print(f"Saved: {class_balance_path}")

    # ---- Image size distribution ----
    print("\n--- Image Size Distribution ---")
    print(f"Width  — min: {sizes_df['width'].min()}, "
          f"max: {sizes_df['width'].max()}, "
          f"mean: {sizes_df['width'].mean():.1f}, "
          f"median: {sizes_df['width'].median():.1f}")
    print(f"Height — min: {sizes_df['height'].min()}, "
          f"max: {sizes_df['height'].max()}, "
          f"mean: {sizes_df['height'].mean():.1f}, "
          f"median: {sizes_df['height'].median():.1f}")

    size_dist_path = os.path.join(RESULTS_DIR, "eda_image_size_distribution.png")
    plot_image_size_distribution(sizes_df, size_dist_path)
    print(f"Saved: {size_dist_path}")

    # ---- Save summary CSV (per-class counts, for a report table) ----
    summary_path = os.path.join(RESULTS_DIR, "eda_summary.csv")
    counts_df.to_csv(summary_path, index=False)
    print(f"Saved: {summary_path}")

    print("\nDone. All EDA outputs are in the results/ folder — "
          "ready to drop into the report's EDA section.")


if __name__ == "__main__":
    main()
