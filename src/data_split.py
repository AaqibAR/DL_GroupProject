"""
Shared train/val/test split logic — used by ALL models so the comparison
in the report stays fair (rubric requirement: "evaluated under fair and
comparable experimental conditions").

TODO: implement once the dataset is finalized.
- Fix a random seed (log it in configs/) for reproducibility.
- Split BEFORE any preprocessing/augmentation to avoid data leakage.
- Save the split (e.g. as file lists or indices) so every notebook loads
  the exact same train/val/test partition.
"""

RANDOM_SEED = 42

def load_and_split(data_dir, val_size=0.15, test_size=0.15):
    """
    Returns train, val, test splits.
    TODO: implement.
    """
    raise NotImplementedError
