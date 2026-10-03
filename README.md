# SE4050 – Deep Learning Group Assignment (2026)

**Category:** Supervised Deep Learning — Image Classification
**Group Leader:** Aaqib Ahamed Rasmy (IT23228276)

## Team & Model Assignments

| Member   | Model          | Notebook |
|----------|----------------|----------|
| Naweedh  | Custom CNN     | `notebooks/custom_cnn_naweedh.ipynb` |
| Sajalee  | ResNet50       | `notebooks/resnet50_sajalee.ipynb` |
| Aaqib    | MobileNetV3    | `notebooks/mobilenetv3_aaqib.ipynb` |
| Rahman   | DenseNet121 | `notebooks/densenet121_rahman.ipynb` |

## Dataset

**GTSRB — German Traffic Sign Recognition Benchmark**
- **Source:** [Kaggle — GTSRB German Traffic Sign](https://www.kaggle.com/datasets/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign)
- **Original creator:** INI Benchmark Website, presented at IJCNN 2011
- **License:** CC0: Public Domain
- **Size:** 50,000+ images across 43 classes
- **Task:** Multi-class, single-image classification of German traffic signs
- **Why this dataset:** genuinely complex real-world images (varying lighting, motion blur, partial occlusion, multiple image sizes) with a well-known class imbalance (some classes have 10x+ more samples than others), which we address explicitly in preprocessing rather than ignoring — see `src/data_split.py` and `results/class_distribution.png`.

Full access instructions and citation details are in `data/README.md`.

## Project Structure
├── README.md
├── requirements.txt
├── .gitignore
├── data/ # dataset access instructions (raw data not committed — see data/README.md)
├── notebooks/ # one notebook per member/model
├── src/ # shared code: preprocessing, dataset loading, evaluation metrics
├── configs/ # saved hyperparameters + random seeds per model, for reproducibility
└── results/ # metrics, confusion matrices, learning curve plots per model

## Setup Instructions

1. Clone the repo: git clone https://github.com/AaqibAR/DL_GroupProject.git
cd DL_groupproj

2. Create a virtual environment and install dependencies: python -m venv venv 

venv\Scripts\activate # Windows
source venv/bin/activate # macOS/Linux

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt

   > Note: the `--index-url` above installs the CUDA 12.8 build for NVIDIA GPUs. If your machine has no NVIDIA GPU, just run `pip install torch torchvision torchaudio` instead (CPU-only).
3. Download the GTSRB dataset from Kaggle (see `data/README.md`) and place it under `data/raw/` as `Train/`, `Test/`, `Meta/`, `Test.csv`.
4. Run the shared preprocessing pipeline once: python src/data_split.py

   This generates `data/processed/gtsrb_data.npz` — the exact train/val/test split every model trains on. (Ask the group leader for the pre-generated `.npz` file if you'd rather skip regenerating it.)
5. Open the relevant notebook under `notebooks/` for your assigned model and load the split:
```python
   import numpy as np
   data = np.load("data/processed/gtsrb_data.npz")
   X_train, y_train = data["X_train"], data["y_train"]
   X_val, y_val     = data["X_val"], data["y_val"]
   X_test, y_test   = data["X_test"], data["y_test"]
```

## Reproducibility

- All models use the identical train/validation/test split, generated once by `src/data_split.py` and shared across the team as `data/processed/gtsrb_data.npz` — nobody regenerates their own split.
- Random seed fixed at `42` (set in `src/data_split.py`), logged in `configs/`.
- Stratified splitting preserves each class's proportion across train/val, given GTSRB's known class imbalance.
- Preprocessing and hyperparameter tuning use only the training and validation sets — the official test set (`Test.csv`) is untouched until final evaluation.

## Evaluation Metrics

Accuracy, Precision, Recall, F1-score, ROC-AUC, and Confusion Matrix — reported per model in `results/`.

## Acknowledgements

Dataset, pretrained weights, and any AI-assisted content are credited in the final report (`Report.pdf`) and in `data/README.md`.