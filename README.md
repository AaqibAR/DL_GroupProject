# SE4050 – Deep Learning Group Assignment (2026)

**Category:** Supervised Deep Learning — Image Classification
**Group Leader:** Aaqib Ahamed Rasmy (IT23228276)

## Team & Model Assignments

| Member   | Model         | Notebook |
|----------|---------------|----------|
| Naweedh  | Custom CNN    | `notebooks/custom_cnn_naweedh.ipynb` |
| Sajalee  | ResNet50      | `notebooks/resnet50_sajalee.ipynb` |
| Aaqib    | MobileNetV3   | `notebooks/mobilenetv3_aaqib.ipynb` |
| Rahman   | EfficientNetb0 | `notebooks/efficientnetb0_rahman.ipynb`|

## Dataset

_TBD — will be filled in once the group finalizes the dataset. Must include: source, license, size, classes, and a link/citation._

## Project Structure

```
├── README.md
├── requirements.txt
├── .gitignore
├── data/               # dataset access instructions (raw data not committed — see data/README.md)
├── notebooks/          # one notebook per member/model
├── src/                # shared code: preprocessing, dataset loading, evaluation metrics
├── configs/            # saved hyperparameters + random seeds per model, for reproducibility
└── results/            # metrics, confusion matrices, learning curve plots per model
```

## Setup Instructions

1. Clone the repo:
   ```
   git clone <repo-url>
   cd se4050-repo
   ```
2. Create a virtual environment and install dependencies:
   ```
   python -m venv venv
   source venv/bin/activate     # Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```
3. Follow `data/README.md` to download/access the dataset.
4. Open the relevant notebook under `notebooks/` for your assigned model.

## Reproducibility

- All models use the same train/validation/test split (see `src/data_split.py` once added).
- Random seeds are fixed and logged in `configs/`.
- Preprocessing and hyperparameter tuning use only the training and validation sets — the test set is untouched until final evaluation.

## Evaluation Metrics

Accuracy, Precision, Recall, F1-score, ROC-AUC, and Confusion Matrix — reported per model in `results/`.

## Acknowledgements

Dataset, pretrained weights, and any AI-assisted content are credited in the final report (`Report.pdf`) and in `data/README.md`.
