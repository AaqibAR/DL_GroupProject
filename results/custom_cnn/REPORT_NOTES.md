# Custom CNN — results and notes for the report

The source for every number is `metrics.json` and the CSVs in this folder, produced by
`notebooks/custom_cnn_naweedh.ipynb`. The run was on 2026-09-28, on an Apple M1 (MPS) with PyTorch 2.14.0, seed 42.
It uses the **shared 85/15 split** from `gtsrb_data.npz`, as-is: train 33,327, val 5,882, test 12,630.

## 1. Headline results (official GTSRB test set, 12,630 images)

| Metric | Test | Validation |
|---|---|---|
| **Accuracy** (primary) | **99.49%** | 100.00% |
| Macro precision | 99.22% | 100.00% |
| Macro recall | 99.35% | 100.00% |
| Macro F1 | 99.27% | 100.00% |
| Misclassified | 64 / 12,630 | 0 / 5,882 |

Supplementary (test): weighted F1 about 99.5%. One-vs-rest macro ROC-AUC rounds to 1.0000; it is saturated and says
little at this accuracy, which is why the group template leaves it out.

## 2. Training behaviour

- Early stopping ended training at epoch 29 of 30. The best (lowest val loss) epoch was **24**, with val loss 0.00006
  and val accuracy 100%.
- Training took **53.0 min** on the M1 GPU, about 109 s per epoch (batch 32). The laptop was also in normal use,
  so treat the timing as approximate.
- Training accuracy stays *below* validation accuracy all the way through (99.3% vs 100% at epoch 24). This is not
  underfitting: training metrics are measured on augmented images with dropout active, while validation images
  are clean and dropout is off. There is **no sign of overfitting**, because val loss keeps falling or stays flat.
  See `learning_curves.png`.
- After about epoch 5 the validation set is essentially solved (≥ 99.9%). From then on, early stopping and
  checkpoint choice depend on tiny val-loss changes (around 1e-3 to 1e-4), so they carry little information.

## 3. Critical-analysis points

**a) Validation overstates generalisation (100% val vs 99.49% test).**
The GTSRB training set is made of *tracks*: about 30 consecutive frames of the same physical sign. A random
stratified split puts near-identical frames of one sign in both train and val, so validation measures
near-duplicate recognition. The official test set contains different physical signs, so it is the honest
estimate. All four models share this split, so the comparison stays fair, but the report should quote
**test** numbers and name this as a limitation. A track-aware split would fix it.

**b) Augmentation clearly helps, but only the test set shows it** (`ablation_augmentation.csv`)

| Run | Val acc | Test acc | Test macro F1 | Test errors | Best / last epoch |
|---|---|---|---|---|---|
| With augmentation | 100.00% | **99.49%** | **99.27%** | 64 | 24 / 29 |
| Without augmentation | 99.86% | 98.69% | 98.32% | 166 | 9 / 14 |

Augmentation cuts test errors by about 61% (166 → 64), yet validation barely moves (+0.14 pt) because of the
near-duplicate effect in (a). Without augmentation the model fits its training data faster (99.6% train acc by
epoch 9), stops early, and generalises worse to unseen signs. With augmentation, training keeps finding useful
signal for much longer (best epoch 24).

**c) Errors come from image quality and look-alike signs, and they cluster on single physical signs.**
The Spearman correlation between a class's training count and its test F1 is **+0.11**, which is weak. Rare classes
are not the weak ones, so class weighting was not needed. The top confusions (`top_confusions.csv`):

| True → predicted | Count | Why (from `misclassified_examples.png`) |
|---|---|---|
| Right-of-way at next intersection → Beware of ice/snow | 12 | triangular warning signs with small central pictograms; most errors are frames of **one** sign in front of a building |
| Priority road → No vehicles | 8 | **overexposed**: the yellow centre is washed out to white, so the sign looks like a plain white shape |
| Double curve → Wild animals crossing | 5 | motion blur smears the pictogram |
| General caution / Road work → Wild animals crossing | 3 + 2 | triangular warning signs, dark or low-contrast images |
| End of speed limit 80 → Turn right ahead / other "end of" signs | 3 + 2 + 2 + 2 | very dark images; the diagonal stripe is the only visible feature |
| Speed limit 60 → 80 | 2 | digit confusion |

Weakest classes by F1: Beware of ice/snow (0.945), End of speed limit 80 (0.969), Double curve (0.971),
Wild animals crossing (0.980), No vehicles (0.981). The GTSRB test set is also made of tracks, so when the model
fails on one hard physical sign it tends to fail on several of its frames. The 64 errors are therefore far
fewer *independent* failures than the count suggests.

**d) Confidence.** Mean softmax confidence is 0.998 on correct predictions and 0.788 on wrong ones. However,
**28 of the 64 errors had confidence > 0.9**, many of them the overexposed Priority-road frames at confidence 1.00.
A confidence threshold alone would not catch these. Such confident mistakes are a safety concern for
driver-assistance use, and they motivate exposure-robust augmentation or preprocessing such as histogram
equalisation (`confidence_histogram.png`).

**e) Robustness to the split.** An earlier run with an 80/20 split gave 99.43% test accuracy (vs 99.49% here),
so the result does not depend much on the exact train/val split. The specific misclassified signs did change
between runs, which shows that the individual confusions are partly run-dependent.

## 4. Efficiency and complexity

| Measure | Value |
|---|---|
| Parameters | 2,396,171 (all trainable, trained from scratch) |
| Share in the 8192→256 dense layer | 87.5% (conv layers are only ~0.29M) |
| Model size (weights file) | 9.16 MB |
| Compute | 0.157 GMACs per 64×64 image |
| Inference (M1 GPU) | 1.39 ms/image at batch 1; 0.55 ms/image at batch 256 |
| Training time | 53.0 min (29 epochs) |

Almost all the parameters are in the fully connected head. Replacing Flatten with global average pooling would
cut the model to about 0.3M parameters, which is a possible improvement to mention. Timings come from the M1; if
teammates train on a Colab T4, compare training *time* with care, or re-time everything on one device.

## 5. Notes for the group

1. **Split.** This run uses the shared `.npz` 85/15 split, the same as the other models. The Finalized
   Experiment Configuration still says 80/20 and should be updated to 85/15 so the report matches.
2. **Framework.** PyTorch. "Sparse categorical cross-entropy + softmax output" is implemented as
   `nn.CrossEntropyLoss` on logits with integer labels, with softmax applied at prediction time. This is
   mathematically the same as the Keras set-up.
3. **"Mild brightness"** was interpreted as a brightness factor between 0.8 and 1.2.
4. Input is the shared [0, 1] scaling with no extra normalisation.

## Files in this folder

`metrics.json`, `history_main.csv`, `history_no_aug.csv`, `per_class_metrics.csv`, `top_confusions.csv`,
`confusion_matrix.csv/.png`, `learning_curves.png`, `per_class_f1.png`, `misclassified_examples.png`,
`confidence_histogram.png`, `f1_vs_class_frequency.png`, `augmentation_examples.png`,
`ablation_augmentation.csv`, `ablation_augmentation_curves.png`, `layer_parameters.csv`,
`split_class_counts.csv`. The config is in `configs/custom_cnn.json`. Checkpoints are in `checkpoints/` and are gitignored.
