# Finalized Experiment Configuration

SE4050 Deep Learning group project, GTSRB traffic sign classification (PyTorch).
All four models must follow this configuration so the comparison is fair.
Any change to this file must be agreed by the whole group and committed with a clear message.

| Parameter           | Fixed Value                                                                          |
| ------------------- | ------------------------------------------------------------------------------------ |
| Dataset             | GTSRB (Kaggle, CC0)                                                                  |
| Classes             | 43                                                                                   |
| Input               | RGB                                                                                  |
| Image size          | **64 x 64 x 3**                                                                      |
| Training/validation | **85% / 15%** stratified split of the official training set (seed 42)                |
| Test                | Official GTSRB test set (Test.csv), unseen until final evaluation                    |
| Data source         | Shared file `data/processed/gtsrb_data.npz` produced by `src/data_split.py`. Nobody re-splits or regenerates their own copy |
| Batch size          | 32                                                                                   |
| Maximum epochs      | 30 per phase                                                                         |
| Early stopping      | Patience = 5 (monitor: validation loss), applied within each phase                   |
| Checkpoint          | Save the lowest validation-loss model (retained across both phases)                  |
| Optimizer           | Adam                                                                                 |
| Loss                | Cross-entropy on integer labels (`nn.CrossEntropyLoss`)                              |
| Class imbalance     | No class weights, no oversampling (unweighted loss). Imbalance is analysed in the report via macro metrics and per-class results |
| Random seed         | 42 (`random`, `numpy`, `torch`, `torch.cuda`)                                        |
| Output layer        | 43 neurons, raw logits. No softmax layer in the model, because `CrossEntropyLoss` applies it internally. Softmax is applied only at evaluation time (probabilities, ROC-AUC) |
| Normalization       | Pixels scaled to [0, 1] in the `.npz`, then ImageNet mean/std applied to all models and to train, val and test identically. Mean = [0.485, 0.456, 0.406], Std = [0.229, 0.224, 0.225] |
| Transfer learning   | ImageNet pretrained weights: ResNet50, MobileNetV3Large, DenseNet121                 |
| Custom CNN          | Trained from scratch                                                                 |

## Model assignments

| Member  | Model            | Training mode                    |
| ------- | ---------------- | -------------------------------- |
| Naweedh | Custom CNN       | From scratch, constant LR 0.001  |
| Sajalee | ResNet50         | Transfer learning, two phases    |
| Aaqib   | MobileNetV3Large | Transfer learning, two phases    |
| Rahman  | DenseNet121      | Transfer learning, two phases    |

## Learning rate schedule (transfer-learning models only)

```
PHASE 1 - Freeze pretrained backbone
  Optimizer: Adam, LR = 0.001
  Train only the new classification head
  Keep frozen BatchNorm layers in eval mode

PHASE 2 - Unfreeze selected upper layers
  Optimizer: Adam, LR = 0.00001 (1e-5)
  Fine-tune carefully, protect pretrained features

Retain the checkpoint with the lowest validation loss across both phases
```

Custom CNN trains at a constant LR = 0.001 for the full run (no freeze/unfreeze phases,
since there are no pretrained weights to protect).

### Layers unfrozen in Phase 2 (each owner records theirs before training)

| Model            | Unfrozen in Phase 2                         |
| ---------------- | ------------------------------------------- |
| ResNet50         | TO BE FILLED BY SAJALEE                     |
| MobileNetV3Large | TO BE FILLED BY AAQIB                       |
| DenseNet121      | TO BE FILLED BY RAHMAN                      |

## Augmentation (training set only, applied on the fly)

| Transform         | Setting                             |
| ----------------- | ----------------------------------- |
| Rotation          | +/-10 degrees                       |
| Width shift       | +/-10%                              |
| Height shift      | +/-10%                              |
| Zoom              | +/-10%                              |
| Brightness        | Mild variation                      |
| Horizontal flip   | **No** (changes sign meaning)       |
| Vertical flip     | **No** (changes sign meaning)       |

Validation and test data are never augmented.

## Evaluation (every model saves the same outputs)

| Category    | Metrics                                                                       |
| ----------- | ----------------------------------------------------------------------------- |
| Primary     | Accuracy                                                                      |
| Other       | Macro Precision, Macro Recall, Macro F1, macro one-vs-rest ROC-AUC, Confusion Matrix, per-class precision/recall/F1 |
| Efficiency  | Total and trainable parameter count, training time per epoch and in total, inference time per image, model size on disk (MB) |
| Curves      | Training vs validation loss and accuracy per epoch                            |

Test data is used once, at the very end, with the saved best checkpoint.
Hyperparameter and design decisions use only the training and validation sets.

## Required output files (in `results/`)

```
<model>_test_metrics.json       all metrics above, including efficiency numbers and config used
<model>_confusion_matrix.png
<model>_learning_curves.png
```

Models: `custom_cnn`, `resnet50`, `mobilenetv3`, `densenet121`.
