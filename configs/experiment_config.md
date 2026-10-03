| Parameter           | Fixed Value                                                              |
| ------------------- | ------------------------------------------------------------------------ |
| Dataset             | GTSRB                                                                    |
| Classes             | 43                                                                       |
| Input               | RGB                                                                      |
| Image size          | **64 × 64 × 3**                                                          |
| Training/validation | 85% / 15% stratified split of official training set                      |
| Test                | Official GTSRB test set (unseen until final evaluation)                  |
| Batch size          | 32                                                                       |
| Maximum epochs      | 30                                                                       |
| Early stopping      | Patience = 5 (monitor: validation loss)                                  |
| Checkpoint          | Save lowest validation-loss model                                        |
| Optimizer           | Adam                                                                     |
| Loss                | Sparse Categorical Cross-Entropy                                         |
| Random seed         | 42                                                                       |
| Output layer        | 43 neurons + Softmax                                                     |
| Transfer learning   | ImageNet pretrained weights — ResNet50, MobileNetV3Large, DenseNet121    |
| Custom CNN          | Trained from scratch                                                     |
| Primary metric      | Accuracy                                                                 |
| Other metrics       | Macro Precision, Macro Recall, Macro F1, Confusion Matrix                |

### Learning rate schedule (transfer-learning models only)

```
PHASE 1 — Freeze pretrained backbone
  Optimizer: Adam, LR = 0.001
  Train only the new classification head

PHASE 2 — Unfreeze selected upper layers
  Optimizer: Adam, LR = 0.00001 (1e-5)
  Fine-tune carefully, protect pretrained features

Retain checkpoint with lowest validation loss across both phases
```

Custom CNN trains at a constant LR = 0.001 for the full run (no freeze/unfreeze phases
since there are no pretrained weights to protect).

### Augmentation (training set only)

| Transform         | Setting            |
|--------------------|---------------------|
| Rotation           | ±10°               |
| Width shift        | ±10%               |
| Height shift       | ±10%               |
| Zoom               | ±10%               |
| Brightness         | Mild variation      |
| Horizontal flip    | **No** — changes sign meaning |
| Vertical flip      | **No** — changes sign meaning |
