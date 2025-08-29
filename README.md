# CeCo (CVPR 2023)

Unofficial Implementation of Understanding Imbalanced Semantic Segmentation Through Neural Collapse (CVPR 2023).

## Experiment Results

| Model | Dataset | Training Config | mIoU | Overall Acc | Mean Acc | FreqW Acc |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Deeplabv3+MobileNet | PascalVOC+aug | CE loss, lr 0.01, bs 8 | 0.704069 | 0.920524 | 0.834106 | 0.861695 |
| Deeplabv3+MobileNet | PascalVOC+aug | CE loss, lr 0.01, bs 16 | 0.710832 | 0.922556 | 0.827111 | 0.863603 |
| Deeplabv3+MobileNet | PascalVOC+aug | CE + 0.4 (Batch-level ETF loss), lr 0.01, bs 8, Logits scaling 5.0 | 0.712657 | 0.922651 | 0.841459 | 0.864801 |
| Deeplabv3+MobileNet | PascalVOC+aug | CE + 0.4 (Batch-level ETF loss), lr 0.01, bs 16, Logits scaling 5.0 | **0.720822** | **0.926161** | 0.835748 | **0.869288** |
| Deeplabv3+MobileNet | PascalVOC+aug | CE + 0.4 (Region-level ETF loss), lr 0.01, bs 16 | 0.718548 | 0.925502 | **0.841944** | 0.868942 |
| Deeplabv3+MobileNet | PascalVOC+aug | CE + 0.4 (Region-level ETF loss), lr 0.01, bs 8, Logits scaling 5.0 | 0.705387 | 0.920016 | 0.839736 | 0.86102 |
| Deeplabv3+MobileNet | Cityscapes | CE Loss, lr 0.1, bs 16 | 0.720683 | 0.952485 | 0.800548 | 0.912763 |
| Deeplabv3+MobileNet | Cityscapes | CE + 0.4 (Batch-level ETF loss), lr 0.1, bs 16, Logits scaling 5.0 | **0.729483** | **0.952627** | **0.811717**| **0.913013** |****

**Credits**

DeepLab Code from https://github.com/VainF/DeepLabV3Plus-Pytorch


