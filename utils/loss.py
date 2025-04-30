import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
from scipy.stats import ortho_group
import math

class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=0, size_average=True, ignore_index=255):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ignore_index = ignore_index
        self.size_average = size_average

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(
            inputs, targets, reduction='none', ignore_index=self.ignore_index)
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * ce_loss
        if self.size_average:
            return focal_loss.mean()
        else:
            return focal_loss.sum()
        

class CeCoLoss(nn.Module):
    def __init__(self, num_classes: int, img_w_list, etf_prototypes: torch.Tensor, 
                 dropout_ratio: float = 0.1, 
                 img_cls_weight: float = 0.4, smooth: float = 1.0, 
                 ignore_index: int = 255,
                 scale: float = 5.0):
        super(CeCoLoss, self).__init__()

        self.num_classes = num_classes
        self.img_cls_weight = img_cls_weight
        self.smooth = smooth
        self.ignore_index = ignore_index
        self.scale = scale

        weight = torch.tensor(img_w_list, dtype=torch.float32)
        weight = weight / weight.sum()
        self.register_buffer('cls_freq', weight.view(1, num_classes))  # Automatically moved with model.to()

        # Frozen projection blocks
        self.reduce = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(128)
        )
        self.gain = nn.Sequential(
            nn.Linear(128, 512),
            nn.ReLU(inplace=True)
        )

        for p in (*self.reduce.parameters(), *self.gain.parameters()):
            p.requires_grad = False
        
        # ETF classifier
        self.img_cls = nn.Linear(512, num_classes, bias=True)
        with torch.no_grad():
            w_etf = F.normalize(etf_prototypes, p=2, dim=1)
            self.img_cls.weight.copy_(w_etf)       # (C,512)
            self.img_cls.bias.zero_()
        self.img_cls.requires_grad_(False)

        self.dropout = nn.Dropout2d(dropout_ratio) if dropout_ratio > 0 else nn.Identity()
    
    def forward(self, features: torch.Tensor, seg_label: torch.Tensor) -> torch.Tensor:

        if seg_label.ndim == 4:          # (N,1,H,W) → (N,H,W)
            seg_label = seg_label.squeeze(1)

        N, _, Hf, Wf = features.shape
        device       = features.device
        seg_label    = seg_label.to(device)

        # 1. pixel features → 128‑D, aligned to GT resolution
        feat = self.dropout(features)
        feat = self.reduce(feat)                                   # (N,128,H',W')
        feat = F.interpolate(feat, size=seg_label.shape[-2:], mode='bilinear',
                             align_corners=True)                   # (N,128,H,W)
        feat = feat.permute(0,2,3,1).contiguous()                  # (N,H,W,128)

        # 2. keep valid pixels
        mask       = seg_label != self.ignore_index
        y_valid    = seg_label[mask]                               # (Nv,)
        feat_valid = feat[mask]                                    # (Nv,128)
        if y_valid.numel() == 0:                                   # all ignore?
            return torch.zeros((), device=device, dtype=features.dtype)

        # 3. mean feature per class present in the batch
        y_onehot   = F.one_hot(y_valid, num_classes=self.num_classes).float()  # (Nv,C)
        class_sum  = y_onehot.T @ feat_valid                               # (C,128)
        class_cnt  = y_onehot.sum(dim=0)                                   # (C,)
        present    = class_cnt > 0
        class_feat = class_sum[present] / class_cnt[present].unsqueeze(1)  # (M,128)
        scene_lbls = present.nonzero(as_tuple=False).squeeze(1)            # (M,)

        # 4. 512‑D projection (+ optional ETF normalisation)
        img_feat = self.gain(class_feat)                        # (M,512)
        img_feat = F.normalize(img_feat, p=2, dim=1)
        logits   = self.img_cls(img_feat) * self.scale                       # (M,C)

        # 5. smoothed targets
        tgt = torch.zeros_like(logits)
        tgt[torch.arange(scene_lbls.size(0)), scene_lbls] = 1.0
        tgt = self.smooth * tgt + (1.0 - self.smooth)/(self.num_classes - 1) * (1.0 - tgt)

        # 6. re‑balanced image loss
        logprob = F.log_softmax(logits + torch.log(self.cls_freq + 1e-12), dim=1)
        img_loss = -(tgt * logprob).sum() / (tgt.sum() + 1e-12)
        img_loss = img_loss * self.img_cls_weight               # scalar

        return img_loss
    

#### REGION-BASED CECO/ETF LOSS ####


# class CeCoLoss(nn.Module):
#     def __init__(self, num_classes: int, img_w_list, etf_prototypes: torch.Tensor, 
#                  dropout_ratio: float = 0.1, 
#                  img_cls_weight: float = 0.4, smooth: float = 1.0, 
#                  ignore_index: int = 255,
#                  scale: float = 5.0):
#         super(CeCoLoss, self).__init__()

#         self.num_classes = num_classes
#         self.img_cls_weight = img_cls_weight
#         self.smooth = smooth
#         self.ignore_index = ignore_index
#         self.scale = scale

#         weight = torch.tensor(img_w_list, dtype=torch.float32)
#         weight = weight / weight.sum()
#         self.register_buffer('cls_freq', weight.view(1, num_classes))  # Automatically moved with model.to()

#         # Frozen projection blocks
#         self.reduce = nn.Sequential(
#             nn.Conv2d(256, 128, kernel_size=1, padding=0, bias=False),
#             nn.BatchNorm2d(128)
#         )
#         self.gain = nn.Sequential(
#             nn.Linear(128, 512),
#             nn.ReLU(inplace=True)
#         )

#         for p in (*self.reduce.parameters(), *self.gain.parameters()):
#             p.requires_grad = False
        
#         # ETF classifier
#         self.img_cls = nn.Linear(512, num_classes, bias=True)
#         with torch.no_grad():
#             w_etf = F.normalize(etf_prototypes, p=2, dim=1)
#             self.img_cls.weight.copy_(w_etf)       # (C,512)
#             self.img_cls.bias.zero_()
#         self.img_cls.requires_grad_(False)

#         self.dropout = nn.Dropout2d(dropout_ratio) if dropout_ratio > 0 else nn.Identity()
    
#     def forward(self, features: torch.Tensor, seg_label: torch.Tensor) -> torch.Tensor:

#         if seg_label.ndim == 4:          # (N,1,H,W) → (N,H,W)
#             seg_label = seg_label.squeeze(1)

#         N, _, Hf, Wf = features.shape
#         device       = features.device
#         seg_label    = seg_label.to(device)

#         # 1. pixel features → 128‑D, aligned to GT resolution
#         feat = self.dropout(features)
#         feat = self.reduce(feat)                                   # (N,128,H',W')
#         feat = F.interpolate(feat, size=seg_label.shape[-2:], mode='bilinear',
#                              align_corners=True)                   # (N,128,H,W)
#         feat = feat.permute(0,2,3,1).contiguous()                  # (N,H,W,128)

#         region_feats  = []
#         region_labels = []

#         for n in range(N):                                           # loop over images
#             y_img   = seg_label[n]                                   # (H,W)
#             mask    = y_img != self.ignore_index
#             if not mask.any():                                       # skip fully ignored image
#                 continue

#             feat_img = feat[n][mask]                                 # (P,128)
#             y_flat   = y_img[mask]                                   # (P,)

#             # pixel -> class pooling for this image
#             y_onehot   = F.one_hot(y_flat, num_classes=self.num_classes).float()  # (P,C)
#             class_sum  = y_onehot.T @ feat_img                       # (C,128)
#             class_cnt  = y_onehot.sum(0)                             # (C,)

#             present    = class_cnt > 0
#             if present.any():
#                 cls_feat  = class_sum[present] / class_cnt[present].unsqueeze(1)  # (M_i,128)
#                 cls_label = present.nonzero(as_tuple=False).squeeze(1)            # (M_i,)
#                 region_feats.append(cls_feat)
#                 region_labels.append(cls_label)

#         # If no valid regions in the whole batch
#             if len(region_feats) == 0:
#                 return torch.zeros((), device=device, dtype=features.dtype)

#             region_feats  = torch.cat(region_feats,  dim=0)              # (R,128)
#             region_labels = torch.cat(region_labels, dim=0)              # (R,)

#             # ----------------------------------------------------------
#             # Classify each region feature
#             # ----------------------------------------------------------
#             img_feat = self.gain(region_feats)                           # (R,512)
#             img_feat = F.normalize(img_feat, p=2, dim=1)

#             logits = self.img_cls(img_feat) * self.scale                 # (R,C)

#             # ----------------------------------------------------------
#             # Label‑smoothing + class‑frequency re‑balancing
#             # ----------------------------------------------------------
#             tgt = F.one_hot(region_labels, num_classes=self.num_classes).float()
#             tgt = self.smooth * tgt + \
#                 (1.0 - self.smooth) / (self.num_classes - 1) * (1.0 - tgt)

#             log_prob = F.log_softmax(logits + torch.log(self.cls_freq + 1e-12), dim=1)
#             loss = -(tgt * log_prob).sum() / (tgt.sum() + 1e-12)
#             loss = loss * self.img_cls_weight

#             return loss