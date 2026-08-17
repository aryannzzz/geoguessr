"""Backbone + dual-head architecture.

Backbone: DINOv2-small (timm `vit_small_patch14_dinov2.lvd142m`) -- a
general-purpose self-supervised ViT. Its pretraining objective is
self-distillation on generic image crops, NOT geolocation, so it's an
allowed starting point under the competition's foundation-model rule.
Partial fine-tune: only the last N transformer blocks + final norm are
unfrozen, everything before that stays frozen to fit the compute budget.

Head A: linear geocell classifier (softmax over `num_cells`).
Head B: linear tangent-plane offset regressor (dx_km, dy_km / offset_scale),
         offset from the predicted-or-true cell's centroid.
"""
import timm
import torch
import torch.nn as nn


class GeoModel(nn.Module):
    def __init__(self, num_cells, backbone_name="vit_small_patch14_dinov2.lvd142m",
                 img_size=224, n_unfrozen_blocks=2, pretrained=True):
        super().__init__()
        self.backbone = timm.create_model(
            backbone_name, pretrained=pretrained, img_size=img_size, num_classes=0)
        feat_dim = self.backbone.num_features

        self._freeze_backbone(n_unfrozen_blocks)

        self.cell_head = nn.Linear(feat_dim, num_cells)
        self.offset_head = nn.Sequential(
            nn.Linear(feat_dim, 128),
            nn.GELU(),
            nn.Linear(128, 2),
        )

    def _freeze_backbone(self, n_unfrozen_blocks):
        for p in self.backbone.parameters():
            p.requires_grad = False
        blocks = self.backbone.blocks
        for blk in blocks[-n_unfrozen_blocks:]:
            for p in blk.parameters():
                p.requires_grad = True
        for p in self.backbone.norm.parameters():
            p.requires_grad = True

    def trainable_parameters(self):
        return [p for p in self.parameters() if p.requires_grad]

    def forward(self, x):
        feat = self.backbone(x)
        cell_logits = self.cell_head(feat)
        offset = self.offset_head(feat)
        return cell_logits, offset


if __name__ == "__main__":
    m = GeoModel(num_cells=147)
    n_total = sum(p.numel() for p in m.parameters())
    n_train = sum(p.numel() for p in m.parameters() if p.requires_grad)
    print(f"total params: {n_total/1e6:.1f}M, trainable: {n_train/1e6:.1f}M "
          f"({100*n_train/n_total:.1f}%)")
    x = torch.randn(4, 3, 224, 224)
    logits, offset = m(x)
    print("cell_logits:", logits.shape, "offset:", offset.shape)
