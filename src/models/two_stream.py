from typing import Literal

import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights

from .attention import CBAM, LightSE


class ResNet50CBAMBackbone(nn.Module):
    def __init__(self, pretrained: bool = True):
        super().__init__()
        weights = ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        model = resnet50(weights=weights)
        self.stem = nn.Sequential(model.conv1, model.bn1, model.relu, model.maxpool)
        self.layer1 = model.layer1
        self.layer2 = model.layer2
        self.layer3 = model.layer3
        self.layer4 = model.layer4
        self.cbam = CBAM(channels=2048)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return self.cbam(x)


class TwoStreamCBAMResNet50SE(nn.Module):
    """
    输入:
      top_img  : [B, 3, H, W]
      side_img : [B, 3, H, W]
    输出:
      bw_pred  : [B]
    """

    def __init__(
        self,
        pretrained_backbone: bool = True,
        fusion_mode: Literal["concat"] = "concat",
        dropout: float = 0.2,
    ):
        super().__init__()
        if fusion_mode != "concat":
            raise ValueError("Only 'concat' fusion is currently supported.")

        self.top_backbone = ResNet50CBAMBackbone(pretrained=pretrained_backbone)
        self.side_backbone = ResNet50CBAMBackbone(pretrained=pretrained_backbone)

        self.fusion_se = LightSE(channels=4096, reduction=8)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(4096, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(1024, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, 1),
        )

    def forward(self, top_img: torch.Tensor, side_img: torch.Tensor) -> torch.Tensor:
        feat_top = self.top_backbone(top_img)
        feat_side = self.side_backbone(side_img)
        feat = torch.cat([feat_top, feat_side], dim=1)
        feat = self.fusion_se(feat)
        feat = self.pool(feat)
        pred = self.head(feat).squeeze(1)
        return pred
