import torch
import torch.nn as nn
from models.agpn_resnet import AGPNResNet50

class MVFFModel(nn.Module):
    def __init__(self, num_classes=5, fusion="avg"):
        super().__init__()

        self.backbone = AGPNResNet50(num_classes=num_classes)
        self.fusion = fusion

        if fusion == "concat":
            self.classifier = nn.Sequential(
                nn.Dropout(0.6),
                nn.Linear(2048 * 2, num_classes)
            )
        else:
            self.classifier = nn.Sequential(
                nn.Dropout(0.5),
                nn.Linear(2048, num_classes)
            )

    def forward(self, img1, img2):

        f1 = self.backbone(img1)
        f2 = self.backbone(img2)

        if self.fusion == "avg":
            fused = (f1 + f2) / 2
        else:
            fused = torch.cat([f1, f2], dim=1)

        out = self.classifier(fused)

        return out