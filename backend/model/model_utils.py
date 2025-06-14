# backend/model/models.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class FeatureExtractor(nn.Module):
    def __init__(self, embed_dim=256):
        super().__init__()
        base = models.resnet18(weights=None)
        self.cnn = nn.Sequential(*list(base.children())[:-3])
        self.conv_to_tokens = nn.Conv2d(128, embed_dim, kernel_size=1)

        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=8, dim_feedforward=512)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x):
        x = self.cnn(x)
        x = self.conv_to_tokens(x)
        B, C, H, W = x.shape
        x = x.view(B, C, -1).permute(2, 0, 1)
        x = self.transformer(x)
        x = self.pool(x.permute(1, 2, 0)).squeeze(-1)
        return x


class SiameseNetwork(nn.Module):
    def __init__(self, embed_dim=256):
        super().__init__()
        self.feature_extractor = FeatureExtractor(embed_dim=embed_dim)
        self.backbone = models.resnet18(pretrained=True)
        self.backbone.fc = nn.Identity()

        self.projector = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128)
        )

    def forward_once(self, x):
        x = self.backbone(x)
        x = self.projector(x)
        return x

    def forward(self, x1, x2):
        out1 = self.forward_once(x1)
        out2 = self.forward_once(x2)
        return F.cosine_similarity(out1, out2)
