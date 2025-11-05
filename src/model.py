import torch
import torch.nn as nn
import torchvision.models as models

class DualHeadClassifier(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()
        backbone = models.resnet18(pretrained=True)
        self.feature_extractor = nn.Sequential(*list(backbone.children())[:-1])  # (B,512,1,1)
        in_features = backbone.fc.in_features

        self.head_a = nn.Linear(in_features, num_classes)
        self.head_b = nn.Linear(in_features, num_classes)

        self.dropout = nn.Dropout(p=0.2)

        nn.init.xavier_uniform_(self.head_a.weight); nn.init.zeros_(self.head_a.bias)
        nn.init.xavier_uniform_(self.head_b.weight); nn.init.zeros_(self.head_b.bias)

    def forward(self, x):
        x = self.feature_extractor(x)
        if x.ndim == 4:
            x = torch.flatten(x, 1)
        x = self.dropout(x)
        out_a = self.head_a(x)
        out_b_logits = self.head_b(x)
        return out_a, out_b_logits
