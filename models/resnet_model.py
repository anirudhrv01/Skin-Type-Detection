from distro import name
import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import ResNet50_Weights

class SkinTypeResNet(nn.Module):
    def __init__(self, num_classes=3):
        super(SkinTypeResNet, self).__init__()

        self.model = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)

        # Freeze backbone
        for name, param in self.model.named_parameters():
            if "layer4" not in name:
                param.requires_grad = False

        # Replace classifier
        in_features = self.model.fc.in_features
        self.model.fc = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        return self.model(x)