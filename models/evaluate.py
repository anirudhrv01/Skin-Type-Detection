import torch
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

from .resnet_model import SkinTypeResNet
from utils.data_loader import get_data_loaders

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

_, _, test_loader = get_data_loaders(
    "data/processed/train",
    "data/processed/val",
    "data/processed/test"
)

model = SkinTypeResNet(num_classes=3)
model.load_state_dict(torch.load("models/best_model.pth"))
model.to(device)
model.eval()

all_preds = []
all_labels = []

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        outputs = model(images)
        _, preds = torch.max(outputs, 1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.numpy())

print(classification_report(all_labels, all_preds))
print(confusion_matrix(all_labels, all_preds))
