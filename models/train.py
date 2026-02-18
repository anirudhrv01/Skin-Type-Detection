import torch
import torch.nn as nn
import torch.optim as optim
from resnet_model import SkinTypeResNet
from utils.data_loader import get_data_loaders

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_loader, val_loader, _ = get_data_loaders(
    "data/train",
    "data/val",
    "data/test"
)

model = SkinTypeResNet(num_classes=3).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.model.fc.parameters(), lr=0.001)

epochs = 10

for epoch in range(epochs):
    model.train()
    total_loss = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch [{epoch+1}/{epochs}] Loss: {total_loss/len(train_loader)}")

torch.save(model.state_dict(), "models/best_model.pth")
