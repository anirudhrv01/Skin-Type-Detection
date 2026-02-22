import torch
from torchvision import transforms
from PIL import Image
from .resnet_model import SkinTypeResNet
from torchvision import datasets

dataset = datasets.ImageFolder("data/processed/train")

import collections
print(collections.Counter(dataset.targets))

device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

model=SkinTypeResNet(num_classes=4)
model.load_state_dict(torch.load("models/best_model.pth"))
model.to(device)
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

image_path = "data/single/OilySkin.jpg"
image = Image.open(image_path).convert("RGB")
image = transform(image).unsqueeze(0).to(device)

with torch.no_grad():
    outputs = model(image)
    print("Raw outputs:", outputs)
    probabilities = torch.softmax(outputs, dim=1)
    print("Probabilities:", probabilities)

    _, predicted = torch.max(outputs, 1)
classes = ['combination', 'dry', 'normal', 'oily']

print("Predicted Skin Type:", classes[predicted.item()])