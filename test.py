from torchvision import datasets

dataset=datasets.ImageFolder("data/raw/train")
print(dataset.classes)
print(len(dataset))
