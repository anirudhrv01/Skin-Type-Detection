import os

train_files = set()
test_files = set()

for root, _, files in os.walk("data/raw/train"):
    for f in files:
        train_files.add(f)

for root, _, files in os.walk("data/raw/test"):
    for f in files:
        test_files.add(f)

overlap = train_files.intersection(test_files)

print("Number of overlapping files:", len(overlap))