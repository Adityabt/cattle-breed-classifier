import os
import shutil
import random

random.seed(42)

source_dir = "dataset_raw"
target_dir = "dataset"

train_ratio = 0.7
val_ratio = 0.15
test_ratio = 0.15

os.makedirs(target_dir, exist_ok=True)

for split in ["train", "val", "test"]:
    os.makedirs(os.path.join(target_dir, split), exist_ok=True)

for breed in os.listdir(source_dir):
    breed_path = os.path.join(source_dir, breed)
    if not os.path.isdir(breed_path):
        continue

    images = os.listdir(breed_path)
    random.shuffle(images)

    total = len(images)
    train_end = int(total * train_ratio)
    val_end = train_end + int(total * val_ratio)

    splits = {
        "train": images[:train_end],
        "val": images[train_end:val_end],
        "test": images[val_end:]
    }

    for split, split_images in splits.items():
        split_breed_dir = os.path.join(target_dir, split, breed)
        os.makedirs(split_breed_dir, exist_ok=True)

        for img in split_images:
            src = os.path.join(breed_path, img)
            dst = os.path.join(split_breed_dir, img)
            shutil.copy(src, dst)

print("Dataset split completed successfully.")
