import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import os
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import numpy as np

from models.mvff_model import MVFFModel

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print("Using device:", device)


class MVFFDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.samples = []
        self.transform = transform

        self.classes = sorted(os.listdir(root_dir))

        for label, cls in enumerate(self.classes):
            class_path = os.path.join(root_dir, cls)

            files = os.listdir(class_path)
            base_images = [f for f in files if "_c" not in f]

            for img in base_images:
                pair_name = img.replace(".jpg", "_c.jpg")

                img1_path = os.path.join(class_path, img)

                if pair_name in files:
                    img2_path = os.path.join(class_path, pair_name)
                else:
                    img2_path = img1_path

                self.samples.append((img1_path, img2_path, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img1_path, img2_path, label = self.samples[idx]

        img1 = Image.open(img1_path).convert("RGB")
        img2 = Image.open(img2_path).convert("RGB")

        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)

        return img1, img2, label


transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


train_dataset = MVFFDataset("dataset/train", transform=transform)
val_dataset = MVFFDataset("dataset/val", transform=transform)
test_dataset = MVFFDataset("dataset/test", transform=transform)

print("Train samples:", len(train_dataset))
print("Val samples:", len(val_dataset))


train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)


model = MVFFModel(num_classes=5, fusion="avg")

agpn_weights = torch.load("agpn_model.pth", map_location=device)
model.backbone.load_state_dict(agpn_weights, strict=False)

for param in model.backbone.parameters():
    param.requires_grad = False


model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0002, weight_decay=1e-4)

num_epochs = 20
best_val_acc = 0

train_losses = []
val_losses = []
val_accuracies = []


for epoch in range(num_epochs):

    model.train()
    running_loss = 0.0

    for img1, img2, labels in train_loader:

        img1 = img1.to(device)
        img2 = img2.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(img1, img2)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    train_loss = running_loss / len(train_loader)
    train_losses.append(train_loss)

    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for img1, img2, labels in val_loader:

            img1 = img1.to(device)
            img2 = img2.to(device)
            labels = labels.to(device)

            outputs = model(img1, img2)
            loss = criterion(outputs, labels)

            val_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    val_loss /= len(val_loader)
    val_acc = 100 * correct / total

    val_losses.append(val_loss)
    val_accuracies.append(val_acc)

    print(f"Epoch [{epoch+1}/{num_epochs}] "
          f"Train Loss: {train_loss:.4f} "
          f"Val Loss: {val_loss:.4f} "
          f"Val Acc: {val_acc:.2f}%")

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), "mvff_model.pth")


print("Best Validation Accuracy:", best_val_acc)


# ------------------------
# 📊 LOSS CURVE
# ------------------------
plt.plot(train_losses, label="Train Loss")
plt.plot(val_losses, label="Validation Loss")
plt.legend()
plt.title("MVFF Loss Curve")
plt.savefig("mvff_loss_curve.png")
plt.close()


# ------------------------
# 📊 ACCURACY CURVE
# ------------------------
plt.plot(val_accuracies, label="Validation Accuracy")
plt.legend()
plt.title("MVFF Accuracy Curve")
plt.savefig("mvff_accuracy_curve.png")
plt.close()


# ------------------------
# 📊 CONFUSION MATRIX
# ------------------------
all_preds = []
all_labels = []

model.eval()

with torch.no_grad():
    for img1, img2, labels in test_loader:

        img1 = img1.to(device)
        img2 = img2.to(device)

        outputs = model(img1, img2)
        _, predicted = torch.max(outputs, 1)

        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.numpy())


cm = confusion_matrix(all_labels, all_preds)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=train_dataset.classes
)

fig, ax = plt.subplots(figsize=(8, 6))

disp.plot(cmap=plt.cm.Blues, ax=ax, colorbar=True)

# 🔥 FIX TITLE SPACING
plt.title("MVFF Confusion Matrix", fontsize=16, pad=20)

# Rotate labels nicely
plt.xticks(rotation=45, ha='right')

# 🔥 IMPORTANT: prevent clipping
plt.tight_layout()

# Extra top margin (THIS is the real fix)
plt.subplots_adjust(top=0.88)

plt.savefig("mvff_confusion_matrix.png", bbox_inches="tight")
plt.close()


print("All graphs saved successfully 🚀")