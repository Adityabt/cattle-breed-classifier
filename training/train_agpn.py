import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import numpy as np
import os
import json

from models.agpn_resnet import AGPNResNet50

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print("Using device:", device)

data_dir = "dataset"

# ------------------------
# TRANSFORMS
# ------------------------
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

val_test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# ------------------------
# DATASETS
# ------------------------
train_dataset = datasets.ImageFolder(os.path.join(data_dir, "train"), transform=train_transform)
val_dataset = datasets.ImageFolder(os.path.join(data_dir, "val"), transform=val_test_transform)
test_dataset = datasets.ImageFolder(os.path.join(data_dir, "test"), transform=val_test_transform)

# 🔥 DEBUG PRINT (VERY IMPORTANT)
print("Classes:", train_dataset.classes)
print("Number of classes:", len(train_dataset.classes))

num_classes = len(train_dataset.classes)

# ------------------------
# SAVE CLASS NAMES (CRUCIAL FOR APP)
# ------------------------
with open("class_names.json", "w") as f:
    json.dump(train_dataset.classes, f)

# ------------------------
# DATALOADERS
# ------------------------
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

# ------------------------
# MODEL
# ------------------------
model = AGPNResNet50(num_classes=num_classes)
model = model.to(device)

# Freeze layers
for name, param in model.named_parameters():
    if "layer3" in name or "layer4" in name or "cbam" in name or "fc" in name:
        param.requires_grad = True
    else:
        param.requires_grad = False

criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=1e-4,
    weight_decay=1e-4
)

num_epochs = 15

train_losses = []
val_losses = []
val_accuracies = []

best_val_acc = 0.0

# ------------------------
# TRAINING LOOP
# ------------------------
for epoch in range(num_epochs):

    model.train()
    running_loss = 0.0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    epoch_train_loss = running_loss / len(train_loader)
    train_losses.append(epoch_train_loss)

    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    epoch_val_loss = val_loss / len(val_loader)
    val_accuracy = 100 * correct / total

    val_losses.append(epoch_val_loss)
    val_accuracies.append(val_accuracy)

    print(f"Epoch [{epoch+1}/{num_epochs}] "
          f"Train Loss: {epoch_train_loss:.4f} "
          f"Val Loss: {epoch_val_loss:.4f} "
          f"Val Accuracy: {val_accuracy:.2f}%")

    if val_accuracy > best_val_acc:
        best_val_acc = val_accuracy
        torch.save(model.state_dict(), "agpn_model.pth")

print("Best Validation Accuracy:", best_val_acc)

# ------------------------
# LOSS CURVE
# ------------------------
plt.figure()
plt.plot(train_losses, label="Train Loss")
plt.plot(val_losses, label="Validation Loss")
plt.legend()
plt.title("AGPN Loss Curve")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.savefig("agpn_loss_curve.png")
plt.close()

# ------------------------
# ACCURACY CURVE
# ------------------------
plt.figure()
plt.plot(val_accuracies, label="Validation Accuracy")
plt.legend()
plt.title("AGPN Accuracy Curve")
plt.xlabel("Epoch")
plt.ylabel("Accuracy (%)")
plt.savefig("agpn_accuracy_curve.png")
plt.close()

# ------------------------
# CONFUSION MATRIX
# ------------------------
all_preds = []
all_labels = []

model.load_state_dict(torch.load("agpn_model.pth"))
model.eval()

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)

        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.numpy())

cm = confusion_matrix(all_labels, all_preds)

classes = train_dataset.classes

fig, ax = plt.subplots(figsize=(8, 6))
im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)

ax.set_xticks(np.arange(len(classes)))
ax.set_yticks(np.arange(len(classes)))

ax.set_xticklabels(classes, rotation=45, ha="right")
ax.set_yticklabels(classes)

ax.set_xlabel("Predicted Label")
ax.set_ylabel("True Label")
ax.set_title("AGPN Confusion Matrix", fontsize=16, pad=15)

fig.colorbar(im)

for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        ax.text(j, i, str(cm[i, j]),
                ha="center", va="center",
                color="white" if cm[i, j] > cm.max()/2 else "black")

plt.tight_layout()
plt.savefig("agpn_confusion_matrix.png")
plt.close()

print("AGPN graphs + model saved successfully 🚀")