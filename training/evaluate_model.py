import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import os

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print("Using device:", device)

data_dir = "dataset"

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

test_dataset = datasets.ImageFolder(os.path.join(data_dir, "test"), transform=transform)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

model = models.resnet50(weights=None)
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, 5)

model.load_state_dict(torch.load("baseline_model.pth", map_location=device))
model = model.to(device)
model.eval()

all_preds = []
all_labels = []

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.numpy())

cm = confusion_matrix(all_labels, all_preds)

fig, ax = plt.subplots(figsize=(10, 8))

disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                              display_labels=test_dataset.classes)

disp.plot(ax=ax, cmap=plt.cm.Blues)

plt.xticks(rotation=45, ha='right')

ax.set_title("Confusion Matrix", pad=20)

plt.tight_layout()

plt.savefig("confusion_matrix.png", bbox_inches="tight")
plt.close()

print("Updated confusion matrix saved successfully.")