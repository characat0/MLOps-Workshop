import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision
from torchvision import transforms, models
from torch.utils.data import random_split, DataLoader

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')


current_file = __file__


data_root = os.path.join(os.path.dirname(current_file), "..", "data", "PetImages")
model_path = os.path.join(data_root, "models", "dog_vs_cat_model.pth")

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],  # Imagenet means
        std=[0.229, 0.224, 0.225]    # Imagenet stds
    )
])

full_data = torchvision.datasets.ImageFolder(root=data_root, transform=transform)

total_len = len(full_data)
train_len = 4 * total_len // 5
val_len = total_len - train_len
generator = torch.Generator().manual_seed(42)
train_data, val_data = random_split(full_data, [train_len, val_len], generator=generator)

train_load = DataLoader(train_data, batch_size=100, shuffle=True)
val_load = DataLoader(val_data, batch_size=100, shuffle=True)

# Load pretrained ResNet18
model = models.resnet18(pretrained=True)

# Freeze all layers except final fully connected layer (optional)
for param in model.parameters():
    param.requires_grad = False

# Replace the final layer for binary classification
# ResNet18 fc layer input features = 512
model.fc = nn.Linear(512, 1)  # Output single logit for binary classification

# Unfreeze the final layer parameters for training
for param in model.fc.parameters():
    param.requires_grad = True

model = model.to(device)

criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.fc.parameters(), lr=0.001)  # Only fine-tuning last layer

epochs = 3
loss_tr, loss_val = [], []
acc_tr, acc_val = [], []

for epoch in range(epochs):
    print(f"Epoch {epoch+1}/{epochs}")

    # Training
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for x, y in train_load:
        x, y = x.to(device), y.to(device).float().unsqueeze(1)
        optimizer.zero_grad()
        outputs = model(x)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        preds = (torch.sigmoid(outputs) > 0.5).float()
        correct += (preds == y).sum().item()
        total += y.size(0)

    train_loss = running_loss / len(train_load)
    train_acc = 100 * correct / total
    loss_tr.append(train_loss)
    acc_tr.append(train_acc)

    # Validation
    model.eval()
    running_val_loss = 0.0
    correct_val = 0
    total_val = 0

    with torch.no_grad():
        for x, y in val_load:
            x, y = x.to(device), y.to(device).float().unsqueeze(1)
            outputs = model(x)
            loss = criterion(outputs, y)
            running_val_loss += loss.item()
            preds = (torch.sigmoid(outputs) > 0.5).float()
            correct_val += (preds == y).sum().item()
            total_val += y.size(0)

    val_loss = running_val_loss / len(val_load)
    val_acc = 100 * correct_val / total_val
    loss_val.append(val_loss)
    acc_val.append(val_acc)

    print(f"Train Loss: {train_loss:.4f}, Accuracy: {train_acc:.2f}%")
    print(f"Val   Loss: {val_loss:.4f}, Accuracy: {val_acc:.2f}%\n")

torch.save(model.state_dict(), model_path)