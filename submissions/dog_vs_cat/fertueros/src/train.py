import os, pathlib
import torch, torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
import torch.nn as nn, torch.optim as optim

current_file = __file__
data_folder = os.path.join(os.path.dirname(current_file), "..", "data")
model_path  = os.path.join(data_folder, "models", "catsdogs_model.pth")
pathlib.Path(os.path.join(data_folder, "models")).mkdir(parents=True, exist_ok=True)

tf_train = transforms.Compose([
    transforms.Resize((128,128)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3),
])
tf_eval = transforms.Compose([
    transforms.Resize((128,128)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3),
])

train = torchvision.datasets.OxfordIIITPet(
    root=data_folder,
    split="trainval",
    target_types="binary-category",  # 0=Cat, 1=Dog
    download=True,
    transform=tf_train
)
test = torchvision.datasets.OxfordIIITPet(
    root=data_folder,
    split="test",
    target_types="binary-category",
    download=True,
    transform=tf_eval
)

train_loader = DataLoader(train, batch_size=64, shuffle=True,  num_workers=2)
test_loader  = DataLoader(test,  batch_size=64, shuffle=False, num_workers=2)

model = nn.Sequential(
    nn.Conv2d(3,16,3,padding=1), nn.ReLU(), nn.MaxPool2d(2),
    nn.Conv2d(16,32,3,padding=1), nn.ReLU(), nn.MaxPool2d(2),
    nn.Conv2d(32,64,3,padding=1), nn.ReLU(),
    nn.AdaptiveAvgPool2d((1,1)),
    nn.Flatten(),
    nn.Linear(64,2)
)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

for epoch in range(2):
    model.train()
    running = 0.0
    for x,y in train_loader:
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        running += loss.item()
    print(f"[epoch {epoch+1}] loss={running/len(train_loader):.4f}")

# evaluación rápida
model.eval()
correct, total = 0, 0
with torch.no_grad():
    for x,y in test_loader:
        pred = model(x).argmax(1)
        correct += (pred==y).sum().item()
        total   += y.size(0)
acc = correct/total if total else 0.0
print(f"Test acc (aprox): {acc:.3f}")

torch.save(model.state_dict(), model_path)
print("Modelo guardado en:", model_path)