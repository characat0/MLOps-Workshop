import torch, torchvision
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import os


current_file = __file__


data_folder = os.path.join(os.path.dirname(current_file), "..", "data")
model_path = os.path.join(data_folder, "models", "mnist_model.pth")

train = torchvision.datasets.MNIST(root=data_folder, train=True, download=True, transform=torchvision.transforms.ToTensor())
train_loader = DataLoader(train, batch_size=64, shuffle=True)

model = nn.Sequential(
    nn.Flatten(),
    nn.Linear(28*28,128),
    nn.ReLU(),
    nn.Linear(128,64),
    nn.ReLU(),
    nn.Linear(64,10)
)


criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

for epoch in range(15):  # örnek 5 epoch
    running_loss = 0.0
    for images, labels in train_loader:
        optimizer.zero_grad()      # gradientleri sıfırla
        outputs = model(images)    # forward pass
        loss = criterion(outputs, labels)  # loss hesapla
        loss.backward()            # backprop
        optimizer.step()           # ağırlıkları güncelle

        running_loss += loss.item()
    print(f"Epoch {epoch+1}, Loss: {running_loss/len(train_loader):.4f}")


torch.save(model.state_dict(), model_path)
