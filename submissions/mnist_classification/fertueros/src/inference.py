import PIL.Image
import PIL
import requests
import torch
import torch.nn as nn
from io import BytesIO
from torchvision import transforms
import os
import numpy as np
import sys

url = sys.argv[1]
response = requests.get(url)
image = PIL.Image.open(BytesIO(response.content))

transform = transforms.Compose([
    transforms.Grayscale(),          # ensure single channel
    transforms.Resize((28, 28)),     # safety resize
    transforms.ToTensor(),           # -> [0,1] float
    transforms.Lambda(lambda x: 1 - x),  # invert to match MNIST's black-on-white
])

current_file = __file__
data_folder = os.path.join(os.path.dirname(current_file), "..", "data")
model_path = os.path.join(data_folder, "models", "mnist_model.pth")

model = nn.Sequential(
    nn.Flatten(),
    nn.Linear(28*28,128),
    nn.ReLU(),
    nn.Linear(128,64),
    nn.ReLU(),
    nn.Linear(64,10)
)

model.load_state_dict(torch.load(model_path))


x = transform(image).unsqueeze(0)

y = model(x).detach().numpy()
predicted_digit = np.argmax(y)
print(f"Predicted digit: {predicted_digit}")