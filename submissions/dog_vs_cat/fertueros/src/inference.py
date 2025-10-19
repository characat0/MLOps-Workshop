# src/inference.py
import sys, os
from io import BytesIO
from PIL import Image, ImageFile
import requests
import torch, torch.nn as nn
from torchvision import transforms
import numpy as np

ImageFile.LOAD_TRUNCATED_IMAGES = True  # por si llega una imagen truncada

if len(sys.argv) < 2:
    print("Uso: python src/inference.py <url_imagen>")
    sys.exit(1)

url = sys.argv[1]

# 1) Descargar con User-Agent + validaciones
headers = {
    "User-Agent": "catsdogs-inference/1.0 (+your.email@example.com)"
}
resp = requests.get(url, headers=headers, timeout=20)
resp.raise_for_status()  # lanza si 4xx/5xx
ct = resp.headers.get("Content-Type", "")
if not ct.startswith("image/"):
    raise ValueError(f"La URL no devolvió una imagen. Content-Type={ct}")

# 2) Abrir imagen de forma robusta
img = Image.open(BytesIO(resp.content))
if img.mode != "RGB":
    img = img.convert("RGB")

# 3) Preproceso
tf = transforms.Compose([
    transforms.Resize((128,128)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3),
])
x = tf(img).unsqueeze(0)

# 4) Cargar modelo (igual que en train.py)
current_file = __file__
data_folder  = os.path.join(os.path.dirname(current_file), "..", "data")
model_path   = os.path.join(data_folder, "models", "catsdogs_model.pth")

model = nn.Sequential(
    nn.Conv2d(3,16,3,padding=1), nn.ReLU(), nn.MaxPool2d(2),
    nn.Conv2d(16,32,3,padding=1), nn.ReLU(), nn.MaxPool2d(2),
    nn.Conv2d(32,64,3,padding=1), nn.ReLU(),
    nn.AdaptiveAvgPool2d((1,1)),
    nn.Flatten(),
    nn.Linear(64,2)
)
if not os.path.exists(model_path):
    sys.exit(f"Modelo no encontrado en {model_path}. "
             "Entrena primero: docker run -v \"$(pwd)/data:/app/data\" --shm-size=1g catsdogs src/train.py")

model.load_state_dict(torch.load(model_path, map_location="cpu"))
model.eval()

with torch.no_grad():
    logits = model(x).squeeze(0).numpy()
    probs  = np.exp(logits - logits.max()); probs = probs / probs.sum()
    label  = int(np.argmax(probs))
    name   = ["Cat","Dog"][label]
    print(f"Predicted: {name}  (p={probs[label]:.2f})")