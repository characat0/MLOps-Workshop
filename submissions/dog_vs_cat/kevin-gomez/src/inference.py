import os
import sys
import requests
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
from io import BytesIO

# Set device (GPU if available, else CPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load and preprocess image
url = sys.argv[1]
response = requests.get(url)
image = Image.open(BytesIO(response.content)).convert('RGB')  # Ensure RGB format

# Define the same transforms as in training
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],  # Imagenet means
        std=[0.229, 0.224, 0.225]    # Imagenet stds
    )
])

# Get model path
current_file = __file__
data_root = os.path.join(os.path.dirname(current_file), "..", "data", "PetImages")
model_path = os.path.join(data_root, "models", "dog_vs_cat_model.pth")

# Initialize model (same architecture as in training)
model = models.resnet18(pretrained=False)
model.fc = nn.Linear(512, 1)  # Binary classification
model = model.to(device)

# Load trained weights
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()  # Set to evaluation mode

# Process image and make prediction
with torch.no_grad():
    x = transform(image).unsqueeze(0).to(device)
    output = model(x)
    probability = torch.sigmoid(output).item()
    predicted_class = 'dog' if probability > 0.5 else 'cat'

print(f"Predicted class: {predicted_class} (confidence: {probability:.2f})")