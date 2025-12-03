import io
from pathlib import Path

import torch
from torchvision import transforms
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image

from src.model import Classifier
from src.data_processing import get_dataloaders


MODEL_PATH=Path('./src/model.pt')

CLASSES = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

app = FastAPI(title='Image Classifier API')
model = None
device = None
transform = transforms.Compose([
    transforms.Resize((32, 32)),  # Adjust to your modeljs input size
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Adjust to your normalization
])

@app.on_event('startup')
async def load_model():
    """
    Load model on server start.
    """
    global model, device

    # instantiate model and load in weights
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Classifier()
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.to(device)
    model.eval()

    print('Model loaded.')

@app.post('/predict')
async def predict(file: UploadFile = File(...)):
    # read in image
    _image = await file.read()
    image = Image.open(io.BytesIO(_image))

    image_t = transform(image).unsqueeze(0)
    with torch.no_grad():
        out = model(image_t)
        _, predicted = torch.max(out, 1)

    predicted_class = CLASSES[predicted.item()]
    return {"predicted_class": predicted_class}

