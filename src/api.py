import io
from pathlib import Path
from collections import deque
import asyncio

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
CONFIDENCE_THRESHOLD = 0.5

app = FastAPI(title='Image Classifier API')
model = None
device = None
transform = transforms.Compose([
    transforms.Resize((32, 32)),  # Adjust to your modeljs input size
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Adjust to your normalization
])

request_queue = deque()
BATCH_SIZE = 16
BATCH_TIMEOUT = 0.1
processing_task = None

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

    processing_task = asyncio.create_task(process_batches())

    print('Model loaded.')

async def process_batches():
    """
        Processes queued requests.
    """
    while True:
        if len(request_queue) == 0:
            await asyncio.sleep(1)
            continue

        batch = []
        future = []

        # add samples to batch if batch_size not met
        while len(batch) < BATCH_SIZE and len(request_queue) > 0:
            image_t, _future = request_queue.popleft()
            batch.append(image_t)
            future.append(_future)

        # process batch
        batch_t = torch.cat(batch, dim=0)

        with torch.no_grad():
            out = model(batch_t)
            prob = torch.nn.functional.softmax(out, dim=1)
            confidence, pred = torch.max(prob, 1)

        for i, f in enumerate(future):
            if confidence[i].item() < CONFIDENCE_THRESHOLD:
                predicted_class = 'Unknown'
            else:
                predicted_class = CLASSES[pred[i].item()]
            result = {
                'predicted_class': predicted_class
            }
            f.set_result(result)
        await asyncio.sleep(0.01)
            
@app.post("/predict_batched")
async def predict_batched(file: UploadFile = File(...)):
    "Batched endpoint"

    _image = await file.read()
    image = Image.open(io.BytesIO(_image))
    image_t = transform(image).unsqueeze(0)

    future = asyncio.Future()
    request_queue.append((image_t, future))

    result = await future
    return result

@app.post('/predict')
async def predict(file: UploadFile = File(...)):
    # read in image
    _image = await file.read()
    image = Image.open(io.BytesIO(_image))

    image_t = transform(image).unsqueeze(0)
    with torch.no_grad():
        out = model(image_t)
        confidence, predicted = torch.max(out, 1)

    if confidence.item() < CONFIDENCE_THRESHOLD:
        predicted_class = 'Unknown'
    else:
        predicted_class = CLASSES[predicted.item()]

    return {"predicted_class": predicted_class}

