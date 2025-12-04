import torch
from torchvision import transforms
import io
import requests
import pathlib
from pathlib import Path 
from absl import flags, app
from ml_collections import config_flags
from PIL import Image

import src.utils.flags as cfg_flags


API_URL = 'http://127.0.0.1:8000/predict'
FLAGS = flags.FLAGS

CONFIG = config_flags.DEFINE_config_file('config')
RESULTS_PATH = cfg_flags.DEFINE_path('results_path', '/Users/user/Projects/deployable-classifier/data/', 'Directory to store any logged outputs.')

def add_noise(path_to_image: Path, noise_scale: float=0.1):
    # load in image
    img = Image.open(path_to_image).convert('RGB')

    # create transforms
    to_tensor = transforms.ToTensor()
    to_pil = transforms.ToPILImage()

    # add noise here
    img_t = to_tensor(img)
    img_t = torch.clamp(img_t + (torch.randn_like(img_t) * noise_scale), 0, 1) 

    img = to_pil(img_t)
    img_bytes = io.BytesIO()
    img.save(img_bytes, format='JPEG')
    img_bytes.seek(0)

    return img_bytes

def predict_image(image_path: Path, api_url:str=API_URL, noise: bool=False):
    if noise:
        img_bytes = add_noise(image_path)
        files = {
            'file': (image_path.name, img_bytes, 'image/jpeg')
        }
        response = requests.post(api_url, files=files, timeout=10)
    else:
        with open(image_path, 'rb') as f:
            files = {
                'file': (image_path.name, f, 'image/jpeg')
            }
            response = requests.post(api_url, files=files, timeout=10)

    response.raise_for_status()
    return response.json()

def main(_):
    config = FLAGS.config

    total = 0
    correct_clean = 0
    correct_noisy = 0

    print('Running adversarial noise test with noise_scale=0.1...')
    
    for category in Path.iterdir(Path(config.inference.data_path)):
        true_label = category.name

        for img_path in Path.iterdir(category):
            clean_result = predict_image(img_path, noise=False)['predicted_class']
            noisy_result = predict_image(img_path, noise=True)['predicted_class']

            if clean_result == true_label:
                correct_clean += 1
            if noisy_result == true_label:
                correct_noisy += 1

            total += 1

    clean_acc = 100 * correct_clean / total if total > 0 else 0
    noisy_acc = 100 * correct_noisy / total if total > 0 else 0
    robustness_drop = clean_acc - noisy_acc

    print(f"Total images tested: {total}")
    print(f"Clean Accuracy: {clean_acc:.2f}%")
    print(f"Noisy Accuracy (noise_scale={0.1}): {noisy_acc:.2f}%")
    print(f"Robustness Drop: {robustness_drop:.2f}%")

if __name__=='__main__':
    app.run(main)

