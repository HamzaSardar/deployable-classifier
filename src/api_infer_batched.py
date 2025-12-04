import requests
import pathlib
from pathlib import Path 
from absl import flags, app
from ml_collections import config_flags
from concurrent.futures import ThreadPoolExecutor, as_completed

import src.utils.flags as cfg_flags


API_URL = 'http://127.0.0.1:8000/predict_batched'
FLAGS = flags.FLAGS

CONFIG = config_flags.DEFINE_config_file('config')
RESULTS_PATH = cfg_flags.DEFINE_path('results_path', '/Users/user/Projects/deployable-classifier/data/', 'Directory to store any logged outputs.')

def predict_image(path_to_image: Path, api_url:str=API_URL):
    image_path = path_to_image

    with open(image_path, 'rb') as f:
        files = {
            'file': (image_path.name, f, 'image/jpeg')
        }

        response = requests.post(api_url, files=files, timeout=10)

    response.raise_for_status()
    return response.json()

def main(_):
    config = FLAGS.config

    all_ims = []
    for category in Path.iterdir(Path(config.inference.data_path)):
        for img_path in Path.iterdir(category):
            all_ims.append(img_path)

    with ThreadPoolExecutor(max_workers=16) as executor:
        futures = [executor.submit(predict_image, img) for img in all_ims]
        for future in as_completed(futures):
            result = future.result()
            print(result)

if __name__=='__main__':
    app.run(main)

