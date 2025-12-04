import requests
import pathlib
from pathlib import Path 
from absl import flags, app
from ml_collections import config_flags

import src.utils.flags as cfg_flags


API_URL = 'http://127.0.0.1:8000/predict'
FLAGS = flags.FLAGS

CONFIG = config_flags.DEFINE_config_file('config')
RESULTS_PATH = cfg_flags.DEFINE_path('results_path', '/Users/user/Projects/deployable-classifier/data/', 'Directory to store any logged outputs.')


def predict_image(path_to_image: Path, api_url:str=API_URL):
    image_path = Path(path_to_image)


    with open(image_path, 'rb') as f:
        files = {
            'file': (image_path.name, f, 'image/jpeg')
        }

        response = requests.post(api_url, files=files, timeout=10)


    response.raise_for_status()
    return response.json()

def main(_):
    config = FLAGS.config
    for category in Path.iterdir(Path(config.inference.data_path)):
        for img_path in Path.iterdir(category):
            print(predict_image(img_path))

if __name__=='__main__':
    app.run(main)

