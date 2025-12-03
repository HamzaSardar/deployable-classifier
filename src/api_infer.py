import requests
from pathlib import Path 


API_URL = 'http://127.0.0.1:8000/predict'

def predict_image(path_to_image: str, api_url:str=API_URL):
    image_path = Path(path_to_image)


    with open(image_path, 'rb') as f:
        files = {
            'file': (image_path.name, f, 'image/jpeg')
        }

        response = requests.post(api_url, files=files, timeout=10)


    response.raise_for_status()
    return response.json()

if __name__=='__main__':
    result = predict_image('/Users/user/Downloads/cifar10_airplane1.jpg')
    print(result)

