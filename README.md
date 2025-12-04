# deployable-classifier [![python](https://img.shields.io/badge/Python-3.12.1-3776AB.svg?style=flat&logo=python&logoColor=white)](https://www.python.org)
A modular, pytorch-based CNN classifier for CIFAR-10, packaged into a RESTful API service using FastAPI. 

## TODO:
- [x] ~~Train and test basic model.~~
- [x] ~~Write the api-based inference script.~~
- [x] ~~Test how the model handles OOD images.~~
- [x] ~~Add batching or queueing to inference script.~~
- [ ] Add detection for performance degradation/increasing error.
- [ ] Test robustness to adversarial inputs.

## Quickstart 

1. Install dependencies:
```
uv pip install -r requirements.txt
```

2. Train model:
```
accelerate launch -m src.train
```

3. Start the API server:
```
uvicorn src.api:app --reload &
```

4. Run test inference scripts:
```
python -m src.api_infer_serial --config=src/configs/common.py # for serial mode
python -m src.api_infer_batched --config=src/configs/common.py # for batched mode
```

