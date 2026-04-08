# Deployable Classifier — Code Notes

## What this project is

A CIFAR-10 image classifier (standard 10-class benchmark: plane, car, bird, cat, deer, dog, frog, horse, ship, truck) packaged behind a FastAPI server so it can accept image uploads over HTTP and return predictions. The ML side is a small CNN trained from scratch; the interesting part is the deployment infrastructure around it.

---

## File-by-file walkthrough

### `src/model.py` — The network

A two-block conv net taken directly from the PyTorch CIFAR-10 tutorial:

```
Input (3×32×32)
→ Conv2d(3→6, kernel=5) + ReLU + MaxPool(2×2)   → (6×14×14)
→ Conv2d(6→16, kernel=5) + ReLU + MaxPool(2×2)  → (16×5×5)
→ Flatten → FC(400→120) → FC(120→84) → FC(84→10)
```

Returns **raw logits** (no softmax). That is intentional and correct — `CrossEntropyLoss` applies softmax internally during training.

The architecture is hardcoded for 32×32 inputs (the `16*5*5` in `fc1`). If you ever swap to a larger dataset you'd need to recalculate that.

---

### `src/data_processing.py` — Dataset loading

Downloads CIFAR-10 via torchvision, applies:
- `ToTensor()` — scales pixel values from [0, 255] to [0.0, 1.0]
- `Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))` — shifts to [-1.0, 1.0]

Returns two `DataLoader`s (train and test).

> **Minor issue:** `test_loader` is created with `shuffle=True`. Shuffling a test set makes no difference to accuracy metrics but is unconventional — evaluation sets are usually left unshuffled for reproducibility.

---

### `src/train.py` — Training loop

Standard supervised loop: forward pass → CrossEntropyLoss → backward → Adam step. Uses `accelerate` to handle device placement (CPU/GPU/MPS) without you having to write `.to(device)` everywhere — handy pattern.

Saves the final `model.state_dict()` to `model.pt`.

**Issues:**

1. **Logging cadence is too sparse.** It prints every 2000 iterations (`i % 2000 == 0`). CIFAR-10 with `batch_size=32` gives ~1562 batches per epoch, so this only ever prints at iteration 0 (the very start of each epoch). Change to `i % 200 == 0` to get ~8 log lines per epoch.

2. **Variable shadowing.** `loss = 0` is set before the loop, then `loss = loss_fn(out, label)` inside it — the inner `loss` shadows the outer one. Not a bug (the outer value is never used after the loop), but confusing to read.

---

### `src/inference.py` — Local evaluation script

Runs the saved model against the CIFAR-10 test set and prints overall accuracy plus per-class breakdown. Not used by the API — this is a standalone script for checking model quality after training.

**Issues:**

1. **Integer division for accuracy.** Line 38 uses `//` (floor division): `100 * correct // total`. For 10000 test images this truncates the decimal. Should be `100 * correct / total`.

2. **`torch.load` without `weights_only=True`.** Newer PyTorch versions emit a deprecation warning here. Fix:
   ```python
   model.load_state_dict(torch.load(MODEL_PATH, weights_only=True))
   ```

3. **Two full passes over the test set.** The overall accuracy loop and the per-class accuracy loop are separate. They can be merged into one.

---

### `src/api.py` — The FastAPI server

This is the core of the deployment. Two endpoints.

#### How FastAPI works (quick orientation)

FastAPI is a Python web framework built on top of Starlette. You decorate functions with `@app.post('/route')` to register them as HTTP handlers. `async def` handlers run in an async event loop (non-blocking), which matters for the batching logic below. `UploadFile` is FastAPI's type for a multipart file upload — `await file.read()` reads the raw bytes asynchronously.

#### `/predict` — single-image endpoint

```
POST /predict  (multipart form, field: "file")
→ reads image bytes → PIL → resize 32×32 → ToTensor → Normalize
→ model forward pass (no_grad)
→ returns {"predicted_class": "cat"} or {"predicted_class": "Unknown"}
```

**Bug — confidence check uses raw logits:**
```python
confidence, predicted = torch.max(out, 1)   # out is raw logits
if confidence.item() < CONFIDENCE_THRESHOLD:  # comparing logit to 0.5
```
`torch.max` on raw logits gives you the highest logit value, not a probability. A logit of 0.5 has no meaningful interpretation. You need softmax first:
```python
prob = torch.nn.functional.softmax(out, dim=1)
confidence, predicted = torch.max(prob, 1)
```
The `/predict_batched` path does this correctly — the inconsistency means the two endpoints behave differently.

#### `/predict_batched` — async queue + batching endpoint

This is the more sophisticated endpoint. The idea: instead of running inference on one image at a time, collect images from multiple concurrent requests and process them together as a batch (better GPU utilisation).

The mechanism:
1. `request_queue` is a `deque` (double-ended queue, thread-safe for appends/poplefts).
2. Each request to `/predict_batched` puts its tensor + an `asyncio.Future` into the queue, then `await future` — suspending that request handler until a result is ready.
3. `process_batches()` runs as a background task in the same event loop. It drains up to 16 items from the queue, stacks them into a batch tensor, runs inference, applies softmax, and calls `future.set_result(...)` for each — which wakes up the suspended request handlers.

This is a clean implementation of the "dynamic batching" pattern. The asyncio model makes it work without threads on the server side.

**Bug — `processing_task` is never assigned to the global:**
```python
processing_task = None  # global

@app.on_event('startup')
async def load_model():
    ...
    processing_task = asyncio.create_task(process_batches())  # local variable!
```
The assignment inside the function creates a *local* `processing_task`, not the global one. The task itself still runs (asyncio keeps a reference internally), but the global stays `None`. If you ever wanted to cancel or inspect the task from elsewhere it wouldn't work. Fix:
```python
global processing_task
processing_task = asyncio.create_task(process_batches())
```

**`@app.on_event('startup')` is deprecated** in FastAPI >= 0.93. The modern replacement is a `lifespan` context manager:
```python
from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    # startup
    await load_model()
    yield
    # shutdown (nothing needed here)

app = FastAPI(title='Image Classifier API', lifespan=lifespan)
```
It still works as-is, but you'll see deprecation warnings.

**Minor:** line 27 has a stale comment `# Adjust to your modeljs input size` — looks like a copy-paste artifact.

---

### `src/api_infer_single_image.py` — Simplest client

Sends one image to `/predict` and prints the result. Nothing fancy.

> Has a hardcoded path `/Users/user/Downloads/cifar10_airplane1.jpg` in `__main__` — swap this to your own path.

---

### `src/api_infer_serial.py` — Serial batch client

Iterates a directory of images (organised as `data_path/category/image.jpg`) and sends them one-by-one to `/predict`. Uses `absl` flags and `ml_collections` config to get the data path from the command line:

```bash
python -m src.api_infer_serial --config=src/configs/config.py
```

> **Missing dependencies:** `absl-py` and `ml_collections` are used here (and in the batched/adversarial scripts) but are **not in `requirements.txt` or `pyproject.toml`**. Add them.

---

### `src/api_infer_batched.py` — Parallel batch client

Same directory-walk as the serial client, but fires all requests concurrently using `ThreadPoolExecutor(max_workers=16)` pointing at `/predict_batched`.

The threading here is on the **client** side (16 HTTP connections in parallel). The batching is on the **server** side (the async queue collecting them). The two work together: the 16 threads hammer the server simultaneously, the server groups them into batches of up to 16, runs one GPU forward pass, and responds to all 16 at once.

---

### `src/adversarial_noise.py` — Robustness test

Tests how much accuracy drops when you add Gaussian noise to images before sending them. For each image it:
1. Sends the clean image → records if prediction matches the folder name (used as ground truth label)
2. Adds `N(0, 0.1)` noise (clamped to [0,1]) → sends noisy version → records accuracy

Prints clean accuracy, noisy accuracy, and the drop between them as a robustness metric. This is a reasonable sanity check — a model that drops 40% accuracy under mild noise is not well-generalised.

> The `noise_scale` parameter exists on `add_noise()` but is hardcoded to `0.1` at the call site in `main`. You could expose it as an `absl` flag to sweep over multiple noise levels.

---

### `src/configs/common.py` + `src/configs/config.py` — Config system

`ml_collections` is a Google-internal config library used heavily in JAX research code (you've likely seen it in diffusion model repos). It creates a nested `ConfigDict` that can be passed around and overridden from the command line.

`common.py` defines the base config — currently just one field:
```python
config.inference.data_path = Path('/Users/user/Projects/CIFAR-10-images/test')
```

> **Hardcoded path** — this needs to be changed to your own CIFAR-10 test images directory before running any of the inference/adversarial scripts.

`config.py` just calls `common.get_config()` and returns it. The pattern is: `config.py` is the "experiment config" you point `--config` at, and it layers overrides on top of `common`. Since there are no overrides yet it's essentially a pass-through, but the structure is there if you want to add dataset-specific or run-specific settings.

---

### `src/utils/flags.py` — Custom flag type

Adds a `Path` flag type to `absl-flags`. By default absl only has string/int/float/bool flags; this parser converts a string CLI argument into a `pathlib.Path` object automatically. Used for `RESULTS_PATH` in the inference scripts (though `RESULTS_PATH` is defined but never actually used in any script currently).

---

## Issues summary

| # | File | Issue | Severity |
|---|------|-------|----------|
| 1 | `api.py:114` | `/predict` confidence uses raw logits, not softmax probabilities — threshold comparison is meaningless | Bug |
| 2 | `api.py:50` | `processing_task` assigned to local variable in startup handler, global never updated | Bug |
| 3 | `api.py:36` | `@app.on_event('startup')` deprecated in FastAPI ≥0.93 | Warning |
| 4 | `api.py:27` | Stale comment `# Adjust to your modeljs input size` | Cosmetic |
| 5 | `inference.py:18` | `torch.load` without `weights_only=True` — deprecation warning in newer torch | Warning |
| 6 | `inference.py:38` | `//` integer division truncates accuracy percentage | Minor |
| 7 | `train.py:34` | Log every 2000 iters = once per epoch on CIFAR-10/batch32 — too sparse | Minor |
| 8 | `data_processing.py:22` | Test loader shuffled — unconventional | Minor |
| 9 | `pyproject.toml` | `absl-py` and `ml-collections` missing from dependencies | Bug |
| 10 | `configs/common.py:11` | `data_path` hardcoded to `/Users/user/...` | Usability |
| 11 | `api_infer_single_image.py:23` | Image path hardcoded to `/Users/user/...` | Usability |
| 12 | `adversarial_noise.py` | `noise_scale` not exposed as a flag — hardcoded to 0.1 | Minor |
