# sps_genai

A FastAPI project that serves several generative AI / deep learning models:
- Bigram text generator
- Word embeddings (spaCy)
- CNN image classifier trained on CIFAR-10 (Assignment 2)

## Setup
```bash
uv sync
```

## Run the API
```bash
uv run fastapi dev app/main.py
```
Interactive docs: http://127.0.0.1:8000/docs

## Endpoints
- `POST /generate` — JSON body `{"start_word": "bigram", "length": 5}`
- `POST /embedding` — JSON body `{"word": "apple"}`
- `POST /classify` — upload an image file (multipart/form-data, field name `file`).
  Returns the predicted CIFAR-10 class and its confidence, e.g.
  `{"filename": "cat.jpg", "class": "cat", "confidence": 0.8123}`

  Example with curl:
```bash
  curl -X POST -F "file=@cat.jpg" http://127.0.0.1:8000/classify
```

## CNN Classifier (Assignment 2)

### Architecture (`app/cnn_model.py`)
Input 64×64×3 → Conv(16, 3×3, stride 1, pad 1) → ReLU → MaxPool(2×2, stride 2)
→ Conv(32, 3×3, stride 1, pad 1) → ReLU → MaxPool(2×2, stride 2)
→ Flatten → FC(100) → ReLU → FC(10)

### Training (`train_cnn.py`)
CIFAR-10 images (32×32) are resized to 64×64 to match the required input size.
Trained for 10 epochs with Adam (lr = 0.0005), batch size 32.
Test accuracy: **66.09%**.

The trained weights are included at `models/cnn_cifar10.pth`, so retraining is not required.
To retrain:
```bash
uv run python train_cnn.py
```

### Project structure
```
app/
  main.py          # FastAPI app and endpoints
  bigram_model.py  # bigram text generator
  embeddings.py    # spaCy word embeddings
  cnn_model.py     # CNN architecture definition
  classifier.py    # loads trained weights and runs prediction
models/
  cnn_cifar10.pth  # trained CNN weights
train_cnn.py       # training script
```