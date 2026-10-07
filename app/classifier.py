import io
import torch
import torchvision.transforms as transforms
from PIL import Image
from app.cnn_model import CNNClassifier

CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]

MODEL_PATH = "models/cnn_cifar10.pth"
device = torch.device("cpu")

transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
])

model = CNNClassifier()
state_dict = torch.load(MODEL_PATH, map_location=device)
model.load_state_dict(state_dict)  
model.to(device)
model.eval() 


def predict_image(image_bytes: bytes) -> dict:
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(tensor)
        probabilities = torch.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probabilities, 1)

    return {
        "class": CLASSES[predicted.item()],
        "confidence": round(confidence.item(), 4),
         }