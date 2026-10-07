EPOCHS = 10

import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from tqdm import tqdm 
from app.cnn_model import CNNClassifier

torch.manual_seed(42)


device = (
    torch.device("mps")
    if torch.backends.mps.is_available()
    else torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
)
print(f"Using device: {device}")

BATCH_SIZE = 32
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),  ])

train_dataset = torchvision.datasets.CIFAR10(
    root="./data", train=True, download=True, transform=transform
)
test_dataset = torchvision.datasets.CIFAR10(
    root="./data", train=False, download=True, transform=transform
)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

model = CNNClassifier().to(device)  # [改动] 原来是 EnhancedCNN()
print(model)

criterion = nn.CrossEntropyLoss()                      
optimizer = optim.Adam(model.parameters(), lr=0.0005)  

for epoch in range(EPOCHS):
    running_loss = 0.0
    running_correct, running_total = 0, 0

    model.train() 
   
    train_loader_with_progress = tqdm(
        iterable=train_loader, ncols=120, desc=f"Epoch {epoch+1}/{EPOCHS}"
    )
    for batch_number, (inputs, labels) in enumerate(train_loader_with_progress):
        inputs = inputs.to(device)   
        labels = labels.to(device)
        optimizer.zero_grad()        
        outputs = model(inputs)      
        _, predicted = torch.max(outputs.data, 1)  

        loss = criterion(outputs, labels)  
        loss.backward()                    
        optimizer.step()                   

   
        running_correct += (predicted == labels).sum().item()
        running_total += labels.size(0)
        running_loss += loss.item()


        if batch_number % 100 == 99:
            train_loader_with_progress.set_postfix(
                {
                    "avg accuracy": f"{running_correct/running_total:.3f}",
                    "avg loss": f"{running_loss/(batch_number+1):.4f}",
                }
            )

test_correct = 0
test_total = 0
model.eval()           
with torch.no_grad(): 
    for test_images, test_labels in test_loader:
        test_images = test_images.to(device)
        test_labels = test_labels.to(device)
        test_outputs = model(test_images)
        _, test_predicted = torch.max(test_outputs.data, 1)
        test_total += test_labels.size(0)
        test_correct += (test_predicted == test_labels).sum().item()

test_accuracy = 100 * test_correct / test_total
print(f"Accuracy of the network on the 10000 test images: {test_accuracy:.2f}%")

os.makedirs("models", exist_ok=True)
torch.save(model.state_dict(), "models/cnn_cifar10.pth")
print("Model saved to models/cnn_cifar10.pth")