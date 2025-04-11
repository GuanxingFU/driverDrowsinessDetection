import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import time
from tqdm import tqdm  
import pandas as pd

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

torch.backends.cudnn.benchmark = True

data_dir = "/kaggle/input/driver-fatigue-detection"

transform = transforms.Compose([
    transforms.Resize((224, 224)),  
    transforms.RandomHorizontalFlip(),  
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  
    transforms.RandomRotation(10),  
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  
])

dataset = datasets.ImageFolder(root=data_dir, transform=transform)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4, pin_memory=True)

from torchvision.models import efficientnet_b0
model = efficientnet_b0(weights="IMAGENET1K_V1")
model.classifier[1] = nn.Linear(model.classifier[1].in_features, 2)
model = model.to(device)
print("Model loaded on:", next(model.parameters()).device)

num_params = sum(p.numel() for p in model.parameters()) / 1e6
print(f"Total Parameters: {num_params:.2f}M")

criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
optimizer = optim.AdamW(model.parameters(), lr=0.0005, weight_decay=1e-4)

scaler = torch.amp.GradScaler()
num_epochs = 1 
total_training_time = 0
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    start_time = time.time()
    loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")

    for images, labels in loop:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        with torch.amp.autocast(device_type="cuda"):
            outputs = model(images)
            loss = criterion(outputs, labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        running_loss += loss.item()
        loop.set_postfix(loss=loss.item())
    
    epoch_time = time.time() - start_time  
    total_training_time += epoch_time
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}, Time: {epoch_time:.2f}s")

model.eval()
correct = 0
total = 0
total_inference_time = 0
num_batches = 0
inference_times = []

with torch.no_grad():
    for images, labels in tqdm(val_loader, desc="Evaluating Model"):
        start_time = time.time()
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        
        end_time = time.time()
        inference_times.append(time.time() - start_time)
        
        total_inference_time += (end_time - start_time)
        num_batches += 1
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = 100 * correct / total
avg_inference_time = total_inference_time / num_batches
avg_inference_time_new = sum(inference_times) / len(inference_times)
print(f"efficientnet_b0 new Average Inference Time per Batch: {avg_inference_time:.6f}s")

print(f"Validation Accuracy: {accuracy:.2f}%")
print(f"Total Training Time: {total_training_time:.2f}s")
print(f"Total Inference Time: {total_inference_time:.4f}s")
print(f"Avg Inference Time per Batch: {avg_inference_time:.6f}s")

results = pd.DataFrame({
    "Model": ["EfficientNet-B0"],
    "Parameters (M)": [num_params],
    "Training Time (s)": [total_training_time],
    "Inference Time (s)": [total_inference_time],
    "Accuracy (%)": [accuracy]
})
print(results)