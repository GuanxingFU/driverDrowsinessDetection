import torch
import torch.nn as nn
import timm
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import time
from tqdm import tqdm
import torch.optim as optim
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

data_dir = "/kaggle/input/driver-fatigue-detection"
resize_transform = transforms.Resize((256, 256))
flip_transform = transforms.RandomHorizontalFlip(p=1.0)
to_tensor_transform = transforms.ToTensor()
normalize_transform = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

sample_idx = 0
sample_path, sample_label = datasets.ImageFolder(root=data_dir).samples[sample_idx]
sample_image = datasets.folder.default_loader(sample_path)

resized_image = resize_transform(sample_image)
flipped_image = flip_transform(resized_image)
tensor_image = to_tensor_transform(flipped_image)  # 转换为Tensor
tensor_image = tensor_image*0.8
normalized_image = normalize_transform(tensor_image)  # 归一化

def show_images(images, titles):
    fig, axes = plt.subplots(1, 5, figsize=(15, 4))
    for ax, img, title in zip(axes, images, titles):
        ax.imshow(img)
        ax.set_title(title)
        ax.axis("off")
    plt.show()

tensor_image_np = tensor_image.permute(1, 2, 0).numpy()
normalized_image_np = normalized_image.permute(1, 2, 0).numpy()

show_images(
    [sample_image, resized_image, flipped_image, tensor_image_np, normalized_image_np],
    ["Original", "Resized", "Flipped", "ToTensor()", "Normalized"]
)
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


dataset = datasets.ImageFolder(root=data_dir, transform=transform)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=2, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=2, pin_memory=True)

print("Class Mapping:", dataset.class_to_idx)

class MobileViT_S(nn.Module):
    def __init__(self, num_classes=2):
        super(MobileViT_S, self).__init__()
        self.model = timm.create_model('mobilevit_s', pretrained=True)
        num_ftrs = self.model.head.in_features
        self.model.head = nn.Identity()
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(num_ftrs, num_classes)

    def forward(self, x):
        x = self.model.forward_features(x)
        x = self.global_pool(x)
        x = torch.flatten(x, start_dim=1)
        x = self.fc(x)
        return x

model = MobileViT_S(num_classes=2).to(device, memory_format=torch.channels_last)
print("Model loaded on:", next(model.parameters()).device)

total_params = sum(p.numel() for p in model.parameters()) / 1e6
print(f"Total Parameters: {total_params:.2f}M")

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
scaler = torch.amp.GradScaler()

# total_training_time = 0
# num_epochs = 10
# for epoch in range(num_epochs):
#     model.train()
#     running_loss = 0.0
#     start_time = time.time()
    
#     for images, labels in tqdm(train_loader, desc=f"Training Epoch {epoch+1}/{num_epochs}"):
#         images, labels = images.to(device), labels.to(device)
#         optimizer.zero_grad()
        
#         with torch.amp.autocast(device_type='cuda'):
#             outputs = model(images)
#             loss = criterion(outputs, labels)
        
#         scaler.scale(loss).backward()
#         scaler.step(optimizer)
#         scaler.update()
#         running_loss += loss.item()
    
#     epoch_time = time.time() - start_time
#     total_training_time += epoch_time
#     print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}, Time: {epoch_time:.2f}s")

# total_inference_time = 0
# model.eval()
# correct = 0
# total = 0

# with torch.no_grad():
#     start_inference = time.time()
#     for images, labels in val_loader:
#         images, labels = images.to(device), labels.to(device)
#         outputs = model(images)
#         _, predicted = torch.max(outputs, 1)
#         total += labels.size(0)
#         correct += (predicted == labels).sum().item()
#     total_inference_time = time.time() - start_inference

# accuracy = 100 * correct / total

# print("\nModel Performance Summary")
# print("+----------------+------------+---------------+-------------+----------------+")
# print("| Model Name     | Params (M) | Train Time (s)| Infer Time (s)| Accuracy (%)   |")
# print("+----------------+------------+---------------+-------------+----------------+")
# print(f"| MobileViT-S    | {total_params:.2f}       | {total_training_time:.2f}     | {total_inference_time:.2f}   | {accuracy:.2f}       |")
# print("+----------------+------------+---------------+-------------+----------------+")
total_training_time = 0
num_epochs = 10
train_losses = []
val_losses = []

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    start_time = time.time()
    
    for images, labels in tqdm(train_loader, desc=f"Training Epoch {epoch+1}/{num_epochs}"):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        
        with torch.amp.autocast(device_type='cuda'):
            outputs = model(images)
            loss = criterion(outputs, labels)
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        running_loss += loss.item()
    
    train_loss = running_loss / len(train_loader)
    train_losses.append(train_loss)
    epoch_time = time.time() - start_time
    total_training_time += epoch_time
    
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
    
    val_loss /= len(val_loader)
    val_losses.append(val_loss)
    
    print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Time: {epoch_time:.2f}s")

plt.figure(figsize=(10, 5))
plt.plot(range(1, num_epochs+1), train_losses, label='Training Loss')
plt.plot(range(1, num_epochs+1), val_losses, label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.show()