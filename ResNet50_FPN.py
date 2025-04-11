import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import time
from tqdm import tqdm  # 进度条

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

data_dir = "/kaggle/input/driver-fatigue-detection"

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(0.2, 0.2, 0.2, 0.1),
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
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class ResNet50_FPN(nn.Module):
    def __init__(self, num_classes=2):
        super(ResNet50_FPN, self).__init__()

        resnet = models.resnet50(weights="IMAGENET1K_V2")
        
        self.stem = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool
        )
        
        self.layer1 = resnet.layer1  # C2 (256通道)
        self.layer2 = resnet.layer2  # C3 (512通道)
        self.layer3 = resnet.layer3  # C4 (1024通道)
        self.layer4 = resnet.layer4  # C5 (2048通道)

        self.toplayer = nn.Conv2d(2048, 256, kernel_size=1)  # P5降维
        self.latlayer1 = nn.Conv2d(1024, 256, kernel_size=1) # C4修正
        self.latlayer2 = nn.Conv2d(512, 256, kernel_size=1)  # C3修正
        self.latlayer3 = nn.Conv2d(256, 256, kernel_size=1)  # C2修正

        self.c5_reduce = nn.Linear(2048, 256)  # 将原始C5的2048维特征降维到256

        self.smooth1 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.smooth2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.smooth3 = nn.Conv2d(256, 256, kernel_size=3, padding=1)

        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)

        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(256 * 5, num_classes)
        )

    def _upsample_add(self, x, y):
        _, _, H, W = y.size()
        return F.interpolate(x, size=(H, W), mode='bilinear', align_corners=True) + y

    def forward(self, x):
        x = self.stem(x)       # [B,64,56,56]
        c2 = self.layer1(x)    # [B,256,56,56]
        c3 = self.layer2(c2)   # [B,512,28,28]
        c4 = self.layer3(c3)   # [B,1024,14,14]
        c5 = self.layer4(c4)   # [B,2048,7,7]

        p5 = self.toplayer(c5)                      # [B,256,7,7]
        p4 = self._upsample_add(p5, self.latlayer1(c4))  # [B,256,14,14]
        p3 = self._upsample_add(p4, self.latlayer2(c3))  # [B,256,28,28]
        p2 = self._upsample_add(p3, self.latlayer3(c2))  # [B,256,56,56]

        p4 = self.smooth1(p4)
        p3 = self.smooth2(p3)
        p2 = self.smooth3(p2)

        p2_pool = self.global_avg_pool(p2).flatten(1)  # [B,256]
        p3_pool = self.global_avg_pool(p3).flatten(1)  # [B,256]
        p4_pool = self.global_avg_pool(p4).flatten(1)  # [B,256]
        p5_pool = self.global_avg_pool(p5).flatten(1)  # [B,256]

        c5_pool = self.global_avg_pool(c5).flatten(1)  # [B,2048]
        c5_reduced = self.c5_reduce(c5_pool)           # [B,256] 

        features = torch.cat([p2_pool, p3_pool, p4_pool, p5_pool, c5_reduced], dim=1)

        out = self.classifier(features)
        return out

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ResNet50_FPN().to(device)

dummy_input = torch.randn(2, 3, 224, 224).to(device)
output = model(dummy_input)

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc, precision_recall_curve
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F

writer = SummaryWriter('runs/fpn_experiment')

criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
num_epochs = 100

train_losses = []
val_losses = []
val_accuracies = []
all_targets = []
all_preds = []
all_probs = []

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
    
    for images, labels in loop:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        loop.set_postfix(loss=loss.item())
    
    epoch_train_loss = running_loss / len(train_loader)
    train_losses.append(epoch_train_loss)
    
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    epoch_targets = []
    epoch_preds = []
    epoch_probs = []
    
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            
            probs = F.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs, 1)
            
            epoch_targets.extend(labels.cpu().numpy())
            epoch_preds.extend(predicted.cpu().numpy())
            epoch_probs.extend(probs.cpu().numpy())
            
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    all_targets.extend(epoch_targets)
    all_preds.extend(epoch_preds)
    all_probs.extend(epoch_probs)
    
    epoch_val_loss = val_loss / len(val_loader)
    epoch_val_acc = 100 * correct / total
    val_losses.append(epoch_val_loss)
    val_accuracies.append(epoch_val_acc)
    
    writer.add_scalar('Loss/train', epoch_train_loss, epoch)
    writer.add_scalar('Loss/val', epoch_val_loss, epoch)
    writer.add_scalar('Accuracy/val', epoch_val_acc, epoch)
    
    print(f"Epoch [{epoch+1}/{num_epochs}] "
          f"Train Loss: {epoch_train_loss:.4f} | "
          f"Val Loss: {epoch_val_loss:.4f} | "
          f"Val Acc: {epoch_val_acc:.2f}%")

writer.close()

plt.figure(figsize=(18, 10))

plt.subplot(2, 3, 1)
plt.plot(train_losses, label='Training Loss')
plt.plot(val_losses, label='Validation Loss')
plt.title('Training & Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.subplot(2, 3, 2)
plt.plot(val_accuracies, color='green')
plt.title('Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy (%)')

plt.subplot(2, 3, 3)
cm = confusion_matrix(all_targets, all_preds)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
           xticklabels=['Normal', 'Fatigue'],
           yticklabels=['Normal', 'Fatigue'])
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')

plt.subplot(2, 3, 4)
fpr, tpr, _ = roc_curve(all_targets, np.array(all_probs)[:, 1])
roc_auc = auc(fpr, tpr)
plt.plot(fpr, tpr, color='darkorange', lw=2, 
         label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")

plt.subplot(2, 3, 5)
precision, recall, _ = precision_recall_curve(all_targets, np.array(all_probs)[:, 1])
plt.plot(recall, precision, color='blue', lw=2)
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')

plt.subplot(2, 3, 6)
report = classification_report(all_targets, all_preds, target_names=['Normal', 'Fatigue'], output_dict=True)
metrics = ['precision', 'recall', 'f1-score']
normal_metrics = [report['Normal'][m] for m in metrics]
fatigue_metrics = [report['Fatigue'][m] for m in metrics]

angles = np.linspace(0, 2*np.pi, len(metrics), endpoint=False).tolist()
angles += angles[:1]

fig = plt.gcf()
ax = fig.add_subplot(2, 3, 6, polar=True)

ax.plot(angles, normal_metrics + [normal_metrics[0]], 'o-', label='Normal')
ax.plot(angles, fatigue_metrics + [fatigue_metrics[0]], 'o-', label='Fatigue')
ax.fill(angles, normal_metrics + [normal_metrics[0]], alpha=0.25)
ax.fill(angles, fatigue_metrics + [fatigue_metrics[0]], alpha=0.25)
ax.set_theta_offset(np.pi/2)
ax.set_theta_direction(-1)
ax.set_thetagrids(np.degrees(angles[:-1]), metrics)
plt.title('Classification Metrics Radar')
plt.legend(loc='upper right')

plt.tight_layout()
plt.savefig('training_analysis.png', dpi=300)
plt.show()

# 打印分类报告
print("\nClassification Report:")
print(classification_report(all_targets, all_preds, target_names=['Normal', 'Fatigue']))