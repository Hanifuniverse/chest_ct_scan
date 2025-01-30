# chest_ct_scan

# Chest CT Scan Classification using ResNet18

This project implements a **deep learning-based image classification model** using **ResNet18** for classifying chest CT scan images.

## Dataset
The dataset is stored in Google Drive and is structured as follows:
```
/content/drive/MyDrive/chest-ctscan-images/Data/
    train/
    test/
    valid/
```

## Dependencies
```python
!pip install torch torchvision torchinfo
```

## Mount Google Drive
```python
from google.colab import drive
drive.mount('/content/drive')
```

## Import Libraries
```python
import torch
from torch import nn
import numpy as np
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.models import resnet18, ResNet18_Weights
```

## Load Pretrained ResNet18 Model
```python
resnet18_model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
resnet18_model.to(device)
```

## Data Preprocessing
```python
data_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.7),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
])

# Load datasets
dataset_root = "/content/drive/MyDrive/chest-ctscan-images/Data"
train_data = datasets.ImageFolder(root=f"{dataset_root}/train", transform=data_transform)
test_data = datasets.ImageFolder(root=f"{dataset_root}/test", transform=data_transform)
validation_data = datasets.ImageFolder(root=f"{dataset_root}/valid", transform=data_transform)
```

## Modify Class Names
```python
class_dict = train_data.class_to_idx
# Update class names
class_dict['adenocarcinoma_left.lower.lobe'] = class_dict.pop('adenocarcinoma_left.lower.lobe_T2_N0_M0_Ib')
class_dict['large.cell.carcinoma_left.hilum'] = class_dict.pop('large.cell.carcinoma_left.hilum_T2_N2_M0_IIIa')
class_dict['squamous.cell.carcinoma_left.hilum'] = class_dict.pop('squamous.cell.carcinoma_left.hilum_T1_N2_M0_IIIa')
```

## DataLoader
```python
BATCH_SIZE = 32
train_dataloader = DataLoader(dataset=train_data, batch_size=BATCH_SIZE, num_workers=1, shuffle=True)
test_dataloader = DataLoader(dataset=test_data, batch_size=BATCH_SIZE, num_workers=1, shuffle=False)
```

## Modify Model Architecture
```python
resnet18_model.fc = nn.Sequential(
    nn.Linear(512, 512),
    nn.Dropout(0.2),
    nn.Linear(512, 256),
    nn.Linear(256, len(train_data.classes))
)
```

## Training Functions
```python
def train_step(model, dataloader, loss_fn, optimizer, device):
    model.train()
    train_loss, train_acc = 0, 0
    for X, y in dataloader:
        X, y = X.to(device), y.to(device)
        y_pred = model(X)
        loss = loss_fn(y_pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        train_acc += (y_pred.argmax(1) == y).sum().item() / len(y_pred)
    return train_loss / len(dataloader), train_acc / len(dataloader)
```

## Train the Model
```python
NUM_EPOCHS = 15
optimizer = torch.optim.Adam(resnet18_model.parameters(), lr=0.01)
loss_fn = nn.CrossEntropyLoss()

def train(model, train_dataloader, test_dataloader, optimizer, loss_fn, epochs):
    for epoch in range(epochs):
        train_loss, train_acc = train_step(model, train_dataloader, loss_fn, optimizer, device)
        print(f"Epoch {epoch+1}: Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
    return model

trained_model = train(resnet18_model, train_dataloader, test_dataloader, optimizer, loss_fn, NUM_EPOCHS)
```

## Save the Model
```python
torch.save(trained_model.state_dict(), "chest-ctscan_model.pth")
```

## Prediction Function
```python
def predict_image(model, class_names):
    from google.colab import files
    uploaded = files.upload()
    image_path = list(uploaded.keys())[0]
    img = Image.open(image_path).convert("RGB")
    img_tensor = data_transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(img_tensor)
        _, predicted = torch.max(outputs, 1)
    return class_names[predicted.item()]
```

## Model Evaluation
```python
def calculate_accuracy(dataloader, model):
    correct, total = 0, 0
    model.eval()
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    print(f'Accuracy: {accuracy:.2f}%')
    return accuracy

calculate_accuracy(test_dataloader, trained_model)
