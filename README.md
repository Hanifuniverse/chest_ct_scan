# Chest CT Scan Classification with ResNet18

## Overview
This project implements a deep learning model using ResNet18 for classifying chest CT scans to detect cancerous conditions. The dataset is loaded from Google Drive and preprocessed using PyTorch's torchvision library. The model is trained, evaluated, and used for predictions.

## Requirements
- Python
- PyTorch
- Google Colab (for training)
- Google Drive (to store dataset)
- Torchvision
- PIL (Python Imaging Library)
- Matplotlib

## Dataset
The dataset consists of labeled CT scan images stored in the following directory structure:
```
Data/
    train/
    test/
    valid/
```

## Model Architecture
The model is based on **ResNet18**, with modifications to the fully connected (fc) layer to match the number of classes in the dataset.

```python
import torch
from torch import nn
from torchvision.models import resnet18
from torchvision.models import ResNet18_Weights

# Load pre-trained ResNet18 model
resnet18_model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
resnet18_model.fc = nn.Sequential(
    nn.Linear(512, 512),
    nn.Dropout(0.2),
    nn.Linear(512, 256),
    nn.Linear(256, len(class_names))
)
```

## Training
The model is trained using the following settings:
- **Optimizer**: SGD with momentum
- **Loss Function**: CrossEntropyLoss
- **Batch Size**: 32
- **Epochs**: 15

```python
from torch.optim import lr_scheduler
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(resnet18_model.parameters(), lr=0.01, momentum=0.9)
```

## Prediction
The trained model can be used to classify uploaded CT scan images. Below is a sample result:

![CT Scan Prediction](image.png)

**Predicted Label:** squamous.cell.carcinoma_left.hilum  
**Diagnosis:** Positive for Cancer  
**Accuracy:** 82.22%

## Usage
To run predictions on new images:
```python
predicted_label, diagnosis = predict_image(model, class_names)
print(f"Predicted label: {predicted_label}")
print(f"Diagnosis: {diagnosis}")
```
To evaluate model accuracy on test data:
```python
calculate_accuracy(test_dataloader, model)
```

## Conclusion
This project demonstrates how to use deep learning for medical image classification. The model achieved an accuracy of **82.22%** on the test dataset. Further improvements can be made by using data augmentation, hyperparameter tuning, and more advanced architectures.

---
**Author:** Muhammad Hanif  
**Project:** Chest CT Scan Classification  
**Tools:** PyTorch, torchvision, Google Colab

