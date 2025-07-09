# src/automl/model.py

import torch.nn as nn
from torchvision import models

def freeze_layers(model,num_layers_to_freeze:int):
   
    layer_count = 0
    for child in model.children():
        for param in child.parameters():
            if layer_count < num_layers_to_freeze:
                param.requires_grad = False
                layer_count += 1
            else:
                return model  # ✅ safe exit once limit reached
    return model  # ✅ return model in all cases


def get_resnet50(num_classes, num_layers_to_freeze: int = 10, grayscale=True):
    """Returns a ResNet50 model, adapted for grayscale input and number of classes."""
    model = models.resnet50(pretrained=True)

    if grayscale:
        # Adjust first conv layer for 1-channel input
        model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

    # Replace final classifier
    model.fc = nn.Linear(model.fc.in_features, num_classes)

    # Freeze specified layers
    model = freeze_layers(model, num_layers_to_freeze)

    return model
