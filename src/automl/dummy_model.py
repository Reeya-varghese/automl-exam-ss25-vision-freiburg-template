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


def get_resnet50(num_classes, num_layers_to_freeze=0, grayscale=True):
    """Mimics the Keras ResNet50 + Dense head architecture."""

    # Load pretrained resnet50 without the top FC layer
    base_model = models.resnet50(pretrained=True)

    if grayscale:
        base_model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

    # Remove the top layer (fc) and use the output of avgpool (2048 features)
    features_dim = base_model.fc.in_features
    base_model.fc = nn.Identity()  # remove FC layer

    # Build a classifier head similar to your Keras setup
    classifier = nn.Sequential(
        nn.Flatten(),
        nn.BatchNorm1d(features_dim),
        nn.Linear(features_dim, 2048),
        nn.ReLU(),
        nn.BatchNorm1d(2048),
        nn.Linear(2048, 1024),
        nn.ReLU(),
        nn.BatchNorm1d(1024),
        nn.Linear(1024, num_classes)
    )

    # Final model: ResNet50 backbone + custom classifier
    model = nn.Sequential(
        base_model,
        classifier
    )

    # Optionally freeze layers in base model
    model[0] = freeze_layers(model[0], num_layers_to_freeze)

    return model