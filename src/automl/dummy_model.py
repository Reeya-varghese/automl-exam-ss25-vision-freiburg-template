import torch
import torch.nn as nn
from torchvision import models, transforms
#from autogluon.multimodal
import optuna
from optuna.samplers import TPESampler
from optuna.pruners import SuccessiveHalvingPruner


# --- Utility for device selection ---
def get_device():
    return 'cuda' if torch.cuda.is_available() else 'cpu'

# --- Augmentation and Normalization pipeline ---
def get_train_transforms(mean, std):
    return transforms.Compose([
        transforms.RandomRotation(degrees=15),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

def get_val_transforms(mean, std):
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

# --- Flexible ResNet Backbone ---
def get_resnet_model(backbone_name, num_classes, grayscale=False, num_layers_to_freeze=0):
    assert backbone_name in ["resnet18", "resnet50"], "Only resnet18 or resnet50 supported"
    base_model = getattr(models, backbone_name)(pretrained=True)
    if grayscale:
        base_model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    features_dim = base_model.fc.in_features
    base_model.fc = nn.Identity()
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
    model = nn.Sequential(base_model, classifier)
    # Optionally freeze layers
    model[0] = freeze_layers(model[0], num_layers_to_freeze)
    return model

# --- Freezing layers helper ---
def freeze_layers(model, num_layers_to_freeze: int):
    layer_count = 0
    for child in model.children():
        for param in child.parameters():
            if layer_count < num_layers_to_freeze:
                param.requires_grad = False
                layer_count += 1
            else:
                return model
    return model

# --- AutoGluon Training Wrapper ---
def get_autogluon_predictor(train_data, time_limit=3600, save_path="autogluon_predictor", 
                            batch_size=None, learning_rate=None, optimizer=None, 
                            epochs=None, backbone='resnet18'):
    """
    Trains and returns an AutoGluon ImagePredictor.
    train_data: pandas DataFrame with 'image' and 'label' columns.
    """
    hyperparameters = {
        "model": backbone,
        "batch_size": batch_size if batch_size else 32,
        "lr": learning_rate if learning_rate else 1e-3,
        "optimizer": optimizer if optimizer else "adam",
        "epochs": epochs if epochs else 10
    }
    predictor = MultiModalPredictor(path=save_path)
    predictor.fit(
        train_data=train_data,
        hyperparameters=hyperparameters,
        time_limit=time_limit,
        num_gpus=1 if torch.cuda.is_available() else 0
    )
    return predictor
