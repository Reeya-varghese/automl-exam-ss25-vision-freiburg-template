import torch
import torch.nn as nn
from torchvision import models, transforms
import timm

class GrayscaleToRGBAdapter(nn.Module):
    def __init__(self):
        super().__init__()
        # 1x1 Conv to map 1-channel → 3-channel
        self.adapter = nn.Conv2d(1, 3, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        return self.adapter(x)

def get_device():
    return 'cuda' if torch.cuda.is_available() else 'cpu'

def get_transforms(mean, std, phase="train", backbone_name="resnet18"):
   
    tf = [transforms.Resize((224, 224))]

    if phase == "train":
        tf += [
            transforms.RandomRotation(15),
            transforms.RandomHorizontalFlip()
        ]

    tf.append(transforms.ToTensor())  # convert PIL to tensor first


    tf.append(transforms.Normalize(mean, std))

    return transforms.Compose(tf)



def load_resnet18(grayscale=False):
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    return model


def load_efficientnet_b0(grayscale=False):
    model = timm.create_model('efficientnet_b0', pretrained=True)

    return model


def load_vit(grayscale=False):
    model = timm.create_model('vit_base_patch16_224', pretrained=True)
    
    return model

def load_swin(grayscale=False):
        model = timm.create_model('swin_tiny_patch4_window7_224', pretrained=True)
        return model

def load_convnext(grayscale=False):
        model = timm.create_model('convnext_tiny', pretrained=True)
        return model

def get_backbone_loader(backbone_name):
    loaders = {
        "resnet18": load_resnet18,
        "efficientnet_b0": load_efficientnet_b0,
        "vit_base_patch16_224": load_vit,
        "swin_tiny_patch4_window7_224": load_swin,
        "convnext_tiny": load_convnext,
    }
    assert backbone_name in loaders, f"Unsupported backbone: {backbone_name}"
    return loaders[backbone_name]
def get_model(backbone_name, num_classes, grayscale=False, custom_head=None):
    device = get_device()

    # Load the base backbone as RGB
    base_backbone = get_backbone_loader(backbone_name)(grayscale=False).to(device)

    if grayscale:
        print(f"✅ Using trainable grayscale → RGB adapter for {backbone_name}")
        adapter = GrayscaleToRGBAdapter().to(device)
    else:
        adapter = nn.Identity()

    # Extract features_dim
    if backbone_name.startswith("resnet"):
        features_dim = base_backbone.fc.in_features
        base_backbone.fc = nn.Identity()
        backbone = base_backbone

    elif "efficientnet" in backbone_name:
        features_dim = base_backbone.classifier.in_features
        base_backbone.classifier = nn.Identity()
        backbone = base_backbone

    elif "vit" in backbone_name:
        features_dim = base_backbone.head.in_features
        base_backbone.head = nn.Identity()
        backbone = base_backbone

    elif "swin" in backbone_name or "convnext" in backbone_name:
        features_dim = base_backbone.num_features

        # ✅ wrap forward_features in a module
        class Wrapper(nn.Module):
            def __init__(self, model):
                super().__init__()
                self.model = model
            def forward(self, x):
                return self.model.forward_features(x)

        backbone = Wrapper(base_backbone)

    else:
        raise ValueError(f"Unsupported backbone: {backbone_name}")

    # Custom head
    if custom_head is None:
        head = nn.Sequential(
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
    else:
        head = custom_head

    # Assemble full model
    model = nn.Sequential(adapter, backbone, head).to(device)
    return model
