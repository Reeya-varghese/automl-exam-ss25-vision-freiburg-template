import torch
import torch.nn as nn
from torchvision import models, transforms
import timm


def get_device():
    return 'cuda' if torch.cuda.is_available() else 'cpu'

def get_transforms(mean, std, phase="train", backbone_name="resnet18"):
    is_transformer = any(k in backbone_name.lower() for k in ["vit", "swin", "convnext"])

    tf = [transforms.Resize((224, 224))]

    if phase == "train":
        tf += [
            transforms.RandomRotation(15),
            transforms.RandomHorizontalFlip()
        ]

    tf.append(transforms.ToTensor())  # convert PIL to tensor first

    if is_transformer:
        # Now it's a tensor, so .repeat works
        tf.append(transforms.Lambda(lambda x: x.repeat(3, 1, 1) if x.shape[0] == 1 else x))

    tf.append(transforms.Normalize(mean, std))

    return transforms.Compose(tf)



def load_resnet18(grayscale=False):
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    if grayscale:
        conv1 = model.conv1
        model.conv1 = nn.Conv2d(1, conv1.out_channels, conv1.kernel_size,
                                stride=conv1.stride, padding=conv1.padding, bias=False)
        with torch.no_grad():
            model.conv1.weight = nn.Parameter(conv1.weight.sum(dim=1, keepdim=True))
    return model


def load_efficientnet_b0(grayscale=False):
    model = timm.create_model('efficientnet_b0', pretrained=True)
    if grayscale:
        conv = model.conv_stem
        model.conv_stem = nn.Conv2d(1, conv.out_channels, conv.kernel_size,
                                    stride=conv.stride, padding=conv.padding, bias=False)
        with torch.no_grad():
            model.conv_stem.weight = nn.Parameter(conv.weight.sum(dim=1, keepdim=True))
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
    backbone = get_backbone_loader(backbone_name)(grayscale=grayscale)

    # Determine feature dimension based on backbone type
    if backbone_name.startswith("resnet"):
        features_dim = backbone.fc.in_features
        backbone.fc = nn.Identity()

    elif "efficientnet" in backbone_name:
        features_dim = backbone.classifier.in_features
        backbone.classifier = nn.Identity()

    elif "vit" in backbone_name:
        if grayscale:
            print(f"⚠️ Grayscale + ViT: input will be repeated to 3 channels")
        features_dim = backbone.head.in_features
        backbone.head = nn.Identity()

    elif "swin" in backbone_name or "convnext" in backbone_name:
        # Use dummy forward pass
      
        dummy_input = torch.randn(1, 3, 224, 224).to(device)
        backbone.eval()
        with torch.no_grad():
            dummy_output = backbone(dummy_input)
            features_dim = dummy_output.view(1, -1).shape[1]

    else:
        raise ValueError(f"Unsupported or unknown backbone: {backbone_name}")


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
    head = head.to(device)  # 
    model = nn.Sequential(backbone, head)
  
    return model

