import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from copy import deepcopy
from model import get_backbone_loader, GrayscaleToRGBAdapter

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def wrap_backbone(name, model):
    if any(k in name for k in ["vit", "swin", "convnext", "efficientnet"]):
        class Wrapper(nn.Module):
            def __init__(self, m):
                super().__init__()
                self.model = m
            def forward(self, x):
                return self.model.forward_features(x)
        return Wrapper(model)
    return model
class ZeroCostCandidateGenerator:
 
    
    def __init__(self, real_input, real_target, num_candidates=100, top_k=10, num_classes=7):
        self.real_input = real_input.to(DEVICE)
        self.real_target = real_target.to(DEVICE)
        self.num_candidates = num_candidates
        self.top_k = top_k
        self.num_classes = num_classes
        self.device = DEVICE
        grayscale = self.real_input.shape[1] == 1

        self.BACKBONE_NAMES = [
        "resnet18",
        "efficientnet_b0",
        "vit_base_patch16_224",
        "swin_tiny_patch4_window7_224",
        "convnext_tiny"
]
    
    
        self.backbones = {
            name: wrap_backbone(name, get_backbone_loader(name)(grayscale=False).to(self.device).eval())
            for name in self.BACKBONE_NAMES
        }
        if grayscale:
            self.rgb_adapter = GrayscaleToRGBAdapter().to(self.device)
        else:
            self.rgb_adapter = None
    
    
    # ------------------ Scoring Functions ------------------ #
    def get_jacobian_score(self, model, input_tensor):
        model.eval()
        input_tensor = input_tensor.clone().detach().requires_grad_(True)
        output = model(input_tensor)
        jacobian = torch.autograd.grad(outputs=output.sum(), inputs=input_tensor, create_graph=True)[0]
        return jacobian.norm().item()

    def get_gradnorm_score(self, model, input_tensor, target_tensor):
        model.train()
        model.zero_grad()
        output = model(input_tensor)
        loss = F.cross_entropy(output, target_tensor)
        loss.backward()
        return sum(p.grad.norm().item() for p in model.parameters() if p.grad is not None)

    def generate_random_head(self, input_dim):
        hidden_dim = random.choice([
            [1024, 512],         # 2-layer MLP
            [2048, 1024, 512],   # 3-layer MLP
            [2048, 1024]         # simplified but deep
        ])
        dropout = random.choice([0.0, 0.1, 0.2])
        use_bn = random.choice([True, False])
        activation = nn.GELU() if random.random() < 0.5 else nn.ReLU()

        layers = [nn.Flatten()]
        prev_dim = input_dim

        for hidden_dim in hidden_dim:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            if use_bn:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(activation)
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim

    # Final classification layer
        layers.append(nn.Linear(prev_dim, self.num_classes))

        return nn.Sequential(*layers)

    def normalize(self, score_list):
        min_val, max_val = min(score_list), max(score_list)
        return [(s - min_val) / (max_val - min_val + 1e-8) for s in score_list]
    
    def extract_features(self, backbone, backbone_name, input_tensor):
        with torch.no_grad():
            x = backbone(input_tensor)

        # Handle common output shapes
            if x.ndim == 4:  # e.g., [B, C, H, W]
                return F.adaptive_avg_pool2d(x, 1).reshape(x.size(0), -1)
            elif x.ndim == 3:  # e.g., ViT with [B, Tokens, D]
                return x.mean(dim=1)
            elif x.ndim == 2:  # already flattened features
                return x
            else:
                raise ValueError(f"Unsupported feature shape from {backbone_name}: {x.shape}")


    # ------------------ Feature Dimension Extraction ------------------ #
    def get_feature_dim(self, model, backbone_name):
        model.eval()
        in_channels = 1 if self.rgb_adapter else 3
        dummy_input = torch.randn(1, in_channels, 224, 224).to(self.device)
        if self.rgb_adapter:
            dummy_input = self.rgb_adapter(dummy_input)
        feats = self.extract_features(model, backbone_name, dummy_input)
        return feats.shape[-1]

    # ------------------ Candidate Generation ------------------ #
    
    def get_top_k_candidates(self):
        candidates = []
    
        for i in range(self.num_candidates):
            backbone_name = random.choice(self.BACKBONE_NAMES)

            input_tensor = self.real_input
            if self.rgb_adapter:
                input_tensor = self.rgb_adapter(input_tensor)


            backbone = self.backbones[backbone_name]
            feat_dim = self.get_feature_dim(backbone, backbone_name)

            head = self.generate_random_head(feat_dim).to(self.device)

            feats = self.extract_features(backbone, backbone_name, input_tensor)
            feats = feats.to(self.device)

            jac = self.get_jacobian_score(head, feats)
            grad = self.get_gradnorm_score(head, feats, self.real_target)
            
            candidates.append({
                "backbone": backbone_name,
                "head": deepcopy(head),
                "jacobian_score": jac,
                "gradnorm_score": grad,
                "id": f"{backbone_name}_{i}"
            })

        # Normalize and score
        jac_norm = self.normalize([c["jacobian_score"] for c in candidates])
        grad_norm = self.normalize([c["gradnorm_score"] for c in candidates])
        for i, c in enumerate(candidates):
            c["combined_score"] = jac_norm[i] +  grad_norm[i]

        ranked = sorted(candidates, key=lambda x: x["combined_score"], reverse=True)
        return ranked[:self.top_k] 