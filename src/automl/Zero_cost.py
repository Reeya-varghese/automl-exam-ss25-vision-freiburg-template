import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from copy import deepcopy
from model import get_backbone_loader, GrayscaleToRGBAdapter

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def wrap_backbone(name, model):
    if name.startswith("resnet"):
        model.fc = nn.Identity()
    elif "efficientnet" in name:
        model.classifier = nn.Identity()
    elif "vit" in name:
        model.head = nn.Identity()
    elif "swin" in name or "convnext" in name:
        model.head = nn.Identity()  # 🔧 This line was missing

    class Wrapper(nn.Module):
        def __init__(self, m):
            super().__init__()
            self.model = m

        def forward(self, x):
            if hasattr(self.model, "forward_features"):
                x = self.model.forward_features(x)
            else:
                x = self.model(x)

            if x.ndim == 4:  # [B, C, H, W]
                x = F.adaptive_avg_pool2d(x, 1).view(x.size(0), -1)
            elif x.ndim == 3:  # [B, N, D] from ViT/Swin/ConvNeXt
                x = x.mean(dim=1)
            elif x.ndim == 2:
                pass  # already flattened
            else:
                raise ValueError(f"Unexpected shape from model: {x.shape}")
            return x

    return Wrapper(model)




def has_adapter(model):
    return any(isinstance(m, GrayscaleToRGBAdapter) for m in model.modules())


class ZeroCostCandidateGenerator:
    def __init__(self, real_input, real_target, num_candidates=100, top_k=10, num_classes=7):
        self.real_input = real_input.to(DEVICE)
        self.real_target = real_target.to(DEVICE)
        self.num_candidates = num_candidates
        self.top_k = top_k
        self.num_classes = num_classes
        self.device = DEVICE
        self.grayscale = self.real_input.shape[1] == 1

        self.BACKBONE_NAMES = [
            "resnet18", "efficientnet_b0", "vit_base_patch16_224",
            "swin_tiny_patch4_window7_224", "convnext_tiny"
        ]

        self.backbones = {
            name: wrap_backbone(name, get_backbone_loader(name)(grayscale=False).to(self.device).eval())
            for name in self.BACKBONE_NAMES
        }

        self.rgb_adapter = GrayscaleToRGBAdapter().to(self.device) if self.grayscale else None

    def adapt_input(self, x):
        return self.rgb_adapter(x) if self.rgb_adapter else x

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
            [1024, 512], [2048, 1024, 512], [2048, 1024]
        ])
        dropout = random.choice([0.0, 0.1, 0.2])
        use_bn = random.choice([True, False])
        activation = nn.GELU() if random.random() < 0.5 else nn.ReLU()

        layers = [nn.Flatten()]
        prev_dim = input_dim

        for hidden in hidden_dim:
            layers.append(nn.Linear(prev_dim, hidden))
            if use_bn:
                layers.append(nn.BatchNorm1d(hidden))
            layers.append(activation)
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev_dim = hidden

        layers.append(nn.Linear(prev_dim, self.num_classes))
        return nn.Sequential(*layers)

    def normalize(self, scores):
        min_val, max_val = min(scores), max(scores)
        return [(s - min_val) / (max_val - min_val + 1e-8) for s in scores]

    def get_feature_dim(self, model, backbone_name):
        dummy = torch.randn(1, 1 if self.grayscale else 3, 224, 224).to(self.device)
        dummy = self.adapt_input(dummy) if not has_adapter(model) else dummy
        with torch.no_grad():
            return model(dummy).shape[1]

    def get_top_k_candidates(self):
        candidates = []

        for i in range(self.num_candidates):
            backbone_name = random.choice(self.BACKBONE_NAMES)
            backbone = self.backbones[backbone_name]
            input_tensor = self.adapt_input(self.real_input) if not has_adapter(backbone) else self.real_input

            feat_dim = self.get_feature_dim(backbone, backbone_name)
            feats = backbone(input_tensor)

            print(f"[DEBUG] Feature shape for {backbone_name}: {feats.shape}")

            head = self.generate_random_head(feat_dim).to(self.device)
            jac = self.get_jacobian_score(head, feats)
            grad = self.get_gradnorm_score(head, feats, self.real_target)

            candidates.append({
                "backbone": backbone_name,
                "head": deepcopy(head),
                "jacobian_score": jac,
                "gradnorm_score": grad,
                "id": f"{backbone_name}_{i}"
            })

        # Normalize and rank
        jac_norm = self.normalize([c["jacobian_score"] for c in candidates])
        grad_norm = self.normalize([c["gradnorm_score"] for c in candidates])

        for i, c in enumerate(candidates):
            c["combined_score"] = jac_norm[i] + grad_norm[i]

        ranked = sorted(candidates, key=lambda x: x["combined_score"], reverse=True)
        return ranked[:self.top_k]
