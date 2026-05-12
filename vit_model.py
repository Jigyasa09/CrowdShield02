# vit_model.py
import torch
import torch.nn as nn
import timm

class ViTAnomalyModel(nn.Module):
    """
    ViT-Base fine-tuned for binary anomaly detection.
    Input:  (B, T, C, H, W) — batch of 16-frame clips
    Output: (B, 2) — logits for [Normal, Anomaly]
    """
    def __init__(self):
        super().__init__()
        self.backbone = timm.create_model(
            'vit_base_patch16_224',
            pretrained=False,   # False when loading saved weights
            num_classes=0
        )
        self.classifier = nn.Sequential(
            nn.LayerNorm(768),
            nn.Linear(768, 256),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(256, 2)
        )

    def forward(self, x):
        B, T, C, H, W = x.shape
        x        = x.view(B * T, C, H, W)
        features = self.backbone(x)           # (B*T, 768)
        features = features.view(B, T, -1)    # (B, T, 768)
        features = features.mean(dim=1)       # (B, 768) — mean pool over time
        return self.classifier(features)


def load_vit_model(weights_path: str, device: str = 'cpu'):
    """
    Load trained ViT anomaly model from weights file.

    Args:
        weights_path: path to best_model.pt
        device:       'cuda' or 'cpu'

    Returns:
        model (eval mode, ready for inference)
    """
    model = ViTAnomalyModel()
    state = torch.load(weights_path, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    print(f"✅ ViT model loaded from {weights_path}")
    return model