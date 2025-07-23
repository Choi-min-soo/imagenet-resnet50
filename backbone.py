# backbone.py
from torchvision.models import resnet50, ResNet50_Weights
import torch.nn as nn

def build_backbone(cfg):
    model_cfg = cfg.get("model", {})
    weights = ResNet50_Weights.DEFAULT if model_cfg.get("pretrained", False) else None
    model = resnet50(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, model_cfg.get("num_classes", 1000))
    return model