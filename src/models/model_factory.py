import torch.nn as nn
from torchvision import models


def _replace_resnet18_head(model: nn.Module, num_classes: int) -> None:
    model.fc = nn.Linear(model.fc.in_features, num_classes)


def _replace_efficientnet_b0_head(model: nn.Module, num_classes: int) -> None:
    model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)


def _replace_convnext_base_head(model: nn.Module, num_classes: int) -> None:
    model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)


def _replace_vit_base16_head(model: nn.Module, num_classes: int) -> None:
    model.heads.head = nn.Linear(model.heads.head.in_features, num_classes)


def _replace_swin_base_head(model: nn.Module, num_classes: int) -> None:
    model.head = nn.Linear(model.head.in_features, num_classes)


MODEL_REGISTRY = {
    "resnet18": {
        "constructor": models.resnet18,
        "weights": models.ResNet18_Weights.IMAGENET1K_V1,
        "replace_head": _replace_resnet18_head,
    },
    "efficientnet_b0": {
        "constructor": models.efficientnet_b0,
        "weights": models.EfficientNet_B0_Weights.IMAGENET1K_V1,
        "replace_head": _replace_efficientnet_b0_head,
    },
    "convnext_base": {
        "constructor": models.convnext_base,
        "weights": models.ConvNeXt_Base_Weights.IMAGENET1K_V1,
        "replace_head": _replace_convnext_base_head,
    },
    "vit_base16": {
        "constructor": models.vit_b_16,
        "weights": models.ViT_B_16_Weights.IMAGENET1K_V1,
        "replace_head": _replace_vit_base16_head,
    },
    "swin_base": {
        "constructor": models.swin_b,
        "weights": models.Swin_B_Weights.IMAGENET1K_V1,
        "replace_head": _replace_swin_base_head,
    },
}


def create_model(
    model_name: str, num_classes: int, pretrained: bool = True
) -> nn.Module:
    if model_name not in MODEL_REGISTRY:
        supported = ", ".join(MODEL_REGISTRY.keys())
        raise ValueError(f"Unknown model '{model_name}'. Supported models: {supported}")

    entry = MODEL_REGISTRY[model_name]
    weights = entry["weights"] if pretrained else None
    model = entry["constructor"](weights=weights)
    entry["replace_head"](model, num_classes)
    return model
