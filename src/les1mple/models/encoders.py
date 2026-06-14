from __future__ import annotations

from types import SimpleNamespace

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint


def vit_hf(
    *,
    size: str,
    patch_size: int,
    image_size: int,
    pretrained: bool,
    use_mask_token: bool,
) -> nn.Module:
    from transformers import ViTConfig, ViTModel

    size_configs = {
        "tiny": {"hidden_size": 192, "num_hidden_layers": 12, "num_attention_heads": 3},
        "small": {"hidden_size": 384, "num_hidden_layers": 12, "num_attention_heads": 6},
        "base": {"hidden_size": 768, "num_hidden_layers": 12, "num_attention_heads": 12},
        "large": {
            "hidden_size": 1024,
            "num_hidden_layers": 24,
            "num_attention_heads": 16,
        },
        "huge": {
            "hidden_size": 1280,
            "num_hidden_layers": 32,
            "num_attention_heads": 16,
        },
    }
    if size not in size_configs:
        raise ValueError(f"unknown ViT size {size!r}")

    if pretrained:
        model_name = f"google/vit-{size}-patch{patch_size}-{image_size}"
        model = ViTModel.from_pretrained(
            model_name,
            add_pooling_layer=False,
            use_mask_token=use_mask_token,
        )
    else:
        config_params = dict(size_configs[size])
        config_params["intermediate_size"] = config_params["hidden_size"] * 4
        config_params["image_size"] = image_size
        config_params["patch_size"] = patch_size
        config = ViTConfig(**config_params)
        model = ViTModel(config, add_pooling_layer=False, use_mask_token=use_mask_token)

    model.config.interpolate_pos_encoding = True
    return model


class TinyEncoder(nn.Module):
    def __init__(self, emb_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=5, stride=4, padding=2),
            nn.GELU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(32, emb_dim),
        )

    def forward(
        self,
        pixels: torch.Tensor,
        interpolate_pos_encoding: bool = False,
    ) -> SimpleNamespace:
        del interpolate_pos_encoding
        return SimpleNamespace(last_hidden_state=self.net(pixels)[:, None, :])


class TimmEncoder(nn.Module):
    def __init__(self, model_name: str, emb_dim: int) -> None:
        super().__init__()
        import timm

        self.backbone = timm.create_model(model_name, pretrained=False, num_classes=0)
        feature_dim = self.backbone.num_features
        self.project = (
            nn.Identity() if feature_dim == emb_dim else nn.Linear(feature_dim, emb_dim)
        )

    def forward(
        self,
        pixels: torch.Tensor,
        interpolate_pos_encoding: bool = False,
    ) -> SimpleNamespace:
        del interpolate_pos_encoding
        return SimpleNamespace(
            last_hidden_state=self.project(self.backbone(pixels))[:, None, :]
        )


class ChunkedEncoder(nn.Module):
    def __init__(
        self,
        encoder: nn.Module,
        *,
        chunk_size: int,
        checkpoint_encoder: bool,
    ) -> None:
        super().__init__()
        if chunk_size < 1:
            raise ValueError("chunk_size must be positive")
        self.encoder = encoder
        self.chunk_size = chunk_size
        self.checkpoint_encoder = checkpoint_encoder

    def forward(
        self,
        pixels: torch.Tensor,
        interpolate_pos_encoding: bool = False,
    ) -> SimpleNamespace:
        chunks = pixels.split(self.chunk_size, dim=0)
        hidden_states = [
            self._encode_chunk(chunk, interpolate_pos_encoding=interpolate_pos_encoding)
            for chunk in chunks
        ]
        return SimpleNamespace(last_hidden_state=torch.cat(hidden_states, dim=0))

    def _encode_chunk(
        self,
        pixels: torch.Tensor,
        *,
        interpolate_pos_encoding: bool,
    ) -> torch.Tensor:
        if self.checkpoint_encoder and self.training and torch.is_grad_enabled():

            def encode(pixels_chunk: torch.Tensor) -> torch.Tensor:
                return self.encoder(
                    pixels_chunk,
                    interpolate_pos_encoding=interpolate_pos_encoding,
                ).last_hidden_state

            return checkpoint(encode, pixels, use_reentrant=False)

        return self.encoder(
            pixels,
            interpolate_pos_encoding=interpolate_pos_encoding,
        ).last_hidden_state
