from __future__ import annotations

from types import SimpleNamespace

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint


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
