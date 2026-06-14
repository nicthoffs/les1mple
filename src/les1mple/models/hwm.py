from __future__ import annotations

from collections.abc import Mapping

import torch
from stable_worldmodel.wm.lewm import LeWM
from stable_worldmodel.wm.lewm.module import Embedder, MLP, Predictor
from torch import nn


class MacroActionEncoder(nn.Module):
    """Encode primitive action chunks into one latent macro-action per chunk."""

    def __init__(
        self,
        *,
        input_dim: int,
        smoothed_dim: int,
        emb_dim: int,
        pooling: str = "mean",
    ) -> None:
        super().__init__()
        if pooling not in {"mean", "last"}:
            raise ValueError(f"unknown macro-action pooling {pooling!r}")
        self.primitive_encoder = Embedder(
            input_dim=input_dim,
            smoothed_dim=smoothed_dim,
            emb_dim=emb_dim,
        )
        self.pooling = pooling

    def forward(self, action_chunks: torch.Tensor) -> torch.Tensor:
        """Return macro-actions shaped ``B,N,D`` from chunks shaped ``B,N,K,A``."""

        if action_chunks.ndim != 4:
            raise ValueError(
                "action_chunks must be shaped B,N,K,A; "
                f"got {tuple(action_chunks.shape)}"
            )
        batch_size, num_chunks, chunk_size, action_dim = action_chunks.shape
        actions = action_chunks.reshape(batch_size * num_chunks, chunk_size, action_dim)
        encoded = self.primitive_encoder(actions)
        if self.pooling == "mean":
            macro = encoded.mean(dim=1)
        else:
            macro = encoded[:, -1]
        return macro.reshape(batch_size, num_chunks, -1)


class HierarchicalLeWM(nn.Module):
    """High-level latent world model wrapped around a frozen low-level ``LeWM``."""

    def __init__(
        self,
        *,
        low: LeWM,
        macro_action_encoder: nn.Module,
        high_predictor: nn.Module,
        high_pred_proj: nn.Module | None = None,
        freeze_low: bool = True,
    ) -> None:
        super().__init__()
        self.low = low
        self.macro_action_encoder = macro_action_encoder
        self.high_predictor = high_predictor
        self.high_pred_proj = high_pred_proj or nn.Identity()
        self.freeze_low = freeze_low
        if freeze_low:
            self.freeze_low_model()

    def freeze_low_model(self) -> None:
        for parameter in self.low.parameters():
            parameter.requires_grad_(False)
        self.low.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze_low:
            self.low.eval()
        return self

    @torch.no_grad()
    def encode_waypoints(
        self,
        pixels: torch.Tensor,
        action: torch.Tensor | None = None,
        extra: Mapping[str, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        info = {"pixels": pixels}
        if action is not None:
            info["action"] = action
        if extra:
            info.update(extra)
        return self.low.encode(info)["emb"].detach()

    def predict_high(
        self,
        z: torch.Tensor,
        action_chunks: torch.Tensor,
    ) -> torch.Tensor:
        max_steps = getattr(self.high_predictor, "num_frames", None)
        if max_steps is not None and z.shape[1] > max_steps:
            raise ValueError(
                "high-level context is longer than the high predictor was built for: "
                f"context={z.shape[1]} max={max_steps}"
            )
        macro = self.macro_action_encoder(action_chunks)
        preds = self.high_predictor(z, macro)
        batch_size, num_steps, emb_dim = preds.shape
        preds = preds.reshape(batch_size * num_steps, emb_dim)
        preds = self.high_pred_proj(preds)
        return preds.reshape(batch_size, num_steps, -1)

    def forward(self, z: torch.Tensor, action_chunks: torch.Tensor) -> torch.Tensor:
        return self.predict_high(z, action_chunks)


def build_hierarchical_lewm(
    *,
    low: LeWM,
    action_dim: int,
    emb_dim: int,
    high_context_size: int,
    action_smoothed_dim: int,
    predictor_depth: int,
    predictor_heads: int,
    predictor_mlp_dim: int,
    predictor_dim_head: int,
    predictor_dropout: float = 0.1,
    predictor_emb_dropout: float = 0.0,
    projection_hidden_dim: int = 2048,
    macro_pooling: str = "mean",
    freeze_low: bool = True,
) -> HierarchicalLeWM:
    return HierarchicalLeWM(
        low=low,
        macro_action_encoder=MacroActionEncoder(
            input_dim=action_dim,
            smoothed_dim=action_smoothed_dim,
            emb_dim=emb_dim,
            pooling=macro_pooling,
        ),
        high_predictor=Predictor(
            num_frames=high_context_size,
            depth=predictor_depth,
            heads=predictor_heads,
            dim_head=predictor_dim_head,
            mlp_dim=predictor_mlp_dim,
            input_dim=emb_dim,
            hidden_dim=emb_dim,
            output_dim=emb_dim,
            dropout=predictor_dropout,
            emb_dropout=predictor_emb_dropout,
        ),
        high_pred_proj=MLP(
            input_dim=emb_dim,
            output_dim=emb_dim,
            hidden_dim=projection_hidden_dim,
            norm_fn=nn.BatchNorm1d,
        ),
        freeze_low=freeze_low,
    )
