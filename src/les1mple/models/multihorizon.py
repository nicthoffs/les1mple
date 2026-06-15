from __future__ import annotations

from collections.abc import Mapping

import torch
from stable_worldmodel.wm.lewm import LeWM
from stable_worldmodel.wm.lewm.module import Predictor
from torch import nn


class MeanPoolActionWindowEncoder(nn.Module):
    """Encode per-position action windows with a per-action MLP and mean pool."""

    def __init__(
        self,
        *,
        input_dim: int,
        emb_dim: int,
        hidden_dim: int | None = None,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        hidden_dim = hidden_dim or emb_dim * 2
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, emb_dim),
        )

    def forward(self, action_windows: torch.Tensor) -> torch.Tensor:
        """Return conditions shaped ``B,T,D`` from windows shaped ``B,T,H,A``."""

        if action_windows.ndim != 4:
            raise ValueError(
                "action_windows must be shaped B,T,H,A; "
                f"got {tuple(action_windows.shape)}"
            )
        if action_windows.shape[2] < 1:
            raise ValueError("action_windows must have a non-empty horizon dimension")
        return self.net(action_windows.float()).mean(dim=2)


class TransformerActionWindowEncoder(nn.Module):
    """Encode ordered per-position action windows with a small transformer."""

    def __init__(
        self,
        *,
        input_dim: int,
        emb_dim: int,
        max_horizon: int,
        depth: int = 2,
        heads: int = 4,
        mlp_dim: int | None = None,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if max_horizon < 1:
            raise ValueError("max_horizon must be positive")
        if depth < 1:
            raise ValueError("depth must be positive")
        if heads < 1:
            raise ValueError("heads must be positive")
        if emb_dim % heads != 0:
            raise ValueError(
                f"emb_dim={emb_dim} must be divisible by heads={heads}"
            )
        self.max_horizon = max_horizon
        self.input_proj = nn.Linear(input_dim, emb_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, emb_dim))
        self.pos_embedding = nn.Parameter(
            torch.randn(1, max_horizon + 1, emb_dim) * 0.02
        )
        layer = nn.TransformerEncoderLayer(
            d_model=emb_dim,
            nhead=heads,
            dim_feedforward=mlp_dim or emb_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=depth)
        self.output_proj = nn.Sequential(
            nn.LayerNorm(emb_dim),
            nn.Linear(emb_dim, emb_dim),
        )

    def forward(self, action_windows: torch.Tensor) -> torch.Tensor:
        """Return conditions shaped ``B,T,D`` from windows shaped ``B,T,H,A``."""

        if action_windows.ndim != 4:
            raise ValueError(
                "action_windows must be shaped B,T,H,A; "
                f"got {tuple(action_windows.shape)}"
            )
        batch_size, num_positions, horizon, action_dim = action_windows.shape
        if horizon < 1:
            raise ValueError("action_windows must have a non-empty horizon dimension")
        if horizon > self.max_horizon:
            raise ValueError(
                "action window is longer than this encoder supports: "
                f"got {horizon}, max_horizon={self.max_horizon}"
            )

        actions = action_windows.reshape(batch_size * num_positions, horizon, action_dim)
        tokens = self.input_proj(actions.float())
        cls = self.cls_token.expand(tokens.shape[0], -1, -1)
        tokens = torch.cat([cls, tokens], dim=1)
        tokens = tokens + self.pos_embedding[:, : tokens.shape[1]]
        encoded = self.transformer(tokens)
        condition = self.output_proj(encoded[:, 0])
        return condition.reshape(batch_size, num_positions, -1)


class CausalMultiHorizonLeWM(nn.Module):
    """Causal seq-to-seq multi-horizon latent predictor.

    For a fixed horizon ``h``, output position ``t`` predicts ``z[t+h]`` from
    causal latent prefix ``z[:t+1]`` and action window ``a[t:t+h]``.
    """

    def __init__(
        self,
        *,
        low: LeWM,
        action_encoder: nn.Module,
        predictor: Predictor,
        max_horizon: int,
        emb_dim: int,
        freeze_low: bool = True,
        freeze_low_unused: bool = True,
    ) -> None:
        super().__init__()
        if max_horizon < 1:
            raise ValueError("max_horizon must be positive")
        self.low = low
        self.action_encoder = action_encoder
        self.predictor = predictor
        self.horizon_embedding = nn.Embedding(max_horizon + 1, emb_dim)
        self.max_horizon = max_horizon
        self.freeze_low = freeze_low
        self.freeze_low_unused = freeze_low_unused
        if freeze_low:
            self.freeze_low_model()
        elif freeze_low_unused:
            self.freeze_low_unused_modules()

    def freeze_low_model(self) -> None:
        for parameter in self.low.parameters():
            parameter.requires_grad_(False)
        self.low.eval()

    def freeze_low_unused_modules(self) -> None:
        for module_name in ("action_encoder", "predictor", "pred_proj"):
            module = getattr(self.low, module_name, None)
            if module is None:
                continue
            for parameter in module.parameters():
                parameter.requires_grad_(False)
            module.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze_low:
            self.low.eval()
        elif self.freeze_low_unused:
            for module_name in ("action_encoder", "predictor", "pred_proj"):
                module = getattr(self.low, module_name, None)
                if module is not None:
                    module.eval()
        return self

    def encode_latents(
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
        if self.freeze_low:
            with torch.no_grad():
                return self.low.encode(info)["emb"].detach()
        return self.low.encode(info)["emb"]

    def predict_horizon(
        self,
        z_prefix: torch.Tensor,
        action_windows: torch.Tensor,
        horizon: int,
    ) -> torch.Tensor:
        if horizon < 1 or horizon > self.max_horizon:
            raise ValueError(
                f"horizon must be in [1, {self.max_horizon}], got {horizon}"
            )
        max_steps = getattr(self.predictor, "num_frames", None)
        if max_steps is not None and z_prefix.shape[1] > max_steps:
            raise ValueError(
                "context is longer than the causal predictor was built for: "
                f"context={z_prefix.shape[1]} max={max_steps}"
            )
        action_condition = self.action_encoder(action_windows)
        horizon_ids = torch.full(
            action_condition.shape[:2],
            horizon,
            dtype=torch.long,
            device=action_condition.device,
        )
        condition = action_condition + self.horizon_embedding(horizon_ids)
        return self.predictor(z_prefix, condition)

    def forward(
        self,
        z_prefix: torch.Tensor,
        action_windows: torch.Tensor,
        horizon: int,
    ) -> torch.Tensor:
        return self.predict_horizon(z_prefix, action_windows, horizon)


def build_causal_multi_horizon_lewm(
    *,
    low: LeWM,
    action_dim: int,
    emb_dim: int,
    num_frames: int,
    max_horizon: int,
    action_encoder_type: str = "mean-pool",
    action_hidden_dim: int | None = None,
    action_encoder_depth: int = 2,
    action_encoder_heads: int = 4,
    action_encoder_mlp_dim: int | None = None,
    action_dropout: float = 0.0,
    predictor_depth: int = 4,
    predictor_heads: int = 4,
    predictor_mlp_dim: int | None = None,
    predictor_dim_head: int = 64,
    predictor_dropout: float = 0.1,
    predictor_emb_dropout: float = 0.0,
    freeze_low: bool = True,
    freeze_low_unused: bool = True,
) -> CausalMultiHorizonLeWM:
    if action_encoder_type == "mean-pool":
        action_encoder = MeanPoolActionWindowEncoder(
            input_dim=action_dim,
            emb_dim=emb_dim,
            hidden_dim=action_hidden_dim,
            dropout=action_dropout,
        )
    elif action_encoder_type == "transformer":
        action_encoder = TransformerActionWindowEncoder(
            input_dim=action_dim,
            emb_dim=emb_dim,
            max_horizon=max_horizon,
            depth=action_encoder_depth,
            heads=action_encoder_heads,
            mlp_dim=action_encoder_mlp_dim,
            dropout=action_dropout,
        )
    else:
        raise ValueError(f"unknown action_encoder_type {action_encoder_type!r}")

    return CausalMultiHorizonLeWM(
        low=low,
        action_encoder=action_encoder,
        predictor=Predictor(
            num_frames=num_frames,
            depth=predictor_depth,
            heads=predictor_heads,
            dim_head=predictor_dim_head,
            mlp_dim=predictor_mlp_dim or emb_dim * 4,
            input_dim=emb_dim,
            hidden_dim=emb_dim,
            output_dim=emb_dim,
            dropout=predictor_dropout,
            emb_dropout=predictor_emb_dropout,
        ),
        max_horizon=max_horizon,
        emb_dim=emb_dim,
        freeze_low=freeze_low,
        freeze_low_unused=freeze_low_unused,
    )
