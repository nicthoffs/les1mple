from __future__ import annotations

import argparse
from types import SimpleNamespace

import torch
from einops import rearrange
from stable_worldmodel.wm.lewm import LeWM
from stable_worldmodel.wm.lewm.module import Embedder, MLP, Predictor
from torch import nn

from les1mple.data import action_dim
from les1mple.models.encoders import ChunkedEncoder, TimmEncoder, TinyEncoder, vit_hf
from les1mple.models.hwm import HierarchicalLeWM, build_hierarchical_lewm
from les1mple.models.predictors import OfficialMamba3Predictor


class NormalizingLeWM(LeWM):
    def __init__(self, *args, channels_last: bool = False, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.channels_last = channels_last

    def encode(self, info):
        pixels = info["pixels"]
        if torch.is_tensor(pixels) and pixels.dtype == torch.uint8:
            pixels = pixels.float().div(255.0)
        pixels = pixels.to(next(self.encoder.parameters()).dtype)
        batch_size = pixels.size(0)
        pixels = rearrange(pixels, "b t ... -> (b t) ...")
        if self.channels_last:
            pixels = pixels.contiguous(memory_format=torch.channels_last)
        output = self.encoder(pixels, interpolate_pos_encoding=True)
        pixels_emb = output.last_hidden_state[:, 0]
        emb = self.projector(pixels_emb)

        info = dict(info)
        info["emb"] = rearrange(emb, "(b t) d -> b t d", b=batch_size)
        if "action" in info:
            info["act_emb"] = self.action_encoder(info["action"])
        return info


def build_lewm_model(args: argparse.Namespace | SimpleNamespace) -> LeWM:
    return NormalizingLeWM(
        encoder=build_encoder(args),
        action_encoder=Embedder(
            input_dim=action_dim(),
            smoothed_dim=args.action_smoothed_dim,
            emb_dim=args.emb_dim,
        ),
        predictor=build_predictor(args),
        projector=MLP(
            input_dim=args.emb_dim,
            output_dim=args.emb_dim,
            hidden_dim=args.projection_hidden_dim,
            norm_fn=nn.BatchNorm1d,
        ),
        pred_proj=MLP(
            input_dim=args.emb_dim,
            output_dim=args.emb_dim,
            hidden_dim=args.projection_hidden_dim,
            norm_fn=nn.BatchNorm1d,
        ),
        channels_last=getattr(args, "channels_last", False),
    )


def build_hierarchical_lewm_model(
    args: argparse.Namespace | SimpleNamespace,
    *,
    low: LeWM | None = None,
) -> HierarchicalLeWM:
    if low is None:
        low = build_lewm_model(args)
    return build_hierarchical_lewm(
        low=low,
        action_dim=action_dim(),
        emb_dim=args.emb_dim,
        high_context_size=getattr(args, "high_context_size", args.history_size),
        action_smoothed_dim=getattr(
            args, "macro_action_smoothed_dim", args.action_smoothed_dim
        ),
        predictor_depth=getattr(args, "high_predictor_depth", args.predictor_depth),
        predictor_heads=getattr(args, "high_predictor_heads", args.predictor_heads),
        predictor_mlp_dim=getattr(
            args, "high_predictor_mlp_dim", args.predictor_mlp_dim
        ),
        predictor_dim_head=getattr(
            args, "high_predictor_dim_head", args.predictor_dim_head
        ),
        predictor_dropout=getattr(
            args, "high_predictor_dropout", args.predictor_dropout
        ),
        predictor_emb_dropout=getattr(
            args, "high_predictor_emb_dropout", args.predictor_emb_dropout
        ),
        projection_hidden_dim=getattr(
            args, "high_projection_hidden_dim", args.projection_hidden_dim
        ),
        macro_pooling=getattr(args, "macro_action_pooling", "mean"),
        freeze_low=getattr(args, "freeze_low", True),
    )


def build_encoder(args: argparse.Namespace | SimpleNamespace) -> nn.Module:
    encoder: nn.Module
    if args.encoder == "vit-hf":
        if args.emb_dim != _vit_hidden_size(args.vit_size):
            raise SystemExit(
                "--emb-dim must match the upstream HF ViT hidden size for "
                f"--vit-size {args.vit_size!r}; got {args.emb_dim}, "
                f"expected {_vit_hidden_size(args.vit_size)}"
            )
        encoder = vit_hf(
            size=args.vit_size,
            patch_size=args.patch_size,
            image_size=args.img_size,
            pretrained=False,
            use_mask_token=False,
        )
    elif args.encoder == "tiny-cnn":
        encoder = TinyEncoder(args.emb_dim)
    else:
        encoder = TimmEncoder(args.timm_model, args.emb_dim)

    if args.encoder_chunk_size < 0:
        raise SystemExit("--encoder-chunk-size must be non-negative")
    if args.checkpoint_encoder and args.encoder_chunk_size == 0:
        raise SystemExit("--checkpoint-encoder requires --encoder-chunk-size > 0")
    if args.encoder_chunk_size:
        encoder = ChunkedEncoder(
            encoder,
            chunk_size=args.encoder_chunk_size,
            checkpoint_encoder=args.checkpoint_encoder,
        )
    return encoder


def build_predictor(args: argparse.Namespace | SimpleNamespace) -> nn.Module:
    if args.predictor_type == "transformer":
        return Predictor(
            num_frames=args.history_size,
            depth=args.predictor_depth,
            heads=args.predictor_heads,
            dim_head=args.predictor_dim_head,
            mlp_dim=args.predictor_mlp_dim,
            input_dim=args.emb_dim,
            hidden_dim=args.emb_dim,
            output_dim=args.emb_dim,
            dropout=args.predictor_dropout,
            emb_dropout=args.predictor_emb_dropout,
        )
    return OfficialMamba3Predictor(
        num_frames=args.history_size,
        input_dim=args.emb_dim,
        action_dim=args.emb_dim,
        hidden_dim=args.emb_dim,
        output_dim=args.emb_dim,
        depth=args.predictor_depth,
        d_state=args.mamba_d_state,
        headdim=args.mamba_headdim,
    )


def _vit_hidden_size(size: str) -> int:
    hidden_sizes = {
        "tiny": 192,
        "small": 384,
        "base": 768,
        "large": 1024,
        "huge": 1280,
    }
    if size not in hidden_sizes:
        raise SystemExit(f"unknown --vit-size {size!r}")
    return hidden_sizes[size]
