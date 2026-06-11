from __future__ import annotations

from types import SimpleNamespace

from stable_pretraining.backbone.utils import vit_hf
from stable_worldmodel.wm.lewm import LeWM
from stable_worldmodel.wm.lewm.module import Embedder, MLP, Predictor
from torch import nn

from les1mple.data import action_dim
from les1mple.models.encoders import ChunkedEncoder, TimmEncoder, TinyEncoder
from les1mple.models.predictors import OfficialMamba3Predictor


def build_lewm_model(args: SimpleNamespace) -> LeWM:
    return LeWM(
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
    )


def build_encoder(args: SimpleNamespace) -> nn.Module:
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


def build_predictor(args: SimpleNamespace) -> nn.Module:
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
