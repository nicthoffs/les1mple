from __future__ import annotations

import argparse
import csv
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from les1mple.data import LocalTensorSequenceDataset, SyntheticOpenCS2Dataset, action_dim
from les1mple.models import TimmEncoder, TinyEncoder
from stable_worldmodel.wm.lewm import LeWM
from stable_worldmodel.wm.lewm.module import Embedder, Predictor


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a tiny LeWM training smoke test.")
    parser.add_argument("--source", choices=("synthetic", "local"), default="local")
    parser.add_argument("--data-dir", default="/tmp/les1mple-real-opencs2")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--sequence-length", type=int, default=8)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--history-size", type=int, default=4)
    parser.add_argument("--num-preds", type=int, default=4)
    parser.add_argument("--emb-dim", type=int, default=64)
    parser.add_argument("--encoder", choices=("tiny-cnn", "timm"), default="tiny-cnn")
    parser.add_argument("--timm-model", default="vit_tiny_patch16_224")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--checkpoint-path", default=None)
    parser.add_argument("--loss-log", default=None)
    parser.add_argument("--log-every", type=int, default=10)
    args = parser.parse_args()

    if args.source == "synthetic":
        dataset = SyntheticOpenCS2Dataset(
            num_samples=max(args.batch_size, 8),
            sequence_length=args.sequence_length,
            image_size=args.image_size,
        )
    else:
        dataset = LocalTensorSequenceDataset(args.data_dir)

    if args.history_size + args.num_preds > args.sequence_length:
        raise SystemExit("history-size + num-preds must fit inside sequence-length")

    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
    try:
        batch = next(iter(loader))
    except StopIteration:
        raise SystemExit("dataset is too small for the requested batch size")

    if batch["pixels"].shape[1] < args.history_size + args.num_preds:
        raise SystemExit(
            f"batch sequence length {batch['pixels'].shape[1]} is too short for "
            f"history-size={args.history_size}, num-preds={args.num_preds}"
        )

    if args.encoder == "tiny-cnn":
        encoder = TinyEncoder(args.emb_dim)
    else:
        encoder = TimmEncoder(args.timm_model, args.emb_dim)

    model = LeWM(
        encoder=encoder,
        action_encoder=Embedder(
            input_dim=action_dim(),
            smoothed_dim=args.emb_dim,
            emb_dim=args.emb_dim,
        ),
        predictor=Predictor(
            num_frames=args.history_size,
            depth=2,
            heads=2,
            dim_head=32,
            mlp_dim=args.emb_dim * 4,
            input_dim=args.emb_dim,
            hidden_dim=args.emb_dim,
            output_dim=args.emb_dim,
        ),
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    log_file = None
    log_writer = None
    if args.loss_log:
        Path(args.loss_log).parent.mkdir(parents=True, exist_ok=True)
        log_file = open(args.loss_log, "w", newline="")
        log_writer = csv.DictWriter(log_file, fieldnames=("step", "loss"))
        log_writer.writeheader()

    iterator = iter(loader)
    last_loss = None
    for step in range(args.steps):
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            batch = next(iterator)

        batch["action"] = torch.nan_to_num(batch["action"], 0.0)

        output = model.encode(batch)
        emb = output["emb"]
        act_emb = output["act_emb"]

        ctx_emb = emb[:, : args.history_size]
        ctx_act = act_emb[:, : args.history_size]
        tgt_emb = emb[:, args.num_preds : args.num_preds + args.history_size].detach()
        pred_emb = model.predict(ctx_emb, ctx_act)

        loss = (pred_emb - tgt_emb).pow(2).mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        last_loss = loss.item()
        if log_writer:
            log_writer.writerow({"step": step + 1, "loss": last_loss})
        if step == 0 or (step + 1) % args.log_every == 0 or step + 1 == args.steps:
            print(f"step={step + 1} loss={last_loss:.6f}")

    if log_file:
        log_file.close()

    if args.checkpoint_path:
        Path(args.checkpoint_path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "args": vars(args),
                "last_loss": last_loss,
            },
            args.checkpoint_path,
        )
        print(f"checkpoint: {args.checkpoint_path}")

    print(f"pixels: {tuple(batch['pixels'].shape)}")
    print(f"action: {tuple(batch['action'].shape)}")
    print(f"embedding: {tuple(emb.shape)}")
    print(f"prediction: {tuple(pred_emb.shape)}")


if __name__ == "__main__":
    main()
