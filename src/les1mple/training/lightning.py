from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

import lightning as pl
import torch
from stable_worldmodel.wm.loss import SIGReg
from torch.optim.lr_scheduler import LRScheduler

from les1mple.training.objectives import hwm_forward, lejepa_forward


class LinearWarmupCosineAnnealingLR(LRScheduler):
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        *,
        warmup_steps: int,
        max_steps: int,
        warmup_start_lr: float = 0.0,
        eta_min: float = 0.0,
        last_epoch: int = -1,
    ) -> None:
        self.warmup_steps = warmup_steps
        self.max_steps = max_steps
        self.warmup_start_lr = warmup_start_lr
        self.eta_min = eta_min
        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> list[float]:
        if self.last_epoch < self.warmup_steps:
            return [
                self.warmup_start_lr
                + (base_lr - self.warmup_start_lr)
                * self.last_epoch
                / self.warmup_steps
                for base_lr in self.base_lrs
            ]
        return [
            self.eta_min
            + (base_lr - self.eta_min)
            * (
                1
                + math.cos(
                    math.pi
                    * (self.last_epoch - self.warmup_steps)
                    / (self.max_steps - self.warmup_steps)
                )
            )
            / 2
            for base_lr in self.base_lrs
        ]


class LeWMLightningModule(pl.LightningModule):
    def __init__(
        self,
        *,
        model: torch.nn.Module,
        history_size: int,
        target_horizon: int,
        sigreg_weight: float,
        sigreg_knots: int,
        sigreg_num_proj: int,
        sigreg_frame_stride: int,
        lr: float,
        weight_decay: float,
        warmup_steps: int,
        max_steps: int,
        hparams: Mapping[str, Any],
        adamw_fused: str = "auto",
    ) -> None:
        super().__init__()
        self.model = model
        self.sigreg = SIGReg(knots=sigreg_knots, num_proj=sigreg_num_proj)
        self.history_size = history_size
        self.target_horizon = target_horizon
        self.sigreg_weight = sigreg_weight
        self.sigreg_frame_stride = sigreg_frame_stride
        self.lr = lr
        self.weight_decay = weight_decay
        self.adamw_fused = adamw_fused
        self.warmup_steps = warmup_steps
        self.max_steps = max_steps
        self.save_hyperparameters(dict(hparams), ignore=("model",))

    def training_step(self, batch, batch_idx):
        del batch_idx
        output = lejepa_forward(
            self,
            batch,
            history_size=self.history_size,
            target_horizon=self.target_horizon,
            sigreg_weight=self.sigreg_weight,
            sigreg_frame_stride=self.sigreg_frame_stride,
        )
        self._log_losses("fit", output, batch_size=_batch_size(batch))
        return output

    def validation_step(self, batch, batch_idx):
        del batch_idx
        output = lejepa_forward(
            self,
            batch,
            history_size=self.history_size,
            target_horizon=self.target_horizon,
            sigreg_weight=self.sigreg_weight,
            sigreg_frame_stride=self.sigreg_frame_stride,
        )
        self._log_losses("validate", output, batch_size=_batch_size(batch))
        return output

    def _log_losses(
        self,
        stage: str,
        output: Mapping[str, Any],
        *,
        batch_size: int | None,
    ) -> None:
        on_step = stage == "fit"
        on_epoch = stage != "fit"
        self.log(
            f"{stage}/loss",
            output["raw_loss"],
            on_step=on_step,
            on_epoch=on_epoch,
            prog_bar=True,
            batch_size=batch_size,
        )
        self.log(
            f"{stage}/pred_loss",
            output["pred_loss"],
            on_step=on_step,
            on_epoch=on_epoch,
            batch_size=batch_size,
        )
        self.log(
            f"{stage}/sigreg_loss",
            output["sigreg_loss"],
            on_step=on_step,
            on_epoch=on_epoch,
            batch_size=batch_size,
        )

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
            fused=_resolve_adamw_fused(self.adamw_fused),
        )
        scheduler = LinearWarmupCosineAnnealingLR(
            optimizer,
            warmup_steps=self.warmup_steps,
            max_steps=self.max_steps,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
            },
        }


class HierarchicalLeWMLightningModule(pl.LightningModule):
    def __init__(
        self,
        *,
        model: torch.nn.Module,
        macro_stride: int,
        loss_type: str,
        lr: float,
        weight_decay: float,
        warmup_steps: int,
        max_steps: int,
        hparams: Mapping[str, Any],
        adamw_fused: str = "auto",
    ) -> None:
        super().__init__()
        self.model = model
        self.macro_stride = macro_stride
        self.loss_type = loss_type
        self.lr = lr
        self.weight_decay = weight_decay
        self.adamw_fused = adamw_fused
        self.warmup_steps = warmup_steps
        self.max_steps = max_steps
        self.save_hyperparameters(dict(hparams), ignore=("model",))

    def training_step(self, batch, batch_idx):
        del batch_idx
        output = hwm_forward(
            self,
            batch,
            macro_stride=self.macro_stride,
            loss_type=self.loss_type,
        )
        self._log_losses("fit", output, batch_size=_batch_size(batch))
        return output

    def validation_step(self, batch, batch_idx):
        del batch_idx
        output = hwm_forward(
            self,
            batch,
            macro_stride=self.macro_stride,
            loss_type=self.loss_type,
        )
        self._log_losses("validate", output, batch_size=_batch_size(batch))
        return output

    def _log_losses(
        self,
        stage: str,
        output: Mapping[str, Any],
        *,
        batch_size: int | None,
    ) -> None:
        on_step = stage == "fit"
        on_epoch = stage != "fit"
        self.log(
            f"{stage}/loss",
            output["raw_loss"],
            on_step=on_step,
            on_epoch=on_epoch,
            prog_bar=True,
            batch_size=batch_size,
        )
        self.log(
            f"{stage}/pred_loss",
            output["pred_loss"],
            on_step=on_step,
            on_epoch=on_epoch,
            batch_size=batch_size,
        )

    def configure_optimizers(self):
        parameters = [p for p in self.model.parameters() if p.requires_grad]
        if not parameters:
            raise RuntimeError("HWM has no trainable parameters")
        optimizer = torch.optim.AdamW(
            parameters,
            lr=self.lr,
            weight_decay=self.weight_decay,
            fused=_resolve_adamw_fused(self.adamw_fused),
        )
        scheduler = LinearWarmupCosineAnnealingLR(
            optimizer,
            warmup_steps=self.warmup_steps,
            max_steps=self.max_steps,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
            },
        }


def _resolve_adamw_fused(adamw_fused: str) -> bool | None:
    if adamw_fused == "auto":
        return True if torch.cuda.is_available() else None
    if adamw_fused == "true":
        return True
    if adamw_fused == "false":
        return False
    raise ValueError(f"unknown adamw_fused setting {adamw_fused!r}")


def _batch_size(batch) -> int | None:
    if isinstance(batch, Mapping):
        pixels = batch.get("pixels")
        if torch.is_tensor(pixels) and pixels.ndim > 0:
            return int(pixels.shape[0])
    return None
