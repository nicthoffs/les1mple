from __future__ import annotations

import torch
import torch.nn.functional as F


def lejepa_forward(
    self,
    batch,
    *,
    history_size: int,
    target_horizon: int,
    sigreg_weight: float,
    sigreg_frame_stride: int = 1,
):
    batch["action"] = batch["action"].nan_to_num_(0.0)
    output = self.model.encode(batch)

    emb = output["emb"]
    act_emb = output["act_emb"]

    ctx_emb = emb[:, :history_size]
    ctx_act = act_emb[:, :history_size]
    tgt_emb = emb[:, target_horizon:]
    pred_emb = self.model.predict(ctx_emb, ctx_act)

    output["pred_loss"] = (pred_emb - tgt_emb).pow(2).mean()
    sigreg_emb = emb[:, ::sigreg_frame_stride]
    output["sigreg_loss"] = self.sigreg(sigreg_emb.transpose(0, 1))
    output["raw_loss"] = output["pred_loss"] + sigreg_weight * output["sigreg_loss"]
    output["loss"] = output["raw_loss"]
    return output


def hwm_forward(
    self,
    batch,
    *,
    macro_stride: int,
    loss_type: str,
):
    if macro_stride < 1:
        raise ValueError("macro_stride must be positive")

    pixels = batch["pixels"]
    action = batch["action"].nan_to_num_(0.0)
    if pixels.shape[1] <= macro_stride:
        raise ValueError(
            "sequence is too short for high-level training: "
            f"sequence_length={pixels.shape[1]} macro_stride={macro_stride}"
        )

    starts = list(range(0, pixels.shape[1] - macro_stride, macro_stride))
    waypoint_indices = torch.tensor(
        [*starts, starts[-1] + macro_stride],
        device=pixels.device,
    )
    waypoint_pixels = pixels.index_select(1, waypoint_indices)

    action_chunks = torch.stack(
        [action[:, start : start + macro_stride] for start in starts],
        dim=1,
    )

    with torch.no_grad():
        z_waypoints = self.model.encode_waypoints(waypoint_pixels)

    z_in = z_waypoints[:, :-1]
    z_target = z_waypoints[:, 1:].detach()
    z_pred = self.model.predict_high(z_in, action_chunks)

    if loss_type == "l1":
        pred_loss = F.l1_loss(z_pred, z_target)
    elif loss_type == "mse":
        pred_loss = F.mse_loss(z_pred, z_target)
    else:
        raise ValueError(f"unknown HWM loss type {loss_type!r}")

    return {
        "loss": pred_loss,
        "raw_loss": pred_loss,
        "pred_loss": pred_loss,
        "z_pred": z_pred,
        "z_target": z_target,
        "macro_stride": macro_stride,
        "num_waypoints": int(waypoint_indices.numel()),
    }


def causal_multi_horizon_forward(
    self,
    batch,
    *,
    horizons: tuple[int, ...],
    num_positions: int,
    loss_type: str,
    sigreg_weight: float = 0.0,
    sigreg_frame_stride: int = 1,
    eval_baselines: bool = False,
):
    if not horizons:
        raise ValueError("horizons must not be empty")
    if min(horizons) < 1:
        raise ValueError("horizons must be positive")
    if len(set(horizons)) != len(horizons):
        raise ValueError("horizons must be unique")
    if num_positions < 1:
        raise ValueError("num_positions must be positive")
    if sigreg_weight < 0:
        raise ValueError("sigreg_weight must be non-negative")
    if sigreg_frame_stride < 1:
        raise ValueError("sigreg_frame_stride must be positive")

    pixels = batch["pixels"]
    action = batch["action"].nan_to_num_(0.0)
    sequence_length = pixels.shape[1]
    max_horizon = max(horizons)
    if num_positions + max_horizon > sequence_length:
        raise ValueError(
            "num positions plus max horizon must fit in sequence: "
            f"sequence_length={sequence_length} "
            f"num_positions={num_positions} "
            f"max_horizon={max_horizon}"
        )

    z_all = self.model.encode_latents(pixels)
    z_prefix = z_all[:, :num_positions]
    losses = []
    copy_losses = []
    shuffled_losses = []
    zero_losses = []
    output = {
        "num_positions": num_positions,
        "max_horizon": max_horizon,
    }
    for horizon in horizons:
        action_windows = _causal_action_windows(
            action,
            num_positions=num_positions,
            horizon=horizon,
        )
        z_target = z_all[:, horizon : horizon + num_positions].detach()
        z_pred = self.model.predict_horizon(z_prefix, action_windows, horizon)
        pred_loss_per = _latent_loss_per_step(z_pred, z_target, loss_type)
        pred_loss_h = pred_loss_per.mean()
        output[f"pred_loss_h{horizon}"] = pred_loss_h
        losses.append(pred_loss_h)

        if eval_baselines:
            copy_loss_h = _latent_loss_per_step(z_prefix, z_target, loss_type).mean()
            shuffled_pred = self.model.predict_horizon(
                z_prefix,
                _rolled_action_chunks(action_windows),
                horizon,
            )
            shuffled_loss_h = _latent_loss_per_step(
                shuffled_pred,
                z_target,
                loss_type,
            ).mean()
            zero_pred = self.model.predict_horizon(
                z_prefix,
                torch.zeros_like(action_windows),
                horizon,
            )
            zero_loss_h = _latent_loss_per_step(zero_pred, z_target, loss_type).mean()
            output[f"copy_last_loss_h{horizon}"] = copy_loss_h
            output[f"shuffled_action_pred_loss_h{horizon}"] = shuffled_loss_h
            output[f"zero_action_pred_loss_h{horizon}"] = zero_loss_h
            output[f"pred_vs_copy_loss_improvement_h{horizon}"] = (
                1.0 - pred_loss_h / copy_loss_h.clamp_min(1e-12)
            )
            output[f"real_vs_shuffled_action_delta_h{horizon}"] = (
                shuffled_loss_h - pred_loss_h
            )
            output[f"real_vs_zero_action_delta_h{horizon}"] = zero_loss_h - pred_loss_h
            copy_losses.append(copy_loss_h)
            shuffled_losses.append(shuffled_loss_h)
            zero_losses.append(zero_loss_h)

    pred_loss = torch.stack(losses).mean()
    raw_loss = pred_loss
    output["pred_loss"] = pred_loss

    if eval_baselines:
        copy_loss = torch.stack(copy_losses).mean()
        shuffled_loss = torch.stack(shuffled_losses).mean()
        zero_loss = torch.stack(zero_losses).mean()
        output["copy_last_loss"] = copy_loss
        output["shuffled_action_pred_loss"] = shuffled_loss
        output["zero_action_pred_loss"] = zero_loss
        output["pred_vs_copy_loss_improvement"] = (
            1.0 - pred_loss / copy_loss.clamp_min(1e-12)
        )
        output["real_vs_shuffled_action_delta"] = shuffled_loss - pred_loss
        output["real_vs_zero_action_delta"] = zero_loss - pred_loss

    if sigreg_weight > 0:
        sigreg_emb = z_all[:, ::sigreg_frame_stride]
        sigreg_loss = self.sigreg(sigreg_emb.transpose(0, 1))
        raw_loss = raw_loss + sigreg_weight * sigreg_loss
        output["sigreg_loss"] = sigreg_loss

    output["loss"] = raw_loss
    output["raw_loss"] = raw_loss
    return output


def _causal_action_windows(
    action: torch.Tensor,
    *,
    num_positions: int,
    horizon: int,
) -> torch.Tensor:
    return torch.stack(
        [
            action[
                :,
                offset : offset + num_positions,
            ]
            for offset in range(horizon)
        ],
        dim=2,
    )


def _latent_loss_per_step(
    pred: torch.Tensor,
    target: torch.Tensor,
    loss_type: str,
) -> torch.Tensor:
    if loss_type == "l1":
        return (pred - target).abs().mean(dim=-1)
    if loss_type == "mse":
        return (pred - target).pow(2).mean(dim=-1)
    raise ValueError(f"unknown loss type {loss_type!r}")


def _rolled_action_chunks(action_chunks: torch.Tensor) -> torch.Tensor:
    batch_size, num_chunks = action_chunks.shape[:2]
    if batch_size * num_chunks <= 1:
        return action_chunks
    flat = action_chunks.flatten(0, 1)
    return flat.roll(shifts=1, dims=0).reshape_as(action_chunks)
