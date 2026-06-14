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
