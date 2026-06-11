from __future__ import annotations

import torch


def lejepa_forward(
    self,
    batch,
    stage,
    *,
    history_size: int,
    num_preds: int,
    sigreg_weight: float,
    grad_accum_steps: int,
):
    batch["action"] = torch.nan_to_num(batch["action"], 0.0)
    output = self.model.encode(batch)

    emb = output["emb"]
    act_emb = output["act_emb"]

    ctx_emb = emb[:, :history_size]
    ctx_act = act_emb[:, :history_size]
    tgt_emb = emb[:, num_preds:]
    pred_emb = self.model.predict(ctx_emb, ctx_act)

    output["pred_loss"] = (pred_emb - tgt_emb).pow(2).mean()
    output["sigreg_loss"] = self.sigreg(emb.transpose(0, 1))
    output["raw_loss"] = output["pred_loss"] + sigreg_weight * output["sigreg_loss"]
    if stage == "fit":
        output["loss"] = output["raw_loss"] / grad_accum_steps
    else:
        output["loss"] = output["raw_loss"]

    self.log(f"{stage}/loss", output["raw_loss"], on_step=True, prog_bar=True)
    self.log(f"{stage}/pred_loss", output["pred_loss"], on_step=True)
    self.log(f"{stage}/sigreg_loss", output["sigreg_loss"], on_step=True)
    return output
