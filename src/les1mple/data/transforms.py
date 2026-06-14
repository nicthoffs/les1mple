from __future__ import annotations

import torch
from torch.utils.data import Dataset, Subset


class ViewDeltaNormalizedDataset(Dataset):
    def __init__(
        self,
        dataset,
        mean: torch.Tensor,
        std: torch.Tensor,
        *,
        copy_action: bool = True,
    ) -> None:
        self.dataset = dataset
        self.mean = (float(mean[0]), float(mean[1]))
        self.std = (float(std[0]), float(std[1]))
        self.copy_action = copy_action

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        sample = dict(self.dataset[index])
        action = sample["action"].clone() if self.copy_action else sample["action"]
        action[:, 0].sub_(self.mean[0]).div_(self.std[0])
        action[:, 1].sub_(self.mean[1]).div_(self.std[1])
        sample["action"] = action
        return sample


def compute_view_delta_stats(dataset) -> tuple[torch.Tensor, torch.Tensor]:
    values = []
    for index in range(len(dataset)):
        values.append(_action_at(dataset, index)[:, :2])
    stacked = torch.cat(values, dim=0)
    mean = stacked.mean(dim=0)
    std = stacked.std(dim=0).clamp_min(1e-6)
    return mean, std


def _action_at(dataset, index: int) -> torch.Tensor:
    if hasattr(dataset, "action_at"):
        return dataset.action_at(index)
    if isinstance(dataset, Subset):
        return _action_at(dataset.dataset, int(dataset.indices[index]))
    return dataset[index]["action"]
