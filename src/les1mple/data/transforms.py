from __future__ import annotations

import torch


class ViewDeltaNormalizedDataset:
    def __init__(self, dataset, mean: torch.Tensor, std: torch.Tensor) -> None:
        self.dataset = dataset
        self.mean = mean
        self.std = std

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        sample = dict(self.dataset[index])
        action = sample["action"].clone()
        action[:, :2] = (action[:, :2] - self.mean) / self.std
        sample["action"] = action
        return sample


def compute_view_delta_stats(dataset) -> tuple[torch.Tensor, torch.Tensor]:
    values = []
    for index in range(len(dataset)):
        values.append(dataset[index]["action"][:, :2])
    stacked = torch.cat(values, dim=0)
    mean = stacked.mean(dim=0)
    std = stacked.std(dim=0).clamp_min(1e-6)
    return mean, std
