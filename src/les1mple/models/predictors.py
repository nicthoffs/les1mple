from __future__ import annotations

import torch
from torch import nn


def _modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return x * (1 + scale) + shift


class OfficialMamba3Predictor(nn.Module):
    def __init__(
        self,
        *,
        num_frames: int,
        input_dim: int,
        action_dim: int,
        hidden_dim: int,
        output_dim: int,
        depth: int,
        d_state: int,
        headdim: int,
    ) -> None:
        super().__init__()
        _initialize_cuda_before_mamba_import()
        try:
            from mamba_ssm import Mamba3
        except ImportError as exc:
            raise SystemExit(
                "official Mamba3 requires mamba-ssm built from the current "
                "state-spaces/mamba repo with its CUDA/Triton/TileLang kernels. "
                "Install that package, then rerun --predictor-type mamba3."
            ) from exc

        self.num_frames = num_frames
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.cond_proj = nn.Linear(action_dim, hidden_dim)
        self.norms = nn.ModuleList(
            [
                nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-6)
                for _ in range(depth)
            ]
        )
        self.adaLN_modulations = nn.ModuleList(
            [
                nn.Sequential(nn.SiLU(), nn.Linear(hidden_dim, 3 * hidden_dim))
                for _ in range(depth)
            ]
        )
        self.layers = nn.ModuleList(
            [
                Mamba3(
                    d_model=hidden_dim,
                    d_state=d_state,
                    headdim=headdim,
                    is_mimo=False,
                    chunk_size=16,
                )
                for _ in range(depth)
            ]
        )
        self.output_proj = nn.Linear(hidden_dim, output_dim)
        self._init_adaln_zero()

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(x)
        c = self.cond_proj(c)
        for norm, modulation, layer in zip(
            self.norms, self.adaLN_modulations, self.layers, strict=True
        ):
            shift, scale, gate = modulation(c).chunk(3, dim=-1)
            x = x + gate * layer(_modulate(norm(x), shift, scale))
        return self.output_proj(x)

    def _init_adaln_zero(self) -> None:
        for modulation in self.adaLN_modulations:
            nn.init.constant_(modulation[-1].weight, 0)
            nn.init.constant_(modulation[-1].bias, 0)


def _initialize_cuda_before_mamba_import() -> None:
    if torch.cuda.is_available():
        torch.empty(0, device="cuda")
