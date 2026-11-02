"""Minimal architecture required to load the established empirical mask."""

import torch
from torch import nn


class FiLMResidualBlock(nn.Module):
    def __init__(self, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim)
        self.condition = nn.Linear(hidden_dim, hidden_dim * 2)
        self.layers = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )

    def forward(self, hidden: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        scale, shift = self.condition(condition).chunk(2, dim=-1)
        modulated = self.norm(hidden) * (1.0 + scale) + shift
        return hidden + self.layers(modulated)


class SparsityMaskPredictor(nn.Module):
    def __init__(
        self,
        *,
        num_genes: int,
        num_cell_types: int,
        num_batches: int,
        perturbation_dim: int,
        hidden_dim: int = 512,
        condition_dim: int = 128,
        num_blocks: int = 3,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.control_projection = nn.Linear(num_genes, hidden_dim)
        self.cell_embedding = nn.Embedding(num_cell_types, condition_dim)
        self.batch_embedding = nn.Embedding(num_batches, condition_dim)
        self.perturbation_projection = nn.Linear(perturbation_dim, condition_dim)
        self.statistics_projection = nn.Sequential(
            nn.Linear(num_genes * 2, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, condition_dim),
        )
        self.condition_projection = nn.Sequential(
            nn.Linear(condition_dim * 4, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.blocks = nn.ModuleList(
            [FiLMResidualBlock(hidden_dim, dropout) for _ in range(num_blocks)]
        )
        self.output = nn.Sequential(
            nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, num_genes)
        )

    def forward(
        self,
        control: torch.Tensor,
        *,
        cell_idx: torch.Tensor,
        batch_idx: torch.Tensor,
        perturbation: torch.Tensor,
        control_mean: torch.Tensor,
        control_std: torch.Tensor,
    ) -> torch.Tensor:
        condition = self.condition_projection(
            torch.cat(
                [
                    self.cell_embedding(cell_idx.long().view(-1)),
                    self.batch_embedding(batch_idx.long().view(-1)),
                    self.perturbation_projection(perturbation),
                    self.statistics_projection(
                        torch.cat([control_mean, control_std], dim=-1)
                    ),
                ],
                dim=-1,
            )
        )
        hidden = self.control_projection(control)
        for block in self.blocks:
            hidden = block(hidden, condition)
        return self.output(hidden)
