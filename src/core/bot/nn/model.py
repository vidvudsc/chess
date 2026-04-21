from __future__ import annotations

import torch
from torch import nn

from features import DUMMY_FEATURE_INDEX, HALFKP_DIM


class HalfKpValueNet(nn.Module):
    def __init__(self,
                 accumulator_dim: int = 128,
                 hidden_dim: int = 32) -> None:
        super().__init__()
        self.accumulator = nn.EmbeddingBag(
            HALFKP_DIM + 1,
            accumulator_dim,
            mode="sum",
            padding_idx=DUMMY_FEATURE_INDEX,
        )
        self.fc1 = nn.Linear(accumulator_dim * 2, accumulator_dim)
        self.fc2 = nn.Linear(accumulator_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, 1)

        nn.init.normal_(self.accumulator.weight, mean=0.0, std=0.01)
        with torch.no_grad():
            self.accumulator.weight[DUMMY_FEATURE_INDEX].zero_()

    def forward(self,
                white_indices: torch.Tensor,
                white_offsets: torch.Tensor,
                black_indices: torch.Tensor,
                black_offsets: torch.Tensor,
                stm_white: torch.Tensor) -> torch.Tensor:
        white_acc = self.accumulator(white_indices, white_offsets)
        black_acc = self.accumulator(black_indices, black_offsets)

        stm_mask = stm_white.unsqueeze(1)
        front = torch.where(stm_mask, white_acc, black_acc)
        back = torch.where(stm_mask, black_acc, white_acc)

        x = torch.cat([front, back], dim=1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return torch.tanh(self.out(x)).squeeze(1)
