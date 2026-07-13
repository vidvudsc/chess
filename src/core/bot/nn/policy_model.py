from __future__ import annotations

import torch
from torch import nn


MOVE_INDEX_COUNT = 64 * 64


class ConvPolicyValueNet(nn.Module):
    def __init__(self,
                 input_channels: int = 15,
                 channels: int = 64,
                 blocks: int = 3,
                 value_hidden: int = 128) -> None:
        super().__init__()
        layers: list[nn.Module] = [
            nn.Conv2d(input_channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        ]
        for _ in range(blocks):
            layers.extend([
                nn.Conv2d(channels, channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(channels),
                nn.ReLU(inplace=True),
            ])
        self.trunk = nn.Sequential(*layers)
        self.policy = nn.Sequential(
            nn.Conv2d(channels, 8, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(8 * 8 * 8, MOVE_INDEX_COUNT),
        )
        self.value = nn.Sequential(
            nn.Conv2d(channels, 4, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(4 * 8 * 8, value_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(value_hidden, 1),
            nn.Tanh(),
        )

    def forward(self, states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.trunk(states)
        policy_logits = self.policy(x)
        value = self.value(x).squeeze(1)
        return policy_logits, value
