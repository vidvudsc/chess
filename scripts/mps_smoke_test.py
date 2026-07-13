#!/usr/bin/env python3
"""Quick MPS smoke test: create a HalfKP-style NNUE model and run batches."""
from __future__ import annotations

import time
import argparse
import torch
from torch import nn


HALFKP_DIM = 20480
DUMMY_INDEX = HALFKP_DIM


class HalfKpValueNet(nn.Module):
    def __init__(self, accumulator_dim: int = 128, hidden_dim: int = 16) -> None:
        super().__init__()
        self.accumulator = nn.EmbeddingBag(
            HALFKP_DIM + 1, accumulator_dim, mode="sum", padding_idx=DUMMY_INDEX
        )
        self.fc1 = nn.Linear(accumulator_dim * 2, accumulator_dim)
        self.fc2 = nn.Linear(accumulator_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, 1)
        nn.init.normal_(self.accumulator.weight, mean=0.0, std=0.01)
        with torch.no_grad():
            self.accumulator.weight[DUMMY_INDEX].zero_()

    def forward(
        self,
        white_indices: torch.Tensor,
        white_offsets: torch.Tensor,
        black_indices: torch.Tensor,
        black_offsets: torch.Tensor,
        stm_white: torch.Tensor,
    ) -> torch.Tensor:
        white_acc = self.accumulator(white_indices, white_offsets)
        black_acc = self.accumulator(black_indices, black_offsets)
        stm_mask = stm_white.unsqueeze(1)
        front = torch.where(stm_mask, white_acc, black_acc)
        back = torch.where(stm_mask, black_acc, white_acc)
        x = torch.cat([front, back], dim=1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return torch.tanh(self.out(x)).squeeze(1)


def make_batch(batch_size: int, avg_features: int = 32, device: torch.device | None = None):
    """Create a synthetic sparse HalfKP batch."""
    white_features: list[torch.Tensor] = []
    black_features: list[torch.Tensor] = []
    white_offsets = [0]
    black_offsets = [0]
    for _ in range(batch_size):
        n = max(1, torch.randint(avg_features - 8, avg_features + 8, (1,)).item())
        white_features.append(torch.randint(0, HALFKP_DIM, (n,), dtype=torch.long))
        black_features.append(torch.randint(0, HALFKP_DIM, (n,), dtype=torch.long))
        white_offsets.append(white_offsets[-1] + n)
        black_offsets.append(black_offsets[-1] + n)
    white_indices = torch.cat(white_features)
    black_indices = torch.cat(black_features)
    stm_white = torch.randint(0, 2, (batch_size,), dtype=torch.bool)
    target = torch.randn(batch_size)
    if device is not None:
        white_indices = white_indices.to(device)
        white_offsets = torch.tensor(white_offsets[:-1], dtype=torch.long).to(device)
        black_indices = black_indices.to(device)
        black_offsets = torch.tensor(black_offsets[:-1], dtype=torch.long).to(device)
        stm_white = stm_white.to(device)
        target = target.to(device)
    else:
        white_offsets = torch.tensor(white_offsets[:-1], dtype=torch.long)
        black_offsets = torch.tensor(black_offsets[:-1], dtype=torch.long)
    return white_indices, white_offsets, black_indices, black_offsets, stm_white, target


def benchmark(device: torch.device, batch_size: int, batches: int, accumulator_dim: int, hidden_dim: int):
    model = HalfKpValueNet(accumulator_dim=accumulator_dim, hidden_dim=hidden_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    # Warmup
    for _ in range(3):
        batch = make_batch(batch_size, device=device)
        out = model(*batch[:5])
        loss = loss_fn(out, batch[5])
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    if device.type == "mps":
        torch.mps.synchronize()

    start = time.perf_counter()
    for _ in range(batches):
        batch = make_batch(batch_size, device=device)
        out = model(*batch[:5])
        loss = loss_fn(out, batch[5])
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    positions = batch_size * batches
    print(f"device={device} arch={accumulator_dim}x{hidden_dim} batch={batch_size} batches={batches}")
    print(f"  positions/sec = {positions/elapsed:,.0f}")
    print(f"  time = {elapsed:.2f}s for {positions:,} positions")
    return positions / elapsed


def main() -> int:
    parser = argparse.ArgumentParser(description="MPS smoke test for NNUE-style training.")
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--batches", type=int, default=100)
    parser.add_argument("--accumulator-dim", type=int, default=128)
    parser.add_argument("--hidden-dim", type=int, default=16)
    parser.add_argument("--device", default="mps", help="Device to test: mps, cuda, cpu")
    args = parser.parse_args()

    print(f"torch {torch.__version__}")
    print(f"mps available: {torch.backends.mps.is_available()}")
    print(f"cuda available: {torch.cuda.is_available()}")

    device = torch.device(args.device)
    benchmark(device, args.batch_size, args.batches, args.accumulator_dim, args.hidden_dim)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
