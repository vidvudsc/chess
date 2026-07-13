from __future__ import annotations

import math

import torch
from torch import nn

from features import (
    DUMMY_FEATURE_INDEX,
    HALFKP_DIM,
    MIRRORED_HALFKA_DIM,
    MIRRORED_HALFKA_DUMMY_FEATURE_INDEX,
    MIRRORED_DUMMY_FEATURE_INDEX,
    MIRRORED_HALFKP_DIM,
)


def clipped_relu(x: torch.Tensor) -> torch.Tensor:
    return torch.clamp(torch.relu(x), max=1.0)


def squared_clipped_relu(x: torch.Tensor) -> torch.Tensor:
    y = clipped_relu(x)
    return y * y


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


class HalfKpLinearHeadValueNet(nn.Module):
    def __init__(self,
                 accumulator_dim: int = 256,
                 hidden_dim: int = 32) -> None:
        super().__init__()
        self.accumulator = nn.EmbeddingBag(
            HALFKP_DIM + 1,
            accumulator_dim,
            mode="sum",
            padding_idx=DUMMY_FEATURE_INDEX,
        )
        self.fc1 = nn.Linear(accumulator_dim * 2, hidden_dim)
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
        return torch.tanh(self.out(x)).squeeze(1)


class HalfKpLinearHeadClippedValueNet(HalfKpLinearHeadValueNet):
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

        x = clipped_relu(torch.cat([front, back], dim=1))
        x = clipped_relu(self.fc1(x))
        return torch.tanh(self.out(x)).squeeze(1)


class HalfKpLinearHeadSquaredClippedValueNet(HalfKpLinearHeadValueNet):
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

        x = squared_clipped_relu(torch.cat([front, back], dim=1))
        x = squared_clipped_relu(self.fc1(x))
        return torch.tanh(self.out(x)).squeeze(1)


class MirroredHalfKpLinearHeadSquaredClippedValueNet(HalfKpLinearHeadSquaredClippedValueNet):
    def __init__(self,
                 accumulator_dim: int = 128,
                 hidden_dim: int = 16) -> None:
        nn.Module.__init__(self)
        self.accumulator = nn.EmbeddingBag(
            MIRRORED_HALFKP_DIM + 1,
            accumulator_dim,
            mode="sum",
            padding_idx=MIRRORED_DUMMY_FEATURE_INDEX,
        )
        self.fc1 = nn.Linear(accumulator_dim * 2, hidden_dim)
        self.out = nn.Linear(hidden_dim, 1)

        nn.init.normal_(self.accumulator.weight, mean=0.0, std=0.01)
        with torch.no_grad():
            self.accumulator.weight[MIRRORED_DUMMY_FEATURE_INDEX].zero_()


NUM_MATERIAL_BUCKETS = 8


def material_buckets_from_features(indices: torch.Tensor,
                                   offsets: torch.Tensor,
                                   dummy_index: int,
                                   num_buckets: int = NUM_MATERIAL_BUCKETS) -> torch.Tensor:
    """Bucket by total piece count, derived from the HalfKP feature list.

    Each perspective's features enumerate every non-king piece, so the
    per-sample piece count is (non-dummy features) + 2 kings. Deriving the
    bucket from the features keeps training, export parity, and C inference
    on the exact same rule.
    """
    non_dummy = (indices != dummy_index).to(torch.long)
    positions = torch.arange(indices.numel(), device=indices.device)
    segment = torch.bucketize(positions, offsets, right=True) - 1
    counts = torch.zeros(offsets.numel(), dtype=torch.long, device=indices.device)
    counts.index_add_(0, segment, non_dummy)
    piece_count = counts if dummy_index == MIRRORED_HALFKA_DUMMY_FEATURE_INDEX else counts + 2
    return torch.clamp((piece_count - 1) // 4, 0, num_buckets - 1)


class MirroredHalfKpBucketedSquaredClippedValueNet(nn.Module):
    def __init__(self,
                 accumulator_dim: int = 128,
                 hidden_dim: int = 16,
                 num_buckets: int = NUM_MATERIAL_BUCKETS) -> None:
        super().__init__()
        self.num_buckets = num_buckets
        self.accumulator = nn.EmbeddingBag(
            MIRRORED_HALFKP_DIM + 1,
            accumulator_dim,
            mode="sum",
            padding_idx=MIRRORED_DUMMY_FEATURE_INDEX,
        )
        in_dim = accumulator_dim * 2
        self.fc1_weight = nn.Parameter(torch.empty(num_buckets, hidden_dim, in_dim))
        self.fc1_bias = nn.Parameter(torch.empty(num_buckets, hidden_dim))
        self.out_weight = nn.Parameter(torch.empty(num_buckets, hidden_dim))
        self.out_bias = nn.Parameter(torch.empty(num_buckets))

        nn.init.normal_(self.accumulator.weight, mean=0.0, std=0.01)
        with torch.no_grad():
            self.accumulator.weight[MIRRORED_DUMMY_FEATURE_INDEX].zero_()
        fc1_bound = 1.0 / math.sqrt(in_dim)
        nn.init.uniform_(self.fc1_weight, -fc1_bound, fc1_bound)
        nn.init.uniform_(self.fc1_bias, -fc1_bound, fc1_bound)
        out_bound = 1.0 / math.sqrt(hidden_dim)
        nn.init.uniform_(self.out_weight, -out_bound, out_bound)
        nn.init.uniform_(self.out_bias, -out_bound, out_bound)

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

        x = squared_clipped_relu(torch.cat([front, back], dim=1))
        bucket = material_buckets_from_features(
            white_indices,
            white_offsets,
            self.accumulator.padding_idx,
            self.num_buckets,
        )
        hidden = squared_clipped_relu(
            torch.einsum("bhi,bi->bh", self.fc1_weight[bucket], x) + self.fc1_bias[bucket]
        )
        out = (self.out_weight[bucket] * hidden).sum(dim=1) + self.out_bias[bucket]
        return torch.tanh(out)


class MirroredHalfKpBucketedPsqtSquaredClippedValueNet(
        MirroredHalfKpBucketedSquaredClippedValueNet):
    def __init__(self,
                 accumulator_dim: int = 128,
                 hidden_dim: int = 16,
                 num_buckets: int = NUM_MATERIAL_BUCKETS) -> None:
        super().__init__(accumulator_dim, hidden_dim, num_buckets)
        self.psqt = nn.EmbeddingBag(
            MIRRORED_HALFKP_DIM + 1,
            num_buckets,
            mode="sum",
            padding_idx=MIRRORED_DUMMY_FEATURE_INDEX,
        )
        nn.init.zeros_(self.psqt.weight)

    def forward(self,
                white_indices: torch.Tensor,
                white_offsets: torch.Tensor,
                black_indices: torch.Tensor,
                black_offsets: torch.Tensor,
                stm_white: torch.Tensor) -> torch.Tensor:
        white_acc = self.accumulator(white_indices, white_offsets)
        black_acc = self.accumulator(black_indices, black_offsets)
        white_psqt = self.psqt(white_indices, white_offsets)
        black_psqt = self.psqt(black_indices, black_offsets)

        stm_mask = stm_white.unsqueeze(1)
        front = torch.where(stm_mask, white_acc, black_acc)
        back = torch.where(stm_mask, black_acc, white_acc)
        front_psqt = torch.where(stm_mask, white_psqt, black_psqt)

        x = squared_clipped_relu(torch.cat([front, back], dim=1))
        bucket = material_buckets_from_features(
            white_indices,
            white_offsets,
            self.accumulator.padding_idx,
            self.num_buckets,
        )
        hidden = squared_clipped_relu(
            torch.einsum("bhi,bi->bh", self.fc1_weight[bucket], x) + self.fc1_bias[bucket]
        )
        positional = (self.out_weight[bucket] * hidden).sum(dim=1) + self.out_bias[bucket]
        direct = front_psqt.gather(1, bucket.unsqueeze(1)).squeeze(1)
        return torch.tanh(positional + direct)


class MirroredHalfKaBucketedPsqtSquaredClippedValueNet(
        MirroredHalfKpBucketedPsqtSquaredClippedValueNet):
    """King-aware HalfKAv2_hm with the existing bucketed PSQT head."""

    def __init__(self,
                 accumulator_dim: int = 128,
                 hidden_dim: int = 16,
                 num_buckets: int = NUM_MATERIAL_BUCKETS) -> None:
        MirroredHalfKpBucketedSquaredClippedValueNet.__init__(
            self, accumulator_dim, hidden_dim, num_buckets
        )
        self.accumulator = nn.EmbeddingBag(
            MIRRORED_HALFKA_DIM + 1,
            accumulator_dim,
            mode="sum",
            padding_idx=MIRRORED_HALFKA_DUMMY_FEATURE_INDEX,
        )
        self.psqt = nn.EmbeddingBag(
            MIRRORED_HALFKA_DIM + 1,
            num_buckets,
            mode="sum",
            padding_idx=MIRRORED_HALFKA_DUMMY_FEATURE_INDEX,
        )
        nn.init.normal_(self.accumulator.weight, mean=0.0, std=0.01)
        nn.init.zeros_(self.psqt.weight)
        with torch.no_grad():
            self.accumulator.weight[MIRRORED_HALFKA_DUMMY_FEATURE_INDEX].zero_()


class HalfKpBottleneckValueNet(nn.Module):
    def __init__(self,
                 accumulator_dim: int = 128,
                 bottleneck_dim: int = 64,
                 hidden_dim: int = 16) -> None:
        super().__init__()
        self.accumulator = nn.EmbeddingBag(
            HALFKP_DIM + 1,
            accumulator_dim,
            mode="sum",
            padding_idx=DUMMY_FEATURE_INDEX,
        )
        self.fc1 = nn.Linear(accumulator_dim * 2, bottleneck_dim)
        self.fc2 = nn.Linear(bottleneck_dim, hidden_dim)
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


class HalfKpBottleneckClippedValueNet(HalfKpBottleneckValueNet):
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

        x = clipped_relu(torch.cat([front, back], dim=1))
        x = clipped_relu(self.fc1(x))
        x = clipped_relu(self.fc2(x))
        return torch.tanh(self.out(x)).squeeze(1)


class HalfKpBottleneckSquaredClippedValueNet(HalfKpBottleneckValueNet):
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

        x = squared_clipped_relu(torch.cat([front, back], dim=1))
        x = squared_clipped_relu(self.fc1(x))
        x = squared_clipped_relu(self.fc2(x))
        return torch.tanh(self.out(x)).squeeze(1)


def build_value_model(arch: str,
                      accumulator_dim: int,
                      hidden_dim: int,
                      bottleneck_dim: int | None = None) -> nn.Module:
    if arch == "v2":
        return HalfKpValueNet(accumulator_dim=accumulator_dim, hidden_dim=hidden_dim)
    if arch == "linear-head":
        return HalfKpLinearHeadValueNet(accumulator_dim=accumulator_dim, hidden_dim=hidden_dim)
    if arch == "linear-head-crelu":
        return HalfKpLinearHeadClippedValueNet(accumulator_dim=accumulator_dim, hidden_dim=hidden_dim)
    if arch == "linear-head-screlu":
        return HalfKpLinearHeadSquaredClippedValueNet(accumulator_dim=accumulator_dim, hidden_dim=hidden_dim)
    if arch == "linear-head-screlu-hm":
        return MirroredHalfKpLinearHeadSquaredClippedValueNet(
            accumulator_dim=accumulator_dim,
            hidden_dim=hidden_dim,
        )
    if arch == "linear-head-screlu-hm-buckets":
        return MirroredHalfKpBucketedSquaredClippedValueNet(
            accumulator_dim=accumulator_dim,
            hidden_dim=hidden_dim,
        )
    if arch == "linear-head-screlu-hm-buckets-psqt":
        return MirroredHalfKpBucketedPsqtSquaredClippedValueNet(
            accumulator_dim=accumulator_dim,
            hidden_dim=hidden_dim,
        )
    if arch == "linear-head-screlu-halfka-hm-buckets-psqt":
        return MirroredHalfKaBucketedPsqtSquaredClippedValueNet(
            accumulator_dim=accumulator_dim,
            hidden_dim=hidden_dim,
        )
    if arch == "bottleneck-head":
        return HalfKpBottleneckValueNet(
            accumulator_dim=accumulator_dim,
            bottleneck_dim=bottleneck_dim if bottleneck_dim is not None else 64,
            hidden_dim=hidden_dim,
        )
    if arch == "bottleneck-head-crelu":
        return HalfKpBottleneckClippedValueNet(
            accumulator_dim=accumulator_dim,
            bottleneck_dim=bottleneck_dim if bottleneck_dim is not None else 64,
            hidden_dim=hidden_dim,
        )
    if arch == "bottleneck-head-screlu":
        return HalfKpBottleneckSquaredClippedValueNet(
            accumulator_dim=accumulator_dim,
            bottleneck_dim=bottleneck_dim if bottleneck_dim is not None else 64,
            hidden_dim=hidden_dim,
        )
    raise ValueError(f"unsupported NNUE architecture: {arch}")
