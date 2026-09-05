"""The neural network used solely to score candidate moves for ordering."""

from __future__ import annotations

from collections.abc import Sequence
from typing import cast

import chess
import torch
from torch import Tensor, nn

from .halfkp import FEATURE_COUNT, halfkp_indices
from .move_encoder import PROMOTION_VOCAB_SIZE, moves_to_tensor


class MoveOrderingModel(nn.Module):
    """Sum HalfKP embeddings once, then score each candidate move."""

    def __init__(self, position_dim: int = 256, move_dim: int = 64, hidden_dim: int = 256) -> None:
        super().__init__()
        self.position_dim = position_dim
        self.halfkp_embedding = nn.Embedding(FEATURE_COUNT, position_dim)
        self.from_embedding = nn.Embedding(64, move_dim)
        self.to_embedding = nn.Embedding(64, move_dim)
        self.promotion_embedding = nn.Embedding(PROMOTION_VOCAB_SIZE, move_dim)
        self.mlp = nn.Sequential(
            nn.Linear(position_dim + 3 * move_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def encode_features(self, feature_indices: Tensor) -> Tensor:
        """Sum active feature embeddings; ``-1`` is padding in a batched tensor."""
        valid = feature_indices >= 0
        safe_indices = feature_indices.masked_fill(~valid, 0)
        embeddings = self.halfkp_embedding(safe_indices)
        return cast(Tensor, (embeddings * valid.unsqueeze(-1)).sum(dim=-2))

    def encode_position(self, board: chess.Board) -> Tensor:
        """Encode one position once and return a ``[position_dim]`` tensor."""
        device = self.halfkp_embedding.weight.device
        features = torch.tensor(halfkp_indices(board), dtype=torch.long, device=device).unsqueeze(0)
        return self.encode_features(features).squeeze(0)

    def encode_moves(self, moves: Tensor) -> Tensor:
        """Encode ``[N, 3]`` or ``[B, N, 3]`` move components."""
        return torch.cat(
            (
                self.from_embedding(moves[..., 0]),
                self.to_embedding(moves[..., 1]),
                self.promotion_embedding(moves[..., 2]),
            ),
            dim=-1,
        )

    def score_moves(
        self, position_embedding: Tensor, moves: Sequence[chess.Move] | Tensor
    ) -> Tensor:
        """Score all moves for one encoded position without re-encoding it."""
        move_tensor = (
            moves_to_tensor(moves, position_embedding.device)
            if not isinstance(moves, Tensor)
            else moves.to(position_embedding.device)
        )
        move_embedding = self.encode_moves(move_tensor)
        if position_embedding.ndim != 1:
            raise ValueError("score_moves expects one position embedding")
        positions = position_embedding.unsqueeze(0).expand(move_embedding.shape[0], -1)
        return cast(Tensor, self.mlp(torch.cat((positions, move_embedding), dim=-1)).squeeze(-1))

    def forward(self, feature_indices: Tensor, moves: Tensor) -> Tensor:
        """Score paired batches: features ``[B, N]``, moves ``[B, 3]``."""
        position_embedding = self.encode_features(feature_indices)
        move_embedding = self.encode_moves(moves)
        return cast(
            Tensor, self.mlp(torch.cat((position_embedding, move_embedding), dim=-1)).squeeze(-1)
        )
