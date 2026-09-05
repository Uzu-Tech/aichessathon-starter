"""Pairwise learning-to-rank objective."""

from __future__ import annotations

import torch
from torch import Tensor


def pairwise_ranking_loss(good_scores: Tensor, bad_scores: Tensor) -> Tensor:
    """Mean softplus(-(good - bad)); lower is better."""
    return torch.nn.functional.softplus(-(good_scores - bad_scores)).mean()
