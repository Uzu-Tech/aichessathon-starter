"""Train and validate the simple pairwise move-ordering network."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.dataset import MovePairDataset, collate_move_pairs
from src.metrics import top_k_agreement
from src.model import MoveOrderingModel
from src.ranking_loss import pairwise_ranking_loss


def run_epoch(
    model: MoveOrderingModel,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
) -> tuple[float, float]:
    """Return mean ranking loss and pairwise accuracy."""
    model.train(optimizer is not None)
    total_loss = 0.0
    correct = 0
    total = 0
    context = torch.enable_grad() if optimizer is not None else torch.no_grad()
    with context:
        for batch in loader:
            features = batch["features"].to(device)
            good_scores = model(features, batch["good_move"].to(device))
            bad_scores = model(features, batch["bad_move"].to(device))
            loss = pairwise_ranking_loss(good_scores, bad_scores)
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
            batch_size = good_scores.numel()
            total_loss += loss.item() * batch_size
            correct += (good_scores > bad_scores).sum().item()
            total += batch_size
    return total_loss / total, correct / total


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a neural chess move-ordering model.")
    parser.add_argument("--train-data", required=True, type=Path)
    parser.add_argument("--validation-data", required=True, type=Path)
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("checkpoints"))
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--embedding-dim", type=int, default=256)
    parser.add_argument("--move-dim", type=int, default=64)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--device", default="cpu")
    arguments = parser.parse_args()
    device = torch.device(arguments.device)
    train_loader = DataLoader(
        MovePairDataset(arguments.train_data),
        batch_size=arguments.batch_size,
        shuffle=True,
        collate_fn=collate_move_pairs,
    )
    validation_loader = DataLoader(
        MovePairDataset(arguments.validation_data),
        batch_size=arguments.batch_size,
        shuffle=False,
        collate_fn=collate_move_pairs,
    )
    model = MoveOrderingModel(arguments.embedding_dim, arguments.move_dim, arguments.hidden_dim).to(
        device
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=arguments.learning_rate)
    arguments.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_validation_loss = float("inf")
    for epoch in range(1, arguments.epochs + 1):
        train_loss, train_accuracy = run_epoch(model, train_loader, optimizer, device)
        validation_loss, validation_accuracy = run_epoch(model, validation_loader, None, device)
        top_1 = top_k_agreement(model, validation_loader.dataset, 1)
        top_3 = top_k_agreement(model, validation_loader.dataset, 3)
        top_5 = top_k_agreement(model, validation_loader.dataset, 5)
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "validation_loss": validation_loss,
            "model_config": {
                "position_dim": arguments.embedding_dim,
                "move_dim": arguments.move_dim,
                "hidden_dim": arguments.hidden_dim,
            },
        }
        torch.save(checkpoint, arguments.checkpoint_dir / "last.pt")
        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            torch.save(checkpoint, arguments.checkpoint_dir / "best.pt")
        print(
            f"epoch {epoch}: train loss={train_loss:.4f}, pair accuracy={train_accuracy:.3%}; "
            f"validation loss={validation_loss:.4f}, pair accuracy={validation_accuracy:.3%}, "
            f"top-1/3/5={top_1:.3%}/{top_3:.3%}/{top_5:.3%}"
        )


if __name__ == "__main__":
    main()
