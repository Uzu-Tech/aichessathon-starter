# Neural chess move ordering

This package scores a *position plus one legal move* with one unconstrained
scalar.  It is a search-ordering model, not a chess evaluator, win-probability
model, or best-move classifier.  Encode a position once, score its legal moves,
and pass the descending result to alpha-beta.

## HalfKP

The input is the classic 40,960-feature HalfKP representation.  Its active
features are `(our king square, piece square, non-king piece type, relative
piece colour)`.  From Black's perspective the board is flipped vertically and
piece colours become `us`/`them`.  For a non-king piece the index is:

```
piece_slot = (piece_type - 1) * 2 + relative_colour  # us=0, them=1
index = piece_square + (piece_slot + king_square * 10) * 64
```

This agrees with the original Stockfish-style HalfKP layout described in the
[official NNUE training documentation](https://github.com/official-stockfish/nnue-pytorch/blob/master/docs/nnue.md).
Kings select the king bucket and are not themselves features.

`MoveOrderingModel` sums active feature embeddings into one position embedding.
It then concatenates that once-computed vector with learned from-square,
to-square, and promotion embeddings and returns one score per move. There is no
softmax or sigmoid.

## Data and training

Use Lichess PGNs only to obtain varied positions; do not label human moves.
Run a local Stockfish executable offline to make every ordered pair:

```bash
cd challenger/MoveOrdering
python scripts/generate_data.py --pgn games.pgn --stockfish /path/to/stockfish \
  --depth 10 --output data/train.jsonl
python scripts/generate_data.py --pgn games.pgn --stockfish /path/to/stockfish \
  --depth 10 --output data/validation.jsonl --seed 1
python scripts/train.py --train-data data/train.jsonl --validation-data data/validation.jsonl \
  --epochs 10 --batch-size 256 --checkpoint-dir checkpoints
```

Each JSONL record holds `fen`, `good_move`, and `bad_move` in UCI. The objective
is `softplus(-(score_good - score_bad))`, so labels are strictly relative.
Training reports validation pairwise ranking accuracy plus top-1, top-3, and
top-5 agreement. Top-k assumes files made by the included generator (all strict
pairs for each FEN).

Stockfish is used only by `scripts/generate_data.py`; it is not a runtime or
submission dependency. Do not package it with a competition agent.

## Inference and alpha-beta

```python
from src.inference import order_moves

moves = order_moves(board, model)
for move in moves:
    board.push(move)
    score = alpha_beta(board, depth - 1)  # evaluates leaves with your existing model
    board.pop()
```

`order_moves` does not run alpha-beta or evaluate positions. It returns only
legal `python-chess` moves, sorted by descending neural score.

## Tests

```bash
cd challenger/MoveOrdering
python -m pytest tests
```
