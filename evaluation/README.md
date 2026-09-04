# Evaluation training toolkit

`training.py` is self-contained and uses the synchronized NPZ shards produced by
`prepare_datasets.py`. It never changes those shards.

## Command-line pipeline

Use the command-line entry point for real runs. This is important on Windows because
PyTorch data-loader workers start reliably from a guarded Python command, while notebook
worker processes are easy to start recursively by accident.

```powershell
# Verify the dataset and see the automatically selected worker count.
python -m evaluation.pipeline inspect

# Establish fixed validation baselines.
python -m evaluation.pipeline baseline

# Quick end-to-end smoke run.
python -m evaluation.pipeline train --objective wdl --architecture 512x32x1 `
  --epochs 1 --max-train-batches 20 --max-validation-batches 5 --workers auto `
  --run-name smoke_512x32

# Train directly for all three targets with identical settings.
python -m evaluation.pipeline train-suite --architecture 512x32x1 --epochs 50 `
  --batch-size 8192 --workers auto --patience 5
```

The suite trains WDL, RMSE, and MAE models sequentially so they do not contend for one
GPU. Within each run, shards are decompressed and prepared in parallel by persistent
data-loader workers. Override the automatic choice with `--workers N` after measuring
your own disk and memory limits.

Compare checkpoints on the untouched validation split, then benchmark the exact
single-position and maintained-accumulator paths:

```powershell
python -m evaluation.pipeline compare evaluation/models/model_wdl.pt `
  evaluation/models/model_rmse.pt evaluation/models/model_mae.pt

python -m evaluation.pipeline benchmark evaluation/models/model_wdl.pt `
  --device cpu --iterations 1000
```

The benchmark forces one CPU thread. It reports a full single-position inference, the
dense head with an already-maintained accumulator, and a legal 20-ply game simulation
that includes accumulator updates, captures, and castling. Deployment should still use
a compact integer/Numba evaluator; PyTorch call overhead is significant at batch size one.

From a notebook, the equivalent full suite is one call:

```python
from evaluation.pipeline import train_suite

runs = train_suite(architecture="512x32x1", workers="auto")
```

Use validation for model selection. Only run test-set evaluation after choosing the
final configuration, with `--test` or the `evaluate` command.

## Quantized incremental evaluators

The selected HalfKP `768x32x1` and `1024x32x1` checkpoints have separate entry
modules. Change only the import to compare them:

```python
import chess
from nnues.nnue_768 import position  # or: from nnues.nnue_1024 import position

nnue = position(chess.STARTING_FEN)
nnue.push(chess.Move.from_uci("e2e4"))
score_cp = nnue.evaluate()          # side-to-move centipawns
nnue.pop()                          # restores the accumulator without rebuilding it
```

For an isolated position, use `evaluate_fen(fen)`. The feature table is int8, the
incremental accumulator is int32, and the ONNX head uses int8 weights. Both modules
load their assets and warm Numba once during import.

Regenerate the deployable assets from the saved PyTorch checkpoints with:

```powershell
python -m evaluation.export_onnx_nnue
```

## Fast dataset preparation

For a new dataset, one call streams the compressed source into bounded batches and
writes both HalfKP and Piece768 features:

```python
from evaluation.prepare_datasets import prepare_dataset

prepared = prepare_dataset(examples=10_000_000)
print(prepared)
```

`prepare(...)` is a short alias for `prepare_dataset(...)`. A run can be restarted with
the same arguments: completed batches are tracked in an atomic resume state, and each
shard becomes visible only after its temporary file has been fully written. New shards
also store `white_to_move` and `is_mate`, so training does not need to parse FEN strings
to recover those facts.

The existing interrupted run already contains 9.8 million usable legacy examples. It
does **not** need to be regenerated. Validate those shards and add the completion
metadata expected by `train()` with:

```python
from evaluation.prepare_datasets import finalize_existing

prepared = finalize_existing()
print(prepared)
```

Finalization reads and validates the existing shard payloads, then writes metadata; it
does not rewrite their feature arrays. Legacy shards have no `white_to_move` or
`is_mate` columns, which remains supported: the training loader derives side to move
from their stored FEN values.

## Quick start

Wait until dataset preparation has written `metadata.json`, then:

```python
from evaluation.training import inspect_data, train

print(inspect_data())
run = train()  # HalfKP_hm 256x2 -> 32 -> 32 -> 1

print(run.test_metrics)
print(run.info())
print(run.benchmark())
```

The defaults find `evaluation/data`, choose CUDA/MPS when available, save the best
validation checkpoint under `evaluation/models`, and test that best checkpoint once.
`train()` refuses to start while completion metadata is missing.

Use the other synchronized representation with one option:

```python
piece_run = train(representation="piece768", epochs=10)
```

Useful controls:

```python
run = train(
    architecture="256x32x32x1",
    epochs=20,
    batch_size=4096,
    patience=3,
    workers=0,       # safest default in Windows notebooks
    test=False,      # use this while comparing experiments
    run_name="halfkp_trial_01",
)
```

Once the validation winner is chosen, evaluate its test set exactly once:

```python
metrics = run.evaluate("test")
model = run.load()
```

Standalone helpers accept either a checkpoint path, a loaded model, or a training result:

```python
from evaluation.training import (
    benchmark_game,
    benchmark_inference,
    evaluate,
    load_model,
    model_info,
    predict,
)

model = load_model(run.model_path)
metrics = evaluate(model, split="validation")
budget = model_info(model)
full_speed = benchmark_inference(model)
game_speed = benchmark_game(model)  # HalfKP only: legal move + update + dense head

# Encoded arrays matching the model; results default to side-to-move centipawns.
scores = predict(model, encoded_features, white_to_move)
```

`train_model(...)` and `plot_history(...)` remain available for the existing experiment
notebook.

## Why this NNUE shape

The first stage is an affine transform of sparse binary chess features. For HalfKP,
`training.py` computes it by summing embedding rows into two separate accumulators using
one shared weight table. It then orders them `[side to move, opponent]`, applies clipped
ReLU, and uses a very small dense head.

The existing shard IDs are converted to a horizontally mirrored 20,480-feature HalfKP
table at load time. With the default width of 256, the whole float32 model is about
20.1 MiB (about 10.0 MiB with 16-bit first-layer storage), leaving room beneath the
Chessathon's 50 MB expanded submission limit. The architecture is also suitable for a
later incremental evaluator: a move adds/subtracts only the affected first-layer rows.

Training targets are changed from the dataset's White point of view to side-to-move point
of view. Model selection uses bounded WDL-space loss rather than letting mate-sized
centipawn values dominate; reports still include centipawn RMSE, MAE, clipped RMSE, and
sign accuracy.

Primary design references:

- [Official Stockfish NNUE guide](https://github.com/official-stockfish/nnue-pytorch/blob/master/docs/nnue.md)
- [Current Stockfish NNUE architecture](https://github.com/official-stockfish/Stockfish/blob/master/src/nnue/nnue_architecture.h)
- [Original NNUE paper](https://github.com/ynasu87/nnue/blob/master/docs/nnue.pdf)
- [Lichess evaluation schema](https://github.com/lichess-org/api/blob/master/doc/specs/schemas/CloudEval.yaml)
