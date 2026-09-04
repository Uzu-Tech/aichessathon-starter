# Quantized NNUE evaluators

`evaluate.py` has the same interface as the starter's evaluator, but uses the
quantized 768-wide NNUE:

```python
import chess
from nnues.evaluate import evaluate

score_cp = evaluate(chess.Board(fen))
```

Import `evaluate` from `nnues.evaluate_1024` to test the larger model. Both
functions return centipawns from the side-to-move viewpoint.

`agent.py` copies the starter minimax baseline's search depth and replaces only
its material-and-mobility evaluation with `nnues.evaluate`. `agent_1024.py`
uses the same search with `nnues.evaluate_1024`. The root `visualizer.ipynb`
is configured to play 1024 as White against 768 as Black.

Compare the 1024 and 768 models with identical minimax searches:

```powershell
.venv\Scripts\python -m harness.arena `
  --agent nnues/player_1024 `
  --opponent nnues/player_768 `
  --games 20
```
