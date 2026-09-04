# How the 768 and 1024 NNUEs were derived

Both models use horizontally mirrored `halfkp32` features and a `32x1` dense
head. They were trained against the WDL objective on 9.8 million positions:
7.9 million train, 900,000 validation, and 1 million test. The shard files are
local data and are intentionally not committed; `evaluation/data/metadata.json`
records their schema and counts.

## Training

The finalist sweep trained each architecture for 10 epochs with batch size
8192, learning rate `1e-3`, no test-set evaluation, and CUDA. The two finalists
were then resumed from their best sweep checkpoints for 50 epochs using:

```python
train(
    representation="halfkp32",
    architecture=architecture,
    resume_from=initial_checkpoint,
    epochs=50,
    learning_rate=3e-4,
    patience=None,
    batch_size=8192,
    workers=0,
    device="cuda",
    test=False,
)
```

The saved final checkpoints are self-describing:

- `halfkp_hm_768x32x1_wdl_20260904_184251_038752.pt`
  - best continuation epoch: 1
  - validation WDL loss: `0.006436309427354071`
- `halfkp_hm_1024x32x1_wdl_20260904_190246_569992.pt`
  - best continuation epoch: 1
  - validation WDL loss: `0.006235590752247307`

Their matching JSON files contain every epoch's training and validation
history. The exact notebook calls are retained in `evaluation/nnue_pipeline.ipynb`.

## Quantized deployment export

Run this from the repository root:

```powershell
.venv\Scripts\python -m evaluation.export_onnx_nnue
```

The exporter converts each large HalfKP feature table to per-channel int8,
keeps its bias and incremental accumulator in int32, and dynamically quantizes
the ONNX dense-head weights to int8. The resulting files live in
`nnues/weights/` and are loaded by `nnue_768.py` and `nnue_1024.py`.
