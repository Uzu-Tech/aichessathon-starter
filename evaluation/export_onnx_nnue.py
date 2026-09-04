"""Export int8 HalfKP transformers and dynamically quantized ONNX heads."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import cast

import numpy as np
import torch
from onnxruntime.quantization import QuantType, quantize_dynamic
from torch import nn

from evaluation.training import NNUE, load_model

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "nnues/weights"
CHECKPOINTS = {
    768: ROOT
    / "evaluation/models/halfkp_hm_768x32x1_wdl_20260904_184251_038752.pt",
    1024: ROOT
    / "evaluation/models/halfkp_hm_1024x32x1_wdl_20260904_190246_569992.pt",
}


class _Head(nn.Module):
    def __init__(self, model: NNUE):
        super().__init__()
        self.head = model.head
        self.scale = model.output_scale_cp

    def forward(self, accumulator: torch.Tensor) -> torch.Tensor:
        return self.head(accumulator) * self.scale


def _quantize_transformer(model: NNUE, destination: Path) -> tuple[Path, Path, Path]:
    width = model.architecture.widths[0]
    transformer = cast(nn.EmbeddingBag, model.feature_transformer)
    weights = transformer.weight.detach().numpy()
    maximum = np.max(np.abs(weights), axis=0)
    scales = np.maximum(maximum / 127.0, np.finfo(np.float32).eps).astype(np.float32)

    transformer_path = destination / f"halfkp_{width}_transformer_q8.npy"
    quantized = np.lib.format.open_memmap(
        transformer_path,
        mode="w+",
        dtype=np.int8,
        shape=weights.shape,
    )
    for start in range(0, len(weights), 1024):
        stop = min(start + 1024, len(weights))
        quantized[start:stop] = np.clip(
            np.rint(weights[start:stop] / scales), -127, 127
        ).astype(np.int8)
    quantized.flush()
    del quantized

    bias = cast(torch.Tensor, model.feature_bias).detach().numpy()
    quantized_bias = np.rint(bias / scales).astype(np.int32)
    bias_path = destination / f"halfkp_{width}_bias_q32.npy"
    scale_path = destination / f"halfkp_{width}_scale_f32.npy"
    np.save(bias_path, quantized_bias, allow_pickle=False)
    np.save(scale_path, scales, allow_pickle=False)
    return transformer_path, bias_path, scale_path


def export_model(
    checkpoint: str | Path, output_dir: str | Path = OUTPUT_DIR
) -> tuple[Path, Path, Path, Path]:
    model = load_model(checkpoint, device="cpu")
    if model.representation != "halfkp32" or model.architecture.widths[1:] != (32, 1):
        raise ValueError("export expects a HalfKP WIDTHx32x1 checkpoint")
    width = model.architecture.widths[0]
    destination = Path(output_dir).resolve()
    destination.mkdir(parents=True, exist_ok=True)

    transformer_path, bias_path, scale_path = _quantize_transformer(model, destination)

    float_head_path = destination / f"halfkp_{width}_head_float_temp.onnx"
    head_path = destination / f"halfkp_{width}_head_q8.onnx"
    head = _Head(model).eval()
    example = torch.zeros((1, width * 2), dtype=torch.float32)
    torch.onnx.export(
        head,
        (example,),
        float_head_path,
        input_names=["accumulator"],
        output_names=["score_cp"],
        opset_version=17,
        dynamo=False,
        external_data=False,
    )
    try:
        quantize_dynamic(
            float_head_path,
            head_path,
            per_channel=True,
            reduce_range=False,
            weight_type=QuantType.QInt8,
        )
    finally:
        float_head_path.unlink(missing_ok=True)
    return transformer_path, bias_path, scale_path, head_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("widths", nargs="*", type=int, default=sorted(CHECKPOINTS))
    args = parser.parse_args()
    for width in args.widths:
        if width not in CHECKPOINTS:
            raise ValueError(f"unknown width {width}; choose {sorted(CHECKPOINTS)}")
        paths = export_model(CHECKPOINTS[width])
        print(f"{width}: {', '.join(path.name for path in paths)}")


if __name__ == "__main__":
    main()
