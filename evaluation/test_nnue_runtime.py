"""Correctness tests for the incremental Numba + ONNX evaluators."""

from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path

import chess
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RUNTIME_AVAILABLE = (
    importlib.util.find_spec("numba") is not None
    and importlib.util.find_spec("onnxruntime") is not None
)
ASSETS_AVAILABLE = all(
    (ROOT / f"nnues/weights/halfkp_{width}_{name}").exists()
    for width in (768, 1024)
    for name in (
        "transformer_q8.npy",
        "bias_q32.npy",
        "scale_f32.npy",
        "head_q8.onnx",
    )
)


@unittest.skipUnless(
    RUNTIME_AVAILABLE and ASSETS_AVAILABLE,
    "Numba, ONNX Runtime, and exported model assets are required",
)
class IncrementalRuntimeTests(unittest.TestCase):
    @staticmethod
    def _backend(width: int):
        from nnues.nnue_runtime import NNUEBackend

        prefix = f"halfkp_{width}"
        return NNUEBackend(
            ROOT / f"nnues/weights/{prefix}_transformer_q8.npy",
            ROOT / f"nnues/weights/{prefix}_bias_q32.npy",
            ROOT / f"nnues/weights/{prefix}_scale_f32.npy",
            ROOT / f"nnues/weights/{prefix}_head_q8.onnx",
            width,
        )

    def _assert_line(self, width: int, fen: str, moves: tuple[str, ...]) -> None:
        backend = self._backend(width)
        incremental = backend.position(fen)
        root_score = incremental.evaluate()
        for uci in moves:
            incremental.push(chess.Move.from_uci(uci))
            rebuilt = backend.position(incremental.board.fen())
            self.assertAlmostEqual(
                incremental.evaluate(), rebuilt.evaluate(), delta=0.01
            )
        while incremental.board.move_stack:
            incremental.pop()
        self.assertAlmostEqual(incremental.evaluate(), root_score, delta=0.01)

    def test_push_pop_and_special_moves_match_full_rebuild(self) -> None:
        lines = (
            (
                chess.STARTING_FEN,
                (
                    "e2e4",
                    "e7e5",
                    "g1f3",
                    "b8c6",
                    "f1b5",
                    "a7a6",
                    "b5c6",
                    "d7c6",
                    "e1g1",
                    "g8f6",
                    "d2d4",
                    "e5d4",
                    "d1d4",
                    "f8e7",
                    "f1e1",
                    "e8g8",
                ),
            ),
            (chess.STARTING_FEN, ("e2e4", "a7a6", "e4e5", "d7d5", "e5d6")),
            ("4k3/P7/8/8/8/8/8/4K3 w - - 0 1", ("a7a8q",)),
        )
        for width in (768, 1024):
            for fen, moves in lines:
                with self.subTest(width=width, last_move=moves[-1]):
                    self._assert_line(width, fen, moves)

    def test_null_move_matches_full_rebuild(self) -> None:
        for width in (768, 1024):
            with self.subTest(width=width):
                backend = self._backend(width)
                incremental = backend.position()
                incremental.push(chess.Move.null())
                rebuilt = backend.position(incremental.board.fen())
                self.assertAlmostEqual(
                    incremental.evaluate(), rebuilt.evaluate(), delta=0.01
                )
                incremental.pop()

    def test_quantized_outputs_stay_close_to_float_checkpoints(self) -> None:
        import torch

        from evaluation.training import load_model

        checkpoints = {
            768: "halfkp_hm_768x32x1_wdl_20260904_184251_038752.pt",
            1024: "halfkp_hm_1024x32x1_wdl_20260904_190246_569992.pt",
        }
        with np.load(ROOT / "evaluation/data/validation/shard_0000.npz") as shard:
            features = torch.from_numpy(np.asarray(shard["halfkp32"][:32]))
            fens = [str(fen) for fen in shard["fen"][:32]]
        sides = torch.tensor([fen.split()[1] == "w" for fen in fens])

        for width, checkpoint in checkpoints.items():
            with self.subTest(width=width):
                model = load_model(ROOT / "evaluation/models" / checkpoint, device="cpu")
                with torch.inference_mode():
                    reference = model(features, sides).numpy()
                backend = self._backend(width)
                quantized = np.asarray(
                    [backend.position(fen).evaluate() for fen in fens],
                    dtype=np.float32,
                )
                error = np.abs(quantized - reference)
                self.assertLess(float(error.mean()), 10.0)
                self.assertLess(float(error.max()), 30.0)


if __name__ == "__main__":
    unittest.main()
