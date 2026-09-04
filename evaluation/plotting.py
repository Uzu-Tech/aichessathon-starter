"""Small dependency-free plots for comparing trained NNUE models."""

from __future__ import annotations

import html
import json
import math
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from .training import TrainingResult, benchmark_game, benchmark_inference


@dataclass(frozen=True)
class ModelTradeoff:
    representation: str
    architecture: str
    validation_wdl: float
    latency_microseconds: float
    positions_per_second: float
    memory_mib: float
    timing_path: Literal["full", "incremental game"]


@dataclass(frozen=True)
class TrainingCurve:
    architecture: str
    epochs: tuple[int, ...]
    validation_wdl: tuple[float, ...]
    best_epoch: int
    best_wdl: float


def benchmark_tradeoffs(
    runs: Sequence[TrainingResult],
    *,
    device: str = "cpu",
) -> list[ModelTradeoff]:
    """Collect accuracy, practical latency, and FP32 memory for trained runs."""
    results = []
    for run in runs:
        if run.representation == "halfkp32":
            timing = benchmark_game(run.model_path, device=device)
            latency = timing.total_microseconds
            speed = timing.positions_per_second
            path: Literal["full", "incremental game"] = "incremental game"
        else:
            timing = benchmark_inference(run.model_path, device=device)
            latency = timing.microseconds_per_position
            speed = timing.positions_per_second
            path = "full"
        results.append(
            ModelTradeoff(
                representation=run.representation,
                architecture=run.architecture,
                validation_wdl=run.best_validation_loss,
                latency_microseconds=latency,
                positions_per_second=speed,
                memory_mib=run.model_size_mib,
                timing_path=path,
            )
        )
    return results


def _range(values: Sequence[float]) -> tuple[float, float]:
    low, high = min(values), max(values)
    padding = (high - low) * 0.1 if high > low else max(abs(low) * 0.1, 1.0)
    return low - padding, high + padding


def tradeoff_svg(results: Sequence[ModelTradeoff]) -> str:
    """Render an accuracy/speed scatter plot; bubble area represents FP32 memory."""
    if not results:
        raise ValueError("results cannot be empty")
    width, height = 960, 560
    left, right, top, bottom = 88, 190, 76, 72
    plot_width = width - left - right
    plot_height = height - top - bottom
    x_low, x_high = _range([item.latency_microseconds for item in results])
    y_low, y_high = _range([item.validation_wdl for item in results])

    def x(value: float) -> float:
        return left + (value - x_low) * plot_width / (x_high - x_low)

    def y(value: float) -> float:
        return top + (y_high - value) * plot_height / (y_high - y_low)

    def radius(memory_mib: float) -> float:
        return math.sqrt(25 + 5 * memory_mib)

    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" ',
        'role="img" aria-label="NNUE accuracy, game latency, and FP32 memory comparison" ',
        'style="max-width:100%;height:auto;color:currentColor;font-family:system-ui,sans-serif">',
        '<text x="24" y="28" fill="currentColor" font-size="18" font-weight="600">',
        "NNUE accuracy vs in-game evaluation time</text>",
        '<text x="24" y="50" fill="currentColor" opacity="0.7" font-size="12">',
        "Lower-left is better · bubble area shows FP32 model memory</text>",
        f'<rect x="{left}" y="{top}" width="{plot_width}" height="{plot_height}" ',
        'fill="none" stroke="currentColor" opacity="0.35"/>',
    ]

    for index in range(5):
        fraction = index / 4
        x_value = x_low + fraction * (x_high - x_low)
        x_position = left + fraction * plot_width
        y_value = y_low + fraction * (y_high - y_low)
        y_position = top + (1 - fraction) * plot_height
        svg.extend(
            [
                f'<line x1="{x_position:.1f}" y1="{top}" x2="{x_position:.1f}" ',
                f'y2="{height - bottom}" stroke="currentColor" opacity="0.1"/>',
                f'<text x="{x_position:.1f}" y="{height - bottom + 22}" ',
                'fill="currentColor" opacity="0.75" font-size="11" text-anchor="middle">',
                f"{x_value:.0f}</text>",
                f'<line x1="{left}" y1="{y_position:.1f}" x2="{width - right}" ',
                f'y2="{y_position:.1f}" stroke="currentColor" opacity="0.1"/>',
                f'<text x="{left - 12}" y="{y_position + 4:.1f}" ',
                'fill="currentColor" opacity="0.75" font-size="11" text-anchor="end">',
                f"{y_value:.4f}</text>",
            ]
        )

    for item in results:
        colour = "#4c78a8" if item.representation == "piece768" else "#f58518"
        short_representation = "P" if item.representation == "piece768" else "H"
        cx, cy, r = x(item.latency_microseconds), y(item.validation_wdl), radius(item.memory_mib)
        label_y = cy - r - 5 if item.representation == "piece768" else cy + r + 13
        label = html.escape(f"{short_representation} {item.architecture}")
        details = html.escape(
            f"{item.representation} {item.architecture}: WDL {item.validation_wdl:.7f}, "
            f"{item.latency_microseconds:.1f} us, {item.positions_per_second:,.0f}/s, "
            f"{item.memory_mib:.1f} MiB FP32, {item.timing_path}"
        )
        svg.extend(
            [
                f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="{r:.1f}" fill="{colour}" ',
                f'fill-opacity="0.72" stroke="{colour}" stroke-width="2"><title>',
                f"{details}</title></circle>",
                f'<text x="{cx:.1f}" y="{label_y:.1f}" fill="currentColor" font-size="11" ',
                f'text-anchor="middle">{label}</text>',
            ]
        )

    axis_middle = top + plot_height / 2
    svg.extend(
        [
            f'<text x="{left + plot_width / 2:.1f}" y="{height - 18}" fill="currentColor" ',
            'font-size="12" text-anchor="middle">CPU time per evaluated position (µs)</text>',
            f'<text x="20" y="{axis_middle:.1f}" fill="currentColor" font-size="12" ',
            f'text-anchor="middle" transform="rotate(-90 20 {axis_middle:.1f})">',
            "Validation WDL loss</text>",
            f'<circle cx="{width - 155}" cy="{top + 15}" r="6" fill="#4c78a8"/>',
            f'<text x="{width - 142}" y="{top + 19}" fill="currentColor" font-size="12">',
            "Piece768 · full</text>",
            f'<circle cx="{width - 155}" cy="{top + 42}" r="6" fill="#f58518"/>',
            f'<text x="{width - 142}" y="{top + 46}" fill="currentColor" font-size="12">',
            "HalfKP · game</text>",
            f'<text x="{width - 165}" y="{top + 87}" fill="currentColor" ',
            'font-size="12" font-weight="600">FP32 memory</text>',
        ]
    )
    for offset, memory in enumerate((1.0, 20.0, 40.0, 80.0)):
        cy = top + 120 + offset * 48
        r = radius(memory)
        svg.extend(
            [
                f'<circle cx="{width - 145}" cy="{cy}" r="{r:.1f}" fill="currentColor" ',
                'fill-opacity="0.16" stroke="currentColor" stroke-opacity="0.45"/>',
                f'<text x="{width - 112}" y="{cy + 4}" fill="currentColor" ',
                f'font-size="11">{memory:.0f} MiB</text>',
            ]
        )
    svg.append("</svg>")
    return "".join(svg)


def plot_tradeoffs(runs: Sequence[TrainingResult], *, device: str = "cpu") -> Any:
    """Benchmark runs and return a notebook-displayable SVG plot."""
    from IPython.display import SVG

    return SVG(tradeoff_svg(benchmark_tradeoffs(runs, device=device)))


def load_training_curves(paths: Sequence[str | Path]) -> list[TrainingCurve]:
    """Load validation curves from completed training-result JSON files."""
    curves = []
    for path in paths:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        history = payload["epochs"]
        curves.append(
            TrainingCurve(
                architecture=str(payload["architecture"]),
                epochs=tuple(int(item["epoch"]) for item in history),
                validation_wdl=tuple(float(item["validation_loss"]) for item in history),
                best_epoch=int(payload["best_epoch"]),
                best_wdl=float(payload["best_validation_loss"]),
            )
        )
    if not curves:
        raise ValueError("paths cannot be empty")
    return curves


def training_curves_svg(curves: Sequence[TrainingCurve]) -> str:
    """Render comparable validation-WDL curves without a plotting dependency."""
    if not curves or any(not curve.epochs for curve in curves):
        raise ValueError("curves cannot be empty")
    width, height = 900, 500
    left, right, top, bottom = 82, 28, 70, 64
    plot_width = width - left - right
    plot_height = height - top - bottom
    all_epochs = [epoch for curve in curves for epoch in curve.epochs]
    all_losses = [loss for curve in curves for loss in curve.validation_wdl]
    x_low, x_high = min(all_epochs), max(all_epochs)
    y_low, y_high = _range(all_losses)
    colours = ("#4c78a8", "#f58518", "#54a24b", "#e45756")

    def x(value: float) -> float:
        return left + (value - x_low) * plot_width / max(x_high - x_low, 1)

    def y(value: float) -> float:
        return top + (y_high - value) * plot_height / (y_high - y_low)

    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" ',
        'role="img" aria-label="Fine-tuning validation WDL loss by epoch" ',
        'style="max-width:100%;height:auto;color:currentColor;font-family:system-ui,sans-serif">',
        '<text x="24" y="28" fill="currentColor" font-size="18" font-weight="600">',
        "HalfKP fine-tuning comparison</text>",
        '<text x="24" y="50" fill="currentColor" opacity="0.7" font-size="12">',
        "Validation WDL loss · lower is better</text>",
        f'<rect x="{left}" y="{top}" width="{plot_width}" height="{plot_height}" ',
        'fill="none" stroke="currentColor" opacity="0.35"/>',
    ]

    x_ticks = (1, 10, 20, 30, 40, 50)
    for epoch in x_ticks:
        if epoch < x_low or epoch > x_high:
            continue
        position = x(epoch)
        svg.extend(
            [
                f'<line x1="{position:.1f}" y1="{top}" x2="{position:.1f}" ',
                f'y2="{height - bottom}" stroke="currentColor" opacity="0.1"/>',
                f'<text x="{position:.1f}" y="{height - bottom + 21}" ',
                'fill="currentColor" opacity="0.75" font-size="11" text-anchor="middle">',
                f"{epoch}</text>",
            ]
        )
    for index in range(5):
        fraction = index / 4
        value = y_low + fraction * (y_high - y_low)
        position = y(value)
        svg.extend(
            [
                f'<line x1="{left}" y1="{position:.1f}" x2="{width - right}" ',
                f'y2="{position:.1f}" stroke="currentColor" opacity="0.1"/>',
                f'<text x="{left - 10}" y="{position + 4:.1f}" fill="currentColor" ',
                f'opacity="0.75" font-size="11" text-anchor="end">{value:.5f}</text>',
            ]
        )

    for index, curve in enumerate(curves):
        colour = colours[index % len(colours)]
        points = [
            (x(epoch), y(loss))
            for epoch, loss in zip(curve.epochs, curve.validation_wdl, strict=True)
        ]
        path = " ".join(
            f"{'M' if point_index == 0 else 'L'} {px:.1f} {py:.1f}"
            for point_index, (px, py) in enumerate(points)
        )
        best_index = min(
            range(len(curve.validation_wdl)), key=curve.validation_wdl.__getitem__
        )
        best_x, best_y = points[best_index]
        shape = "circle" if index % 2 == 0 else "square"
        marker = (
            f'<circle cx="{best_x:.1f}" cy="{best_y:.1f}" r="5" '
            f'fill="{colour}" stroke="currentColor"/>'
            if shape == "circle"
            else f'<rect x="{best_x - 5:.1f}" y="{best_y - 5:.1f}" width="10" height="10" '
            f'fill="{colour}" stroke="currentColor"/>'
        )
        label = html.escape(curve.architecture)
        svg.extend(
            [
                f'<path d="{path}" fill="none" stroke="{colour}" stroke-width="2.5">',
                f"<title>{label}</title></path>",
                marker,
                f'<text x="{best_x + 9:.1f}" y="{best_y - 9:.1f}" fill="currentColor" ',
                f'font-size="11">best {curve.best_wdl:.6f}</text>',
                f'<line x1="{left + index * 190}" y1="{top - 12}" ',
                f'x2="{left + 24 + index * 190}" y2="{top - 12}" stroke="{colour}" ',
                'stroke-width="3"/>',
                f'<text x="{left + 31 + index * 190}" y="{top - 8}" ',
                f'fill="currentColor" font-size="12">{label}</text>',
            ]
        )

    middle = top + plot_height / 2
    svg.extend(
        [
            f'<text x="{left + plot_width / 2:.1f}" y="{height - 17}" fill="currentColor" ',
            'font-size="12" text-anchor="middle">Fine-tuning epoch</text>',
            f'<text x="18" y="{middle:.1f}" fill="currentColor" font-size="12" ',
            f'text-anchor="middle" transform="rotate(-90 18 {middle:.1f})">',
            "Validation WDL loss</text>",
            "</svg>",
        ]
    )
    return "".join(svg)


def plot_training_curves(paths: Sequence[str | Path]) -> Any:
    """Load result histories and return a notebook-displayable comparison plot."""
    from IPython.display import SVG

    return SVG(training_curves_svg(load_training_curves(paths)))


__all__ = [
    "ModelTradeoff",
    "TrainingCurve",
    "benchmark_tradeoffs",
    "load_training_curves",
    "plot_tradeoffs",
    "plot_training_curves",
    "tradeoff_svg",
    "training_curves_svg",
]
