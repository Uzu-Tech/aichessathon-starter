"""Local arena entry point for the 1024-wide NNUE agent."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from nnues.agent_1024 import get_move  # noqa: E402

__all__ = ["get_move"]
