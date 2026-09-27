"""
tools.py — the tool catalogue for the pointing-gesture menu.

Deliberately tiny and dependency-free so gesture_engine.py (pure NumPy) can
import it. `on_select` is a hook for Step 6+: wire disassembly, assembly,
measuring, etc. by passing a callable, or leave it None and branch on
`tool.name` in main.py after a selection.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple


@dataclass
class Tool:
    name: str                                    # full label, shown on selection
    glyph: str = "?"                              # <=4 chars, drawn inside the wheel slot
    color: Optional[Tuple[int, int, int]] = None   # BGR override; None -> viewport default
    on_select: Optional[Callable[[], None]] = None  # optional side effect, called by main.py


# Placeholder catalogue — extend freely. Disassemble/Assemble are stubbed for
# the next step; give them real `on_select` callbacks once the CAD side exists.
DEFAULT_TOOLS: List[Tool] = [
    Tool("Select", "SEL"),
    Tool("Move", "MOV"),
    Tool("Rotate", "ROT"),
    Tool("Scale", "SCL"),
    Tool("Extrude", "EXT"),
    Tool("Disassemble", "DIS"),
    Tool("Assemble", "ASM"),
    Tool("Measure", "MSR"),
]
