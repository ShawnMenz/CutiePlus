"""
Undo/redo of frame edits in the labeling window. A frame's state is its mask file (bytes; None when
the frame has no mask file) and, in task mode, its anchor entry (None when it is not an anchor).
Propagation is not recorded: it never overwrites anchors, so re-propagating repairs its results.
"""
from dataclasses import dataclass
from typing import List, Optional

HISTORY_LIMIT = 100


@dataclass
class FrameState:
    mask: Optional[bytes]
    anchor: Optional[dict]


@dataclass
class Edit:
    frame: str
    before: FrameState
    after: FrameState


class EditHistory:

    def __init__(self, limit: int = HISTORY_LIMIT):
        self.limit = limit
        self.undo_stack: List[Edit] = []
        self.redo_stack: List[Edit] = []

    def record(self, edit: Edit) -> None:
        self.undo_stack.append(edit)
        del self.undo_stack[:-self.limit]
        self.redo_stack.clear()

    def undo(self) -> Optional[Edit]:
        if not self.undo_stack:
            return None
        edit = self.undo_stack.pop()
        self.redo_stack.append(edit)
        return edit

    def redo(self) -> Optional[Edit]:
        if not self.redo_stack:
            return None
        edit = self.redo_stack.pop()
        self.undo_stack.append(edit)
        return edit
