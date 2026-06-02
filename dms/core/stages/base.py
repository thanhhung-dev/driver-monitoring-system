from typing import Protocol

from core.frame_context import FrameContext


class Stage(Protocol):
    """Protocol defining the interface for pipeline stages."""

    @property
    def name(self) -> str: ...

    def process(self, ctx: FrameContext) -> FrameContext: ...
