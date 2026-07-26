import threading
from queue import Queue, Empty, Full

import torch

from pipeline.context import FrameContext


class ProducerStage:
    """Runs a chain of stages on a dedicated thread, connected by bounded queues.

    Reads FrameContext from input_queue, processes through stages,
    and puts result into output_queue. Supports graceful shutdown via poison pill.
    """

    def __init__(
        self,
        stages: list,
        input_queue: Queue,
        output_queue: Queue,
    ) -> None:
        self._stages = stages
        self._input = input_queue
        self._output = output_queue
        self._running = False
        self._thread: threading.Thread | None = None

    @property
    def name(self) -> str:
        return "producer"

    def start(self) -> None:
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)

    def _loop(self) -> None:
        with torch.no_grad():
            while self._running:
                try:
                    ctx = self._input.get(timeout=0.1)
                except Empty:
                    continue
                if ctx is None:  # Poison pill
                    break
                for stage in self._stages:
                    ctx = stage.process(ctx)
                try:
                    self._output.put_nowait(ctx)
                except Full:
                    # Drop frame: discard oldest and insert new
                    try:
                        self._output.get_nowait()
                    except Empty:
                        pass
                    try:
                        self._output.put_nowait(ctx)
                    except Full:
                        pass
