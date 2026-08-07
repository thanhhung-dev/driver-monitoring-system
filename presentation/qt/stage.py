import threading

import cv2
from PySide6.QtCore import QObject, Signal
from pipeline.context import FrameContext
from presentation.opencv.stage import VizStage
from presentation.opencv.visualizer import Visualizer
from infrastructure.camera import VideoCapture

class QtVizStage(VizStage, QObject):
    """
    A Qt-compatible version of VizStage.
    Instead of calling cv2.imshow, it emits a PySide6 Signal with the processed frame 
    and the context so the main thread can update the GUI.
    """
    # Signal that carries the drawn frame and the FrameContext
    frame_ready = Signal(object, object)

    def __init__(self, visualizer: Visualizer | None, capture: VideoCapture):
        # Initialize both parent classes
        VizStage.__init__(self, visualizer, capture)
        QObject.__init__(self)
        self._frame_pending = False
        self._frame_pending_lock = threading.Lock()

    def frame_consumed(self) -> None:
        """Allow the worker to queue the next frame after the GUI is done."""
        with self._frame_pending_lock:
            self._frame_pending = False

    def process(self, ctx: FrameContext) -> FrameContext:
        """
        Processes the frame (drawing meshes, etc.) and emits it via a Qt Signal.
        """
        # Let the original VizStage do all the drawing.
        # But we need to intercept before cv2.imshow is called.
        # The easiest way is to temporarily patch visualizer.show
        if self._visualizer is None:
            return ctx

        # A queued Qt signal has no built-in queue bound. Drop visualization
        # work while one frame is waiting instead of exhausting RAM.
        with self._frame_pending_lock:
            if self._frame_pending:
                return ctx
            self._frame_pending = True

        original_show = self._visualizer.show

        drawn_frame = None

        def mock_show(window_name, frame):
            nonlocal drawn_frame
            drawn_frame = frame.copy()

        self._visualizer.show = mock_show

        try:
            # Call the parent process which draws and calls our mock_show.
            super().process(ctx)
        except Exception:
            self.frame_consumed()
            raise
        finally:
            self._visualizer.show = original_show

        # Emit the signal to the GUI thread
        if drawn_frame is not None:
            try:
                self.frame_ready.emit(drawn_frame, ctx)
            except Exception:
                self.frame_consumed()
                raise
        else:
            self.frame_consumed()

        return ctx
