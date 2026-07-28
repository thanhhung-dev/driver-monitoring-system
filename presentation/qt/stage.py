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

    def process(self, ctx: FrameContext) -> FrameContext:
        """
        Processes the frame (drawing meshes, etc.) and emits it via a Qt Signal.
        """
        # Let the original VizStage do all the drawing.
        # But we need to intercept before cv2.imshow is called.
        # The easiest way is to temporarily patch visualizer.show
        if self._visualizer is None:
            return ctx
            
        original_show = self._visualizer.show
        
        drawn_frame = None
        
        def mock_show(window_name, frame):
            nonlocal drawn_frame
            drawn_frame = frame.copy()
            
        self._visualizer.show = mock_show
        
        # Call the parent process which will do the drawing and call our mock_show
        super().process(ctx)
        
        # Restore original show
        self._visualizer.show = original_show
        
        # Emit the signal to the GUI thread
        if drawn_frame is not None:
            self.frame_ready.emit(drawn_frame, ctx)
            
        return ctx
