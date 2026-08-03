import os
import sys


project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import cv2
import numpy as np

from PySide6.QtWidgets import QApplication
from PySide6.QtUiTools import QUiLoader
from PySide6.QtCore import QFile, QIODevice, Slot, QTimer, Qt, QObject, QThread
from PySide6.QtGui import QImage, QPixmap

from app.pipeline_factory import create_application
from presentation.opencv.stage import VizStage
from presentation.qt.stage import QtVizStage
from pipeline.context import FrameContext


QSS_STYLE = """
QMainWindow {
    background-color: #1a202c;
}

QLabel#notifyBar {
    background-color: #1e2638;
    color: #fc8181;
    font-size: 15px;
    font-weight: bold;
}
QLabel#captionBar {
    background-color: #1e2638;
    color: #a0aec0;
    font-size: 12px;
    font-weight: normal;
    border-top: 1px solid #2d3748;
}
QFrame#sidebarFrame {
    background-color: #1e2638;
    border-right: 2px solid #2d3748;
}
QFrame#profileFrame, QFrame#distractionFrame, QFrame#drowsyFrame,
QFrame#actionFrame, QFrame#expressionFrame, QFrame#eyeOpFrame,
QFrame#blinkFrame, QFrame#headLocFrame, QFrame#eyeLocFrame,
QFrame#headDirFrame, QFrame#gazeDirFrame, QFrame#gazeZoneFrame, QFrame#headZoneFrame {
    background-color: #2a3449;
    border-radius: 8px;
    border: 1px solid #4a5568;
    padding: 6px;
}
QLabel {
    color: #a0aec0;
    font-family: "Segoe UI", Arial, sans-serif;
    font-size: 11px;
    font-weight: bold;
}
QLabel#lblDistractionVal, QLabel#lblDrowsyVal {
    color: #ffffff;
    font-size: 20px;
    font-weight: bold;
}
QLabel#lblDistractionTitle, QLabel#lblDrowsyTitle {
    color: #63b3ed;
    font-size: 9px;
}
QLabel#driverName {
    color: #ffffff;
    font-size: 14px;
}
QLabel#lblActionTitle, QLabel#lblExpTitle, QLabel#lblEyeOpTitle, 
QLabel#lblBlinkTitle, QLabel#lblHeadLocTitle, QLabel#lblEyeLocTitle,
QLabel#lblHeadDirTitle, QLabel#lblGazeDirTitle, QLabel#lblGazeZoneTitle, QLabel#lblHeadZoneTitle {
    color: #4fd1c5;
    font-size: 9px;
}
QLabel#lblActionIcons {
    font-size: 14px;
}
QLabel#lblExpVal, QLabel#lblEyeOpL, QLabel#lblEyeOpR, QLabel#lblBlinkVal,
QLabel#lblHeadLocVal, QLabel#lblEyeLocVal, QLabel#lblHeadDirVal, QLabel#lblGazeDirVal,
QLabel#lblGazeZoneVal, QLabel#lblHeadZoneVal {
    color: #f7fafc;
    font-size: 11px;
}
QPushButton {
    background-color: #3182ce;
    color: white;
    border: none;
    border-radius: 4px;
    padding: 8px;
    font-weight: bold;
}
QPushButton:hover {
    background-color: #2b6cb0;
}
"""

class PipelineWorker(QThread):
    """
    Runs the AI Pipeline in a background thread to prevent GUI freezing.
    """
    def __init__(self, qt_viz_stage: QtVizStage):
        super().__init__()
        self.qt_viz_stage = qt_viz_stage
        self.application = None

    def run(self):
        self.application = create_application()
        
        pipeline = self.application.pipeline
        stages = pipeline._stages
        
        for i, stage in enumerate(stages):
            if isinstance(stage, VizStage) and not isinstance(stage, QtVizStage):
                self.qt_viz_stage._visualizer = stage._visualizer
                self.qt_viz_stage._capture = stage._capture
                stages[i] = self.qt_viz_stage
                break
                
        # Start the pipeline (this will block until stopped)
        self.application.run()

    def stop(self):
        # This will simulate a quit command to the pipeline (like pressing 'q' or raising KeyboardInterrupt)
        # We can forcibly break the loop by raising KeyboardInterrupt or closing the capture
        if self.application and self.application.capture:
            # Releasing capture causes cv2.read() to fail and exit loop eventually
            self.application.capture.release()


class GUIManager(QObject):
    def __init__(self):
        super().__init__()
        
        # Load the .ui file dynamically
        ui_file_path = os.path.join(os.path.dirname(__file__), "main_window.ui")
        ui_file = QFile(ui_file_path)
        if not ui_file.open(QIODevice.ReadOnly):
            print(f"Cannot open {ui_file_path}: {ui_file.errorString()}")
            sys.exit(-1)
            
        loader = QUiLoader()
        self.window = loader.load(ui_file)
        ui_file.close()
        
        self.window.setStyleSheet(QSS_STYLE)
        
        # Setup AI Pipeline Thread
        # Create QtVizStage with dummy objects (will be populated inside worker)
        self.qt_viz_stage = QtVizStage(None, None)
        self.qt_viz_stage.frame_ready.connect(self.on_frame_ready)
        
        self.worker = PipelineWorker(self.qt_viz_stage)
        
        # Auto-start pipeline shortly after initialization
        QTimer.singleShot(100, self.start_pipeline)
        
        # Intercept close event to clean up camera
        self._original_close_event = self.window.closeEvent
        self.window.closeEvent = self.close_event

    def show(self):
        self.window.show()

    @Slot()
    def start_pipeline(self):
        if not self.worker.isRunning():
            self.window.videoLabel.setText("Starting AI Pipeline... Please wait.")
            self.worker.start()

    @Slot()
    def stop_pipeline(self):
        if self.worker.isRunning():
            self.worker.stop()
            self.worker.wait() # Wait for thread to finish
        self.window.videoLabel.clear()
        self.window.videoLabel.setText("Pipeline Stopped")

    @Slot(object, object)
    def on_frame_ready(self, frame: np.ndarray, ctx: FrameContext):
        """
        Slot called every time the AI pipeline finishes processing a frame.
        """
        # --- Update Video Frame ---
        # Convert OpenCV BGR format to RGB
        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        
        # Convert to QImage and display on QLabel
        qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qt_image)
        
        # Scale pixmap to fit the label size while keeping aspect ratio
        scaled_pixmap = pixmap.scaled(
            self.window.videoLabel.size(), 
            Qt.KeepAspectRatio, 
            Qt.SmoothTransformation
        )
        self.window.videoLabel.setPixmap(scaled_pixmap)
        
        # --- Update Metrics Sidebar ---
        self.update_metrics(ctx)

    def update_metrics(self, ctx: FrameContext):
        """
        Update UI values from the backend Context data.
        """
        # Driver Profile
        if ctx.driver_name:
            self.window.driverName.setText(ctx.driver_name)
        
        # Distraction Level
        if ctx.distraction_score is not None:
            val = int(ctx.distraction_score * 100)
            self.window.lblDistractionVal.setText(f"{val}%")
        
        # Drowsy Level
        if ctx.drowsiness_score is not None:
            val = int(ctx.drowsiness_score * 100)
            self.window.lblDrowsyVal.setText(f"{val}%")
            
        # Expression
        if ctx.attribs and 'emotion' in ctx.attribs: # Assuming dict keys if available
            pass # You can parse emotion here if the backend provides it
            
        # Eye Openness / Blink
        if ctx.attribs:
            # Example if eye_openness is in attribs
            if 'eye_openness' in ctx.attribs:
                pass
                
        # Head Pose
        if ctx.head_pose:
            yaw, pitch, roll = ctx.head_pose
            self.window.lblHeadDirVal.setText(f"{pitch:+.0f}°   {yaw:+.0f}°   {roll:+.0f}°")
            
        # Gaze Direction
        if ctx.gaze_vec_world is not None:
            # Quick conversion to angles just for display if needed
            # For now, just show vectors or placeholders
            gx, gy, gz = ctx.gaze_vec_world
            self.window.lblGazeDirVal.setText(f"{gx:+.2f}   {gy:+.2f}")

    def close_event(self, event):
        self.stop_pipeline()
        self._original_close_event(event)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    manager = GUIManager()
    manager.show()
    sys.exit(app.exec())
