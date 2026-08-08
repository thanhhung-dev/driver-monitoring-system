import os
import sys
from pathlib import Path


project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import math
import cv2
import numpy as np
import qtawesome as qta

from PySide6.QtWidgets import QApplication, QLabel
from PySide6.QtUiTools import QUiLoader
from PySide6.QtCore import QFile, QIODevice, QSize, Slot, QTimer, Qt, QObject, QThread
from PySide6.QtGui import QIcon, QImage, QPixmap

from app.pipeline_factory import create_application
from presentation.opencv.stage import VizStage
from presentation.qt.stage import QtVizStage
from pipeline.context import FrameContext


class PipelineWorker(QThread):
    """
    Runs the AI Pipeline in a background thread to prevent GUI freezing.
    """

    def __init__(self, qt_viz_stage: QtVizStage):
        super().__init__()
        self.qt_viz_stage = qt_viz_stage
        self.application = None
        self._stop_requested = False

    def run(self):
        try:
            self.application = create_application()
        except Exception as e:
            print(f"[PipelineWorker] Failed to start pipeline: {e}")
            return

        pipeline = self.application.pipeline
        stages = pipeline._stages

        for i, stage in enumerate(stages):
            if isinstance(stage, VizStage) and not isinstance(stage, QtVizStage):
                self.qt_viz_stage._visualizer = stage._visualizer
                self.qt_viz_stage._capture = stage._capture
                stages[i] = self.qt_viz_stage
                break

        try:
            self.application.run()
        except Exception as e:
            print(f"[PipelineWorker] Pipeline crashed: {e}")

    def stop(self):
        """
        Yêu cầu dừng một cách hợp tác. Ưu tiên gọi application.stop()
        (hoặc application.request_stop()) nếu pipeline của bạn có hàm đó —
        KHÔNG release() capture trực tiếp từ thread khác vì cv2.VideoCapture
        không thread-safe và có thể crash khi đang read() dở.
        """
        self._stop_requested = True
        if self.application is not None:
            if hasattr(self.application, "stop"):
                self.application.stop()
            elif hasattr(self.application, "request_stop"):
                self.application.request_stop()
            else:
                if self.application.capture:
                    self.application.capture.release()


class GUIManager(QObject):
    def __init__(self):
        super().__init__()

        ui_file_path = os.path.join(os.path.dirname(__file__), "main_window.ui")
        ui_file = QFile(ui_file_path)
        if not ui_file.open(QIODevice.ReadOnly):
            print(f"Cannot open {ui_file_path}: {ui_file.errorString()}")
            sys.exit(-1)

        loader = QUiLoader()
        self.window = loader.load(ui_file)
        ui_file.close()

        self.window.mainSplitter.setStretchFactor(0, 0)
        self.window.mainSplitter.setStretchFactor(1, 1)
        self.window.mainSplitter.setSizes([180, self.window.width() - 180])

        self.window.lblDistractionTitle.raise_()
        self.window.lblDrowsyTitle.raise_()

        self._passenger_icons = []
        self._icon_states = {}
        self.setup_icons()
        # self.window.setFixedSize(1366, 702)  

        self.qt_viz_stage = QtVizStage(None, None)
        self.qt_viz_stage.frame_ready.connect(
            self.on_frame_ready, type=Qt.QueuedConnection
        )

        self.worker = PipelineWorker(self.qt_viz_stage)
        self.worker.finished.connect(self.on_worker_finished)

        QTimer.singleShot(100, self.start_pipeline)

        self._original_close_event = self.window.closeEvent
        self.window.closeEvent = self.close_event

    def setup_icons(self):
        """
        Gán icon SVG nội bộ và Font Awesome cho các QLabel/QPushButton
        để trống trong main_window.ui. Tên icon Font Awesome xem tại
        https://fontawesome.com/icons (bản v5 free) hoặc chạy
        `python -c "import qtawesome as qta; qta.icon_browser()"` để duyệt.
        """
        muted = "#a0aec0"
        accent = "#4fd1c5"
        status_icons = {
            "iconGlasses": "glasses.svg",
            "iconDevice": "face-mask.svg",
            "iconPerson": "user-slash.svg",
        }
        for name, filename in status_icons.items():
            self._set_asset_icon(name, filename, size=25)
        title_icons = {
            "iconActionTitle": "user-viewfinder.svg",
            "iconExpression": "user-emotion.svg",
            "iconEyeOpenness": "eye.svg",
            "iconEyeBlink": "eye.svg",
            "iconHeadLoc": "ruler.svg",
            "iconEyeLoc": "ruler.svg",
            "iconHeadDir": "arrow-up.svg",
            "iconGazeDir": "arrow-up.svg",
            "iconGazeZone": "eye.svg",
            "iconHeadZone": "head-side.svg",
        }
        for name, icon_name in title_icons.items():
            self._set_label_icon(name, icon_name, color=accent, size=17)

        self.action_icon_defs = {
            "iconActionPhone": "phone-call.svg",
            "iconActionDrink": "drinking.svg",
            "iconActionSmoke": "smoking.svg",
            "iconActionYawn": "emotion-yawn.svg",
        }
        for name, icon_name in self.action_icon_defs.items():
            self._set_label_icon(name, icon_name, color=muted, size=20)

        # -- 2 nút dưới cùng --
        asset_dir = Path(__file__).resolve().parent / "access"
        if hasattr(self.window, "btnAddDriver"):
            self.window.btnAddDriver.setIcon(QIcon(str(asset_dir / "add-driver.svg")))
            self.window.btnAddDriver.setIconSize(QSize(35, 35))
        if hasattr(self.window, "btnLogout"):
            self.window.btnLogout.setIcon(QIcon(str(asset_dir / "exit.svg")))
            self.window.btnLogout.setIconSize(QSize(35, 35))

    @staticmethod
    def _set_icon_pixmap(label, icon: QIcon, size: int):
        device_pixel_ratio = label.devicePixelRatioF()
        label.setScaledContents(False)
        label.setPixmap(icon.pixmap(QSize(size, size), device_pixel_ratio))

    def _set_asset_icon(self, label_name: str, filename: str, size: int = 20):
        label = getattr(self.window, label_name, None)
        if label is None:
            return
        icon_path = Path(__file__).resolve().parent / "access" / filename
        self._set_icon_pixmap(label, QIcon(str(icon_path)), size)

    def _set_label_icon(self, label_name: str, icon_name: str, color: str, size: int = 18):
        if icon_name.lower().endswith(".svg"):
            self._set_asset_icon(label_name, icon_name, size)
            return

        label = getattr(self.window, label_name, None)
        if label is None:
            return
        icon = qta.icon(icon_name, color=color)
        self._set_icon_pixmap(label, icon, size)

    def set_icon_state(self, label_name: str, icon_name: str, active: bool):
        """
        Đổi màu icon theo trạng thái, dùng khi cập nhật realtime
        (vd kính được detect -> xanh, mất tracking -> xám).
        Gọi từ update_metrics() nếu cần.
        """
        label = getattr(self.window, label_name, None)
        if label is None:
            return
        if self._icon_states.get(label_name) == active:
            return
        color = "#4fd1c5" if active else "#4a5568"
        self._set_label_icon(label_name, icon_name, color=color, size=24)
        self._icon_states[label_name] = active

    def _set_passenger_icons(self, passenger_count: int):
        self.window.userLayout.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        while len(self._passenger_icons) < passenger_count:
            label = QLabel(self.window.userFrame)
            label.setAlignment(Qt.AlignCenter)
            self._set_icon_pixmap(
                label,
                QIcon(str(Path(__file__).resolve().parent / "access" / "user.svg")),
                20,
            )
            self.window.userLayout.addWidget(label)
            self._passenger_icons.append(label)

        for index, label in enumerate(self._passenger_icons):
            label.setVisible(index < passenger_count)

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
            self.worker.wait(timeout=3000) 
        self.window.videoLabel.clear()
        self.window.videoLabel.setText("Pipeline Stopped")

    @Slot()
    def on_worker_finished(self):
        # pipeline tự thoát (lỗi hoặc capture hết) mà không qua stop_pipeline()
        if self.window.videoLabel.pixmap() is None:
            self.window.videoLabel.setText("Pipeline stopped unexpectedly")

    @Slot(object, object)
    def on_frame_ready(self, frame: np.ndarray, ctx: FrameContext):
        """
        Slot called every time the AI pipeline finishes processing a frame.
        """
        try:
            rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_image.shape
            bytes_per_line = ch * w

            qt_image = QImage(
                rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888
            ).copy()
            pixmap = QPixmap.fromImage(qt_image)

            scaled_pixmap = pixmap.scaled(
                self.window.videoLabel.size(),
                Qt.KeepAspectRatioByExpanding,
                Qt.SmoothTransformation,
            )
            self.window.videoLabel.setPixmap(scaled_pixmap)

            self.update_metrics(ctx)
        finally:
            self.qt_viz_stage.frame_consumed()

    def update_metrics(self, ctx: FrameContext):
        """
        Update UI values from the backend Context data.
        """
        # Driver Profile
        if getattr(ctx, "driver_name", None):
            self.window.driverName.setText(ctx.driver_name)

        detections = getattr(ctx, "face_detections", None)
        passenger_count = len(detections) if detections is not None else 0
        if getattr(ctx, "driver_face_index", None) is not None:
            passenger_count = max(0, passenger_count - 1)
        self._set_passenger_icons(passenger_count)

        # Distraction Level
        if ctx.distraction_score is not None:
            val = int(ctx.distraction_score * 100)
            self.window.lblDistractionVal.setValue(val)

        # Drowsy Level
        if ctx.drowsiness_score is not None:
            val = int(ctx.drowsiness_score * 100)
            self.window.lblDrowsyVal.setValue(val)

        attribs = getattr(ctx, "attribs", None) or {}

        # Regular action icons (phone/drink/smoke/yawn) - đổi màu icon
        # theo action nào đang active. TODO: đổi 'active_actions' cho đúng
        # tên field thật trong FrameContext (vd set các action string).
        active_actions = getattr(ctx, "active_actions", None) or set()
        action_key_map = {
            "iconActionPhone": "calling",
            "iconActionDrink": "drinking",
            "iconActionSmoke": "smoking",
            "iconActionYawn": "yawning",
        }
        for label_name, action_key in action_key_map.items():
            icon_name = self.action_icon_defs.get(label_name)
            if icon_name:
                self.set_icon_state(label_name, icon_name, action_key in active_actions)

        # Expression
        # TODO: đổi 'emotion' cho đúng key thật trong attribs của bạn
        if "emotion" in attribs:
            self.window.lblExpVal.setText(str(attribs["emotion"]).upper())

        # Eye Openness (trái/phải)
        # TODO: đổi 'eye_openness_l' / 'eye_openness_r' cho đúng key thật
        if "eye_openness_l" in attribs and "eye_openness_r" in attribs:
            l = int(attribs["eye_openness_l"] * 100)
            r = int(attribs["eye_openness_r"] * 100)
            self.window.lblEyeOpL.setText(f"{l}%")
            self.window.lblEyeOpR.setText(f"{r}%")

        # Blink rate
        blink_left = attribs.get("blink_rate_l")
        blink_right = attribs.get("blink_rate_r")
        if blink_left is not None and blink_right is not None:
            self.window.lblBlinkVal.setText(f"{blink_left:.1f}/s")
            self.window.lblBlinkRVal.setText(f"{blink_right:.1f}/s")
        elif "blink_rate" in attribs:
            self.window.lblBlinkVal.setText(f"{attribs['blink_rate']:.1f}/s")
            self.window.lblBlinkRVal.setText("--")

        # Head location (mm)
        if getattr(ctx, "head_location_mm", None) is not None:
            x, y, z = ctx.head_location_mm
            self.window.lblHeadLocVal.setText(f"{x:.0f}")
            self.window.lblHeadLocYVal.setText(f"{y:.0f}")
            self.window.lblHeadLocZVal.setText(f"{z:.0f}")

        # Eye location (mm) - trái/phải
        if getattr(ctx, "eye_location_mm", None) is not None:
            (lx, ly, lz), (rx, ry, rz) = ctx.eye_location_mm
            eye_values = (
                (self.window.lblEyeLocVal, lx),
                (self.window.lblEyeLocLYVal, ly),
                (self.window.lblEyeLocLZVal, lz),
                (self.window.lblEyeLocRXVal, rx),
                (self.window.lblEyeLocRYVal, ry),
                (self.window.lblEyeLocRZVal, rz),
            )
            for label, value in eye_values:
                label.setText(f"{value:.0f}")

        # Head Pose (pitch, yaw, roll)
        if ctx.head_pose:
            yaw, pitch, roll = ctx.head_pose
            self.window.lblHeadDirVal.setText(f"{pitch:+.0f}°")
            self.window.lblHeadDirYawVal.setText(f"{yaw:+.0f}°")
            self.window.lblHeadDirRollVal.setText(f"{roll:+.0f}°")

        # Gaze Direction — convert vector -> góc (pitch/yaw) để dễ đọc hơn số vector thô
        if ctx.gaze_vec_world is not None:
            gx, gy, gz = ctx.gaze_vec_world
            gaze_yaw = math.degrees(math.atan2(gx, -gz))
            gaze_pitch = math.degrees(math.atan2(gy, -gz))
            self.window.lblGazeDirVal.setText(f"{gaze_pitch:+.0f}°")
            self.window.lblGazeDirYawVal.setText(f"{gaze_yaw:+.0f}°")

        # Gaze zone / Head zone
        # TODO: đổi tên field cho đúng ctx thật (vd ctx.gaze_zone, ctx.head_zone)
        if getattr(ctx, "gaze_zone", None):
            self.window.lblGazeZoneVal.setText(str(ctx.gaze_zone))
        if getattr(ctx, "head_zone", None):
            self.window.lblHeadZoneVal.setText(str(ctx.head_zone))

    def close_event(self, event):
        self.stop_pipeline()
        self._original_close_event(event)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    manager = GUIManager()
    manager.show()
    sys.exit(app.exec())
