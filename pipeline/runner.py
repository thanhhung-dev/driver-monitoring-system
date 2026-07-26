import sys
from queue import Queue, Full

import cv2
import numpy as np
import torch

from infrastructure.camera import CameraNotFoundError, VideoCapture
from pipeline.context import FrameContext
from pipeline.event_bus import EventBus
from pipeline.threading_stage import ProducerStage


class DMSPipeline:
    """Orchestrates the DMS processing loop using a chain of stages.

    Supports two modes:
    - Sequential (default): all stages run on the main thread.
    - Threaded: capture on Thread 1, inference on Thread 2, viz on main thread.

    Threading mode uses bounded Queue(2) between threads for natural frame-drop.
    """

    def __init__(
        self,
        stages: list,
        event_bus: EventBus | None = None,
        threaded: bool = False,
    ) -> None:
        self._stages = stages
        self._bus = event_bus
        self._threaded = threaded

    def start(self, capture: VideoCapture) -> None:
        """Open the capture source and run the main processing loop."""
        try:
            capture.open()
            if self._threaded:
                self._run_threaded(capture)
            else:
                self._run_sequential(capture)
        except CameraNotFoundError as e:
            print(f"Camera error: {e}", file=sys.stderr)
            sys.exit(1)
        finally:
            capture.release()
            cv2.destroyAllWindows()

    def _run_sequential(self, capture: VideoCapture) -> None:
        """Sequential mode: all stages on main thread."""
        with torch.no_grad():
            while True:
                ctx = FrameContext(frame=np.empty(0), frame_number=0)
                for stage in self._stages:
                    ctx = stage.process(ctx)
                    if self._bus:
                        self._bus.publish(f"{stage.name}_complete", ctx)

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

    def _run_threaded(self, capture: VideoCapture) -> None:
        """Threaded mode: capture + inference on separate threads, viz on main.

        Thread 1 (Capture): CaptureStage -> queue_capture
        Thread 2 (Inference): Detect -> Landmark -> HeadPose -> Gaze -> Attrib -> Drowsiness -> Debug -> queue_viz
        Main Thread (Viz): VizStage -> show
        """
        # Split stages: first = capture, last = viz, middle = inference
        capture_stage = self._stages[0]
        viz_stage = self._stages[-1]
        inference_stages = self._stages[1:-1]

        queue_capture: Queue = Queue(maxsize=2)
        queue_viz: Queue = Queue(maxsize=2)

        # Inference producer thread
        inference_producer = ProducerStage(inference_stages, queue_capture, queue_viz)
        inference_producer.start()

        try:
            with torch.no_grad():
                while True:
                    # Capture on main thread (VideoCapture is not thread-safe)
                    ctx = capture_stage.process(
                        FrameContext(frame=np.empty(0), frame_number=0)
                    )

                    # Feed to inference thread
                    try:
                        queue_capture.put_nowait(ctx)
                    except Full:
                        # Drop oldest frame
                        try:
                            queue_capture.get_nowait()
                        except Exception:
                            pass
                        try:
                            queue_capture.put_nowait(ctx)
                        except Full:
                            pass

                    # Get inference result and visualize
                    try:
                        result = queue_viz.get(timeout=0.5)
                        viz_stage.process(result)
                    except Exception:
                        pass

                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
        finally:
            # Shutdown inference thread
            try:
                queue_capture.put(None)  # Poison pill
            except Full:
                pass
            inference_producer.stop()
