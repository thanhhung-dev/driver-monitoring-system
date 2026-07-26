from dataclasses import dataclass
from logging import Logger

from infrastructure.camera import VideoCapture
from infrastructure.event_logger import EventLogger
from pipeline.runner import DMSPipeline


@dataclass
class Application:
    """Own the start/stop lifecycle of the assembled DMS application."""

    pipeline: DMSPipeline
    capture: VideoCapture
    logger: Logger
    event_logger: EventLogger | None = None

    def run(self) -> None:
        if self.event_logger:
            self.event_logger.start()

        try:
            self.pipeline.start(self.capture)
        except KeyboardInterrupt:
            self.logger.info("User stopped the system.")
        finally:
            if self.event_logger:
                self.event_logger.stop()
            self.logger.info("System shutdown")
