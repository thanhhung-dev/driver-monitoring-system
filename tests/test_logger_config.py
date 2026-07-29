import logging
import tempfile
import unittest
from pathlib import Path

from utils.logger import setup_logger


class LoggerConfigTests(unittest.TestCase):
    def test_setup_logger_handles_invalid_bytes_in_config(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "config.yaml"
            config_path.write_bytes(b"system:\n  log_level: INFO\n  log_file: logs/test.log\n\x90\n")

            logger = setup_logger("test_logger_invalid_bytes", str(config_path))

            self.assertIsNotNone(logger)
            self.assertEqual(logger.level, logging.INFO)
            self.assertTrue(any(isinstance(handler, logging.StreamHandler) for handler in logger.handlers))

            for handler in list(logger.handlers):
                handler.close()
                logger.removeHandler(handler)


if __name__ == "__main__":
    unittest.main()
