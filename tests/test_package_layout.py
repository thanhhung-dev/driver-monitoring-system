import unittest
from pathlib import Path
import re


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class PackageLayoutTest(unittest.TestCase):
    def test_feature_first_packages_exist(self) -> None:
        expected_files = (
            "app/bootstrap.py",
            "app/application.py",
            "app/pipeline_factory.py",
            "pipeline/context.py",
            "pipeline/stage.py",
            "pipeline/runner.py",
            "features/face/detector.py",
            "features/landmarks/detector.py",
            "features/head_pose/stage.py",
            "features/gaze/estimator.py",
            "features/drowsiness/analyzer.py",
            "features/distraction/analyzer.py",
            "features/risk/engine.py",
            "infrastructure/camera.py",
            "infrastructure/database.py",
            "infrastructure/event_logger.py",
            "alerting/alert_manager.py",
            "presentation/opencv/visualizer.py",
        )

        missing = [path for path in expected_files if not (PROJECT_ROOT / path).is_file()]

        self.assertEqual([], missing)

    def test_legacy_source_packages_are_removed(self) -> None:
        legacy_packages = ("dms", "action", "analysis", "core", "detection", "input", "storage", "test")
        remaining = [path for path in legacy_packages if (PROJECT_ROOT / path).exists()]

        self.assertEqual([], remaining)

    def test_source_does_not_import_legacy_packages(self) -> None:
        legacy_import = re.compile(
            r"^(?:from|import)\s+(?:dms|action|analysis|core|detection|input|storage)(?:\.|\s|$)",
            re.MULTILINE,
        )
        source_roots = (
            "alerting",
            "app",
            "features",
            "infrastructure",
            "pipeline",
            "presentation",
            "scripts",
            "utils",
        )
        offenders = []
        for source_root in source_roots:
            for path in (PROJECT_ROOT / source_root).rglob("*.py"):
                if legacy_import.search(path.read_text(encoding="utf-8")):
                    offenders.append(path.relative_to(PROJECT_ROOT).as_posix())

        self.assertEqual([], offenders)


if __name__ == "__main__":
    unittest.main()
