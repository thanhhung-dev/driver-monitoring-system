import sys
from pathlib import Path

from PySide6.QtWidgets import QApplication

project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from presentation.qt.gui_manager import GUIManager


def main() -> None:
    app = QApplication(sys.argv)
    manager = GUIManager()
    manager.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
