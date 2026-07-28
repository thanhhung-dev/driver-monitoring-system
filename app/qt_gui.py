import sys

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication, QLabel, QMainWindow


class QtMainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Driver Monitoring System - Qt Launcher")
        self.setFixedSize(800, 600)

        label = QLabel("Hello from Qt! This is the DMS Qt starter window.")
        label.setAlignment(Qt.AlignCenter)
        self.setCentralWidget(label)


def main() -> None:
    app = QApplication(sys.argv)
    window = QtMainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
