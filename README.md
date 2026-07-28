# Driver Monitoring System (DMS)

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![ONNX Runtime](https://img.shields.io/badge/ONNX-Runtime-005CED?logo=onnx&logoColor=white)](https://onnxruntime.ai/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.x-5C3EE8?logo=opencv&logoColor=white)](https://opencv.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

> **Driver Monitoring System (DMS)** is a real-time driver monitoring system based on Computer Vision and Deep Learning. It detects faces, estimates head pose, analyzes facial attributes (eyes, glasses, mask, etc.), and raises alerts when the driver shows signs of **drowsiness**, **distraction**, or **not watching the road**.


<p align="center">
  If you find this project useful, please <b>star</b> the repo to support the author.
</p>

## Qt Integration

This project can use Qt for a GUI launcher window via `PySide6`.

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the Qt demo window:

```bash
python app/qt_gui.py
```

This will open a Qt-based IDE-style application window for the project.
