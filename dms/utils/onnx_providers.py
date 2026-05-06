"""
ONNX Runtime provider selection helper.

Ưu tiên thứ tự: CUDA → DirectML → CoreML → CPU.
Dùng chung cho mọi module ONNX để code 1 chỗ, đổi GPU không phải sửa nhiều file.

Cài đặt cho từng nền tảng:
    Windows + NVIDIA (GTX/RTX):  pip install onnxruntime-directml
        (DirectML chạy được trên mọi GPU DirectX 12, đơn giản nhất, không cần CUDA toolkit)
    Windows/Linux + NVIDIA + CUDA 11.8/12.x:  pip install onnxruntime-gpu
    macOS Apple Silicon:         onnxruntime mặc định đã có CoreML
    Mặc định / fallback:         pip install onnxruntime  (CPU)
"""
from __future__ import annotations

import onnxruntime as ort

_PRIORITY = (
    "CUDAExecutionProvider",
    "DmlExecutionProvider",       # DirectML (Windows)
    "DirectMLExecutionProvider",  # alias cũ
    "CoreMLExecutionProvider",
    "CPUExecutionProvider",
)


def best_providers() -> list[str]:
    """Trả về danh sách EP khả dụng theo thứ tự ưu tiên (đã filter)."""
    available = set(ort.get_available_providers())
    return [p for p in _PRIORITY if p in available] or ["CPUExecutionProvider"]


def make_session(
    model_or_bytes,
    *,
    enable_all_optim: bool = True,
) -> ort.InferenceSession:
    """Tạo InferenceSession với provider tốt nhất + graph optim ORT_ENABLE_ALL."""
    so = ort.SessionOptions()
    if enable_all_optim:
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(
        model_or_bytes,
        sess_options=so,
        providers=best_providers(),
    )
