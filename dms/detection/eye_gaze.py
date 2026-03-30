import os
class EyeGazeEstimation:
    """Qualcomm AI Hub Eye_Gaze_Estimation (eye_gaze) ONNX wrapper.
    """
    def __init__(self, model_dir: str = "models/eye-gaze-detection",
    ) -> None:
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        model_dir = os.path.join(base_dir, model_dir)

        onnx_path = os.path.join(model_dir, "model.onnx")
        if not os.path.exists(onnx_path):
            raise FileNotFoundError(f"ONNX model not found: {onnx_path} ")