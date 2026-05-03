import os
class EyeGazeEstimation:
    INPUT_SIZE = (96,160)
    def __init__(
        self,
        model_dir: str = "dms\models\eye-gaze-dectecion",
    ) -> None:
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        model_dir = os.path.join(base_dir, model_dir)

        onnx_path = os.path.join(model_dir, "eyegaze.onnx")
        if not os.path.exists(onnx_path):
            raise FileNotFoundError(f"ONNX Model not found: " {onnx_path})
        