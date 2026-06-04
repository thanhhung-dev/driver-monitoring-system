"""Export L2CS-Net (ResNet-50, Gaze360 pretrained) to ONNX.

L2CS-Net architecture:
  - ResNet-50 backbone → 2048-dim feature
  - Two FC heads: pitch (90 bins), yaw (90 bins)
  - Softmax → bin classification → angle in degrees

Pretrained weights: Google Drive (from Ahmednull/L2CS-Net)

Usage:
  python scripts/export_l2cs.py --weights path/to/l2cs.pkl --output dms/models/l2cs-net/l2cs.onnx
"""

import argparse
import os
import sys

import numpy as np
import torch
import torch.nn as nn
from torchvision import models


# Number of angle bins (L2CS-Net default)
NUM_BINS = 90


class L2CSNet(nn.Module):
    """L2CS-Net: ResNet-50 backbone with dual pitch/yaw classification heads."""

    def __init__(self, num_bins: int = NUM_BINS):
        super().__init__()
        # ResNet-50 backbone (remove original FC)
        resnet = models.resnet50(weights=None)
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])  # → (B, 2048, 1, 1)
        self.fc_yaw = nn.Linear(2048, num_bins)
        self.fc_pitch = nn.Linear(2048, num_bins)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            x: (B, 3, 224, 224) RGB face crop, ImageNet-normalized.

        Returns:
            (yaw_softmax, pitch_softmax): each (B, 90) softmax probabilities.
        """
        feat = self.backbone(x)          # (B, 2048, 1, 1)
        feat = feat.view(feat.size(0), -1)  # (B, 2048)
        yaw = self.fc_yaw(feat)           # (B, 90)
        pitch = self.fc_pitch(feat)       # (B, 90)
        return torch.softmax(yaw, dim=1), torch.softmax(pitch, dim=1)


def bins_to_angle(bin_probs: np.ndarray) -> float:
    """Convert 90-bin softmax probabilities to angle in degrees.

    Bins represent angles from -180 to +180 in 4-degree increments.
    Uses expected value (weighted sum) for smooth output.
    """
    # Bin centers: -178, -174, ..., 178  (90 bins, 4° apart)
    bin_centers = np.linspace(-180, 180, NUM_BINS, endpoint=False) + 2.0
    return float(np.dot(bin_centers, bin_probs))


def main():
    parser = argparse.ArgumentParser(description="Export L2CS-Net to ONNX")
    parser.add_argument("--weights", required=True, help="Path to L2CS .pkl weights")
    parser.add_argument("--output", default="dms/models/l2cs-net/l2cs.onnx", help="Output ONNX path")
    parser.add_argument("--validate", action="store_true", help="Run validation after export")
    args = parser.parse_args()

    # Load model
    print(f"Loading L2CS-Net weights from: {args.weights}")
    model = L2CSNet(num_bins=NUM_BINS)

    # Load state dict (handle different checkpoint formats)
    checkpoint = torch.load(args.weights, map_location="cpu", weights_only=False)
    if isinstance(checkpoint, dict):
        if "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        elif "model" in checkpoint:
            state_dict = checkpoint["model"]
        else:
            state_dict = checkpoint
    else:
        state_dict = checkpoint

    # Remove "module." prefix if present (DataParallel)
    cleaned = {}
    for k, v in state_dict.items():
        key = k.replace("module.", "") if k.startswith("module.") else k
        cleaned[key] = v

    model.load_state_dict(cleaned, strict=False)
    model.eval()
    print("Weights loaded successfully.")

    # Export to ONNX
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    dummy = torch.randn(1, 3, 224, 224)
    print(f"Exporting to: {args.output}")

    torch.onnx.export(
        model,
        dummy,
        args.output,
        input_names=["face_crop"],
        output_names=["yaw_softmax", "pitch_softmax"],
        dynamic_axes={
            "face_crop": {0: "batch"},
            "yaw_softmax": {0: "batch"},
            "pitch_softmax": {0: "batch"},
        },
        opset_version=17,
    )
    print("ONNX export complete.")

    # Validate
    if args.validate:
        import onnx
        import onnxruntime as ort

        print("Validating ONNX model...")
        onnx_model = onnx.load(args.output)
        onnx.checker.check_model(onnx_model)
        print("ONNX check passed.")

        session = ort.InferenceSession(args.output)
        test_input = np.random.randn(1, 3, 224, 224).astype(np.float32)
        yaw_out, pitch_out = session.run(None, {"face_crop": test_input})

        print(f"Output shapes: yaw={yaw_out.shape}, pitch={pitch_out.shape}")
        print(f"Yaw sum:   {yaw_out.sum():.4f} (should be ~1.0)")
        print(f"Pitch sum: {pitch_out.sum():.4f} (should be ~1.0)")

        yaw_angle = bins_to_angle(yaw_out[0])
        pitch_angle = bins_to_angle(pitch_out[0])
        print(f"Random input → yaw={yaw_angle:.1f}°, pitch={pitch_angle:.1f}°")
        print("Validation PASSED.")


if __name__ == "__main__":
    main()
