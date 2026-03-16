import os
from functools import partial
from typing import Any, Callable, List, Optional, Sequence

import cv2
import numpy as np
import torch
from torch import nn, Tensor
from torchvision.models import MobileNet_V3_Large_Weights, MobileNet_V3_Small_Weights, WeightsEnum

from utils.general import compute_rotation_matrix_from_ortho6d
from detection.common import _make_divisible, Conv2dNormActivation, load_filtered_state_dict,  SqueezeExcitation as SElayer


__all__ = ["mobilenet_v3_large", "mobilenet_v3_small"]


class InvertedResidualConfig:
    # Stores information listed at Tables 1 and 2 of the MobileNetV3 paper
    def __init__(
        self,
        input_channels: int,
        kernel: int,
        expanded_channels: int,
        out_channels: int,
        use_se: bool,
        activation: str,
        stride: int,
        dilation: int,
        width_mult: float,
    ):
        self.input_channels = self.adjust_channels(input_channels, width_mult)
        self.kernel = kernel
        self.expanded_channels = self.adjust_channels(expanded_channels, width_mult)
        self.out_channels = self.adjust_channels(out_channels, width_mult)
        self.use_se = use_se
        self.use_hs = activation == "HS"
        self.stride = stride
        self.dilation = dilation

    @staticmethod
    def adjust_channels(channels: int, width_mult: float):
        return _make_divisible(channels * width_mult, 8)


class InvertedResidual(nn.Module):
    # Implemented as described at section 5 of MobileNetV3 paper
    def __init__(
        self,
        cnf: InvertedResidualConfig,
        se_layer: Callable[..., nn.Module] = partial(SElayer, scale_activation=nn.Hardsigmoid),
    ):
        super().__init__()
        if cnf.stride not in [1, 2]:
            raise ValueError(f"stride should be 1 or 2 instead of {cnf.stride}")

        self.use_res_connect = cnf.stride == 1 and cnf.input_channels == cnf.out_channels

        layers: List[nn.Module] = []
        activation_layer = nn.Hardswish if cnf.use_hs else nn.ReLU

        # expand
        if cnf.expanded_channels != cnf.input_channels:
            layers.append(
                Conv2dNormActivation(
                    cnf.input_channels,
                    cnf.expanded_channels,
                    kernel_size=1,
                    activation_layer=activation_layer,
                )
            )

        # depthwise
        stride = 1 if cnf.dilation > 1 else cnf.stride
        layers.append(
            Conv2dNormActivation(
                cnf.expanded_channels,
                cnf.expanded_channels,
                kernel_size=cnf.kernel,
                stride=stride,
                dilation=cnf.dilation,
                groups=cnf.expanded_channels,
                activation_layer=activation_layer,
            )
        )
        if cnf.use_se:
            squeeze_channels = _make_divisible(cnf.expanded_channels // 4, 8)
            layers.append(se_layer(cnf.expanded_channels, squeeze_channels))

        # project
        layers.append(
            Conv2dNormActivation(cnf.expanded_channels, cnf.out_channels, kernel_size=1, activation_layer=None)
        )

        self.block = nn.Sequential(*layers)
        self.out_channels = cnf.out_channels
        self._is_cn = cnf.stride > 1

    def forward(self, input: Tensor) -> Tensor:
        result = self.block(input)
        if self.use_res_connect:
            result += input
        return result


class MobileNetV3(nn.Module):
    def __init__(
        self,
        inverted_residual_setting: List[InvertedResidualConfig],
        last_channel: int,
        num_classes: int = 1000,
        dropout: float = 0.2,
        **kwargs: Any,
    ) -> None:
        """
        MobileNet V3 main class

        Args:
            inverted_residual_setting (List[InvertedResidualConfig]): Network structure
            last_channel (int): The number of channels on the penultimate layer
            num_classes (int): Number of classes
            dropout (float): The droupout probability
        """
        super().__init__()

        if not inverted_residual_setting:
            raise ValueError("The inverted_residual_setting should not be empty")
        elif not (
            isinstance(inverted_residual_setting, Sequence)
            and all([isinstance(s, InvertedResidualConfig) for s in inverted_residual_setting])
        ):
            raise TypeError("The inverted_residual_setting should be List[InvertedResidualConfig]")

        layers: List[nn.Module] = []

        # building first layer
        firstconv_output_channels = inverted_residual_setting[0].input_channels
        layers.append(
            Conv2dNormActivation(
                3,
                firstconv_output_channels,
                kernel_size=3,
                stride=2,
                activation_layer=nn.Hardswish,
            )
        )

        # building inverted residual blocks
        for cnf in inverted_residual_setting:
            layers.append(InvertedResidual(cnf))

        # building last several layers
        lastconv_input_channels = inverted_residual_setting[-1].out_channels
        lastconv_output_channels = 6 * lastconv_input_channels
        layers.append(
            Conv2dNormActivation(
                lastconv_input_channels,
                lastconv_output_channels,
                kernel_size=1,
                activation_layer=nn.Hardswish,
            )
        )

        self.features = nn.Sequential(*layers)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # build classifier
        # self.classifier = nn.Sequential(
        #     nn.Linear(lastconv_output_channels, last_channel),
        #     nn.Hardswish(inplace=True),
        #     nn.Dropout(p=dropout, inplace=True),
        #     nn.Linear(last_channel, num_classes),
        # )

        self.linear_reg = nn.Sequential(
            nn.Linear(lastconv_output_channels, last_channel),
            nn.Hardswish(inplace=True),
            nn.Linear(last_channel, num_classes)
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x: Tensor) -> Tensor:
        x = self.features(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)

        # classification layer
        # x = self.classifier(x)

        x = self.linear_reg(x)

        # Return 6D representation (not rotation matrix!)
        # The conversion to rotation matrix should be done in the estimator
        return x


def _mobilenet_v3_conf(
    arch: str, width_mult: float = 1.0, reduced_tail: bool = False, dilated: bool = False, **kwargs: Any
):
    reduce_divider = 2 if reduced_tail else 1
    dilation = 2 if dilated else 1

    bneck_conf = partial(InvertedResidualConfig, width_mult=width_mult)
    adjust_channels = partial(InvertedResidualConfig.adjust_channels, width_mult=width_mult)

    if arch == "mobilenet_v3_large":
        inverted_residual_setting = [
            bneck_conf(16, 3, 16, 16, False, "RE", 1, 1),
            bneck_conf(16, 3, 64, 24, False, "RE", 2, 1),  # C1
            bneck_conf(24, 3, 72, 24, False, "RE", 1, 1),
            bneck_conf(24, 5, 72, 40, True, "RE", 2, 1),  # C2
            bneck_conf(40, 5, 120, 40, True, "RE", 1, 1),
            bneck_conf(40, 5, 120, 40, True, "RE", 1, 1),
            bneck_conf(40, 3, 240, 80, False, "HS", 2, 1),  # C3
            bneck_conf(80, 3, 200, 80, False, "HS", 1, 1),
            bneck_conf(80, 3, 184, 80, False, "HS", 1, 1),
            bneck_conf(80, 3, 184, 80, False, "HS", 1, 1),
            bneck_conf(80, 3, 480, 112, True, "HS", 1, 1),
            bneck_conf(112, 3, 672, 112, True, "HS", 1, 1),
            bneck_conf(112, 5, 672, 160 // reduce_divider, True, "HS", 2, dilation),  # C4
            bneck_conf(160 // reduce_divider, 5, 960 // reduce_divider, 160 // reduce_divider, True, "HS", 1, dilation),
            bneck_conf(160 // reduce_divider, 5, 960 // reduce_divider, 160 // reduce_divider, True, "HS", 1, dilation),
        ]
        last_channel = adjust_channels(1280 // reduce_divider)  # C5
    elif arch == "mobilenet_v3_small":
        inverted_residual_setting = [
            bneck_conf(16, 3, 16, 16, True, "RE", 2, 1),  # C1
            bneck_conf(16, 3, 72, 24, False, "RE", 2, 1),  # C2
            bneck_conf(24, 3, 88, 24, False, "RE", 1, 1),
            bneck_conf(24, 5, 96, 40, True, "HS", 2, 1),  # C3
            bneck_conf(40, 5, 240, 40, True, "HS", 1, 1),
            bneck_conf(40, 5, 240, 40, True, "HS", 1, 1),
            bneck_conf(40, 5, 120, 48, True, "HS", 1, 1),
            bneck_conf(48, 5, 144, 48, True, "HS", 1, 1),
            bneck_conf(48, 5, 288, 96 // reduce_divider, True, "HS", 2, dilation),  # C4
            bneck_conf(96 // reduce_divider, 5, 576 // reduce_divider, 96 // reduce_divider, True, "HS", 1, dilation),
            bneck_conf(96 // reduce_divider, 5, 576 // reduce_divider, 96 // reduce_divider, True, "HS", 1, dilation),
        ]
        last_channel = adjust_channels(1024 // reduce_divider)  # C5
    else:
        raise ValueError(f"Unsupported model type {arch}")

    return inverted_residual_setting, last_channel


def _mobilenet_v3(
    inverted_residual_setting: List[InvertedResidualConfig],
    last_channel: int,
    weights: Optional[WeightsEnum],
    progress: bool,
    **kwargs: Any,
) -> MobileNetV3:

    model = MobileNetV3(inverted_residual_setting, last_channel, **kwargs)

    if weights is not None:
        state_dict = weights.get_state_dict(progress=progress, check_hash=True)
        load_filtered_state_dict(model, state_dict)

    return model


def mobilenet_v3_large(*, pretrained: bool = True, progress: bool = True, **kwargs: Any) -> MobileNetV3:
    if pretrained:
        weights = MobileNet_V3_Large_Weights.verify(MobileNet_V3_Large_Weights.IMAGENET1K_V2)
    else:
        weights = None

    inverted_residual_setting, last_channel = _mobilenet_v3_conf("mobilenet_v3_large", **kwargs)
    return _mobilenet_v3(inverted_residual_setting, last_channel, weights, progress, **kwargs)


def mobilenet_v3_small(*, pretrained: bool = True, progress: bool = True, **kwargs: Any) -> MobileNetV3:
    if pretrained:
        weights = MobileNet_V3_Small_Weights.verify(MobileNet_V3_Small_Weights.IMAGENET1K_V1)
    else:
        weights = None

    inverted_residual_setting, last_channel = _mobilenet_v3_conf("mobilenet_v3_small", **kwargs)
    return _mobilenet_v3(inverted_residual_setting, last_channel, weights, progress, **kwargs)


class HeadPoseResult:
    """Container for head pose estimation results."""
    def __init__(self, yaw: float, pitch: float, roll: float):
        self.yaw = yaw
        self.pitch = pitch
        self.roll = roll

    def __repr__(self):
        return f"HeadPoseResult(yaw={self.yaw:.2f}, pitch={self.pitch:.2f}, roll={self.roll:.2f})"


class HeadPoseEstimator:
    """
    Head Pose Estimator using MobileNetV3-Small (6DRepNet).

    Loads a pre-trained MobileNetV3-Small model fine-tuned for head pose estimation.
    The model takes a face crop as input and outputs yaw, pitch, and roll angles.

    Reference: https://yakhyo.github.io/head-pose-estimation/
    """
    def __init__(
        self,
        weights_path: str = "models/mobilenetv3_small.pt",
        input_size: tuple = (224, 224),
        device: str = "cpu"
    ):
        """
        Initialize the Head Pose Estimator.

        Args:
            weights_path: Path to the pre-trained model weights.
            input_size: Input image size as (width, height). Default is 224x224 (ImageNet).
            device: Device to run inference on ('cpu' or 'cuda').
        """
        self.input_size = input_size
        self.device = device

        # Build MobileNetV3-Small model for head pose (6D output)
        self.model = mobilenet_v3_small(pretrained=False, num_classes=6)
        self.model = self.model.to(self.device)
        self.model.eval()

        # Load pre-trained weights if available
        if weights_path and os.path.exists(weights_path):
            try:
                state_dict = torch.load(weights_path, map_location=self.device, weights_only=True)
                # Handle different checkpoint formats
                if 'model' in state_dict:
                    state_dict = state_dict['model']
                elif 'state_dict' in state_dict:
                    state_dict = state_dict['state_dict']

                # Try to load filtered state dict
                load_filtered_state_dict(self.model, state_dict)
                print(f"Loaded head pose weights from {weights_path}")
            except Exception as e:
                print(f"Warning: Could not load custom weights: {e}")
                print("Using random weights (results will be incorrect)")
        else:
            print(f"Weights file not found: {weights_path}")

        # Preprocessing parameters (ImageNet stats)
        # Note: 6DRepNet uses ImageNet normalization
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def estimate_pose(self, face_crop: np.ndarray) -> Optional[HeadPoseResult]:
        """
        Estimate head pose from a face crop.

        Args:
            face_crop: Cropped face image (BGR format from OpenCV).

        Returns:
            HeadPoseResult with yaw, pitch, roll angles in degrees, or None if inference fails.
        """
        if face_crop is None or face_crop.size == 0:
            return None

        try:
            # Preprocess: resize to 224x224 and normalize
            input_img = cv2.resize(face_crop, self.input_size)
            input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB)  # BGR -> RGB
            input_img = input_img.astype(np.float32) / 255.0  # [0, 255] -> [0, 1]

            # Normalize using ImageNet mean/std
            input_img = (input_img - self.mean) / self.std
            input_img = input_img.transpose(2, 0, 1)  # HWC -> CHW
            input_tensor = torch.from_numpy(input_img).unsqueeze(0).float().to(self.device)

            # Inference - get 6D representation
            with torch.no_grad():
                pred_6d = self.model(input_tensor)  # Shape: (1, 6)

            # Convert 6D to rotation matrix
            rotation_matrix = self._ortho6d_to_rotation_matrix(pred_6d)  # Shape: (3, 3)

            # Convert rotation matrix to Euler angles
            euler_angles = self._rotation_matrix_to_euler(rotation_matrix)

            yaw, pitch, roll = euler_angles

            # Convert to degrees (note: yaw and pitch are swapped in some conventions)
            # In 6DRepNet: pitch = X-axis, yaw = Y-axis, roll = Z-axis
            pitch = np.degrees(pitch)
            yaw = np.degrees(yaw)
            roll = np.degrees(roll)

            return HeadPoseResult(yaw=yaw, pitch=pitch, roll=roll)

        except Exception as e:
            print(f"Error in head pose estimation: {e}")
            import traceback
            traceback.print_exc()
            return None

    def _ortho6d_to_rotation_matrix(self, d6: torch.Tensor) -> np.ndarray:
        """
        Convert 6D orthogonal representation to rotation matrix.

        Reference: Zhou et al., "On the Continuity of Rotation Representations in Neural Networks"

        Args:
            d6: Tensor of shape (1, 6) containing 6D vectors.

        Returns:
            3x3 rotation matrix as numpy array.
        """
        # Get the two 3D vectors from 6D representation
        x_raw = d6[0, 0:3]  # First 3D vector
        y_raw = d6[0, 3:6]  # Second 3D vector

        # Normalize x
        x = x_raw / (torch.norm(x_raw) + 1e-8)

        # Orthogonalize y w.r.t x using Gram-Schmidt
        y = y_raw - torch.sum(x * y_raw) * x
        y = y / (torch.norm(y) + 1e-8)

        # Compute z as cross product (specify dim for compatibility)
        z = torch.cross(x, y, dim=0)

        # Stack to form rotation matrix [x, y, z] as columns
        # This gives us the rotation matrix where columns are the basis vectors
        R = torch.stack([x, y, z], dim=0)  # Shape: (3, 3)

        return R.cpu().numpy()

    def _rotation_matrix_to_euler(self, R: np.ndarray) -> tuple:
        """
        Convert rotation matrix to Euler angles (yaw, pitch, roll).

        Uses ZYX convention (yaw-pitch-roll), common in head pose estimation.

        Args:
            R: 3x3 rotation matrix.

        Returns:
            Tuple of (yaw, pitch, roll) in radians.
            Tuple of (yaw, pitch, roll) in radians.
        """
        # Extract Euler angles from rotation matrix
        # Using ZYX convention (yaw-pitch-roll)
        sy = np.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)

        singular = sy < 1e-6

        if not singular:
            roll = np.arctan2(R[2, 1], R[2, 2])
            pitch = np.arctan2(-R[2, 0], sy)
            yaw = np.arctan2(R[1, 0], R[0, 0])
        else:
            roll = np.arctan2(-R[1, 2], R[1, 1])
            pitch = np.arctan2(-R[2, 0], sy)
            yaw = 0

        return yaw, pitch, roll 