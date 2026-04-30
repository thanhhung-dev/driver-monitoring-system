from typing import Optional, Tuple
import math
import cv2
import numpy as np
from skimage.transform import SimilarityTransform

# Reference alignment for facial landmarks (ArcFace)
reference_alignment: np.ndarray = np.array(
    [
        [38.2946, 51.6963],
        [73.5318, 51.5014],
        [56.0252, 71.7366],
        [41.5493, 92.3655],
        [70.7299, 92.2041]
    ],
    dtype=np.float32
)


def estimate_norm(landmark: np.ndarray, image_size: int = 112) -> Tuple[np.ndarray, np.ndarray]:
    """
    Estimate the normalization transformation matrix for facial landmarks.

    Args:
        landmark (np.ndarray): Array of shape (5, 2) representing the coordinates of the facial landmarks.
        image_size (int, optional): The size of the output image. Default is 112.

    Returns:
        np.ndarray: The 2x3 transformation matrix for aligning the landmarks.
        np.ndarray: The 2x3 inverse transformation matrix for aligning the landmarks.

    Raises:
        ValueError: If the input landmark array does not have the shape (5, 2)
                    or if image_size is not a multiple of 112 or 128.
    """
    if landmark.shape != (5, 2):
        raise ValueError(f"Landmark array must have shape (5, 2), got {landmark.shape}.")
    if image_size % 112 != 0 and image_size % 128 != 0:
        raise ValueError(f"Image size must be a multiple of 112 or 128, got {image_size}.")

    if image_size % 112 == 0:
        ratio = float(image_size) / 112.0
        diff_x = 0.0
    else:
        ratio = float(image_size) / 128.0
        diff_x = 8.0 * ratio

    # Adjust reference alignment based on ratio and diff_x
    alignment = reference_alignment * ratio
    alignment[:, 0] += diff_x

    # Compute the transformation matrix
    transform = SimilarityTransform()
    transform.estimate(landmark, alignment)

    matrix = transform.params[0:2, :]
    inverse_matrix = np.linalg.inv(transform.params)[0:2, :]

    return matrix, inverse_matrix


def face_alignment(image: np.ndarray, landmark: np.ndarray, image_size: int = 112) -> Tuple[np.ndarray, np.ndarray]:
    """
    Align the face in the input image based on the given facial landmarks.

    Args:
        image (np.ndarray): Input image as a NumPy array.
        landmark (np.ndarray): Array of shape (5, 2) representing the coordinates of the facial landmarks.
        image_size (int, optional): The size of the aligned output image. Default is 112.

    Returns:
        np.ndarray: The aligned face as a NumPy array.
        np.ndarray: The 2x3 transformation matrix used for alignment.
    """
    # Get the transformation matrix
    M, M_inv = estimate_norm(landmark, image_size)

    # Warp the input image to align the face
    warped = cv2.warpAffine(image, M, (image_size, image_size), borderValue=0.0)

    return warped, M_inv


def distance2bbox(
    points: np.ndarray,
    distance: np.ndarray,
    max_shape: Optional[Tuple[int, int]] = None,
) -> np.ndarray:
    """Decode distance prediction to bounding box.

    Args:
        points (np.ndarray): Shape (n, 2), [x, y].
        distance (np.ndarray): Distance from the given point to 4
            boundaries (left, top, right, bottom).
        max_shape (tuple, optional): Shape of the image as (height, width).

    Returns:
        np.ndarray: Decoded bounding boxes with shape (n, 4).
    """
    x1 = points[:, 0] - distance[:, 0]
    y1 = points[:, 1] - distance[:, 1]
    x2 = points[:, 0] + distance[:, 2]
    y2 = points[:, 1] + distance[:, 3]
    if max_shape is not None:
        x1 = np.clip(x1, 0, max_shape[1])
        y1 = np.clip(y1, 0, max_shape[0])
        x2 = np.clip(x2, 0, max_shape[1])
        y2 = np.clip(y2, 0, max_shape[0])
    return np.stack([x1, y1, x2, y2], axis=-1)


def distance2kps(
    points: np.ndarray,
    distance: np.ndarray,
    max_shape: Optional[Tuple[int, int]] = None,
) -> np.ndarray:
    """Decode distance prediction to keypoints.

    Args:
        points (np.ndarray): Shape (n, 2), [x, y].
        distance (np.ndarray): Distance from the given point to keypoint
            offsets.
        max_shape (tuple, optional): Shape of the image as (height, width).

    Returns:
        np.ndarray: Decoded keypoints with shape (n, 2k).
    """
    preds = []
    for i in range(0, distance.shape[1], 2):
        px = points[:, i % 2] + distance[:, i]
        py = points[:, i % 2 + 1] + distance[:, i + 1]
        if max_shape is not None:
            px = np.clip(px, 0, max_shape[1])
            py = np.clip(py, 0, max_shape[0])
        preds.append(px)
        preds.append(py)
    return np.stack(preds, axis=-1)


def compute_similarity(feat1: np.ndarray, feat2: np.ndarray) -> np.float32:
    """Computing Similarity between two faces.

    Args:
        feat1 (np.ndarray): Face features.
        feat2 (np.ndarray): Face features.

    Returns:
        np.float32: Cosine similarity between face features.
    """
    feat1 = feat1.ravel()
    feat2 = feat2.ravel()
    similarity = np.dot(feat1, feat2) / (np.linalg.norm(feat1) * np.linalg.norm(feat2))
    return similarity


def draw_bbox(
    image: np.ndarray,
    bbox: list[int],
    color: Tuple[int, int, int] = (255, 0, 255),
    thickness: int = 2,
    proportion: float = 0.2,
) -> None:
    """Draw a bounding box with corner accents on the image (in-place).

    Args:
        image (np.ndarray): Frame to draw on.
        bbox: Bounding box coordinates [x1, y1, x2, y2].
        color: BGR color tuple.
        thickness: Corner line thickness.
        proportion: Corner accent length as fraction of the shorter bbox side.
    """
    x1, y1, x2, y2 = map(int, bbox)
    width = x2 - x1
    height = y2 - y1

    corner_length = int(proportion * min(width, height))

    # Draw the rectangle
    cv2.rectangle(image, (x1, y1), (x2, y2), color, 1)

    # Top-left corner
    cv2.line(image, (x1, y1), (x1 + corner_length, y1), color, thickness)
    cv2.line(image, (x1, y1), (x1, y1 + corner_length), color, thickness)

    # Top-right corner
    cv2.line(image, (x2, y1), (x2 - corner_length, y1), color, thickness)
    cv2.line(image, (x2, y1), (x2, y1 + corner_length), color, thickness)

    # Bottom-left corner
    cv2.line(image, (x1, y2), (x1, y2 - corner_length), color, thickness)
    cv2.line(image, (x1, y2), (x1 + corner_length, y2), color, thickness)

    # Bottom-right corner
    cv2.line(image, (x2, y2), (x2, y2 - corner_length), color, thickness)
    cv2.line(image, (x2, y2), (x2 - corner_length, y2), color, thickness)


def draw_bbox_info(
    frame: np.ndarray,
    bbox: list[int],
    similarity: float,
    name: str,
    color: Tuple[int, int, int],
) -> None:
    """Draw bounding box with identity label and similarity bar (in-place).

    Args:
        frame (np.ndarray): Frame to draw on.
        bbox: Bounding box coordinates [x1, y1, x2, y2].
        similarity: Cosine similarity score.
        name: Identity label to display.
        color: BGR color tuple.
    """
    x1, y1, x2, y2 = map(int, bbox)

    # Keep text label within frame bounds
    text_y = max(y1 - 10, 15)
    cv2.putText(
        frame,
        f"{name}: {similarity:.2f}",
        org=(x1, text_y),
        fontFace=cv2.FONT_HERSHEY_COMPLEX_SMALL,
        fontScale=1,
        color=color,
        thickness=1,
    )

    # Draw bounding box
    draw_bbox(frame, bbox, color)

    # Draw similarity bar (clamp to [0, 1] to avoid negative height)
    clamped_sim = float(np.clip(similarity, 0.0, 1.0))
    rect_x_start = x2 + 10
    rect_x_end = rect_x_start + 10
    rect_y_end = y2
    rect_height = int(clamped_sim * (y2 - y1))
    rect_y_start = rect_y_end - rect_height

    # Draw the filled rectangle
    cv2.rectangle(frame, (rect_x_start, rect_y_start), (rect_x_end, rect_y_end), color, cv2.FILLED)


# ─────────────────────────────────────────────────────────────────────────────
# Head pose visualization
# ─────────────────────────────────────────────────────────────────────────────

def draw_axis(image: np.ndarray, yaw: float, pitch: float, roll: float,
              bbox: list, size_ratio: float = 0.5,
              corner: str = "top-right", corner_size: int = 60, margin: int = 20) -> None:
    """
    Vẽ hệ trục 3D (X-red yaw, Y-green pitch, Z-blue roll) bằng phép chiếu
    chuẩn từ rotation matrix R = Rz @ Ry @ Rx.

    Sửa các lỗi cũ:
      1. Có phép chiếu 3D→2D đầy đủ (lấy cả thành phần Z, không bị "dẹt").
      2. Đúng thứ tự rotation: R = Rz @ Ry @ Rx.
      3. Convert độ → radian bằng np.deg2rad.
      4. Flip dấu yaw để khớp hệ toạ độ image (x→phải, y→xuống, z→ra ngoài).
      5. Z scale bằng X, Y nên không bị "invisible".
    """
    h, w = image.shape[:2]

    # ── Origin & size ──
    if corner is None:
        x_min, y_min, x_max, y_max = bbox
        tdx = int(x_min + (x_max - x_min) * 0.5)
        tdy = int(y_min + (y_max - y_min) * 0.5)
        bbox_size = min(x_max - x_min, y_max - y_min)
        size = int(bbox_size * size_ratio)
    else:
        if corner == "top-right":
            tdx, tdy = w - margin - corner_size, margin + corner_size
        elif corner == "top-left":
            tdx, tdy = margin + corner_size,     margin + corner_size
        elif corner == "bottom-right":
            tdx, tdy = w - margin - corner_size, h - margin - corner_size
        else:
            tdx, tdy = margin + corner_size,     h - margin - corner_size
        size = corner_size

    # ── (3) degree → radian, (4) flip dấu yaw cho khớp image-space ──
    y = np.deg2rad(-yaw)
    p = np.deg2rad(pitch)
    r = np.deg2rad(roll)

    # ── (2) Rotation matrices đúng thứ tự R = Rz @ Ry @ Rx ──
    Rx = np.array([[1, 0, 0],
                   [0, np.cos(p), -np.sin(p)],
                   [0, np.sin(p),  np.cos(p)]])
    Ry = np.array([[ np.cos(y), 0, np.sin(y)],
                   [ 0,         1, 0       ],
                   [-np.sin(y), 0, np.cos(y)]])
    Rz = np.array([[np.cos(r), -np.sin(r), 0],
                   [np.sin(r),  np.cos(r), 0],
                   [0,          0,         1]])
    R = Rz @ Ry @ Rx

    # ── (5) 3 trục đơn vị cùng scale = size ──
    axes_3d = np.array([[size, 0,    0   ],   # X
                        [0,    size, 0   ],   # Y
                        [0,    0,    size]]).T  # 3×3, mỗi cột là 1 trục

    rotated = R @ axes_3d  # 3×3

    # ── (1) Phép chiếu 3D→2D: lấy x, y; image y hướng xuống nên đảo dấu ──
    def project(col):
        return (int(tdx + rotated[0, col]),
                int(tdy - rotated[1, col]))   # flip y trục image

    x_end = project(0)
    y_end = project(1)
    z_end = project(2)

    # Painter's algorithm: vẽ trục có Z nhỏ trước (xa hơn) để trục gần đè lên
    axes = [
        (x_end, (0, 0, 255), rotated[2, 0]),   # Red   - X (yaw)
        (y_end, (0, 255, 0), rotated[2, 1]),   # Green - Y (pitch)
        (z_end, (255, 0, 0), rotated[2, 2]),   # Blue  - Z (roll)
    ]
    for end, color, _ in sorted(axes, key=lambda a: a[2]):
        cv2.line(image, (tdx, tdy), end, color, 2, cv2.LINE_AA)



def expand_bbox(x_min: int, y_min: int, x_max: int, y_max: int, factor: float = 0.2) -> Tuple[int, int, int, int]:
    """Expand bounding box by a factor to include more context for head pose estimation.

    Args:
        x_min, y_min, x_max, y_max: Bounding box coordinates.
        factor: Expansion factor (0.2 = 20% each side).

    Returns:
        Expanded bounding box as (x_min, y_min, x_max, y_max).
    """
    width = x_max - x_min
    height = y_max - y_min

    x_min_new = x_min - int(factor * height)
    y_min_new = y_min - int(factor * width)
    x_max_new = x_max + int(factor * height)
    y_max_new = y_max + int(factor * width)

    return max(0, x_min_new), max(0, y_min_new), x_max_new, y_max_new