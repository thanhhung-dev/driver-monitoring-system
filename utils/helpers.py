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
    color: Tuple[int, int, int] = (255, 255, 255),
    thickness: int = 2,
    proportion: float = 0.07,
    fixed_size: Tuple[int, int] | None = None,
) -> None:
    """Draw a bounding box with corner accents on the image (in-place).

    Args:
        image (np.ndarray): Frame to draw on.
        bbox: Bounding box coordinates [x1, y1, x2, y2].
        color: BGR color tuple.
        thickness: Corner line thickness.
        proportion: Corner accent length as fraction of the shorter bbox side.
        fixed_size: Optional (width, height) to override bbox size, centered on bbox.
    """
    x1, y1, x2, y2 = map(int, bbox)
    if fixed_size is not None:
        fw, fh = fixed_size
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        x1, y1 = cx - fw // 2, cy - fh // 2
        x2, y2 = cx + fw // 2, cy + fh // 2
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

    y = np.deg2rad(yaw)
    # App convention is pitch+ = looking up; image-space rotation uses the
    # opposite X-axis sign so the projected Z axis points upward.
    p = np.deg2rad(-pitch)
    r = np.deg2rad(roll)

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

    axes_3d = np.array([[size, 0,    0   ], 
                        [0,    size, 0   ],   
                        [0,    0,    size]]).T 

    rotated = R @ axes_3d  # 3×3

    def project(col):
        return (int(tdx + rotated[0, col]),
                int(tdy - rotated[1, col]))   

    x_end = project(0)
    y_end = project(1)
    z_end = project(2)

    axes = [
        (x_end, (0, 255, 0), rotated[2, 0]),   
        (y_end, (255, 0, 0), rotated[2, 1]),   
        (z_end, (0, 0, 255), rotated[2, 2]),  
    ]
    for end, color, _ in sorted(axes, key=lambda a: a[2]):
        if(color == (0,0,255)):
            cv2.arrowedLine(image,(tdx,tdy), end, color,2,cv2.LINE_AA, tipLength=0.25)
        cv2.line(image, (tdx, tdy), end, color, 2, cv2.LINE_AA)


def draw_head_direction_arrow(
    image: np.ndarray,
    landmarks: np.ndarray,
    yaw: float,
    pitch: float,
    roll: float = 0.0,
    length: int = 40,
    color: tuple[int, int, int] = (0, 0, 255),
    thickness: int = 1,
) -> None:
    """Vẽ mũi tên đỏ 3D từ mũi chỉ hướng head (yaw + pitch + roll).

    Dùng rotation matrix R = Rz @ Ry @ Rx để chiếu vector [0,0,-1]
    (hướng trước mặt) sang tọa độ ảnh 2D — mũi tên xoay đúng theo
    cả 3 trục, không còn phẳng 2D.

    Args:
        landmarks: (N, 2|3) pixel coords — tự detect nose tip theo N.
        yaw, pitch, roll: độ (head pose output).
        length: độ dài mũi tên (px).
    """
    lm = np.asarray(landmarks, dtype=np.float32)
    n = len(lm)
    # Nose tip index theo landmark format
    if n >= 468:
        nose_idx = 1       # MediaPipe 468
    elif n >= 68:
        nose_idx = 30      # dlib 68-point
    else:
        nose_idx = 0       # 5-point fallback

    nose = lm[nose_idx, :2]
    nx, ny = int(round(nose[0])), int(round(nose[1]))

    # Rotation matrix đầy đủ
    y = np.deg2rad(yaw)
    # App convention is pitch+ = looking up; invert for image projection.
    p = np.deg2rad(-pitch)
    r = np.deg2rad(roll)

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
    forward = R @ np.array([0.0, 0.0, 1.0])
    dx = length * forward[0]
    # Trục y toán học hướng lên, ảnh hướng xuống → đảo dấu cho khớp
    # convention của draw_axis (tdy - rotated[1]); nếu không sẽ lật up/down.
    dy = -length * forward[1]
    ex, ey = int(round(nx + dx)), int(round(ny + dy))
    line_len = float(np.hypot(dx, dy))
    TIP_PX = 5.0
    tip_ratio = float(np.clip(TIP_PX / line_len, 0.05, 0.5)) if line_len > 1e-3 else 0.3
    cv2.arrowedLine(image, (nx, ny), (ex, ey), color, thickness, cv2.LINE_AA, tipLength=tip_ratio)



_FACE_MODEL_3D = np.array([
    (  0.0,    0.0,    0.0),    # 0: nose tip
    (  0.0, -330.0,  -65.0),    # 1: chin
    (-225.0, 170.0, -135.0),    # 2: left eye outer corner
    ( 225.0, 170.0, -135.0),    # 3: right eye outer corner
    (-150.0,-150.0, -125.0),    # 4: left mouth corner
    ( 150.0,-150.0, -125.0),    # 5: right mouth corner
], dtype=np.float64)


def estimate_head_pose_pnp(image_points: np.ndarray,
                           image_shape: Tuple[int, int]
                           ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray],
                                      Optional[np.ndarray], np.ndarray, np.ndarray]:
    """Estimate head pose from 6 facial landmarks using cv2.solvePnP.

    Args:
        image_points: (6, 2) float array of pixel coordinates in the order
            [nose tip, chin, left-eye outer, right-eye outer,
             left-mouth, right-mouth].
        image_shape:  (height, width) of the source image.

    Returns:
        rvec:           Rodrigues rotation vector (3, 1)  or None on failure.
        tvec:           Translation vector       (3, 1)   or None on failure.
        rot_matrix:     3x3 rotation matrix              or None on failure.
        camera_matrix:  3x3 pinhole intrinsics used.
        dist_coeffs:    (4, 1) zero distortion (assumed).
    """
    image_points = np.asarray(image_points, dtype=np.float64).reshape(-1, 2)
    if image_points.shape[0] < 6:
        raise ValueError(f"Need 6 landmarks, got {image_points.shape[0]}.")

    h, w = image_shape[:2]

    # (4) Pinhole camera intrinsics:
    #     focal length ≈ image width, principal point at image center.
    focal_length = float(w)
    center = (w * 0.5, h * 0.5)
    camera_matrix = np.array([
        [focal_length, 0,            center[0]],
        [0,            focal_length, center[1]],
        [0,            0,            1.0     ],
    ], dtype=np.float64)

    # Assume no lens distortion (good enough for visualization).
    dist_coeffs = np.zeros((4, 1), dtype=np.float64)

    # (3) solvePnP: 3D model points  ↔  2D image points  →  rvec, tvec.
    ok, rvec, tvec = cv2.solvePnP(
        _FACE_MODEL_3D,
        image_points[:6],
        camera_matrix,
        dist_coeffs,
        flags=cv2.SOLVEPNP_SQPNP,
    )
    if not ok:
        return None, None, None, camera_matrix, dist_coeffs

    # Convert rotation vector → 3x3 rotation matrix.
    rot_matrix, _ = cv2.Rodrigues(rvec)
    return rvec, tvec, rot_matrix, camera_matrix, dist_coeffs


def draw_axis_pnp(image: np.ndarray,
                  image_points: np.ndarray,
                  axis_length: float = 100.0) -> bool:
    """Draw 3D head-pose axes on `image` using a real perspective projection.

    Pipeline:
      1. Build a generic 3D face model (6 points).
      2. solvePnP with the matching 2D landmarks → rvec, tvec.
      3. Define the 3 unit axes in 3D (scaled by `axis_length` mm).
      4. cv2.projectPoints  → pixel coords for the axis tips.
      5. Draw red=X, green=Y, blue=Z lines from the nose tip.

    Args:
        image:        BGR frame (modified in-place).
        image_points: (6, 2) landmarks in the order documented in
                      `estimate_head_pose_pnp`.
        axis_length:  Length of each axis in millimeters (model units).

    Returns:
        True if pose was estimated and axes were drawn, False otherwise.
    """
    rvec, tvec, _, camera_matrix, dist_coeffs = estimate_head_pose_pnp(
        image_points, image.shape[:2]
    )
    if rvec is None:
        return False

    # (5) 3D axes in the face's local frame, anchored at the nose tip.
    # Convention chosen for an intuitive head-pose visualization
    # (axes follow where the head "looks"):
    #
    #   +X (red)   : ra phía bên phải mặt
    #   +Y (green) : hướng XUỐNG dưới trong ảnh
    #                (model có +Y up nên ta vẽ endpoint dọc -Y)
    #   +Z (blue)  : hướng RA KHỎI mặt cùng chiều gaze
    #                (model có +Z out-of-face; sau R_align từ solvePnP
    #                 chiều +Z ánh xạ thành "into screen" trong camera,
    #                 nên cúi đầu → Z đi XUỐNG ảnh, quay phải → Z sang PHẢI ảnh)
    axes_3d = np.float64([
        [ axis_length,  0,            0          ],   # +X (right of face)
        [ 0,           -axis_length,  0          ],   # +Y (down in image)
        [ 0,            0,            axis_length],   # +Z (along gaze direction)
    ])

    # (6) Perspective projection of the axis endpoints.
    projected, _ = cv2.projectPoints(
        axes_3d, rvec, tvec, camera_matrix, dist_coeffs
    )
    projected = projected.reshape(-1, 2)

    # Origin on the image = projected nose tip (1st landmark).
    origin = tuple(np.int32(image_points[0]))
    x_pt = tuple(np.int32(projected[0]))
    y_pt = tuple(np.int32(projected[1]))
    z_pt = tuple(np.int32(projected[2]))

    # (7) Draw axes — note OpenCV uses BGR.
    cv2.line(image, origin, x_pt, (255, 0, 0), 3, cv2.LINE_AA)   # X – red
    cv2.line(image, origin, y_pt, (0, 0, 255), 3, cv2.LINE_AA)   # Y – green
    cv2.line(image, origin, z_pt, (0, 255, 0), 3, cv2.LINE_AA)   # Z – blue
    return True


def landmarks68_to_pnp_points(landmarks68) -> np.ndarray:
    """Pick the 6 PnP landmarks from a 68-point (dlib/iBUG) landmark set.

    Indices (0-based): 30 nose tip, 8 chin, 36 left-eye outer,
    45 right-eye outer, 48 left-mouth, 54 right-mouth.
    """
    lm = np.asarray(landmarks68, dtype=np.float64).reshape(-1, 2)
    return np.stack([lm[30], lm[8], lm[36], lm[45], lm[48], lm[54]], axis=0)


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

    x_min_new = x_min - int(factor * width)
    y_min_new = y_min - int(factor * height)
    x_max_new = x_max + int(factor * width)
    y_max_new = y_max + int(factor * height)

    return max(0, x_min_new), max(0, y_min_new), x_max_new, y_max_new
