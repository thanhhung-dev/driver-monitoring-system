"""Cross-frame channel carrying the most recent head pose result.

Vì mỗi frame tạo một FrameContext mới và DetectStage chạy TRƯỚC HeadPoseStage,
DetectStage không thể đọc head_pose của frame hiện tại (luôn None). Holder này
cho HeadPoseStage ghi lại kết quả của frame N để DetectStage đọc ở frame N+1,
nhờ đó cơ chế extreme_pose_mode mới hoạt động được.
"""

from __future__ import annotations


class HeadPoseFeedback:
    """Shared, mutable holder for the latest computed head pose angles."""

    def __init__(self) -> None:
        # (yaw, pitch, roll) in degrees, or None if not yet computed.
        self.last_head_pose: tuple[float, float, float] | None = None
