"""이미지 유틸리티 함수 모음.

- `overlay_png`: 투명 PNG 레이어(fg, BGRA)를 배경 이미지(bg, BGR)에 알파 블렌딩으로 합성
- `boxes_overlap`: 두 bounding box가 겹치는지 확인
- `find_non_overlapping_position`: 다른 얼굴들과 겹치지 않는 이모지 위치 찾기
"""

from __future__ import annotations
import numpy as np
import cv2
from typing import List, Tuple, Optional


def overlay_png(bg: np.ndarray, fg: np.ndarray, x: int, y: int) -> np.ndarray:
    """투명 PNG(fg, BGRA)를 배경(bg, BGR)에 합성"""
    H, W = bg.shape[:2]
    Hf, Wf = fg.shape[:2]
    if x >= W or y >= H:
        return bg
    x1, y1 = max(0, x), max(0, y)
    x2, y2 = min(W, x + Wf), min(H, y + Hf)
    fg_x1, fg_y1 = x1 - x, y1 - y
    fg_x2, fg_y2 = fg_x1 + (x2 - x1), fg_y1 + (y2 - y1)
    if x1 >= x2 or y1 >= y2:
        return bg

    roi = bg[y1:y2, x1:x2]
    fg_crop = fg[fg_y1:fg_y2, fg_x1:fg_x2]
    if fg_crop.shape[2] == 4:
        alpha = fg_crop[:, :, 3:4] / 255.0
        fg_rgb = fg_crop[:, :, :3]
        blended = (alpha * fg_rgb + (1.0 - alpha) * roi).astype(np.uint8)
        bg[y1:y2, x1:x2] = blended
    else:
        bg[y1:y2, x1:x2] = fg_crop[:, :, :3]
    return bg


def boxes_overlap(box1: Tuple[int, int, int, int], box2: Tuple[int, int, int, int]) -> bool:
    """두 bounding box (x, y, w, h)가 겹치는지 확인"""
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    
    # box1의 오른쪽이 box2의 왼쪽보다 왼쪽에 있거나
    # box1의 왼쪽이 box2의 오른쪽보다 오른쪽에 있으면 겹치지 않음
    if x1 + w1 <= x2 or x1 >= x2 + w2:
        return False
    
    # box1의 아래쪽이 box2의 위쪽보다 위에 있거나
    # box1의 위쪽이 box2의 아래쪽보다 아래에 있으면 겹치지 않음
    if y1 + h1 <= y2 or y1 >= y2 + h2:
        return False
    
    return True


def find_non_overlapping_position(
    target_face_box: Tuple[int, int, int, int],
    other_face_boxes: List[Tuple[int, int, int, int]],
    emoji_size: int,
    img_width: int,
    img_height: int,
    y_offset_ratio: float = 0.15
) -> Tuple[int, int]:
    """다른 얼굴들과 겹치지 않는 이모지 위치를 찾아 반환
    
    Args:
        target_face_box: 이모지를 배치할 대상 얼굴의 bbox (x, y, w, h)
        other_face_boxes: 다른 모든 얼굴들의 bbox 리스트
        emoji_size: 이모지 크기 (정사각형)
        img_width, img_height: 이미지 크기
        y_offset_ratio: 기본 Y 오프셋 비율
    
    Returns:
        (px, py): 이모지 배치할 좌표
    """
    x, y, w, h = target_face_box
    y_offset = int(h * y_offset_ratio)
    
    # 기본 위치 (얼굴 위쪽 중앙)
    default_px = int(x + w / 2 - emoji_size / 2)
    default_py = int(y - emoji_size - y_offset)
    
    # 대안 위치들을 시도해볼 순서
    positions = [
        # 기본 위치
        (default_px, default_py),
        # 더 위쪽
        (default_px, int(y - emoji_size - y_offset * 2)),
        # 더 더 위쪽
        (default_px, int(y - emoji_size - y_offset * 3)),
        # 얼굴 오른쪽
        (int(x + w + y_offset), int(y + h / 2 - emoji_size / 2)),
        # 얼굴 왼쪽
        (int(x - emoji_size - y_offset), int(y + h / 2 - emoji_size / 2)),
        # 얼굴 아래쪽
        (default_px, int(y + h + y_offset)),
    ]
    
    for px, py in positions:
        # 이미지 경계 체크
        if px < 0 or py < 0 or px + emoji_size > img_width or py + emoji_size > img_height:
            continue
            
        # 이모지가 배치될 영역
        emoji_box = (px, py, emoji_size, emoji_size)
        
        # 다른 얼굴들과 겹치는지 확인
        overlaps = False
        for other_box in other_face_boxes:
            if boxes_overlap(emoji_box, other_box):
                overlaps = True
                break
        
        if not overlaps:
            return px, py
    
    # 모든 대안 위치에서 겹칠 경우 기본 위치 반환
    return default_px, default_py