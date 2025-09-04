"""감정별 이모지 합성 로직.

- `add_emotion_emojis(...)`: 감정 결과에 맞춰 얼굴 상단에 이모지를 배치하고 저장합니다.
  다른 얼굴과의 겹침을 방지하는 위치 조정 로직 포함.
"""

from __future__ import annotations
import os
import cv2
import numpy as np
from typing import Dict, List
from .utils import overlay_png, find_non_overlapping_position


def add_emotion_emojis(
    image_path: str,
    emotions: List[dict],
    out_path: str = "with_emoji.png",
    emoji_map: Dict[str, str] | None = None,
    size_scale: float = 0.6,
    y_offset_ratio: float = 0.15,
    avoid_overlap: bool = True,
) -> str:
    """감정 결과에 맞춰 얼굴 위에 이모지 합성 후 저장
    
    Args:
        image_path: 원본 이미지 경로
        emotions: 감정 분석 결과 리스트
        out_path: 저장할 파일 경로
        emoji_map: 감정명 -> 이모지 파일 경로 매핑
        size_scale: 얼굴 크기 대비 이모지 크기 비율
        y_offset_ratio: 얼굴 위쪽 오프셋 비율
        avoid_overlap: 다른 얼굴과의 겹침 방지 여부
    
    Returns:
        str: 저장된 파일 경로
    """
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"이미지를 읽을 수 없습니다: {image_path}")
    
    img_height, img_width = img.shape[:2]
    
    if not emoji_map:
        emoji_map = {}

    cache: Dict[str, np.ndarray] = {}
    
    # 모든 얼굴의 bbox를 미리 추출 (겹침 방지용)
    all_face_boxes = []
    if avoid_overlap:
        for e in emotions:
            x, y, w, h = e.get("box", (None, None, None, None))
            if None not in (x, y, w, h):
                all_face_boxes.append((x, y, w, h))

    for i, e in enumerate(emotions):
        emotion = e.get("emotion", "").lower()
        x, y, w, h = e.get("box", (None, None, None, None))
        
        if emotion not in emoji_map or None in (x, y, w, h):
            continue
            
        path = emoji_map[emotion]
        if not os.path.exists(path):
            continue
            
        # 이모지 로드 및 캐싱
        if emotion not in cache:
            png = cv2.imread(path, cv2.IMREAD_UNCHANGED)  # BGRA
            if png is None:
                continue
            cache[emotion] = png
        png = cache[emotion]

        # 이모지 크기 계산 및 리사이즈
        size = max(24, int(h * size_scale))
        png = cv2.resize(png, (size, size), interpolation=cv2.INTER_AREA)
        
        if avoid_overlap and len(all_face_boxes) > 1:
            # 현재 얼굴을 제외한 다른 얼굴들의 bbox
            other_boxes = [box for j, box in enumerate(all_face_boxes) if j != i]
            
            # 겹치지 않는 위치 찾기
            px, py = find_non_overlapping_position(
                target_face_box=(x, y, w, h),
                other_face_boxes=other_boxes,
                emoji_size=size,
                img_width=img_width,
                img_height=img_height,
                y_offset_ratio=y_offset_ratio
            )
        else:
            # 기존 방식 (단순히 얼굴 위쪽 중앙)
            y_off = int(h * y_offset_ratio)
            px = int(x + w / 2 - size / 2)
            py = int(y - size - y_off)
        
        # 이모지 오버레이
        img = overlay_png(img, png, px, py)

    cv2.imwrite(out_path, img)
    return out_path