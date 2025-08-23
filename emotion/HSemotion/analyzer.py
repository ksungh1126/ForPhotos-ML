"""감정 분석기: 얼굴 검출(MTCNN) + 감정 분류(HSEmotion).

- `EmotionAnalyzer.analyze_emotion`: 단일 이미지 내 여러 얼굴의 감정을 추론
- `EmotionAnalyzer.analyze_multiple_images`: 다중 이미지 일괄 처리
"""

from __future__ import annotations
import os
import cv2
import numpy as np
from typing import Dict, List, Tuple
from facenet_pytorch import MTCNN
from hsemotion.facial_emotions import HSEmotionRecognizer


class EmotionAnalyzer:
    """
    HSEmotion + MTCNN 기반 감정 분석기 (AffectNet 8-class: contempt 포함)
    기본 클래스: Anger, Contempt, Disgust, Fear, Happiness, Neutral, Sadness, Surprise
    """

    def __init__(self, device: str = "cpu", model_name: str = "enet_b2_8", mtcnn_kwargs: dict | None = None):
        # 얼굴 검출기 (MTCNN)
        self.mtcnn = MTCNN(keep_all=True, **(mtcnn_kwargs or {}))
        # 감정 분류기 (HSEmotion)
        self.fer = HSEmotionRecognizer(model_name=model_name, device=device)
        self.class_order = [
            "anger",
            "contempt",
            "disgust",
            "fear",
            "happiness",
            "neutral",
            "sadness",
            "surprise",
        ]

    def _clip_box(self, box: np.ndarray, W: int, H: int, pad: int = 0) -> Tuple[int, int, int, int]:
        x1, y1, x2, y2 = box
        x1 = max(0, int(x1 - pad)); y1 = max(0, int(y1 - pad))
        x2 = min(W, int(x2 + pad));  y2 = min(H, int(y2 + pad))
        w = max(0, x2 - x1); h = max(0, y2 - y1)
        return x1, y1, w, h

    def analyze_emotion(self, image_path: str, conf_min: float = 0.0) -> List[dict]:
        """단일 이미지에서 여러 얼굴 감정 분석"""
        bgr = cv2.imread(image_path)
        if bgr is None:
            print(f"이미지 로드 실패: {image_path}")
            return []
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        H, W = rgb.shape[:2]

        # 얼굴 검출
        boxes, _ = self.mtcnn.detect(rgb)
        results: List[dict] = []
        if boxes is None or len(boxes) == 0:
            # 얼굴이 없으면 전체를 하나의 얼굴로 간주(포토부스 실패 대비)
            boxes = np.array([[0, 0, W, H]])

        for box in boxes:
            x1, y1, w, h = self._clip_box(box, W, H, pad=8)
            if w <= 0 or h <= 0:
                continue
            face_rgb = rgb[y1 : y1 + h, x1 : x1 + w].copy()

            # HSEmotion: 확률 반환 원하면 logits=False
            emotion_str, probs = self.fer.predict_emotions(face_rgb, logits=False)
            em = emotion_str.lower()
            conf = float(np.max(probs)) if hasattr(probs, "__len__") else 0.0

            if conf >= conf_min:
                results.append({
                    "emotion": em,
                    "confidence": conf * 100.0,
                    "box": (x1, y1, w, h),
                })
        return results

    def analyze_multiple_images(self, image_paths: List[str], conf_min: float = 0.0) -> Dict[str, List[dict]]:
        out: Dict[str, List[dict]] = {}
        for p in image_paths:
            out[p] = self.analyze_emotion(p, conf_min=conf_min) if os.path.exists(p) else []
        return out