"""감정 분석기: 얼굴 검출(MTCNN) + 감정 분류(HSEmotion).

- `EmotionAnalyzer.analyze_emotion`: 단일 이미지 내 여러 얼굴의 감정을 추론
- `EmotionAnalyzer.analyze_multiple_images`: 다중 이미지 일괄 처리
"""

from __future__ import annotations
import os
import cv2
import numpy as np
from typing import Dict, List, Tuple, Iterable
from facenet_pytorch import MTCNN
from hsemotion.facial_emotions import HSEmotionRecognizer


class EmotionAnalyzer:
    """
    HSEmotion + MTCNN 기반 감정 분석기
    기본 클래스(AffectNet 8): anger, contempt, disgust, fear, happiness, neutral, sadness, surprise
    -> 기본 설정으로 fear, contempt는 제외하여 6-class로 동작합니다.
    """

    def __init__(
        self,
        device: str = "cpu",
        model_name: str = "enet_b2_8",
        mtcnn_kwargs: dict | None = None,
        drop_classes: Iterable[str] | None = None,  # 외부에서 바꾸고 싶으면 주입
    ):
        # 얼굴 검출기 (MTCNN)
        self.mtcnn = MTCNN(keep_all=True, **(mtcnn_kwargs or {}))
        # 감정 분류기 (HSEmotion)
        self.fer = HSEmotionRecognizer(model_name=model_name, device=device)

        # HSEmotion 내 클래스 이름 맵 추출
        if hasattr(self.fer, "idx_to_class"):
            self.idx_to_class = dict(self.fer.idx_to_class)  # {idx: label}
        elif hasattr(self.fer, "classes"):
            self.idx_to_class = {i: name for i, name in enumerate(self.fer.classes)}
        else:
            # 최후의 수단: AffectNet 8-class 가정
            self.idx_to_class = {
                0: "anger", 1: "contempt", 2: "disgust", 3: "fear",
                4: "happiness", 5: "neutral", 6: "sadness", 7: "surprise",
            }

        # ✅ 기본 제외: fear, contempt
        default_drop = {"fear", "contempt"}
        self.drop_classes = set(c.lower() for c in (drop_classes or default_drop))

    def _clip_box(self, box: np.ndarray, W: int, H: int, pad: int = 0) -> Tuple[int, int, int, int]:
        x1, y1, x2, y2 = box
        x1 = max(0, int(x1 - pad)); y1 = max(0, int(y1 - pad))
        x2 = min(W, int(x2 + pad));  y2 = min(H, int(y2 + pad))
        w = max(0, x2 - x1); h = max(0, y2 - y1)
        return x1, y1, w, h

    def analyze_emotion(self, image_path: str, conf_min: float = 0.0) -> List[dict]:
        """단일 이미지에서 여러 얼굴 감정 분석(기본 6-class: fear, contempt 제외)"""
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
            boxes = np.array([[0, 0, W, H]])

        for box in boxes:
            x1, y1, w, h = self._clip_box(box, W, H, pad=8)
            if w <= 0 or h <= 0:
                continue
            face_rgb = rgb[y1 : y1 + h, x1 : x1 + w].copy()

            # HSEmotion 예측(확률)
            emotion_str, probs = self.fer.predict_emotions(face_rgb, logits=False)
            probs = np.asarray(probs, dtype=np.float32).reshape(-1)

            # 제외 클래스 인덱스 마스킹
            allowed_idxs = [i for i, name in self.idx_to_class.items()
                            if name.lower() not in self.drop_classes]
            if not allowed_idxs:
                continue  # 전부 제외되면 skip

            filtered = np.clip(probs[allowed_idxs], 0.0, None)
            s = float(filtered.sum())

            if s > 0.0:
                # 6-class 기준 재정규화 후 top-1
                filtered /= s
                k = int(np.argmax(filtered))
                pred_idx = allowed_idxs[k]
                em = self.idx_to_class[pred_idx].lower()
                conf = float(filtered[k])  # 재정규화된 확률
            else:
                # 모두 0이면 원래 top-1로 fallback (거의 없음)
                pred_idx = int(np.argmax(probs))
                em = self.idx_to_class.get(pred_idx, emotion_str).lower()
                conf = float(probs[pred_idx])

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
