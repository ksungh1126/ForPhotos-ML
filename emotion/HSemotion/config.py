"""애플리케이션 설정 및 이모지 매핑 유틸리티.

- `DEFAULT_EMOJI_FILENAMES`: 감정 → 기본 이모지 파일명 매핑
- `AppConfig`: 이모지 디렉터리를 받아 절대경로 기반의 이모지 맵을 생성
"""

from __future__ import annotations
import os
from dataclasses import dataclass
from typing import Dict

# 기본 이모지 파일명 매핑 (파일 확장자는 .png 가정)
DEFAULT_EMOJI_FILENAMES: Dict[str, str] = {
    "happiness": "happy.png",
    "sadness": "sad.png",
    "anger": "angry.png",
    "surprise": "surprise.png",
    "disgust": "disgust.png",
    "neutral": "neutral.png",
    # "fear": "fear.png",
    # "contempt": "contempt.png",
}

@dataclass
class AppConfig:
    emoji_dir: str = "./examples/emojis"

    def build_emoji_map(self) -> Dict[str, str]:
        """이모지 디렉터리 기반의 절대경로 맵 생성"""
        emo_map = {}
        for k, fname in DEFAULT_EMOJI_FILENAMES.items():
            emo_map[k] = os.path.join(self.emoji_dir, fname)
        return emo_map