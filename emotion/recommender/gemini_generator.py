"""Google Gemini API 기반 SNS 추천 생성기 (무료 API 버전)"""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests
from PIL import Image

from .config import RecommendationConfig


@dataclass
class RecommendationRequest:
    """추천 요청 데이터"""
    image_path: Path
    user_hint: Optional[str] = None


@dataclass
class RecommendationResult:
    """추천 결과 데이터"""
    caption: str
    hashtags: List[str]
    song_title: str
    song_artist: str
    caption_raw: str
    music_raw: str
    music_candidates: List[Dict[str, Any]]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "sns_caption": self.caption,
            "hashtags": self.hashtags,
            "music": {"title": self.song_title, "artist": self.song_artist},
            "raw_text": {
                "caption": self.caption_raw,
                "music": self.music_raw,
            },
            "music_candidates": self.music_candidates,
        }


class GeminiRecommendationEngine:
    """Google Gemini API를 사용한 추천 엔진 (무료)"""

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError(
                "GEMINI_API_KEY 환경 변수가 설정되지 않았습니다. "
                "또는 api_key 파라미터를 전달하세요."
            )

        self.api_url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent"

    def _encode_image(self, image_path: Path) -> Dict[str, str]:
        """이미지를 base64로 인코딩"""
        import base64

        with open(image_path, "rb") as f:
            image_data = base64.b64encode(f.read()).decode("utf-8")

        # MIME type 추출
        suffix = image_path.suffix.lower()
        mime_type = {
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".png": "image/png",
            ".webp": "image/webp",
        }.get(suffix, "image/jpeg")

        return {
            "inline_data": {
                "mime_type": mime_type,
                "data": image_data
            }
        }

    def _build_prompt(self, user_hint: Optional[str] = None) -> str:
        """프롬프트 생성"""
        base_prompt = """이 사진을 분석하여 JSON으로 응답하세요:

{
  "sns_caption": "감성적인 SNS 캡션 (1-2문장)",
  "hashtags": ["해시태그1", "해시태그2", "해시태그3", "해시태그4", "해시태그5"],
  "music": {
    "title": "추천 곡 제목",
    "artist": "아티스트 이름"
  }
}

요구사항:
- 캡션: 사진 분위기에 맞는 감성적 문구
- 해시태그: 5-7개
- 음악: 실제 존재하는 K-POP 또는 인기곡"""

        if user_hint:
            base_prompt += f"\n\n힌트: {user_hint}"

        return base_prompt

    def generate(self, request: RecommendationRequest) -> RecommendationResult:
        """SNS 추천 생성"""

        # 이미지 인코딩
        image_part = self._encode_image(request.image_path)

        # 프롬프트 생성
        prompt = self._build_prompt(request.user_hint)

        # API 요청 구성
        payload = {
            "contents": [{
                "parts": [
                    {"text": prompt},
                    image_part
                ]
            }],
            "generationConfig": {
                "temperature": 0.7,
                "maxOutputTokens": 2048,
                "responseMimeType": "application/json"
            }
        }

        # API 호출
        response = requests.post(
            f"{self.api_url}?key={self.api_key}",
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=30
        )

        if response.status_code != 200:
            raise Exception(f"Gemini API 오류: {response.status_code} - {response.text}")

        result = response.json()

        # 응답 파싱
        try:
            # 응답 구조 확인
            if "candidates" not in result or len(result["candidates"]) == 0:
                raise Exception(f"응답에 candidates가 없습니다: {result}")

            candidate = result["candidates"][0]

            # finishReason 확인
            finish_reason = candidate.get("finishReason", "")
            if finish_reason == "MAX_TOKENS":
                raise Exception("응답이 토큰 제한으로 잘렸습니다. 더 짧은 프롬프트를 사용하거나 maxOutputTokens를 늘려주세요.")

            # content.parts 확인
            if "content" not in candidate or "parts" not in candidate["content"]:
                raise Exception(f"응답에 content.parts가 없습니다: {candidate}")

            text_response = candidate["content"]["parts"][0]["text"]

            # JSON 추출 (마크다운 코드 블록 제거)
            json_match = re.search(r'```json\s*(.*?)\s*```', text_response, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
            else:
                # 코드 블록이 없으면 전체를 JSON으로 파싱 시도
                json_str = text_response.strip()

            data = json.loads(json_str)

            # 데이터 추출
            caption = data.get("sns_caption", "")
            hashtags = data.get("hashtags", [])
            music = data.get("music", {})

            # 해시태그 정규화
            if isinstance(hashtags, str):
                hashtags = [tag.strip() for tag in hashtags.split(",")]
            hashtags = [tag.replace("#", "").strip() for tag in hashtags]

            return RecommendationResult(
                caption=caption,
                hashtags=hashtags,
                song_title=music.get("title", "Unknown"),
                song_artist=music.get("artist", "Unknown"),
                caption_raw=text_response,
                music_raw=json.dumps(music, ensure_ascii=False),
                music_candidates=[music]  # Gemini는 단일 추천만 제공
            )

        except (KeyError, json.JSONDecodeError, IndexError) as e:
            raise Exception(f"Gemini 응답 파싱 실패: {e}\n응답: {result}")


__all__ = ["GeminiRecommendationEngine", "RecommendationRequest", "RecommendationResult"]
