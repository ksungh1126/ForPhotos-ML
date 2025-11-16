from __future__ import annotations

import asyncio
import base64
import os
import tempfile
from functools import lru_cache
from pathlib import Path
from typing import List, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from ..HSemotion.analyzer import EmotionAnalyzer
from ..HSemotion.config import AppConfig
from ..HSemotion.emoji import add_emotion_emojis
from ..recommender.config import RecommendationConfig
from ..recommender.generator import RecommendationEngine, RecommendationRequest

# .env 파일 로드 (emotion 디렉토리에서 찾음)
env_path = Path(__file__).parent.parent / ".env"
if env_path.exists():
    load_dotenv(env_path)
    print(f"✅ .env 파일 로드: {env_path}")
else:
    print(f"⚠️  .env 파일 없음: {env_path} (환경 변수 사용)")

# Gemini API 사용 여부 (환경 변수로 제어)
USE_GEMINI_API = os.getenv("USE_GEMINI_API", "false").lower() == "true"

if USE_GEMINI_API:
    from ..recommender.gemini_generator import GeminiRecommendationEngine


class BoundingBox(BaseModel):
    x: int
    y: int
    w: int
    h: int


class EmotionPayload(BaseModel):
    emotion: str
    confidence: float = Field(..., description="Confidence percentage (0-100)")
    box: BoundingBox


class MusicPayload(BaseModel):
    title: str
    artist: str


class RecommendationPayload(BaseModel):
    sns_caption: str
    hashtags: List[str]
    music: MusicPayload
    raw_text: dict
    music_candidates: List[dict]


class PipelineResponse(BaseModel):
    emotions: List[EmotionPayload]
    recommendation: RecommendationPayload
    emoji_image: Optional[str] = Field(None, description="Base64 encoded image with emoji overlays")


@lru_cache(maxsize=1)
def get_emotion_analyzer() -> EmotionAnalyzer:
    return EmotionAnalyzer()


@lru_cache(maxsize=1)
def get_recommendation_engine():
    """추천 엔진 반환 (로컬 Qwen 또는 Gemini API)"""
    if USE_GEMINI_API:
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("USE_GEMINI_API=true이지만 GEMINI_API_KEY가 설정되지 않았습니다.")
        return GeminiRecommendationEngine(api_key=api_key)
    else:
        config = RecommendationConfig()
        return RecommendationEngine(config)


app = FastAPI(
    title="ForPhotos Emotion & SNS Recommender API",
    version="1.0.0",
    description="Unified API exposing HSemotion-based emotion analysis and SNS/music recommendations.",
)

# CORS 설정 - 프론트엔드에서 API 호출 허용
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 프로덕션에서는 특정 도메인만 허용하도록 변경
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health_check() -> dict:
    """Simple health probe."""
    analyzer_ready = get_emotion_analyzer() is not None
    recommender_ready = get_recommendation_engine() is not None

    # 현재 사용 중인 추천 엔진 타입 반환
    engine_type = "Gemini 2.5 Flash" if USE_GEMINI_API else "Qwen2.5-VL"

    return {
        "status": "ok",
        "emotion_analyzer_ready": analyzer_ready,
        "recommender_ready": recommender_ready,
        "engine_type": engine_type,
        "using_gemini": USE_GEMINI_API
    }


@app.post(
    "/analyze",
    response_model=PipelineResponse,
    summary="Analyze emotions and generate SNS recommendations from an image.",
)
async def analyze_image(
    image: UploadFile = File(..., description="Photobooth-style image to analyze"),
    hint: Optional[str] = Form(None, description="Optional hint to guide the recommender tone"),
    conf_min: float = Form(0.0, description="Minimum confidence threshold (0-1 range)"),
) -> JSONResponse:
    if not image.filename:
        raise HTTPException(status_code=400, detail="Image file is required.")
    suffix = Path(image.filename).suffix or ".jpg"

    tmp_file = None
    tmp_file_path = None  # type: ignore[assignment]
    try:
        tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
        data = await image.read()
        if not data:
            raise HTTPException(status_code=400, detail="Uploaded file is empty.")
        tmp_file.write(data)
        tmp_file.flush()
    finally:
        if tmp_file is not None:
            tmp_file_path = Path(tmp_file.name)
            tmp_file.close()

    if tmp_file_path is None:
        raise HTTPException(status_code=500, detail="Failed to persist uploaded image.")

    emoji_image_b64 = None
    emoji_tmp_path = None

    try:
        analyzer = get_emotion_analyzer()
        emotions_raw = await asyncio.to_thread(analyzer.analyze_emotion, str(tmp_file_path), max(conf_min, 0.0))

        recommender = get_recommendation_engine()
        request = RecommendationRequest(image_path=tmp_file_path, user_hint=hint)
        recommendation = await asyncio.to_thread(recommender.generate, request)

        # 이모지 합성
        if emotions_raw:
            try:
                emoji_tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
                emoji_tmp_path = Path(emoji_tmp.name)
                emoji_tmp.close()

                # 이모지 매핑 설정
                app_config = AppConfig(emoji_dir=str(Path(__file__).parent.parent / "examples" / "emojis"))
                emoji_map = app_config.build_emoji_map()

                # 이모지 합성
                await asyncio.to_thread(
                    add_emotion_emojis,
                    str(tmp_file_path),
                    emotions_raw,
                    str(emoji_tmp_path),
                    emoji_map,
                    size_scale=0.65,
                    y_offset_ratio=0.18,
                    avoid_overlap=True
                )

                # Base64 인코딩
                with open(emoji_tmp_path, "rb") as f:
                    emoji_image_b64 = base64.b64encode(f.read()).decode("utf-8")

            except Exception as emoji_exc:
                # 이모지 합성 실패는 치명적이지 않으므로 로그만 남기고 계속 진행
                print(f"Emoji overlay failed: {emoji_exc}")

    except Exception as exc:  # pragma: no cover - runtime error path
        raise HTTPException(status_code=500, detail=f"Processing failed: {exc}") from exc
    finally:
        try:
            tmp_file_path.unlink(missing_ok=True)
            if emoji_tmp_path:
                emoji_tmp_path.unlink(missing_ok=True)
        except Exception:
            pass

    emotions_payload: List[EmotionPayload] = []
    for item in emotions_raw:
        box = item.get("box") or (0, 0, 0, 0)
        emotions_payload.append(
            EmotionPayload(
                emotion=item.get("emotion", ""),
                confidence=float(item.get("confidence", 0.0)),
                box=BoundingBox(x=int(box[0]), y=int(box[1]), w=int(box[2]), h=int(box[3])),
            )
        )

    recommendation_payload = RecommendationPayload(
        sns_caption=recommendation.caption,
        hashtags=recommendation.hashtags,
        music=MusicPayload(title=recommendation.song_title, artist=recommendation.song_artist),
        raw_text={"caption": recommendation.caption_raw, "music": recommendation.music_raw},
        music_candidates=recommendation.music_candidates,
    )

    response = PipelineResponse(
        emotions=emotions_payload,
        recommendation=recommendation_payload,
        emoji_image=emoji_image_b64
    )
    return JSONResponse(content=response.model_dump())


__all__ = ["app"]
