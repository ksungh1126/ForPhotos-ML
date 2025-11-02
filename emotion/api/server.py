from __future__ import annotations

import asyncio
import tempfile
from functools import lru_cache
from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from ..HSemotion.analyzer import EmotionAnalyzer
from ..recommender.config import RecommendationConfig
from ..recommender.generator import RecommendationEngine, RecommendationRequest


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


@lru_cache(maxsize=1)
def get_emotion_analyzer() -> EmotionAnalyzer:
    return EmotionAnalyzer()


@lru_cache(maxsize=1)
def get_recommendation_engine() -> RecommendationEngine:
    config = RecommendationConfig()
    return RecommendationEngine(config)


app = FastAPI(
    title="ForPhotos Emotion & SNS Recommender API",
    version="1.0.0",
    description="Unified API exposing HSemotion-based emotion analysis and SNS/music recommendations.",
)


@app.get("/health")
def health_check() -> dict:
    """Simple health probe."""
    analyzer_ready = get_emotion_analyzer() is not None
    recommender_ready = get_recommendation_engine() is not None
    return {"status": "ok", "emotion_analyzer_ready": analyzer_ready, "recommender_ready": recommender_ready}


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

    try:
        analyzer = get_emotion_analyzer()
        emotions_raw = await asyncio.to_thread(analyzer.analyze_emotion, str(tmp_file_path), max(conf_min, 0.0))

        recommender = get_recommendation_engine()
        request = RecommendationRequest(image_path=tmp_file_path, user_hint=hint)
        recommendation = await asyncio.to_thread(recommender.generate, request)

    except Exception as exc:  # pragma: no cover - runtime error path
        raise HTTPException(status_code=500, detail=f"Processing failed: {exc}") from exc
    finally:
        try:
            tmp_file_path.unlink(missing_ok=True)
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

    response = PipelineResponse(emotions=emotions_payload, recommendation=recommendation_payload)
    return JSONResponse(content=response.model_dump())


__all__ = ["app"]
