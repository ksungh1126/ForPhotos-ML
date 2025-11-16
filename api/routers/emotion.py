"""감정 분석 및 SNS 추천 API 라우터"""
from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse
import asyncio
import base64
import tempfile
from pathlib import Path
from typing import Optional
import sys

# emotion 모듈 import
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from emotion.HSemotion.analyzer import EmotionAnalyzer
from emotion.HSemotion.emoji import add_emotion_emojis
from emotion.HSemotion.config import AppConfig
from emotion.recommender.generator import RecommendationEngine, RecommendationRequest
from emotion.recommender.config import RecommendationConfig
from emotion.recommender.gemini_generator import GeminiRecommendationEngine
import os
from dotenv import load_dotenv

# .env 로드
env_path = Path(__file__).parent.parent.parent / "emotion" / ".env"
if env_path.exists():
    load_dotenv(env_path)

USE_GEMINI_API = os.getenv("USE_GEMINI_API", "false").lower() == "true"

router = APIRouter(prefix="/emotion", tags=["emotion"])

# 싱글톤 인스턴스
_emotion_analyzer = None
_recommendation_engine = None


def get_emotion_analyzer():
    global _emotion_analyzer
    if _emotion_analyzer is None:
        _emotion_analyzer = EmotionAnalyzer()
    return _emotion_analyzer


def get_recommendation_engine():
    global _recommendation_engine
    if _recommendation_engine is None:
        if USE_GEMINI_API:
            api_key = os.getenv("GEMINI_API_KEY")
            if not api_key:
                raise ValueError("USE_GEMINI_API=true이지만 GEMINI_API_KEY가 설정되지 않았습니다.")
            _recommendation_engine = GeminiRecommendationEngine(api_key=api_key)
        else:
            config = RecommendationConfig()
            _recommendation_engine = RecommendationEngine(config)
    return _recommendation_engine


@router.post("/analyze")
async def analyze_emotion(
    image: UploadFile = File(..., description="분석할 이미지"),
    hint: Optional[str] = Form(None, description="SNS 추천을 위한 힌트"),
    conf_min: float = Form(0.15, description="최소 신뢰도 (0-1)"),
    add_emoji: bool = Form(True, description="이모지 오버레이 추가 여부"),
):
    """
    이미지에서 감정을 분석하고 SNS 캡션/해시태그/음악을 추천합니다.

    Returns:
        - emotions: 감정 분석 결과 (얼굴별)
        - recommendation: SNS 캡션, 해시태그, 음악 추천
        - emoji_image: 이모지가 추가된 이미지 (base64, optional)
    """
    if not image.filename:
        raise HTTPException(status_code=400, detail="이미지 파일이 필요합니다.")

    suffix = Path(image.filename).suffix or ".jpg"
    tmp_file = None
    tmp_file_path = None
    emoji_tmp_path = None

    try:
        # 임시 파일 저장
        tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
        data = await image.read()
        if not data:
            raise HTTPException(status_code=400, detail="업로드된 파일이 비어있습니다.")
        tmp_file.write(data)
        tmp_file.flush()
        tmp_file_path = Path(tmp_file.name)
        tmp_file.close()

        # 감정 분석
        analyzer = get_emotion_analyzer()
        emotions_raw = await asyncio.to_thread(
            analyzer.analyze_emotion, str(tmp_file_path), max(conf_min, 0.0)
        )

        # SNS 추천
        recommender = get_recommendation_engine()
        request = RecommendationRequest(image_path=tmp_file_path, user_hint=hint)
        recommendation = await asyncio.to_thread(recommender.generate, request)

        # 이모지 오버레이
        emoji_image_b64 = None
        if add_emoji and emotions_raw:
            try:
                emoji_tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
                emoji_tmp_path = Path(emoji_tmp.name)
                emoji_tmp.close()

                app_config = AppConfig(
                    emoji_dir=str(Path(__file__).parent.parent.parent / "emotion" / "examples" / "emojis")
                )
                emoji_map = app_config.build_emoji_map()

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

                with open(emoji_tmp_path, "rb") as f:
                    emoji_image_b64 = base64.b64encode(f.read()).decode("utf-8")

            except Exception as emoji_exc:
                print(f"이모지 오버레이 실패: {emoji_exc}")

        # 응답 구성
        emotions_payload = []
        for item in emotions_raw:
            box = item.get("box") or (0, 0, 0, 0)
            emotions_payload.append({
                "emotion": item.get("emotion", ""),
                "confidence": float(item.get("confidence", 0.0)),
                "box": {"x": int(box[0]), "y": int(box[1]), "w": int(box[2]), "h": int(box[3])},
            })

        response_data = {
            "emotions": emotions_payload,
            "recommendation": {
                "sns_caption": recommendation.caption,
                "hashtags": recommendation.hashtags,
                "music": {
                    "title": recommendation.song_title,
                    "artist": recommendation.song_artist
                },
                "raw_text": {
                    "caption": recommendation.caption_raw,
                    "music": recommendation.music_raw
                },
                "music_candidates": recommendation.music_candidates,
            },
            "emoji_image": emoji_image_b64
        }

        return JSONResponse(content=response_data)

    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"처리 실패: {exc}")
    finally:
        # 임시 파일 정리
        try:
            if tmp_file_path:
                tmp_file_path.unlink(missing_ok=True)
            if emoji_tmp_path:
                emoji_tmp_path.unlink(missing_ok=True)
        except Exception:
            pass


@router.get("/health")
def emotion_health():
    """감정 분석 모듈 헬스 체크"""
    analyzer_ready = get_emotion_analyzer() is not None
    recommender_ready = get_recommendation_engine() is not None
    engine_type = "Gemini 2.5 Flash" if USE_GEMINI_API else "Qwen2.5-VL"

    return {
        "status": "ok",
        "emotion_analyzer_ready": analyzer_ready,
        "recommender_ready": recommender_ready,
        "engine_type": engine_type,
        "using_gemini": USE_GEMINI_API
    }
