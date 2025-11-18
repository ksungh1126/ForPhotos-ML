"""ForPhotos-ML 통합 API 서버"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pathlib import Path

# 라우터 import
from api.routers import emotion, pose, filter, split

app = FastAPI(
    title="ForPhotos-ML API",
    version="1.0.0",
    description="""
    포토부스 사진 분석 및 처리를 위한 통합 ML API

    ## 주요 기능
    - 😊 **감정 분석**: 얼굴 감정 인식 + SNS 캡션/해시태그/음악 추천
    - 🧍 **포즈 분석**: 사람 수, 성별, 포즈 타입 분석
    - 🎨 **이미지 필터**: 세피아, 그레이스케일, 빈티지 필터
    - ✂️ **스트립 분할**: 4컷 사진을 개별 이미지로 분리
    """,
)

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 라우터 등록
app.include_router(emotion.router, prefix="/api")
app.include_router(pose.router, prefix="/api")
app.include_router(filter.router, prefix="/api")
app.include_router(split.router, prefix="/api")

# 정적 파일 서빙 (프론트엔드)
frontend_path = Path(__file__).parent.parent / "frontend_integrated"
if frontend_path.exists():
    app.mount("/static", StaticFiles(directory=str(frontend_path)), name="static")


@app.get("/")
def root():
    """루트 엔드포인트 - 프론트엔드 페이지 반환"""
    frontend_index = frontend_path / "index.html"
    if frontend_index.exists():
        return FileResponse(str(frontend_index))
    return {
        "message": "ForPhotos-ML API에 오신 것을 환영합니다!",
        "docs": "/docs",
        "api_endpoints": {
            "emotion": "/api/emotion/analyze",
            "pose": "/api/pose/analyze",
            "filter": "/api/filter/apply",
            "split": "/api/split/photobooth"
        }
    }


@app.get("/health")
def health_check():
    """전체 시스템 헬스 체크"""
    return {
        "status": "ok",
        "service": "ForPhotos-ML API",
        "version": "1.0.0"
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
