"""포즈 분석 API 라우터"""
from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse
import tempfile
from pathlib import Path
import sys
import cv2
import numpy as np

# pose 모듈 import (선택적)
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# 포즈 분석 모듈 선택적 로드
POSE_AVAILABLE = False
try:
    from pose.analysis import get_pose_type_from_array, get_num_people
    POSE_AVAILABLE = True
except ImportError as e:
    print(f"⚠️  포즈 분석 모듈 로드 실패: {e}")
    print("   포즈 분석 기능은 비활성화됩니다. mediapipe, ultralytics, deepface 패키지를 설치하세요.")

router = APIRouter(prefix="/pose", tags=["pose"])


@router.post("/analyze")
async def analyze_pose(
    image: UploadFile = File(..., description="분석할 이미지"),
):
    """
    이미지에서 포즈, 사람 수, 성별 등을 분석합니다.

    Returns:
        - num_people: 검출된 사람 수
        - pose_type: 포즈 타입 (standing, sitting 등)
        - gender_distribution: 성별 분포
    """
    if not POSE_AVAILABLE:
        raise HTTPException(
            status_code=503,
            detail="포즈 분석 기능이 비활성화되어 있습니다. mediapipe, ultralytics, deepface 패키지를 설치하세요."
        )

    if not image.filename:
        raise HTTPException(status_code=400, detail="이미지 파일이 필요합니다.")

    suffix = Path(image.filename).suffix or ".jpg"
    tmp_file = None
    tmp_file_path = None

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

        # 이미지 로드
        img = cv2.imread(str(tmp_file_path))
        if img is None:
            raise HTTPException(status_code=400, detail="이미지를 읽을 수 없습니다.")

        # 사람 수 감지 (YOLO 모델은 pose.analysis 내부에서 로드)
        num_people = get_num_people(str(tmp_file_path))

        # 포즈 타입 분석 (MediaPipe 포즈 결과 활용)
        pose_type = get_pose_type_from_array(img)

        response_data = {
            "num_people": num_people,
            "pose_type": pose_type,
            "message": f"{num_people}명이 검출되었으며, 포즈 타입은 '{pose_type}' 입니다."
        }

        return JSONResponse(content=response_data)

    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"처리 실패: {exc}")
    finally:
        # 임시 파일 정리
        try:
            if tmp_file_path:
                tmp_file_path.unlink(missing_ok=True)
        except Exception:
            pass


@router.get("/health")
def pose_health():
    """포즈 분석 모듈 헬스 체크"""
    yolo_model_path = Path(__file__).parent.parent.parent / "pose" / "yolov8n.pt"
    yolo_available = yolo_model_path.exists() if POSE_AVAILABLE else False

    return {
        "status": "ok" if POSE_AVAILABLE else "disabled",
        "pose_module_available": POSE_AVAILABLE,
        "yolo_model_available": yolo_available,
        "yolo_model_path": str(yolo_model_path) if yolo_available else None,
        "message": "mediapipe, ultralytics, deepface 패키지를 설치하면 포즈 분석을 사용할 수 있습니다." if not POSE_AVAILABLE else None
    }
