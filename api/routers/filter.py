"""이미지 필터 API 라우터"""
from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import Response
import tempfile
from pathlib import Path
import cv2
import numpy as np
from enum import Enum

router = APIRouter(prefix="/filter", tags=["filter"])


class FilterType(str, Enum):
    sepia = "sepia"
    grayscale = "grayscale"
    vintage = "vintage"
    original = "original"


def apply_sepia(img):
    """세피아 필터 적용"""
    sepia_filter = np.array([[0.272, 0.534, 0.131],
                             [0.349, 0.686, 0.168],
                             [0.393, 0.769, 0.189]])
    return cv2.transform(img, sepia_filter)


def apply_grayscale(img):
    """그레이스케일 필터 적용"""
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    return img


def apply_vintage(img):
    """빈티지 필터 적용"""
    # BGR to RGB
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # HSV 조정
    img_hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    img_hsv[:, :, 1] = img_hsv[:, :, 1] * 0.3  # 채도 감소
    img_hsv[:, :, 2] = img_hsv[:, :, 2] * 0.8  # 명도 감소
    img_hsv[:, :, 0] = np.clip(img_hsv[:, :, 0] * 1.1, 0, 179)
    img_vintage = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2RGB)

    # 세피아 톤
    sepia_filter = np.array([[0.393, 0.769, 0.189],
                             [0.349, 0.686, 0.168],
                             [0.272, 0.534, 0.131]])
    img_vintage = cv2.transform(img_vintage, sepia_filter)

    # 대비 및 밝기 조정
    img_vintage = cv2.convertScaleAbs(img_vintage, alpha=0.9, beta=10)

    # 노이즈 추가
    noise = np.random.normal(0, 5, img_vintage.shape).astype(np.uint8)
    img_vintage = cv2.add(img_vintage, noise)

    # 가우시안 블러
    img_vintage = cv2.GaussianBlur(img_vintage, (3, 3), 0.5)

    # RGB to BGR
    return cv2.cvtColor(img_vintage, cv2.COLOR_RGB2BGR)


@router.post("/apply")
async def apply_filter(
    image: UploadFile = File(..., description="필터를 적용할 이미지"),
    filter_type: FilterType = Form(FilterType.sepia, description="적용할 필터 타입"),
):
    """
    이미지에 필터를 적용합니다.

    Args:
        image: 필터를 적용할 이미지
        filter_type: 필터 타입 (sepia, grayscale, vintage, original)

    Returns:
        필터가 적용된 이미지 (PNG)
    """
    if not image.filename:
        raise HTTPException(status_code=400, detail="이미지 파일이 필요합니다.")

    tmp_file = None
    tmp_file_path = None

    try:
        # 임시 파일 저장
        suffix = Path(image.filename).suffix or ".jpg"
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

        # 필터 적용
        if filter_type == FilterType.sepia:
            result_img = apply_sepia(img)
        elif filter_type == FilterType.grayscale:
            result_img = apply_grayscale(img)
        elif filter_type == FilterType.vintage:
            result_img = apply_vintage(img)
        else:  # original
            result_img = img

        # PNG로 인코딩
        success, encoded_image = cv2.imencode('.png', result_img)
        if not success:
            raise HTTPException(status_code=500, detail="이미지 인코딩 실패")

        return Response(content=encoded_image.tobytes(), media_type="image/png")

    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"처리 실패: {exc}")
    finally:
        # 임시 파일 정리
        try:
            if tmp_file_path:
                tmp_file_path.unlink(missing_ok=True)
        except Exception:
            pass


@router.get("/types")
def get_filter_types():
    """사용 가능한 필터 타입 목록 반환"""
    return {
        "filters": [
            {"name": "sepia", "description": "세피아 톤 필터"},
            {"name": "grayscale", "description": "흑백 필터"},
            {"name": "vintage", "description": "빈티지 필터"},
            {"name": "original", "description": "원본 (필터 없음)"}
        ]
    }


@router.get("/health")
def filter_health():
    """필터 모듈 헬스 체크"""
    return {"status": "ok"}
