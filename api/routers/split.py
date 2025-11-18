"""포토부스 스트립 분할 API 라우터"""
from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse
import tempfile
from pathlib import Path
import sys
import cv2
import base64
import zipfile
import io

# pose 모듈 import (선택적)
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# 스트립 분할 모듈 선택적 로드
SPLIT_AVAILABLE = False
try:
    from pose.splitter import split_photobooth_strip
    SPLIT_AVAILABLE = True
except ImportError as e:
    print(f"⚠️  스트립 분할 모듈 로드 실패: {e}")
    print("   스트립 분할 기능은 비활성화됩니다.")

router = APIRouter(prefix="/split", tags=["split"])


@router.post("/photobooth")
async def split_photobooth(
    image: UploadFile = File(..., description="분할할 포토부스 스트립 이미지"),
):
    """
    포토부스 스트립 이미지를 개별 사진으로 분할합니다.

    Returns:
        - num_cuts: 분할된 컷 수
        - cuts: 각 컷의 base64 인코딩된 이미지 리스트
    """
    if not SPLIT_AVAILABLE:
        raise HTTPException(
            status_code=503,
            detail="스트립 분할 기능이 비활성화되어 있습니다."
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

        # 스트립 분할
        cuts = split_photobooth_strip(str(tmp_file_path))

        # 각 컷을 base64로 인코딩
        encoded_cuts = []
        for i, (cut_img, _) in enumerate(cuts):
            success, encoded_img = cv2.imencode('.png', cut_img)
            if success:
                b64_img = base64.b64encode(encoded_img).decode('utf-8')
                encoded_cuts.append({
                    "index": i,
                    "image": b64_img
                })

        response_data = {
            "num_cuts": len(encoded_cuts),
            "cuts": encoded_cuts,
            "message": f"{len(encoded_cuts)}개의 컷으로 분할되었습니다."
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


@router.post("/photobooth/zip")
async def split_photobooth_zip(
    image: UploadFile = File(..., description="분할할 포토부스 스트립 이미지"),
):
    """
    포토부스 스트립 이미지를 개별 사진으로 분할하여 ZIP 파일로 반환합니다.

    Returns:
        ZIP 파일 (각 컷이 PNG 파일로 포함됨)
    """
    if not SPLIT_AVAILABLE:
        raise HTTPException(
            status_code=503,
            detail="스트립 분할 기능이 비활성화되어 있습니다."
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

        # 스트립 분할
        cuts = split_photobooth_strip(str(tmp_file_path))

        # ZIP 파일 생성
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
            for i, (cut_img, _) in enumerate(cuts):
                success, encoded_img = cv2.imencode('.png', cut_img)
                if success:
                    zip_file.writestr(f"cut_{i+1}.png", encoded_img.tobytes())

        # ZIP 파일 반환
        from fastapi.responses import Response
        return Response(
            content=zip_buffer.getvalue(),
            media_type="application/zip",
            headers={"Content-Disposition": "attachment; filename=photobooth_cuts.zip"}
        )

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
def split_health():
    """스트립 분할 모듈 헬스 체크"""
    return {
        "status": "ok" if SPLIT_AVAILABLE else "disabled",
        "split_module_available": SPLIT_AVAILABLE,
        "message": "스트립 분할 기능을 사용할 수 있습니다." if SPLIT_AVAILABLE else "스트립 분할 모듈 로드 실패"
    }
