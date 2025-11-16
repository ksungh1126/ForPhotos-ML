# ForPhotos-ML

포토부스 사진을 위한 **감정 분석 → 이모지 합성 → SNS 캡션/해시태그/음악 추천 → 포즈 분석 → 필터 → 스트립 분할** 워크플로를 한 번에 제공합니다. 백엔드는 FastAPI, 프론트는 순수 HTML/CSS/JS로 작성되어 있어 API, CLI, 브라우저 어디서나 동일한 결과를 얻을 수 있습니다.

---

## 폴더 구조

```
ForPhotos-ML/
├─ api/                    # 통합 FastAPI 서버
│   └─ routers/            # emotion, pose, filter, split 라우터
├─ emotion/                # HSemotion + SNS Recommender 모듈
│   ├─ HSemotion/          # 감정 분석/이모지 합성 파이프라인
│   ├─ recommender/        # Qwen2.5-VL 또는 Gemini 추천 엔진
│   ├─ examples/           # 감정별 기본 이모지 PNG
│   └─ requirements.txt    # emotion 모듈 의존성 (torch, transformers 등)
├─ frontend_integrated/    # FastAPI가 서빙하는 통합 웹 UI
├─ pose/                   # 포즈 분류, 사람 수/성비 분석, 스트립 분할 모듈
├─ photos/                 # 데모용 샘플 이미지
├─ requirements.txt        # 공통(포즈/필터) 의존성
└─ 기타 스크립트          # create_csv.py, extract_frames_DISFA.py 등 실험용 코드
```

---

## 주요 기능

- **감정 분석 & 이모지 합성**  
  facenet-pytorch MTCNN으로 얼굴을 찾고 HSemotion(EfficientNet)으로 6개 감정을 분류한 뒤, 감정별 PNG를 겹쳐 합성합니다.
- **SNS 추천**  
  로컬 Qwen2.5-VL 모델 또는 Google Gemini 2.5 Flash를 사용해 캡션·해시태그·음악을 JSON으로 생성합니다. `.env`에서 `USE_GEMINI_API`를 전환할 수 있습니다.
- **포즈 및 인원 분석**  
  MediaPipe Pose의 33개 랜드마크를 기반으로 `peace_sign`, `thumbs_up`, `cross_arm`, `hands_up`, `big_heart`, `other/no_person`을 판별하고, YOLOv8n + DeepFace로 사람 수와 성비를 계산합니다.
- **이미지 필터**  
  세피아, 흑백, 빈티지, 원본 필터를 REST API 및 프론트에서 바로 적용할 수 있습니다.
- **포토부스 스트립 분할**  
  배경색을 추정해 1x4 · 2x2 레이아웃을 자동 자르거나 ZIP으로 내려줍니다.

---

## 환경 설정

```bash
conda create -n forphotos python=3.10 -y
conda activate forphotos

# 1) 공통 패키지 (포즈·필터·공유 헬퍼)
pip install -r requirements.txt

# 2) Emotion + Recommender (PyTorch, transformers, FastAPI 등)
pip install -r emotion/requirements.txt
```

추가 준비 사항:

- 로컬 Qwen2.5-VL을 쓰려면 `emotion/Qwen2.5-VL/`에 가중치와 토크나이저를 넣습니다.
- `.env`는 `emotion/.env`를 참고하여 `USE_GEMINI_API`, `GEMINI_API_KEY`, `DEVICE`, `DEFAULT_CONF_MIN` 등을 설정합니다.
- `pose/yolov8n.pt`는 저장소에 포함되어 있으며 필요 시 더 큰 모델로 교체 가능합니다.

---

## 실행

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

- 브라우저에서 **http://localhost:8000**을 열면 `frontend_integrated/`가 자동으로 서빙됩니다.  
  하단에는 현재 사용 중인 추천 엔진(Qwen 또는 Gemini)이 표시됩니다.
- **Swagger UI**: http://localhost:8000/docs  
  각 엔드포인트의 form-data 키, 예제 응답을 확인할 수 있습니다.

프론트를 외부 호스트에서 구동하고 싶다면:

```bash
cd frontend_integrated
python -m http.server 3000  # 또는 npx serve
```

이 경우 `app.js`의 `const API_BASE = '/api';`를 필요한 주소(예: `https://api.example.com`)로 변경하세요.

---

## 주요 REST 엔드포인트

| 경로 | 메서드 | 설명 |
|------|--------|------|
| `/health` | GET | 통합 서버 상태 |
| `/api/emotion/analyze` | POST (multipart) | 감정 분석 + SNS 추천 + 이모지 합성 |
| `/api/emotion/health` | GET | HSemotion/추천 엔진 준비 여부 |
| `/api/pose/analyze` | POST | 사람 수, 포즈 라벨 |
| `/api/filter/apply` | POST | 필터 적용 후 PNG 반환 |
| `/api/split/photobooth` | POST | 4컷 이미지를 잘라 base64 PNG 리스트 반환 |
| `/api/split/photobooth/zip` | POST | 컷들을 ZIP 파일로 다운로드 |

모든 POST는 `image` 필드가 필수이며, 각 탭에서 사용하는 추가 파라미터(`hint`, `conf_min`, `filter_type` 등)는 Swagger와 README에서 설명합니다.

---

## Emotion & Recommender 세부 구성

- `emotion/HSemotion/`: `EmotionAnalyzer`, `emoji.py`, `visualize.py`, `config.py`, `cli.py` 등을 포함합니다.  
  CLI 실행 예:
  ```bash
  cd emotion
  python -m HSemotion --images photo1.jpg --device cuda \
    --conf-min 0.15 --panel-out outputs/panel.png --overlay-out-dir outputs
  ```
- `emotion/recommender/`: `RecommendationEngine`, `RecommendationRequest`, `RecommendationResult`, Qwen 템플릿, Gemini 래퍼가 포함되어 있습니다.  
  ```bash
  python -m emotion.recommender.cli photo1.jpg --hint "친구들과 즐거운 시간" \
    --device cuda --temperature 0.6
  ```
- `GEMINI_API_GUIDE.md` / `test_gemini.py`: Gemini 2.5 Flash 설정 방법과 테스트 스크립트.

---

## Pose & Photobooth 모듈

- `pose/analysis.py`: MediaPipe Pose 랜드마크로 제스처를 판별하고 YOLOv8n으로 사람 수를 센 뒤 DeepFace로 성비를 추정합니다. `get_pose_type`, `get_num_people`, `get_gender_distribution`, `get_pose_type_from_array`를 외부에서 사용할 수 있습니다.
- `pose/splitter.py`: 단일 이미지를 1x4/2x2 패널로 자동 분할합니다. FastAPI `/api/split` 라우터가 내부적으로 사용합니다.
- `pose/main.py`: InsightFace, DeepFace, metadata 생성 등 추가 실험 기능을 포함한 확장 스크립트입니다.

---

## 기타 유틸리티

- `create_csv.py`, `extract_frames_DISFA.py`: DISFA Action Unit 레이블 및 프레임 추출 도구.
- `simple_filter.py`: REST 필터 로직을 CLI로 빠르게 시험할 수 있는 스크립트.
- `LibreFace_detect_mediapipe.py`, `test_libreface.py`: InsightFace/MediaPipe 기반 실험 코드.

---

## 트러블슈팅

- **프론트에서 JSON만 보임**  
  `frontend_integrated/` 폴더가 없거나 위치가 바뀐 경우입니다. FastAPI가 정적 자산을 찾을 수 있도록 경로를 유지하세요.
- **포즈 분석 503**  
  `mediapipe`, `ultralytics`, `deepface`가 설치되지 않았을 때 발생합니다. 루트 `requirements.txt`를 다시 설치하세요.
- **추천 엔진이 로딩에서 멈춤**  
  `.env`의 `USE_GEMINI_API`와 `GEMINI_API_KEY`를 확인하거나 Qwen 가중치 폴더가 존재하는지 확인하세요.
- **GPU 메모리 오류**  
  최초 요청에서 모델이 로드되기 때문에 GPU 메모리가 부족할 수 있습니다. `DEVICE=cpu`로 테스트 후 최적화하세요.

---

## 활용 아이디어

- 포토부스 키오스크에 API를 붙여 촬영 직후 자동으로 감정/포즈/캡션을 안내.
- 이벤트 리포트 생성: 한 세션에서 가장 많이 사용된 포즈/감정, 남녀 비율 등을 요약.
- SNS 에디터: 이미지 업로드만으로 곧바로 캡션·하트 이모지 합성·필터·4컷 분할까지 수행.
- 데이터셋 구축: `pose/metadata.py`, `create_csv.py`를 활용해 포즈/감정/Action Unit 레이블을 쌓고 모델을 재학습.

ForPhotos-ML로 포토부스 AI 경험을 자유롭게 확장해 보세요!
