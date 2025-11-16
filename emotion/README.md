# Emotion Toolkit

사진 속 표정을 정밀하게 분석하고, SNS 게시물에 바로 붙여넣을 캡션·해시태그·음악까지 자동으로 만들어 주는 **완전한 워크플로**를 제공합니다. `HSemotion` 패키지로 감정을 분류하고 이모지를 합성한 뒤, `recommender` 모듈이 같은 사진을 기반으로 SNS 카피를 생성합니다. **FastAPI 기반 REST API**와 **웹 프론트엔드**를 통해 쉽게 사용할 수 있습니다.

## 📦 폴더 개요
```
emotion/
├─ HSemotion/       # 감정 분석 · 이모지 합성 모듈 (python -m HSemotion)
├─ recommender/     # Qwen2.5-VL 기반 SNS 캡션 & 음악 추천 (python -m emotion.recommender.cli)
├─ api/             # FastAPI 기반 통합 REST API 서버
├─ frontend/        # 웹 UI (HTML/CSS/JavaScript)
├─ Qwen2.5-VL/      # 로컬에 저장된 Qwen2.5-VL 가중치 + 토크나이저
├─ examples/        # 감정별 기본 이모지 PNG
├─ outputs/         # HSemotion 결과물 기본 저장 경로
├─ requirements.txt # emotion 하위 공통 의존성(torch, transformers 등)
└─ README.md        # 이 문서
```

## 🚀 빠른 시작

가장 쉬운 방법은 **웹 인터페이스**를 사용하는 것입니다:

```bash
# 1. 백엔드 API 서버 실행
cd emotion
uvicorn emotion.api.server:app --host 0.0.0.0 --port 8000

# 2. 새 터미널에서 프론트엔드 실행
cd emotion/frontend
./start.sh

# 3. 브라우저에서 http://localhost:3000 접속
```

이미지를 업로드하면 AI가 자동으로 감정 분석과 SNS 콘텐츠를 생성합니다!

---

## ⚙️ 환경 준비

```bash
conda create -n photo python=3.10 -y
conda activate photo
pip install -r requirements.txt          # ForPhotos-ML 루트에서 실행
pip install -r emotion/requirements.txt  # HSemotion + Recommender 공통 의존성
```

**필수 요구사항:**
- Python 3.10+
- GPU 사용 권장 (CUDA 11.7+)
  - GPU 확인: `python -c "import torch; print(torch.cuda.is_available())"`
- `emotion/Qwen2.5-VL/` 폴더에 Qwen2.5-VL 모델 가중치와 토크나이저 파일 필요

---

## 1. HSemotion – 감정 분석 & 이모지 합성
`HSemotion` 폴더는 PyTorch 기반의 얼굴 감정 분석 파이프라인을 제공합니다. 내부 구성은 아래와 같습니다.

- [`analyzer.py`](HSemotion/analyzer.py): `EmotionAnalyzer` 클래스. MTCNN으로 얼굴을 찾고 HSEmotion 모델(`enet_b2_8` 등)로 감정을 추론합니다. 기본적으로 8개의 AffectNet 클래스 중 `fear`, `contempt`를 제외한 6클래스로 재정규화합니다.
- [`emoji.py`](HSemotion/emoji.py): 감정 결과에 맞춘 이모지 합성. 겹침 방지를 위해 다른 얼굴의 bounding box를 고려해 위치를 조정합니다.
- [`visualize.py`](HSemotion/visualize.py): 감정 박스 및 요약 텍스트 패널 이미지를 생성합니다.
- [`config.py`](HSemotion/config.py): 감정 ↔ 이모지 파일 매핑을 구성하는 `AppConfig`를 제공합니다.
- [`cli.py`](HSemotion/cli.py) / [`__main__.py`](HSemotion/__main__.py): 전체 파이프라인을 실행하는 CLI 엔트리.

### CLI 사용법
```bash
cd emotion
python -m HSemotion \
  --images photo1.jpg photo2.jpg \
  --device cuda \
  --model enet_b2_8 \
  --conf-min 0.15 \
  --emoji-dir ./examples/emojis \
  --panel-out emotion_panel.png \
  --overlay-out-dir ./outputs \
  --emoji-size-scale 0.65 \
  --emoji-y-offset 0.18
```

주요 인자:
- `--images`: 분석할 이미지 경로(여러 장 가능, 필수)
- `--device`: `cpu` 또는 `cuda`
- `--model`: HSEmotion에서 지원하는 모델 이름(`enet_b2_8`, `enet_b0_8` 등)
- `--conf-min`: confidence 임계값. 필터링 후 남은 얼굴만 이모지가 붙습니다.
- `--emoji-dir`: 감정별 PNG를 모아둔 폴더. 기본값은 `examples/emojis`.
- `--emoji-size-scale`, `--emoji-y-offset`: 얼굴 대비 이모지 크기와 세로 위치 보정값.
- `--no-overlap-avoid`: true로 지정하면 겹침 방지 로직을 끄고 단순히 얼굴 위 중앙에 배치합니다.
- `--panel-out`: 감정 패널 PNG 저장 경로. 미지정 시 저장하지 않습니다.
- `--overlay-out-dir`: 이모지 합성 결과를 저장할 디렉터리.
- `--show`: 생성된 패널을 GUI로 확인하고 싶을 때 사용합니다.

### 출력물
- `outputs/<원본파일명>_emoji.png`: 감정별 이모지를 합성한 결과.
- `emotion_panel.png`: 얼굴 bounding box와 감정/신뢰도를 시각화한 레포트.



## 2. Photobooth SNS Recommender – 캡션·음악 생성
`recommender` 폴더는 로컬 Qwen2.5-VL 모델을 감싼 생성 파이프라인입니다.

- [`config.py`](recommender/config.py): `RecommendationConfig`. 모델 디렉터리, 디바이스 자동 판별(`resolve_device`), 생성 파라미터(top-p, temperature 등)와 JSON 스키마 힌트, 시스템 프롬프트를 정의합니다.
- [`generator.py`](recommender/generator.py): `RecommendationEngine`과 데이터 클래스(`RecommendationRequest`, `RecommendationResult`). 이미지를 RGB로 불러오고, 챗 템플릿을 구성해 Qwen으로부터 JSON 응답을 추출합니다. 해시태그 문자열/리스트 모두 처리하며, `raw_text` 원문도 보존합니다.
- [`cli.py`](recommender/cli.py): 커맨드라인 진입점. 옵션을 받아 `RecommendationEngine.generate`를 호출하고 JSON을 출력합니다.
- [`run_batch.sh`](recommender/run_batch.sh): 동일한 입력으로 10회 반복 실행하며 결과를 타임스탬프 기반 파일로 저장합니다.

### CLI 사용법
```bash
cd emotion
python -m emotion.recommender.cli \
  photo1.jpg \
  --hint "따뜻한 감성으로 부탁해요" \
  --device cuda \
  --temperature 0.6 \
  --max-new-tokens 256 \
  --use-fast-processor
```

주요 옵션:
- `image`: 분석할 인생네컷/포토부스 이미지 경로 (위치 인자)
- `--hint`: 촬영 의도나 원하는 분위기 등 추가 컨텍스트
- `--model-dir`: 기본값(`emotion/Qwen2.5-VL`) 이외의 모델 위치를 사용할 때 지정
- `--device`: 강제로 `cpu`, `cuda`, `mps` 등을 지정하고 싶을 때 사용
- `--max-new-tokens`, `--temperature`: 생성 길이와 다양성 제어. `temperature=0`이면 결정적 출력
- `--use-fast-processor`: 최신 PyTorch에서 제공하는 fast image processor를 사용

출력 예시(JSON):
```json
{
  "sns_caption": "기억에 남는 순간들, 함께 웃으며 소중한 추억을 만들어요! 🌟\n#행복한시간 #추억저장 #AirFesta",
  "music": {
    "title": "Dynamite",
    "artist": "BTS"
  },
  "raw_text": "..."  // 모델 원문 응답
}
```


## 3. REST API 서버 – FastAPI 통합 엔드포인트

[`api/server.py`](api/server.py)는 감정 분석과 SNS 추천을 **하나의 HTTP 요청**으로 처리하는 FastAPI 서버입니다.

### 주요 엔드포인트

#### `GET /health`
서버 상태 및 모델 로딩 여부 확인
```bash
curl http://localhost:8000/health
# {"status": "ok", "emotion_analyzer_ready": true, "recommender_ready": true}
```

#### `POST /analyze`
이미지를 업로드하면 감정 분석 + SNS 추천을 동시에 수행

**요청 파라미터:**
- `image` (file, 필수): 분석할 이미지 파일
- `hint` (string, 선택): 촬영 의도나 분위기 (예: "친구들과 즐거운 시간")
- `conf_min` (float, 선택): 최소 신뢰도 (기본값: 0.0)

**응답 예시:**
```json
{
  "emotions": [
    {
      "emotion": "happiness",
      "confidence": 85.3,
      "box": {"x": 120, "y": 80, "w": 150, "h": 180}
    }
  ],
  "recommendation": {
    "sns_caption": "친구들과 함께하는 멋진 에어페스타!",
    "hashtags": ["친구들", "즐거운시간", "airfesta"],
    "music": {
      "title": "FRIEND THE END",
      "artist": "볼빨간사춘기"
    },
    "music_candidates": [...]
  }
}
```

### 서버 실행

```bash
cd emotion
uvicorn emotion.api.server:app --host 0.0.0.0 --port 8000
```

실행 후 다음 URL에서 API 문서 확인:
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

### CORS 설정

프론트엔드에서 API를 호출할 수 있도록 CORS가 활성화되어 있습니다. 프로덕션 환경에서는 [`api/server.py`](api/server.py#L67-L74)에서 `allow_origins`를 특정 도메인으로 제한하세요.

---

## 4. 웹 프론트엔드 – 사용자 친화적 UI

`frontend/` 폴더는 순수 HTML/CSS/JavaScript로 구현된 웹 인터페이스를 제공합니다.

### 주요 기능

- 📸 **드래그 앤 드롭 업로드**: 이미지를 쉽게 업로드
- 😊 **실시간 감정 분석**: 모든 얼굴의 감정과 신뢰도 표시
- ✍️ **SNS 캡션 자동 생성**: AI가 작성한 캡션 제공
- #️⃣ **해시태그 추천**: 관련 해시태그 자동 추출
- 🎵 **음악 추천**: 사진 분위기에 맞는 노래 제안
- 📋 **원클릭 복사**: 캡션/해시태그를 바로 SNS에 활용

### 프론트엔드 실행

**방법 1: 자동 스크립트 (권장)**
```bash
cd emotion/frontend
./start.sh
```

**방법 2: 수동 실행**
```bash
cd emotion/frontend
python3 -m http.server 3000
```

브라우저에서 **http://localhost:3000** 접속

### 사용 방법

1. 이미지를 업로드 영역에 드래그 또는 클릭하여 선택
2. (선택) 촬영 의도 입력 (예: "친구들과 즐거운 시간")
3. "분석 시작" 버튼 클릭 (10~20초 소요)
4. 결과 확인 및 "복사하기" 버튼으로 SNS에 활용

자세한 내용은 [`frontend/README.md`](frontend/README.md)를 참고하세요.

---

## 5. 통합 워크플로

### 옵션 A: 웹 인터페이스 사용 (가장 쉬움)

1. **백엔드 서버 실행**
   ```bash
   cd emotion
   uvicorn emotion.api.server:app --host 0.0.0.0 --port 8000
   ```

2. **프론트엔드 서버 실행** (새 터미널)
   ```bash
   cd emotion/frontend
   ./start.sh
   ```

3. **브라우저에서 사용**
   - http://localhost:3000 접속
   - 이미지 업로드 후 분석 시작
   - 결과를 복사하여 SNS에 활용

### 옵션 B: CLI 사용

1. **감정 분석 + 이모지 합성**
   ```bash
   cd emotion
   python -m HSemotion --images photo1.jpg --device cuda
   ```

2. **SNS 추천**
   ```bash
   python -m emotion.recommender.cli photo1.jpg --hint "친구들과 즐거운 시간"
   ```

### 옵션 C: API 직접 호출

```bash
curl -X POST http://localhost:8000/analyze \
  -F "image=@photo1.jpg" \
  -F "hint=친구들과 즐거운 시간" \
  -F "conf_min=0.15"
```

### 옵션 D: Python 코드에서 사용

```python
from emotion.HSemotion.analyzer import EmotionAnalyzer
from emotion.recommender.generator import RecommendationEngine, RecommendationRequest

# 감정 분석
analyzer = EmotionAnalyzer(device="cuda")
emotions = analyzer.analyze_emotion("photo.jpg", conf_min=0.15)

# SNS 추천
engine = RecommendationEngine()
request = RecommendationRequest(image_path="photo.jpg", user_hint="친구들과")
result = engine.generate(request)

print(result.caption)
print(result.hashtags)
```

---

## ✅ 트러블슈팅

### 일반적인 문제

- **`ModuleNotFoundError: PIL`**
  - `pip install -r emotion/requirements.txt` 설치 여부 확인

- **`No module named 'emotion'`**
  - 프로젝트 루트(ForPhotos-ML)에서 실행하거나 `PYTHONPATH` 설정

- **Qwen 로딩 시간이 길다**
  - 최초 1회만 모델을 디스크에서 읽어옵니다. 이후에는 캐시에 의해 빨라집니다.

- **곡 제목/아티스트 정확도가 낮다**
  - `--temperature`를 낮추거나 `--hint`로 분위기를 명시하세요.

- **GPU 관련 오류**
  - CUDA 드라이버 설치 및 `torch.cuda.is_available()` 결과 확인

### API 서버 관련

- **서버가 시작되지 않음**
  ```bash
  # 포트가 이미 사용 중인 경우
  lsof -ti:8000 | xargs kill -9
  # 또는 다른 포트 사용
  uvicorn emotion.api.server:app --port 8001
  ```

- **`/health` 엔드포인트 응답이 느림**
  - 첫 요청 시 모델을 로딩하므로 시간이 걸립니다 (정상 동작)
  - 이후 요청부터는 빠르게 응답합니다

### 프론트엔드 관련

- **브라우저에서 디렉토리 리스팅만 보임**
  - 올바른 디렉토리에서 서버를 실행했는지 확인
  ```bash
  cd emotion/frontend  # 반드시 frontend 폴더에서
  python3 -m http.server 3000
  ```

- **CORS 오류 발생**
  ```
  Access to fetch has been blocked by CORS policy
  ```
  - 백엔드 서버가 실행 중인지 확인
  - [`api/server.py`](api/server.py#L67-L74)에 CORS 미들웨어가 추가되었는지 확인
  - 브라우저 캐시를 삭제하고 재시도 (`Ctrl+Shift+R`)

- **API 연결 실패 (Failed to fetch)**
  - 백엔드 서버가 `http://localhost:8000`에서 실행 중인지 확인
  - [`frontend/app.js`](frontend/app.js#L2)의 `API_BASE_URL` 설정 확인
  - 방화벽 설정 확인

- **이미지 업로드 후 아무 반응 없음**
  - 브라우저 개발자 도구(F12)의 Console 탭에서 에러 확인
  - Network 탭에서 `/analyze` 요청이 전송되는지 확인

### 성능 관련

- **분석 속도가 너무 느림**
  - GPU 사용 확인: `torch.cuda.is_available()` → `True`여야 함
  - 이미지 크기가 너무 큰 경우 리사이즈 권장 (최대 2000x2000)
  - CPU만 사용 시 10~30초, GPU 사용 시 5~15초 소요

- **메모리 부족 오류**
  - 배치 크기 줄이기 또는 이미지 리사이즈
  - Qwen 모델이 큰 경우 8GB+ VRAM 권장

---

## 📚 추가 리소스

- **API 문서**: http://localhost:8000/docs (서버 실행 후)
- **프론트엔드 가이드**: [`frontend/README.md`](frontend/README.md)
- **HSemotion 모듈**: [`HSemotion/`](HSemotion/)
- **Recommender 모듈**: [`recommender/`](recommender/)

---

## 🏗️ 시스템 아키텍처

```
┌─────────────────────────────────────────────────────────────┐
│                    사용자 인터페이스                          │
├─────────────────────────────────────────────────────────────┤
│  웹 브라우저 (frontend/)        CLI              API 클라이언트 │
│  - HTML/CSS/JS                 - HSemotion      - curl/Python │
│  - 드래그 앤 드롭               - recommender    - JavaScript  │
│  - 실시간 결과 표시                                            │
└──────────────┬──────────────────────┬───────────────────────┘
               │                      │
               │ HTTP POST            │ Direct Call
               │ /analyze             │
               ▼                      ▼
┌─────────────────────────────────────────────────────────────┐
│              FastAPI REST API (api/server.py)               │
│  - CORS 활성화                                               │
│  - JSON 응답                                                 │
│  - 비동기 처리                                                │
└──────────────┬──────────────────────────────────────────────┘
               │
               ├──────────────┬────────────────┐
               ▼              ▼                ▼
┌──────────────────┐  ┌───────────────┐  ┌─────────────────┐
│  EmotionAnalyzer │  │ Recommender   │  │ Music Knowledge │
│  (HSemotion/)    │  │ (recommender/)│  │ Base (optional) │
├──────────────────┤  ├───────────────┤  ├─────────────────┤
│ • MTCNN          │  │ • Qwen2.5-VL  │  │ • Sentence      │
│   (얼굴 검출)     │  │   (VLM)       │  │   Transformers  │
│ • HSEmotion      │  │ • 캡션 생성    │  │ • 음악 검색      │
│   (감정 분류)     │  │ • 해시태그     │  │                 │
│ • 6-class        │  │ • 음악 추천    │  │                 │
│   재정규화        │  │               │  │                 │
└──────────────────┘  └───────────────┘  └─────────────────┘
       │                      │
       ▼                      ▼
┌──────────────────┐  ┌───────────────────────────────────┐
│ 감정 분석 결과    │  │ SNS 추천 결과                     │
├──────────────────┤  ├───────────────────────────────────┤
│ • emotion        │  │ • sns_caption                     │
│ • confidence     │  │ • hashtags []                     │
│ • box (x,y,w,h)  │  │ • music {title, artist}           │
│                  │  │ • music_candidates []             │
└──────────────────┘  └───────────────────────────────────┘
```

### 데이터 흐름

1. **입력**: 사용자가 포토부스 이미지 업로드
2. **얼굴 검출**: MTCNN이 이미지에서 모든 얼굴 위치 추출
3. **감정 분류**: HSEmotion이 각 얼굴의 감정을 6가지로 분류
4. **SNS 생성**: Qwen2.5-VL이 이미지를 분석하여 캡션/해시태그/음악 생성
5. **출력**: 통합된 JSON 응답을 사용자에게 반환

### 주요 기술 스택

| 계층 | 기술 | 역할 |
|------|------|------|
| **프론트엔드** | HTML/CSS/JavaScript | 사용자 인터페이스 |
| **API** | FastAPI, Uvicorn | REST API 서버 |
| **얼굴 검출** | MTCNN (facenet-pytorch) | 얼굴 영역 추출 |
| **감정 분석** | HSEmotion (EfficientNet B2) | 6-class 감정 분류 |
| **VLM** | Qwen2.5-VL | 이미지→텍스트 생성 |
| **음악 검색** | Sentence-Transformers | 시맨틱 검색 |

---

이슈가 지속되면 리포지터리 이슈 트래커에 상황을 공유해주세요.
