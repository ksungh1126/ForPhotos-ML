# Emotion Toolkit

사진 속 표정을 정밀하게 분석하고, SNS 게시물에 바로 붙여넣을 캡션·해시태그·음악까지 자동으로 만들어 주는 워크플로를 제공합니다. `HSemotion` 패키지로 감정을 분류하고 이모지를 합성한 뒤, `recommender` 모듈이 같은 사진을 기반으로 SNS 카피를 생성합니다.

## 📦 폴더 개요
```
emotion/
├─ HSemotion/       # 감정 분석 · 이모지 합성 모듈 (python -m HSemotion)
├─ recommender/     # Qwen2.5-VL 기반 SNS 캡션 & 음악 추천 (python -m emotion.recommender.cli)
├─ Qwen2.5-VL/      # 로컬에 저장된 Qwen2.5-VL 가중치 + 토크나이저
├─ examples/        # 감정별 기본 이모지 PNG
├─ outputs/         # HSemotion 결과물 기본 저장 경로
├─ requirements.txt # emotion 하위 공통 의존성(torch, transformers 등)
└─ README.md        # 이 문서
```

## ⚙️ 환경 준비
```bash
conda create -n photo python=3.10 -y
conda activate photo
pip install -r requirements.txt          # ForPhotos-ML 루트에서 실행
pip install -r emotion/requirements.txt  # HSemotion + Recommender 공통 의존성
```

- GPU를 사용할 경우 `python -c "import torch; print(torch.cuda.is_available())"`로 확인합니다.
- `emotion/Qwen2.5-VL/` 폴더에 Qwen2.5-VL 모델 가중치와 토크나이저 파일이 존재해야 합니다.

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


## 3. 통합 워크플로 (추후 개발)
1. `python -m HSemotion --images ...` 명령으로 감정 분석 리포트와 이모지 합성 이미지를 만듭니다.
2. 같은 원본 이미지에 `python -m emotion.recommender.cli`를 실행해 캡션·해시태그·음악 추천을 얻습니다.
3. 최종 SNS 업로드 시 이모지 합성 결과물과 추천된 문구를 함께 사용하거나, `visualize_results`로 만든 패널 이미지를 요약자료로 공유합니다.

필요하다면 Python 스크립트에서 두 모듈을 조합하여 완전 자동화된 파이프라인을 구성할 수 있습니다.

---

## ✅ 트러블슈팅
- `ModuleNotFoundError: PIL` → `pip install -r emotion/requirements.txt` 설치 여부 확인
- `No module named 'emotion'` → 프로젝트 루트(ForPhotos-ML)에서 실행하거나 `PYTHONPATH`를 설정
- Qwen 로딩 시간이 길다 → 최초 1회만 모델을 디스크에서 읽어옵니다. 이후에는 캐시에 의해 빨라집니다.
- 곡 제목/아티스트 정확도가 낮다 → `--temperature`를 낮추거나 `--hint`로 분위기를 명시하세요.
- GPU 관련 오류 → CUDA 드라이버 설치 및 `torch.cuda.is_available()` 결과 확인

이슈가 지속되면 리포지터리 이슈 트래커에 상황을 공유해주세요.
