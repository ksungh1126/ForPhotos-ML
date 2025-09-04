# Emotion (HSemotion + MTCNN)

Facenet MTCNN으로 얼굴을 검출하고, HSEmotion으로 감정을 분류한 뒤,
감정별 이모지를 얼굴 위에 합성하는 유틸리티입니다.

**주요 기능:**
- 다중 얼굴 검출 및 감정 분석 (8가지 감정: anger, contempt, disgust, fear, happiness, neutral, sadness, surprise)
- 감정별 이모지 자동 합성
- **이모지 겹침 방지**: 여러 사람이 있는 사진에서 이모지가 다른 사람 얼굴을 가리지 않도록 자동 위치 조정
- 분석 결과 시각화 패널 생성

## 📁 프로젝트 구조

```
emotion/
├─ HSemotion/           # 메인 패키지
│  ├─ __init__.py      # 패키지 초기화
│  ├─ __main__.py      # 진입점 (python -m HSemotion)
│  ├─ cli.py           # 커맨드라인 인터페이스
│  ├─ analyzer.py      # 얼굴 검출 + 감정 분석
│  ├─ config.py        # 설정 및 이모지 매핑
│  ├─ emoji.py         # 이모지 합성 (겹침 방지 포함)
│  ├─ utils.py         # 유틸리티 함수들
│  └─ visualize.py     # 결과 시각화
├─ requirements.txt    # 패키지 의존성
├─ README.md          # 이 파일
└─ examples/
   └─ emojis/         # 감정별 이모지 PNG 파일들
      ├─ happy.png    # happiness 감정용
      ├─ sad.png      # sadness 감정용
      ├─ angry.png    # anger 감정용
      ├─ surprise.png # surprise 감정용
      ├─ fear.png     # fear 감정용
      ├─ disgust.png  # disgust 감정용
      ├─ neutral.png  # neutral 감정용
      └─ contempt.png # contempt 감정용
```

## 설치

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cd emotion
```

> ⚠️ **주의**: HSEmotion은 특정 timm 버전을 요구하므로 `timm==0.8.19.dev0` 고정을 권장합니다.

## 실행

```bash
python -m HSemotion \
  --images photo1.jpg photo2.jpg \
  --device cpu \
  --model enet_b2_8 \
  --conf-min 0.0 \
  --emoji-dir ./examples/emojis \
  --panel-out emotion_result.png \
  --overlay-out-dir ./outputs
```

## 명령어 옵션

### 필수 옵션
- `--images`: 분석할 이미지 파일 경로들 (여러 장 동시 처리 가능)

### 모델 설정
- `--device`: 실행 디바이스 (`cpu` 또는 `cuda`, 기본값: `cpu`)
- `--model`: HSEmotion 모델명 (기본값: `enet_b2_8`)
- `--conf-min`: 감정 신뢰도 임계값 0~1 (기본값: `0.0`)

### 이모지 설정
- `--emoji-dir`: 이모지 PNG 파일들이 있는 디렉터리 (기본값: `./examples/emojis`)
- `--emoji-size-scale`: 얼굴 크기 대비 이모지 크기 비율 (기본값: `0.6`)
- `--emoji-y-offset`: 얼굴 위쪽 오프셋 비율 (기본값: `0.15`)
- `--no-overlap-avoid`: 이모지 겹침 방지 기능 비활성화

### 출력 설정
- `--panel-out`: 감정 분석 결과 패널 저장 경로 (옵션, 예: `result_panel.png`)
- `--overlay-out-dir`: 이모지 합성 결과 저장 디렉터리 (기본값: 현재 디렉터리)
- `--show`: 분석 결과 패널을 화면에 표시

## 사용 예제

### 1. 기본 사용 (단일 이미지)
```bash
python -m HSemotion --images family_photo.jpg
```

### 2. 다중 이미지 처리
```bash
python -m HSemotion --images photo1.jpg photo2.jpg photo3.jpg --overlay-out-dir ./results
```

### 3. GPU 사용 + 결과 패널 저장
```bash
python -m HSemotion \
  --images group_photo.jpg \
  --device cuda \
  --panel-out analysis_result.png \
  --show
```

### 4. 이모지 크기 및 위치 조정
```bash
python -m HSemotion \
  --images photo.jpg \
  --emoji-size-scale 0.8 \
  --emoji-y-offset 0.2
```

### 5. 겹침 방지 기능 비활성화
```bash
python -m HSemotion --images photo.jpg --no-overlap-avoid
```

## 출력 파일

실행 후 다음과 같은 파일들이 생성됩니다:

### 1. 감정 분석 패널 (`--panel-out` 옵션 사용 시)
- 원본 이미지 + 얼굴 검출 박스 + 감정 라벨이 표시된 시각화 패널
- 각 얼굴별 감정과 신뢰도를 텍스트로 요약

### 2. 이모지 합성 결과
- 각 입력 이미지별로 `원본파일명_emoji.png` 형태로 저장
- 검출된 얼굴 위에 해당 감정의 이모지가 합성됨
- **자동 위치 조정**: 다른 사람 얼굴과 겹치지 않도록 이모지 위치 자동 조정

## 이모지 매핑

`examples/emojis/` 디렉터리의 파일명과 감정이 다음과 같이 매핑됩니다:

| 감정 (HSEmotion) | 파일명 |
|------------------|--------|
| happiness | happy.png |
| sadness | sad.png |
| anger | angry.png |
| surprise | surprise.png |
| fear | fear.png |
| disgust | disgust.png |
| neutral | neutral.png |
| contempt | contempt.png |

> 💡 **팁**: 이모지 파일은 투명 배경(.png)을 사용하면 더 자연스러운 합성이 가능합니다.

## 이모지 겹침 방지 기능

여러 사람이 있는 사진에서 이모지가 다른 사람의 얼굴을 가리는 것을 방지합니다:

1. **겹침 감지**: 이모지 배치 위치가 다른 사람 얼굴과 겹치는지 자동 확인
2. **대안 위치 탐색**: 다음 순서로 대안 위치 시도
   - 기본 위치 (얼굴 위쪽)
   - 더 위쪽
   - 더 더 위쪽  
   - 얼굴 오른쪽
   - 얼굴 왼쪽
   - 얼굴 아래쪽
3. **최적 배치**: 겹치지 않는 첫 번째 위치에 이모지 배치

## 지원하는 감정 클래스

HSEmotion 모델의 AffectNet 8-class 감정을 지원합니다:
- **Anger** (분노)
- **Contempt** (경멸)  
- **Disgust** (혐오)
- **Fear** (공포)
- **Happiness** (행복)
- **Neutral** (중성)
- **Sadness** (슬픔)
- **Surprise** (놀람)

## 요구사항

- Python 3.8+
- CUDA (GPU 사용 시)
- 충분한 메모리 (이미지 크기에 따라)

자세한 패키지 의존성은 `requirements.txt`를 참조하세요.

## 문제 해결

### Q: "No faces detected" 메시지가 나타납니다
A: MTCNN이 얼굴을 감지하지 못한 경우입니다. `--conf-min` 값을 낮춰보거나 이미지 품질을 확인해보세요.

### Q: GPU에서 실행이 안됩니다
A: CUDA가 설치되어 있고 PyTorch가 CUDA를 인식하는지 확인하세요. `python -c "import torch; print(torch.cuda.is_available())"`

### Q: 이모지가 표시되지 않습니다
A: `--emoji-dir` 경로의 PNG 파일들이 존재하는지 확인하세요. 파일명이 위의 매핑 테이블과 일치해야 합니다.

### Q: 이모지 크기가 너무 큽니다/작습니다
A: `--emoji-size-scale` 옵션으로 크기를 조정하세요 (기본값: 0.6).

---

더 자세한 사용법이나 문제가 있다면 프로젝트 이슈를 등록해주세요!