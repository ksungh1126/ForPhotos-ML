# Emotion Toolkit

사진 속 표정을 분석하고, SNS에 바로 올릴 문구와 음악까지 추천해 주는 두 가지 워크플로를 제공합니다.

- **HSemotion 파이프라인**: Facenet MTCNN으로 얼굴을 검출하고 HSEmotion 모델로 6가지 주요 감정을 분류한 뒤, 감정별 이모지를 자동 합성하며 겹침을 방지합니다.
- **Photobooth SNS Recommender**: 로컬 Qwen2.5-VL 모델로 인생네컷 이미지를 이해하고, 한국어 중심 SNS 캡션과 분위기에 맞는 유명 팝/케이팝 추천곡을 생성합니다.

두 기능을 이어서 사용하면 포토부스 결과물을 감정적으로 꾸미고, 공유용 문구와 음악까지 한 번에 얻을 수 있습니다.

## 📁 디렉터리 구조

```
emotion/
├─ HSemotion/       # 감정 분석 + 이모지 합성 패키지 (python -m HSemotion)
├─ recommender/     # Qwen2.5-VL 기반 SNS 캡션 & 음악 추천 모듈
├─ Qwen2.5-VL/      # 로컬에 저장한 Qwen2.5-VL 체크포인트
├─ examples/        # 감정별 이모지 샘플 PNG
├─ outputs/         # HSemotion 기본 출력 경로
├─ requirements.txt # torch, transformers 등 공통 의존성
└─ README.md        # 이 문서
```

## ⚙️ 설치

```bash
conda create -n photo python=3.10 -y
conda activate photo
pip install -r requirements.txt          # ForPhotos-ML 루트
pip install -r emotion/requirements.txt  # HSemotion + Recommender 의존성
```

> ⚠️ `emotion/Qwen2.5-VL/` 디렉터리에 모델 가중치가 있어야 합니다. GPU를 사용할 경우 `python -c "import torch; print(torch.cuda.is_available())"`로 확인하세요.

---

## 1. HSemotion – 감정 분석 & 이모지 합성

### 동작 요약
1. Facenet MTCNN이 얼굴 위치를 찾습니다.
2. HSEmotion(enet_b2_8 등) 모델이 6가지 감정(anger, disgust, happiness, neutral, sadness, surprise)을 추론합니다.  
   `fear`, `contempt` 감정은 우리 사용 사례에서 빈도가 낮아 제외하여 다른 감정 정확도를 높였습니다.
3. 감정별 이모지를 얼굴 위에 합성하고, 내장된 **겹침 방지 로직**으로 이모지가 다른 얼굴을 가리지 않도록 조정합니다.
4. 필요 시 감정 패널 이미지를 생성해 요약 정보를 제공합니다.

### 실행 예시
```bash
cd emotion
python -m HSemotion \
  --images photo1.jpg photo2.jpg \
  --device cuda \
  --model enet_b2_8 \
  --emoji-dir ./examples/emojis \
  --panel-out emotion_result.png \
  --overlay-out-dir ./outputs
```

### 주요 옵션
- `--images`: 분석할 이미지 경로(여러 장 처리 가능, 필수)
- `--device`: `cpu` / `cuda` (기본값: `cpu`)
- `--model`: HSEmotion 모델명 (기본값: `enet_b2_8`)
- `--conf-min`: 감정 신뢰도 임계값 (0~1, 기본 0.0)
- `--emoji-dir`: 감정 이모지 PNG 폴더 (기본 `./examples/emojis`)
- `--emoji-size-scale`, `--emoji-y-offset`: 이모지 크기 및 위치 조정
- `--no-overlap-avoid`: 겹침 방지 비활성화
- `--panel-out`: 감정 패널 PNG 저장 경로
- `--overlay-out-dir`: 합성 이미지 저장 디렉터리
- `--show`: 결과 패널을 화면에 표시

### 출력물
- `outputs/<원본파일명>_emoji.png`: 감정 이모지가 얹힌 결과물
- `emotion_result.png` (또는 지정 경로): 감정 분포 및 얼굴 박스 시각화 패널

### 감정 ↔ 이모지 매핑
| 감정 | 파일명 |
|------|--------|
| anger | angry.png |
| disgust | disgust.png |
| happiness | happy.png |
| neutral | neutral.png |
| sadness | sad.png |
| surprise | surprise.png |

> `fear`, `contempt`는 입력 데이터에서 거의 등장하지 않아 제외했습니다.

> 💡 PNG 투명 배경을 사용하면 합성 결과가 자연스럽습니다.

---

## 2. Photobooth SNS Recommender

### 동작 요약
1. 로컬 Qwen2.5-VL 모델과 프로세서를 불러옵니다.
2. 사진과 선택적 힌트를 기반으로 커스터마이즈된 시스템/유저 프롬프트를 구성합니다.
3. SNS용 문구(한국어 위주, 영어는 포인트로만)와 3~6개의 한·영 혼합 해시태그, 유명 팝/케이팝 곡의 제목·가수를 생성합니다.
4. 모델의 원문 출력도 함께 반환해 디버깅에 활용할 수 있습니다.

### 실행 예시
```bash
cd emotion
python -m recommender.cli \
  photo1.jpg \
  --hint "따뜻한 감성으로 부탁해요" \
  --device cuda \
  --temperature 0.6 \
  --max-new-tokens 256
```

### 주요 옵션
- `image`(위치 인자): 분석할 이미지 경로
- `--hint`: 촬영 의도나 원하는 분위기를 보조 설명으로 전달 (선택)
- `--device`: `cpu`, `cuda`, `mps` 등 강제 지정
- `--temperature`: 0은 결정적 출력, 값이 높을수록 다양성 증가
- `--max-new-tokens`: 생성 길이 제한 (기본 512)
- `--model-dir`: 다른 위치의 Qwen 가중치를 사용할 때 지정
- `--use-fast-processor`: fast image processor 사용 (최신 PyTorch에서 권장)

### 출력 형식 (JSON)
```json
{
  "sns_caption": "기억에 남는 순간들, 함께 웃으며 소중한 추억을 만들어요! 🌟 #AirFesta #행복한시간 #함께하는시간",
  "music": {
    "title": "Dynamite",
    "artist": "BTS"
  },
  "raw_text": "..."  // 모델 원문 응답
}
```
- `sns_caption`: 1~2문장과 마지막 줄의 한·영 혼합 해시태그(4~8개)
- `music`: 유명 팝/케이팝 곡으로 제목과 아티스트가 정확히 매칭되어야 함
- `raw_text`: 추후 디버깅과 프롬프트 튜닝에 활용

### 팁
- 곡 추천 정확도를 높이고 싶다면 `--temperature 0`으로 실행하거나 힌트를 구체적으로 작성합니다.
- GPU 메모리가 부족하면 `--device cpu`로 전환하여 실험 후 필요 시 저해상도 이미지를 사용하세요.

---

## 3. 통합 워크플로
1. `python -m HSemotion`으로 감정 패널과 이모지 합성 이미지를 생성합니다.
2. 같은 이미지에 대해 `python -m recommender.cli`를 실행해 SNS 문구와 음악 추천을 받습니다.
3. 생성된 이미지를 SNS 캡션과 함께 바로 업로드하거나, 패널 이미지를 사용해 감정 리포트를 공유합니다.

---

## ✅ 트러블슈팅 체크리스트
- **`ModuleNotFoundError: PIL`** → `pip install -r emotion/requirements.txt` 실행 여부 확인.
- **`No module named 'emotion'`** → 프로젝트 루트(ForPhotos-ML)에서 실행하거나 `PYTHONPATH`를 설정.
- **Qwen 로딩이 오래 걸림** → 첫 로딩만 시간이 걸리며 이후엔 캐시로 빨라집니다.
- **곡 제목/아티스트가 어긋남** → `--temperature`를 낮추고, 힌트를 추가하거나 프롬프트를 조정하세요.
- **GPU 사용 오류** → CUDA 설치 및 `torch.cuda.is_available()` 확인.

문제가 지속되면 이슈로 제보해주세요!
