# 무료 Gemini API 사용 가이드

로컬 GPU 대신 **Google Gemini API**(무료)를 사용하여 SNS 추천 기능을 실행할 수 있습니다.

## 🆓 무료 제공량

- **분당 15 요청**
- **하루 1,500 요청**
- **완전 무료** (신용카드 등록 불필요)

## 🚀 빠른 시작 (3단계)

### 1️⃣ API 키 발급

https://makersuite.google.com/app/apikey 접속 → "Get API Key" 클릭 → API 키 복사

### 2️⃣ .env 파일 설정

```bash
# .env 파일 생성
cd /home/work/wonjun/ForPhotos-ML/emotion
cp .env.example .env

# .env 파일 편집
nano .env
```

`.env` 파일에 다음 내용 입력:

```bash
USE_GEMINI_API=true
GEMINI_API_KEY=your_actual_api_key_here
```

### 3️⃣ 서버 실행

```bash
# 의존성 설치 (최초 1회)
pip install python-dotenv

# 서버 실행
uvicorn emotion.api.server:app --host 0.0.0.0 --port 8000
```

서버 시작 시 `✅ .env 파일 로드` 메시지 확인!

---

## 📝 상세 설정 방법

### 방법 1: .env 파일 사용 (권장 ⭐)

가장 쉽고 안전한 방법입니다!

**장점:**
- ✅ API 키를 파일로 안전하게 관리
- ✅ Git에 커밋되지 않음 (.gitignore에 자동 등록)
- ✅ 서버 재시작 시에도 자동 로드
- ✅ 팀 협업 시 공유 쉬움

**설정:**

```bash
cd /home/work/wonjun/ForPhotos-ML/emotion

# 1. .env.example 복사
cp .env.example .env

# 2. .env 파일 편집 (nano, vim, code 등)
nano .env
```

`.env` 파일 내용:

```bash
# Gemini API 사용 활성화
USE_GEMINI_API=true

# API 키 입력 (따옴표 없이)
GEMINI_API_KEY=AIzaSyC...your_actual_key_here...
```

**저장 후 서버 실행:**

```bash
uvicorn emotion.api.server:app --host 0.0.0.0 --port 8000
```

### 방법 2: 환경 변수로 직접 설정

.env 파일 대신 환경 변수로 직접 설정할 수도 있습니다.

#### Linux/Mac

```bash
# 현재 세션에만 적용
export GEMINI_API_KEY="YOUR_API_KEY_HERE"
export USE_GEMINI_API="true"

# 서버 실행
uvicorn emotion.api.server:app --host 0.0.0.0 --port 8000
```

**영구 적용 (~/.bashrc 또는 ~/.zshrc에 추가):**

```bash
echo 'export GEMINI_API_KEY="YOUR_API_KEY_HERE"' >> ~/.bashrc
echo 'export USE_GEMINI_API="true"' >> ~/.bashrc
source ~/.bashrc
```

#### Windows (PowerShell)

```powershell
$env:GEMINI_API_KEY="YOUR_API_KEY_HERE"
$env:USE_GEMINI_API="true"

# 서버 실행
uvicorn emotion.api.server:app --host 0.0.0.0 --port 8000
```

#### Windows (CMD)

```cmd
set GEMINI_API_KEY=YOUR_API_KEY_HERE
set USE_GEMINI_API=true

uvicorn emotion.api.server:app --host 0.0.0.0 --port 8000
```

---

## 🔄 로컬 모델로 다시 전환

### .env 파일 수정

`.env` 파일에서:

```bash
USE_GEMINI_API=false
# GEMINI_API_KEY는 그대로 두어도 됨
```

### 환경 변수 제거

```bash
# 환경 변수 제거
unset USE_GEMINI_API
unset GEMINI_API_KEY

# 또는 false로 설정
export USE_GEMINI_API="false"

# 서버 재시작
uvicorn emotion.api.server:app --host 0.0.0.0 --port 8000
```

---

## ✅ 테스트

API 키가 제대로 설정되었는지 테스트:

```bash
cd /home/work/wonjun/ForPhotos-ML/emotion

# 테스트 스크립트 실행
python test_gemini.py

# 또는 이미지 경로 지정
python test_gemini.py photo1.jpg
```

성공하면:
```
✅ .env 파일 로드: /home/work/wonjun/ForPhotos-ML/emotion/.env
✅ API 키 확인: AIzaSyC...
✅ Gemini 모듈 import 성공
✅ Gemini 엔진 초기화 성공
✅ API 호출 성공!
```

---

## 📊 성능 비교

| 항목 | 로컬 Qwen2.5-VL | Gemini API |
|------|-----------------|------------|
| **비용** | 무료 (하드웨어 소유) | 무료 (사용량 제한) |
| **속도** | 5-15초 (GPU) | 2-5초 ⚡ |
| **GPU 필요** | 필수 (8GB+ VRAM) | 불필요 ✅ |
| **메모리** | ~10GB | ~100MB ✅ |
| **설치** | 모델 다운로드 필요 | API 키만 필요 ✅ |
| **오프라인** | 가능 | 불가능 ❌ |
| **커스터마이징** | 가능 | 제한적 |
| **하루 요청** | 무제한 | 1,500개 |

---

## 🎯 권장 사용 사례

### Gemini API 추천
- ✅ GPU가 없는 환경
- ✅ 빠른 프로토타이핑
- ✅ 메모리 제약이 있는 경우
- ✅ 하루 요청이 1,500개 이하

### 로컬 Qwen 추천
- ✅ GPU가 있는 환경 (8GB+ VRAM)
- ✅ 대량 처리 (하루 1,500개 이상)
- ✅ 오프라인 사용 필요
- ✅ 프롬프트 세밀한 조정 필요

---

## 🔧 트러블슈팅

### .env 파일이 로드되지 않음

```
⚠️  .env 파일 없음
```

**해결:**
1. `.env` 파일이 올바른 위치에 있는지 확인:
   ```bash
   ls -la /home/work/wonjun/ForPhotos-ML/emotion/.env
   ```
2. 없으면 생성:
   ```bash
   cp /home/work/wonjun/ForPhotos-ML/emotion/.env.example \
      /home/work/wonjun/ForPhotos-ML/emotion/.env
   ```

### API 키 오류

```
ValueError: USE_GEMINI_API=true이지만 GEMINI_API_KEY가 설정되지 않았습니다.
```

**해결:**
1. `.env` 파일 확인:
   ```bash
   cat /home/work/wonjun/ForPhotos-ML/emotion/.env
   ```
2. `GEMINI_API_KEY=` 뒤에 실제 API 키가 있는지 확인
3. 따옴표 없이 입력했는지 확인:
   - ✅ 올바름: `GEMINI_API_KEY=AIzaSy...`
   - ❌ 잘못됨: `GEMINI_API_KEY="AIzaSy..."`

### API 할당량 초과

```
429 Too Many Requests
```

**해결:**
- 요청 속도 제한: 분당 15개
- 하루 제한: 1,500개
- 로컬 모델로 전환하거나 시간 간격 두기

### python-dotenv 설치 오류

```
ModuleNotFoundError: No module named 'dotenv'
```

**해결:**
```bash
pip install python-dotenv
```

---

## 📝 .env 파일 예시

완전한 예시:

```bash
# ============================================
# 추천 엔진 설정
# ============================================

# Gemini API 사용 (GPU 불필요)
USE_GEMINI_API=true

# Gemini API 키
GEMINI_API_KEY=AIzaSyDnwO...your_actual_key...

# ============================================
# 서버 설정 (선택)
# ============================================

PORT=8000
HOST=0.0.0.0

# ============================================
# 감정 분석 설정 (선택)
# ============================================

DEVICE=cpu
DEFAULT_CONF_MIN=0.15

# ============================================
# 로깅 (선택)
# ============================================

LOG_LEVEL=INFO
```

---

## 📚 추가 리소스

- [Gemini API 문서](https://ai.google.dev/tutorials/python_quickstart)
- [API 할당량 확인](https://console.cloud.google.com/apis/api/generativelanguage.googleapis.com/quotas)
- [Gemini 모델 비교](https://ai.google.dev/models/gemini)

---

**보안 주의사항:**
- ⚠️ `.env` 파일을 절대 Git에 커밋하지 마세요
- ⚠️ API 키를 공개 저장소에 올리지 마세요
- ✅ `.env.example`은 공유해도 안전합니다 (실제 키 없음)
