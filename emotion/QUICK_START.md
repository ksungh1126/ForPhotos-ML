# 🚀 ForPhotos Emotion - 빠른 시작 가이드

## 1️⃣ 환경 설정 (최초 1회)

```bash
# 의존성 설치
cd /home/work/wonjun/ForPhotos-ML
pip install -r emotion/requirements.txt

# .env 파일 생성
cd emotion
cp .env.example .env
```

## 2️⃣ API 키 설정 (선택)

### GPU가 있는 경우 → 로컬 모델 사용 (기본값)

`.env` 파일 그대로 사용 (수정 불필요)

```bash
USE_GEMINI_API=false
```

### GPU가 없는 경우 → 무료 Gemini API 사용

1. API 키 발급: https://makersuite.google.com/app/apikey
2. `.env` 파일 수정:

```bash
USE_GEMINI_API=true
GEMINI_API_KEY=your_actual_api_key_here
```

## 3️⃣ 서버 실행

```bash
cd /home/work/wonjun/ForPhotos-ML/emotion
uvicorn emotion.api.server:app --host 0.0.0.0 --port 8000
```

## 4️⃣ 웹 UI 실행 (새 터미널)

```bash
cd /home/work/wonjun/ForPhotos-ML/emotion/frontend
./start.sh
```

## 5️⃣ 브라우저 접속

http://localhost:3000

---

## 📋 체크리스트

- [ ] Python 3.10+ 설치
- [ ] 의존성 설치 완료
- [ ] .env 파일 생성
- [ ] API 키 설정 (Gemini 사용 시)
- [ ] 백엔드 서버 실행 (포트 8000)
- [ ] 프론트엔드 서버 실행 (포트 3000)
- [ ] 브라우저 접속 확인

---

## 🔍 자세한 내용

- [전체 문서](README.md)
- [Gemini API 가이드](GEMINI_API_GUIDE.md)
- [프론트엔드 가이드](frontend/README.md)
