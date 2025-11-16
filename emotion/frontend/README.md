# ForPhotos 프론트엔드

ForPhotos Emotion API를 사용하는 간단한 웹 인터페이스입니다.

## 기능

- 📸 **이미지 업로드**: 드래그 앤 드롭 또는 클릭으로 이미지 업로드
- 😊 **감정 분석**: AI가 사진 속 모든 얼굴의 감정을 분석
- ✍️ **SNS 캡션 생성**: 사진에 어울리는 캡션 자동 생성
- #️⃣ **해시태그 추천**: 관련 해시태그 자동 추천
- 🎵 **음악 추천**: 사진 분위기에 맞는 음악 추천

## 실행 방법

### 1. 백엔드 서버 실행

먼저 터미널에서 백엔드 API 서버를 실행합니다:

```bash
cd /home/work/wonjun/ForPhotos-ML/emotion
uvicorn emotion.api.server:app --host 0.0.0.0 --port 8000
```

서버가 실행되면 다음과 같은 메시지가 표시됩니다:
```
INFO:     Uvicorn running on http://0.0.0.0:8000
```

### 2. 프론트엔드 실행

간단한 HTTP 서버를 실행합니다:

#### Python 3 사용 (권장)
```bash
cd /home/work/wonjun/ForPhotos-ML/emotion/frontend
python -m http.server 3000
```

#### Node.js 사용 (선택)
```bash
cd /home/work/wonjun/ForPhotos-ML/emotion/frontend
npx serve -p 3000
```

### 3. 브라우저에서 접속

브라우저를 열고 다음 주소로 접속합니다:
```
http://localhost:3000
```

## 사용 방법

1. **이미지 업로드**
   - 업로드 영역을 클릭하거나
   - 이미지를 드래그 앤 드롭

2. **옵션 설정** (선택)
   - 촬영 의도 또는 분위기 입력 (예: "친구들과 즐거운 시간")
   - 최소 신뢰도 조정 (기본값: 0.15)

3. **분석 시작**
   - "분석 시작" 버튼 클릭
   - 10~20초 정도 소요 (이미지 크기에 따라 다름)

4. **결과 확인**
   - 감정 분석 결과
   - SNS 캡션 및 해시태그
   - 추천 음악

5. **복사 및 활용**
   - 각 섹션의 "복사하기" 버튼으로 내용 복사
   - SNS에 바로 붙여넣기 가능

## 파일 구조

```
frontend/
├── index.html      # 메인 HTML
├── style.css       # 스타일시트
├── app.js          # JavaScript 로직
└── README.md       # 이 문서
```

## 설정 변경

### API 서버 주소 변경

`app.js` 파일의 첫 줄을 수정합니다:

```javascript
// 로컬 개발
const API_BASE_URL = 'http://localhost:8000';

// 프로덕션 (예시)
const API_BASE_URL = 'https://api.yourserver.com';
```

### CORS 설정 (프로덕션)

프로덕션 환경에서는 백엔드의 CORS 설정을 제한해야 합니다.

`emotion/api/server.py`에서:

```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://yourfrontend.com"],  # 특정 도메인만 허용
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

## 트러블슈팅

### CORS 오류 발생 시

브라우저 콘솔에 CORS 오류가 표시되면:

1. 백엔드 서버가 실행 중인지 확인
2. 백엔드에 CORS 미들웨어가 추가되었는지 확인
3. 브라우저 캐시 삭제 후 재시도

### API 연결 실패

```
Failed to fetch
```

오류가 발생하면:

1. 백엔드 서버가 `http://localhost:8000`에서 실행 중인지 확인
2. `app.js`의 `API_BASE_URL` 설정 확인
3. 방화벽 설정 확인

### 이미지 분석이 느릴 때

- GPU를 사용하도록 백엔드 설정 확인
- 이미지 크기가 너무 크면 리사이즈 권장 (최대 2000x2000)

## 개선 사항

향후 추가 가능한 기능:

- [ ] 이미지 자르기/회전 기능
- [ ] 여러 이미지 동시 분석
- [ ] 결과 히스토리 저장
- [ ] 다크 모드
- [ ] 다국어 지원

## 라이센스

ForPhotos-ML 프로젝트의 일부입니다.
