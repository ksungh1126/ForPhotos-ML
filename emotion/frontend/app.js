// API 설정
const API_BASE_URL = 'http://localhost:8000';

// DOM 요소
const uploadArea = document.getElementById('uploadArea');
const imageInput = document.getElementById('imageInput');
const previewImage = document.getElementById('previewImage');
const hintInput = document.getElementById('hintInput');
const confInput = document.getElementById('confInput');
const analyzeBtn = document.getElementById('analyzeBtn');
const loadingSection = document.getElementById('loadingSection');
const resultSection = document.getElementById('resultSection');
const errorSection = document.getElementById('errorSection');

// 전역 변수
let selectedFile = null;
let currentResult = null;

// 이벤트 리스너 설정
function initEventListeners() {
    // 클릭으로 파일 선택
    uploadArea.addEventListener('click', () => {
        imageInput.click();
    });

    // 파일 선택
    imageInput.addEventListener('change', handleFileSelect);

    // 드래그 앤 드롭
    uploadArea.addEventListener('dragover', (e) => {
        e.preventDefault();
        uploadArea.classList.add('drag-over');
    });

    uploadArea.addEventListener('dragleave', () => {
        uploadArea.classList.remove('drag-over');
    });

    uploadArea.addEventListener('drop', (e) => {
        e.preventDefault();
        uploadArea.classList.remove('drag-over');

        const files = e.dataTransfer.files;
        if (files.length > 0) {
            handleFile(files[0]);
        }
    });

    // 분석 버튼
    analyzeBtn.addEventListener('click', analyzeImage);
}

// 파일 선택 처리
function handleFileSelect(e) {
    const file = e.target.files[0];
    if (file) {
        handleFile(file);
    }
}

// 파일 처리
function handleFile(file) {
    if (!file.type.startsWith('image/')) {
        showError('이미지 파일만 업로드할 수 있습니다.');
        return;
    }

    selectedFile = file;

    // 미리보기 표시
    const reader = new FileReader();
    reader.onload = (e) => {
        previewImage.src = e.target.result;
        previewImage.style.display = 'block';
        uploadArea.querySelector('.upload-placeholder').style.display = 'none';
        analyzeBtn.disabled = false;
    };
    reader.readAsDataURL(file);
}

// 이미지 분석
async function analyzeImage() {
    if (!selectedFile) {
        showError('이미지를 먼저 업로드해주세요.');
        return;
    }

    // UI 상태 변경
    resultSection.style.display = 'none';
    errorSection.style.display = 'none';
    loadingSection.style.display = 'block';
    analyzeBtn.disabled = true;

    try {
        // FormData 생성
        const formData = new FormData();
        formData.append('image', selectedFile);

        const hint = hintInput.value.trim();
        if (hint) {
            formData.append('hint', hint);
        }

        const confMin = parseFloat(confInput.value);
        formData.append('conf_min', confMin);

        // API 호출
        const response = await fetch(`${API_BASE_URL}/analyze`, {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            throw new Error(`서버 오류: ${response.status} ${response.statusText}`);
        }

        const data = await response.json();
        currentResult = data;

        // 결과 표시
        displayResults(data);

    } catch (error) {
        console.error('분석 오류:', error);
        showError(`분석 중 오류가 발생했습니다: ${error.message}`);
    } finally {
        loadingSection.style.display = 'none';
        analyzeBtn.disabled = false;
    }
}

// 결과 표시
function displayResults(data) {
    // 이모지 합성 이미지
    if (data.emoji_image) {
        displayEmojiImage(data.emoji_image);
    }

    // 감정 분석 결과
    displayEmotions(data.emotions);

    // SNS 추천
    displayRecommendation(data.recommendation);

    // 결과 섹션 표시
    resultSection.style.display = 'block';

    // 결과로 스크롤
    resultSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

// 이모지 이미지 표시
function displayEmojiImage(base64Image) {
    const emojiImageCard = document.getElementById('emojiImageCard');
    const emojiImage = document.getElementById('emojiImage');

    emojiImage.src = `data:image/png;base64,${base64Image}`;
    emojiImageCard.style.display = 'block';
}

// 감정 결과 표시
function displayEmotions(emotions) {
    const summaryDiv = document.getElementById('emotionSummary');
    const listDiv = document.getElementById('emotionList');

    // 요약
    const emotionCounts = {};
    emotions.forEach(item => {
        emotionCounts[item.emotion] = (emotionCounts[item.emotion] || 0) + 1;
    });

    const emotionEmoji = {
        'happiness': '😊',
        'neutral': '😐',
        'sadness': '😢',
        'anger': '😠',
        'disgust': '🤢',
        'surprise': '😮'
    };

    const summaryText = Object.entries(emotionCounts)
        .map(([emotion, count]) => `${emotionEmoji[emotion] || '🙂'} ${emotion}: ${count}명`)
        .join(' • ');

    summaryDiv.innerHTML = `
        <strong>총 ${emotions.length}개의 얼굴 검출</strong><br>
        ${summaryText}
    `;

    // 상세 목록
    listDiv.innerHTML = emotions.map((item, idx) => `
        <div class="emotion-item">
            <span class="emotion-label">
                ${emotionEmoji[item.emotion] || '🙂'} ${item.emotion} #${idx + 1}
            </span>
            <span class="emotion-confidence">${item.confidence.toFixed(1)}%</span>
        </div>
    `).join('');
}

// SNS 추천 결과 표시
function displayRecommendation(recommendation) {
    // 캡션
    const captionDiv = document.getElementById('captionResult');
    captionDiv.textContent = recommendation.sns_caption;

    // 해시태그
    const hashtagDiv = document.getElementById('hashtagResult');
    hashtagDiv.innerHTML = recommendation.hashtags
        .map(tag => `<span class="hashtag">#${tag}</span>`)
        .join('');

    // 음악
    const musicDiv = document.getElementById('musicResult');
    musicDiv.innerHTML = `
        <div class="music-main">
            <div class="music-icon">🎵</div>
            <div class="music-info">
                <h3>${recommendation.music.title}</h3>
                <p>${recommendation.music.artist}</p>
            </div>
        </div>
    `;

    // 음악 후보
    const candidatesDiv = document.getElementById('musicCandidates');
    if (recommendation.music_candidates && recommendation.music_candidates.length > 1) {
        candidatesDiv.innerHTML = `
            <h4>다른 추천 곡:</h4>
            ${recommendation.music_candidates.slice(1, 6).map(song => `
                <div class="music-candidate-item">
                    🎶 ${song.title} - ${song.artist}
                </div>
            `).join('')}
        `;
    } else {
        candidatesDiv.innerHTML = '';
    }
}

// 오류 표시
function showError(message) {
    const errorMessage = document.getElementById('errorMessage');
    errorMessage.textContent = message;
    errorSection.style.display = 'block';
    resultSection.style.display = 'none';
    loadingSection.style.display = 'none';
}

// 캡션 복사
function copyCaption() {
    if (!currentResult) return;

    const caption = currentResult.recommendation.sns_caption;
    navigator.clipboard.writeText(caption)
        .then(() => alert('캡션이 복사되었습니다!'))
        .catch(() => alert('복사에 실패했습니다.'));
}

// 해시태그 복사
function copyHashtags() {
    if (!currentResult) return;

    const hashtags = currentResult.recommendation.hashtags
        .map(tag => `#${tag}`)
        .join(' ');

    navigator.clipboard.writeText(hashtags)
        .then(() => alert('해시태그가 복사되었습니다!'))
        .catch(() => alert('복사에 실패했습니다.'));
}

// 이모지 이미지 다운로드
function downloadEmojiImage() {
    if (!currentResult || !currentResult.emoji_image) return;

    const emojiImage = document.getElementById('emojiImage');
    const link = document.createElement('a');
    link.href = emojiImage.src;
    link.download = `emoji_result_${Date.now()}.png`;
    link.click();
}

// 초기화
function reset() {
    selectedFile = null;
    currentResult = null;

    previewImage.src = '';
    previewImage.style.display = 'none';
    uploadArea.querySelector('.upload-placeholder').style.display = 'block';

    hintInput.value = '';
    confInput.value = '0.15';

    resultSection.style.display = 'none';
    errorSection.style.display = 'none';
    loadingSection.style.display = 'none';

    // 이모지 이미지 카드 숨기기
    const emojiImageCard = document.getElementById('emojiImageCard');
    if (emojiImageCard) {
        emojiImageCard.style.display = 'none';
    }

    analyzeBtn.disabled = true;

    window.scrollTo({ top: 0, behavior: 'smooth' });
}

// 초기화
document.addEventListener('DOMContentLoaded', () => {
    initEventListeners();
    console.log('ForPhotos 프론트엔드가 로드되었습니다.');
    console.log(`API 서버: ${API_BASE_URL}`);
});
