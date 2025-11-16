// API 베이스 URL
const API_BASE = '/api';

// 전역 변수
let selectedFiles = {
    emotion: null,
    pose: null,
    filter: null,
    split: null
};

// 초기화
document.addEventListener('DOMContentLoaded', () => {
    initTabs();
    initEmotionTab();
    initPoseTab();
    initFilterTab();
    initSplitTab();
    fetchEngineInfo();
    console.log('ForPhotos-ML 프론트엔드 로드 완료');
});

async function fetchEngineInfo() {
    const indicator = document.getElementById('engine-indicator');
    if (!indicator) return;

    indicator.textContent = '추천 엔진 상태 확인 중...';

    try {
        const response = await fetch(`${API_BASE}/emotion/health`);
        if (!response.ok) {
            throw new Error('서버 응답 없음');
        }
        const data = await response.json();
        const engineName = data.engine_type || (data.using_gemini ? 'Gemini API' : 'Qwen2.5-VL');
        const badgeClass = data.recommender_ready ? 'ok' : 'pending';
        const badgeText = data.recommender_ready ? '가동 중' : '로딩 중';
        const analyzer = data.emotion_analyzer_ready ? '감정 모델 준비 완료' : '감정 모델 초기화 중';

        indicator.innerHTML = `<strong>추천 엔진</strong>: ${engineName} <span class="engine-badge ${badgeClass}">${badgeText}</span><br><small>${analyzer}</small>`;
        indicator.classList.remove('error');
    } catch (error) {
        console.error('엔진 상태 조회 실패', error);
        indicator.textContent = '추천 엔진 상태를 불러오지 못했습니다';
        indicator.classList.add('error');
    }
}

// 탭 전환
function initTabs() {
    const tabButtons = document.querySelectorAll('.tab-button');
    const tabContents = document.querySelectorAll('.tab-content');

    tabButtons.forEach(button => {
        button.addEventListener('click', () => {
            const tabName = button.dataset.tab;

            // 모든 탭 버튼과 컨텐츠 비활성화
            tabButtons.forEach(btn => btn.classList.remove('active'));
            tabContents.forEach(content => content.classList.remove('active'));

            // 선택된 탭 활성화
            button.classList.add('active');
            document.getElementById(`${tabName}-tab`).classList.add('active');
        });
    });
}

// ===== 감정 분석 탭 =====
function initEmotionTab() {
    const uploadArea = document.getElementById('emotion-upload');
    const input = document.getElementById('emotion-input');
    const preview = document.getElementById('emotion-preview');
    const analyzeBtn = document.getElementById('emotion-analyze-btn');

    setupFileUpload(uploadArea, input, preview, (file) => {
        selectedFiles.emotion = file;
        analyzeBtn.disabled = false;
    });

    analyzeBtn.addEventListener('click', analyzeEmotion);
}

async function analyzeEmotion() {
    if (!selectedFiles.emotion) return;

    const loading = document.getElementById('emotion-loading');
    const result = document.getElementById('emotion-result');
    const formData = new FormData();

    formData.append('image', selectedFiles.emotion);
    formData.append('hint', document.getElementById('emotion-hint').value);
    formData.append('conf_min', document.getElementById('emotion-conf').value);
    formData.append('add_emoji', 'true');

    loading.style.display = 'block';
    result.style.display = 'none';

    try {
        const response = await fetch(`${API_BASE}/emotion/analyze`, {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.detail || '분석 실패');
        }

        const data = await response.json();
        displayEmotionResult(data);
        result.style.display = 'block';
    } catch (error) {
        alert(`분석 오류: ${error.message}`);
    } finally {
        loading.style.display = 'none';
    }
}

function displayEmotionResult(data) {
    const resultDiv = document.getElementById('emotion-result');

    let html = '<div class="result-section">';

    // 이모지 이미지
    if (data.emoji_image) {
        html += `
            <div class="result-card">
                <h3>🎨 이모지 오버레이</h3>
                <img src="data:image/png;base64,${data.emoji_image}" style="max-width:100%;border-radius:8px;">
            </div>
        `;
    }

    // 감정 분석
    html += `
        <div class="result-card">
            <h3>😊 감정 분석 결과</h3>
            <p><strong>검출된 얼굴: ${data.emotions.length}개</strong></p>
            <div class="emotion-list">
    `;

    data.emotions.forEach((item, idx) => {
        const emoji = {
            'happiness': '😊',
            'neutral': '😐',
            'sadness': '😢',
            'anger': '😠',
            'disgust': '🤢',
            'surprise': '😮',
            'fear': '😨'
        }[item.emotion] || '🙂';

        html += `
            <div class="emotion-item">
                <span>${emoji} ${item.emotion} #${idx + 1}</span>
                <span>${item.confidence.toFixed(1)}%</span>
            </div>
        `;
    });

    html += '</div></div>';

    // SNS 추천
    const rec = data.recommendation;
    html += `
        <div class="result-card">
            <h3>📝 SNS 캡션</h3>
            <p class="caption-text">${rec.sns_caption}</p>
            <button onclick="copyToClipboard('${rec.sns_caption.replace(/'/g, "\\'")}')">복사</button>
        </div>

        <div class="result-card">
            <h3>#️⃣ 해시태그</h3>
            <div class="hashtag-list">
                ${rec.hashtags.map(tag => `<span class="hashtag">#${tag}</span>`).join('')}
            </div>
            <button onclick="copyToClipboard('${rec.hashtags.map(t => '#' + t).join(' ').replace(/'/g, "\\'")}')">복사</button>
        </div>

        <div class="result-card">
            <h3>🎵 추천 음악</h3>
            <div class="music-info">
                <p><strong>${rec.music.title}</strong></p>
                <p>${rec.music.artist}</p>
            </div>
        </div>
    `;

    html += '</div>';
    resultDiv.innerHTML = html;
}

// ===== 포즈 분석 탭 =====
function initPoseTab() {
    const uploadArea = document.getElementById('pose-upload');
    const input = document.getElementById('pose-input');
    const preview = document.getElementById('pose-preview');
    const analyzeBtn = document.getElementById('pose-analyze-btn');

    setupFileUpload(uploadArea, input, preview, (file) => {
        selectedFiles.pose = file;
        analyzeBtn.disabled = false;
    });

    analyzeBtn.addEventListener('click', analyzePose);
}

async function analyzePose() {
    if (!selectedFiles.pose) return;

    const loading = document.getElementById('pose-loading');
    const result = document.getElementById('pose-result');
    const formData = new FormData();

    formData.append('image', selectedFiles.pose);

    loading.style.display = 'block';
    result.style.display = 'none';

    try {
        const response = await fetch(`${API_BASE}/pose/analyze`, {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.detail || '분석 실패');
        }

        const data = await response.json();
        result.innerHTML = `
            <div class="result-section">
                <div class="result-card">
                    <h3>🧍 포즈 분석 결과</h3>
                    <p><strong>검출된 사람 수:</strong> ${data.num_people}명</p>
                    <p><strong>포즈 타입:</strong> ${data.pose_type}</p>
                    <p>${data.message}</p>
                </div>
            </div>
        `;
        result.style.display = 'block';
    } catch (error) {
        alert(`분석 오류: ${error.message}`);
    } finally {
        loading.style.display = 'none';
    }
}

// ===== 이미지 필터 탭 =====
function initFilterTab() {
    const uploadArea = document.getElementById('filter-upload');
    const input = document.getElementById('filter-input');
    const preview = document.getElementById('filter-preview');
    const applyBtn = document.getElementById('filter-apply-btn');
    const downloadBtn = document.getElementById('filter-download-btn');

    setupFileUpload(uploadArea, input, preview, (file) => {
        selectedFiles.filter = file;
        applyBtn.disabled = false;
    });

    applyBtn.addEventListener('click', applyFilter);
    downloadBtn.addEventListener('click', downloadFilterResult);
}

async function applyFilter() {
    if (!selectedFiles.filter) return;

    const loading = document.getElementById('filter-loading');
    const result = document.getElementById('filter-result');
    const formData = new FormData();

    formData.append('image', selectedFiles.filter);
    formData.append('filter_type', document.getElementById('filter-type').value);

    loading.style.display = 'block';
    result.style.display = 'none';

    try {
        const response = await fetch(`${API_BASE}/filter/apply`, {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            throw new Error('필터 적용 실패');
        }

        const blob = await response.blob();
        const url = URL.createObjectURL(blob);

        document.getElementById('filter-result-img').src = url;
        result.style.display = 'block';
    } catch (error) {
        alert(`필터 적용 오류: ${error.message}`);
    } finally {
        loading.style.display = 'none';
    }
}

function downloadFilterResult() {
    const img = document.getElementById('filter-result-img');
    const link = document.createElement('a');
    link.href = img.src;
    link.download = `filtered_${Date.now()}.png`;
    link.click();
}

// ===== 스트립 분할 탭 =====
function initSplitTab() {
    const uploadArea = document.getElementById('split-upload');
    const input = document.getElementById('split-input');
    const preview = document.getElementById('split-preview');
    const splitBtn = document.getElementById('split-btn');
    const downloadZipBtn = document.getElementById('split-download-zip-btn');

    setupFileUpload(uploadArea, input, preview, (file) => {
        selectedFiles.split = file;
        splitBtn.disabled = false;
    });

    splitBtn.addEventListener('click', splitPhotobooth);
    downloadZipBtn.addEventListener('click', downloadPhotoboothZip);
}

async function splitPhotobooth() {
    if (!selectedFiles.split) return;

    const loading = document.getElementById('split-loading');
    const result = document.getElementById('split-result');
    const formData = new FormData();

    formData.append('image', selectedFiles.split);

    loading.style.display = 'block';
    result.style.display = 'none';

    try {
        const response = await fetch(`${API_BASE}/split/photobooth`, {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.detail || '분할 실패');
        }

        const data = await response.json();

        const cutsDiv = document.getElementById('split-cuts');
        cutsDiv.innerHTML = `
            <h3>✂️ ${data.num_cuts}개의 컷으로 분할되었습니다</h3>
            <div class="cuts-grid">
                ${data.cuts.map(cut => `
                    <div class="cut-item">
                        <p>컷 ${cut.index + 1}</p>
                        <img src="data:image/png;base64,${cut.image}">
                    </div>
                `).join('')}
            </div>
        `;

        result.style.display = 'block';
    } catch (error) {
        alert(`분할 오류: ${error.message}`);
    } finally {
        loading.style.display = 'none';
    }
}

async function downloadPhotoboothZip() {
    if (!selectedFiles.split) return;

    const formData = new FormData();
    formData.append('image', selectedFiles.split);

    try {
        const response = await fetch(`${API_BASE}/split/photobooth/zip`, {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            throw new Error('ZIP 생성 실패');
        }

        const blob = await response.blob();
        const url = URL.createObjectURL(blob);
        const link = document.createElement('a');
        link.href = url;
        link.download = `photobooth_cuts_${Date.now()}.zip`;
        link.click();
        URL.revokeObjectURL(url);
    } catch (error) {
        alert(`ZIP 다운로드 오류: ${error.message}`);
    }
}

// ===== 유틸리티 함수 =====
function setupFileUpload(uploadArea, input, preview, onFileSelect) {
    uploadArea.addEventListener('click', () => input.click());

    input.addEventListener('change', (e) => {
        const file = e.target.files[0];
        if (file) handleFile(file, preview, onFileSelect);
    });

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
        const file = e.dataTransfer.files[0];
        if (file) handleFile(file, preview, onFileSelect);
    });
}

function handleFile(file, preview, callback) {
    if (!file.type.startsWith('image/')) {
        alert('이미지 파일만 업로드할 수 있습니다.');
        return;
    }

    const reader = new FileReader();
    reader.onload = (e) => {
        preview.src = e.target.result;
        preview.style.display = 'block';
        preview.parentElement.querySelector('.upload-placeholder').style.display = 'none';
    };
    reader.readAsDataURL(file);

    callback(file);
}

function copyToClipboard(text) {
    navigator.clipboard.writeText(text)
        .then(() => alert('복사되었습니다!'))
        .catch(() => alert('복사 실패'));
}
