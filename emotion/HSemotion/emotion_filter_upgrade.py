import os
import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')  # 서버/SSH 환경에서 저장용
import matplotlib.pyplot as plt
from facenet_pytorch import MTCNN
from hsemotion.facial_emotions import HSEmotionRecognizer

# ===== 감정별 이모지 경로 (투명 PNG 권장) =====
EMOJI_MAP = {
    'happy':    'emojis/happy.png',
    'sad':      'emojis/sad.png',
    'angry':    'emojis/angry.png',
    'surprise': 'emojis/surprise.png',
    'fear':     'emojis/fear.png',
    'disgust':  'emojis/disgust.png',
    'neutral':  'emojis/neutral.png',
    'contempt': 'emojis/contempt.png',  # HSEmotion의 8번째 클래스
}

def overlay_png(bg, fg, x, y):
    """투명 PNG(fg, BGRA)를 배경(bg, BGR)에 합성"""
    H, W = bg.shape[:2]
    Hf, Wf = fg.shape[:2]
    if x >= W or y >= H:
        return bg
    x1, y1 = max(0, x), max(0, y)
    x2, y2 = min(W, x + Wf), min(H, y + Hf)
    fg_x1, fg_y1 = x1 - x, y1 - y
    fg_x2, fg_y2 = fg_x1 + (x2 - x1), fg_y1 + (y2 - y1)
    if x1 >= x2 or y1 >= y2:
        return bg

    roi = bg[y1:y2, x1:x2]
    fg_crop = fg[fg_y1:fg_y2, fg_x1:fg_x2]
    if fg_crop.shape[2] == 4:
        alpha = fg_crop[:, :, 3:4] / 255.0
        fg_rgb = fg_crop[:, :, :3]
        blended = (alpha * fg_rgb + (1.0 - alpha) * roi).astype(np.uint8)
        bg[y1:y2, x1:x2] = blended
    else:
        bg[y1:y2, x1:x2] = fg_crop[:, :, :3]
    return bg

def add_emotion_emojis(image_path, emotions, out_path='with_emoji.png',
                       emoji_map=EMOJI_MAP, size_scale=0.6, y_offset_ratio=0.15):
    """감정 결과에 맞춰 얼굴 위에 이모지 합성"""
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"이미지를 읽을 수 없습니다: {image_path}")
    cache = {}

    for e in emotions:
        emotion = e.get('emotion', '').lower()
        x, y, w, h = e.get('box', (None, None, None, None))
        if emotion not in emoji_map or None in (x, y, w, h):
            continue
        path = emoji_map[emotion]
        if not os.path.exists(path):
            continue
        if emotion not in cache:
            png = cv2.imread(path, cv2.IMREAD_UNCHANGED)  # BGRA
            if png is None:
                continue
            cache[emotion] = png
        png = cache[emotion]

        size = max(24, int(h * size_scale))
        png = cv2.resize(png, (size, size), interpolation=cv2.INTER_AREA)
        y_off = int(h * y_offset_ratio)
        px = int(x + w / 2 - size / 2)
        py = int(y - size - y_off)
        img = overlay_png(img, png, px, py)

    cv2.imwrite(out_path, img)
    return out_path

class EmotionAnalyzer:
    """
    HSEmotion + MTCNN 기반 감정 분석기 (AffectNet 8-class: contempt 포함)
    """
    def __init__(self, device='cpu', model_name='enet_b2_8', mtcnn_kwargs=None):
        # 얼굴 검출기 (MTCNN)
        self.mtcnn = MTCNN(keep_all=True, **(mtcnn_kwargs or {}))
        # 감정 분류기 (HSEmotion)
        self.fer = HSEmotionRecognizer(model_name=model_name, device=device)
        # HSEmotion 기본 클래스: Anger, Contempt, Disgust, Fear, Happiness, Neutral, Sadness, Surprise
        self.class_order = ['anger','contempt','disgust','fear','happiness','neutral','sadness','surprise']

    def _clip_box(self, box, W, H, pad=0):
        x1, y1, x2, y2 = box
        x1 = max(0, int(x1 - pad)); y1 = max(0, int(y1 - pad))
        x2 = min(W, int(x2 + pad));  y2 = min(H, int(y2 + pad))
        w = max(0, x2 - x1); h = max(0, y2 - y1)
        return x1, y1, w, h

    def analyze_emotion(self, image_path, conf_min=0.0):
        """
        단일 이미지에서 여러 얼굴 감정 분석
        Returns: [{'emotion': str, 'confidence': float, 'box': (x,y,w,h)}, ...]
        """
        bgr = cv2.imread(image_path)
        if bgr is None:
            print(f"이미지 로드 실패: {image_path}")
            return []
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        H, W = rgb.shape[:2]

        # 얼굴 검출
        boxes, _ = self.mtcnn.detect(rgb)
        results = []
        if boxes is None or len(boxes) == 0:
            # 얼굴이 없으면 전체를 하나의 얼굴로 간주(포토부스 실패 대비)
            boxes = np.array([[0, 0, W, H]])

        for box in boxes:
            x1, y1, w, h = self._clip_box(box, W, H, pad=8)
            if w <= 0 or h <= 0:
                continue
            face_rgb = rgb[y1:y1+h, x1:x1+w].copy()

            # HSEmotion: 확률 반환 원하면 logits=False
            emotion_str, probs = self.fer.predict_emotions(face_rgb, logits=False)
            # 반환 문자열은 대문자 시작이므로 소문자로 통일
            em = emotion_str.lower()
            conf = float(np.max(probs)) if hasattr(probs, "__len__") else 0.0

            if conf >= conf_min:
                results.append({
                    'emotion': em,
                    'confidence': conf * 100.0,
                    'box': (x1, y1, w, h)
                })
        return results

    def analyze_multiple_images(self, image_paths, conf_min=0.0):
        out = {}
        for p in image_paths:
            out[p] = self.analyze_emotion(p, conf_min=conf_min) if os.path.exists(p) else []
        return out

    def visualize_results(self, results, save_path=None, show_plot=False):
        if not results:
            print("시각화할 결과가 없습니다."); return
        n = len(results)
        fig, axes = plt.subplots(n, 2, figsize=(20, 8 * n))
        if n == 1:
            axes = np.array([axes])

        for i, (path, emos) in enumerate(results.items()):
            bgr = cv2.imread(path); img = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB) if bgr is not None else None
            if img is None:
                axes[i,0].text(0.5,0.5,f"이미지 열기 실패\n{os.path.basename(path)}",ha='center',va='center',transform=axes[i,0].transAxes)
                axes[i,0].axis('off')
            else:
                for j, e in enumerate(emos):
                    x,y,w,h = e['box']
                    cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
                    cv2.putText(img,f"{j+1}:{e['emotion']}({e['confidence']:.1f}%)",(x,y-8),
                                cv2.FONT_HERSHEY_SIMPLEX,0.7,(255,0,0),2)
                axes[i,0].imshow(img); axes[i,0].axis('off')

            axes[i,1].axis('off')
            if emos:
                txt = "result:\n\n" + "\n".join([f"face {k+1}: {e['emotion']} ({e['confidence']:.1f}%)" for k,e in enumerate(emos)])
                axes[i,1].text(0.1,0.9,txt,transform=axes[i,1].transAxes,fontsize=12,va='top',
                               bbox=dict(boxstyle='round,pad=0.5',facecolor='lightblue',alpha=0.85))
            else:
                axes[i,1].text(0.5,0.5,"얼굴을 감지할 수 없습니다.",ha='center',va='center',transform=axes[i,1].transAxes,
                               fontsize=12,bbox=dict(boxstyle='round,pad=0.5',facecolor='lightcoral',alpha=0.85))

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight'); print(f"시각화 저장: {save_path}")
        if show_plot: plt.show()
        else: plt.close(fig)

def main():
    analyzer = EmotionAnalyzer(device='cpu', model_name='enet_b2_8')  # GPU면 'cuda'로
    targets = ['input.jpg']
    existing = [p for p in targets if os.path.exists(p)]
    if not existing:
        # 샘플 생성 (얼굴이 없어 분석은 빈 결과일 수 있음)
        dummy = np.random.randint(0,255,(300,400,3),dtype=np.uint8)
        cv2.imwrite('sample.jpg', dummy); existing = ['sample.jpg']

    # 1) 감정 분석
    results = analyzer.analyze_multiple_images(existing, conf_min=0.0)

    # 2) 패널 저장
    analyzer.visualize_results(results, save_path='emotion_result.png', show_plot=False)

    # 3) 이모지 오버레이 저장
    for img_path, emos in results.items():
        out = os.path.splitext(os.path.basename(img_path))[0] + '_emoji.png'
        add_emotion_emojis(img_path, emos, out_path=out)
        print(f"이모지 오버레이 저장: {out}")

if __name__ == '__main__':
    main()
