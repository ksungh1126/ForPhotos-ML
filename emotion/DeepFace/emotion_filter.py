import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')  # 서버/터미널 환경에서 이미지 저장용
import matplotlib.pyplot as plt
from deepface import DeepFace
import os
from PIL import Image
import warnings
warnings.filterwarnings('ignore')


# === 감정별 이모지 경로 매핑 (투명 배경 PNG 권장) ===
EMOJI_MAP = {
    'happy':    'emojis/happy.png',
    'sad':      'emojis/sad.png',
    'angry':    'emojis/angry.png',
    'surprise': 'emojis/surprise.png',
    'fear':     'emojis/fear.png',
    'disgust':  'emojis/disgust.png',
    'neutral':  'emojis/neutral.png',
}


def overlay_png(bg, fg, x, y):
    """
    투명 PNG(fg, BGRA)를 배경(bg, BGR)에 합성
    (x, y): 배치할 좌상단 좌표
    """
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
        # 알파가 없으면 그냥 덮어씀
        bg[y1:y2, x1:x2] = fg_crop[:, :, :3]

    return bg


def add_emotion_emojis(image_path, emotions, out_path='with_emoji.png',
                       emoji_map=EMOJI_MAP, size_scale=0.6, y_offset_ratio=0.15):
    """
    분석 결과(emotions)에 따라 얼굴 주변에 감정 이모지를 오버레이하여 저장
    emotions: [{'emotion': 'happy', 'confidence': 99.0, 'box': (x,y,w,h)}, ...]
    """
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"이미지를 읽을 수 없습니다: {image_path}")

    emoji_cache = {}
    for e in emotions:
        emotion = e.get('emotion')
        x, y, w, h = e.get('box', (None, None, None, None))
        if emotion not in emoji_map or None in (x, y, w, h):
            continue

        if emotion not in emoji_cache:
            p = emoji_map[emotion]
            if not os.path.exists(p):
                continue
            png = cv2.imread(p, cv2.IMREAD_UNCHANGED)  # BGRA
            if png is None:
                continue
            emoji_cache[emotion] = png
        png = emoji_cache[emotion]

        # 이모지 크기: 얼굴 높이에 비례
        size = max(24, int(h * size_scale))
        png_resized = cv2.resize(png, (size, size), interpolation=cv2.INTER_AREA)

        # 얼굴 상단 중앙에 살짝 띄워 배치
        y_off = int(h * y_offset_ratio)
        place_x = int(x + w / 2 - size / 2)
        place_y = int(y - size - y_off)

        img = overlay_png(img, png_resized, place_x, place_y)

    cv2.imwrite(out_path, img)
    return out_path


class EmotionAnalyzer:
    def __init__(self):
        self.emotions = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
        self.emotion_korean = {
            'angry': '화남', 'disgust': '역겨움', 'fear': '두려움',
            'happy': '행복', 'sad': '슬픔', 'surprise': '놀람', 'neutral': '무표정'
        }

    def analyze_emotion(self, image_path):
        try:
            result = DeepFace.analyze(image_path, actions=['emotion'], enforce_detection=False)
            # DeepFace 버전에 따라 단일/복수 포맷 다름
            faces = result if isinstance(result, list) else [result]
            out = []
            for face in faces:
                emotion = face['dominant_emotion']
                conf = float(face['emotion'][emotion])
                region = face.get('region', {}) or {}
                x, y, w, h = region.get('x'), region.get('y'), region.get('w'), region.get('h')
                out.append({'emotion': emotion, 'confidence': conf, 'box': (x, y, w, h)})
            return out
        except Exception as e:
            print(f"감정 분석 중 오류 발생: {e}")
            return []

    def analyze_multiple_images(self, image_paths):
        results = {}
        for p in image_paths:
            if os.path.exists(p):
                print(f"분석 중: {p}")
                results[p] = self.analyze_emotion(p)
            else:
                print(f"파일을 찾을 수 없습니다: {p}")
                results[p] = []
        return results

    def visualize_results(self, results, save_path=None, show_plot=True):
        if not results:
            print("시각화할 결과가 없습니다."); return
        n = len(results)
        fig, axes = plt.subplots(n, 2, figsize=(20, 8 * n))
        if n == 1:
            axes = np.array([axes])

        for idx, (path, emos) in enumerate(results.items()):
            img_bgr = cv2.imread(path)
            if img_bgr is None:
                axes[idx, 0].text(0.5, 0.5, f"이미지를 열 수 없음\n{os.path.basename(path)}",
                                  ha='center', va='center', transform=axes[idx, 0].transAxes)
                axes[idx, 0].axis('off')
            else:
                img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                for i, e in enumerate(emos or []):
                    x, y, w, h = e.get('box', (None, None, None, None))
                    if None not in (x, y, w, h):
                        cv2.rectangle(img_rgb, (x, y), (x + w, y + h), (0, 255, 0), 2)
                        cv2.putText(img_rgb, f"{i+1}:{e['emotion']}", (x, y - 8),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                axes[idx, 0].imshow(img_rgb); axes[idx, 0].axis('off')

            axes[idx, 1].axis('off')
            txt = "result:\n\n" + "\n".join(
                [f"face {i+1}: {e['emotion']} ({e['confidence']:.1f}%)" for i, e in enumerate(emos or [])]
            ) if emos else "얼굴을 감지할 수 없습니다."
            axes[idx, 1].text(0.1, 0.9, txt, transform=axes[idx, 1].transAxes,
                              fontsize=12, va='top',
                              bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue' if emos else 'lightcoral', alpha=0.85))

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"시각화 결과 저장: {save_path}")
        if show_plot:
            plt.show()
        else:
            plt.close(fig)

    def print_summary(self, results):
        print("\n" + "=" * 50)
        print("감정 분석 결과 요약")
        print("=" * 50)
        total = 0
        cnt = {e: 0 for e in self.emotions}
        for p, emos in results.items():
            print(f"\n📸 {os.path.basename(p)}:")
            if emos:
                for i, e in enumerate(emos):
                    print(f"  얼굴 {i+1}: {e['emotion']} ({e['confidence']:.1f}%)")
                    cnt[e['emotion']] += 1; total += 1
            else:
                print("  ❌ 얼굴을 감지할 수 없습니다.")
        print(f"\n📊 총 얼굴 수: {total}")
        if total:
            print("  감정 분포:")
            for k, v in cnt.items():
                if v:
                    print(f"    {k}: {v}명 ({100*v/total:.1f}%)")


def main():
    analyzer = EmotionAnalyzer()

    test_images = ['input.jpg']
    existing = [p for p in test_images if os.path.exists(p)]
    if not existing:
        print("분석할 이미지 파일이 없습니다. 샘플 이미지를 생성합니다.")
        sample = np.random.randint(0, 255, (300, 400, 3), dtype=np.uint8)
        cv2.imwrite('sample_face.jpg', sample)
        existing = ['sample_face.jpg']

    print(f"분석할 이미지: {existing}")
    results = analyzer.analyze_multiple_images(existing)

    # 1) 패널 시각화 저장
    analyzer.visualize_results(results, save_path="emotion_result.png", show_plot=False)

    # 2) 얼굴 주변에 감정 이모지 오버레이 저장
    for img_path, emos in results.items():
        out_path = os.path.splitext(os.path.basename(img_path))[0] + "_emoji.png"
        saved = add_emotion_emojis(img_path, emos, out_path=out_path)
        print(f"이모지 오버레이 저장: {saved}")

    analyzer.print_summary(results)


if __name__ == "__main__":
    main()