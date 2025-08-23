import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')  # 서버 환경에서도 저장 가능하도록
import matplotlib.pyplot as plt
from deepface import DeepFace
import os
from PIL import Image
import warnings
warnings.filterwarnings('ignore')


class EmotionAnalyzer:
    def __init__(self):
        self.emotions = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
        self.emotion_korean = {
            'angry': '화남',
            'disgust': '역겨움',
            'fear': '두려움',
            'happy': '행복',
            'sad': '슬픔',
            'surprise': '놀람',
            'neutral': '무표정'
        }
    
    def analyze_emotion(self, image_path):
        try:
            result = DeepFace.analyze(image_path, actions=['emotion'], enforce_detection=False)
            if isinstance(result, list):
                emotions = []
                for face_result in result:
                    emotion = face_result['dominant_emotion']
                    confidence = face_result['emotion'][emotion]
                    region = face_result.get('region', None)
                    if region:
                        x, y, w, h = region['x'], region['y'], region['w'], region['h']
                    else:
                        x, y, w, h = None, None, None, None
                    emotions.append({'emotion': emotion, 'confidence': confidence, 'box': (x, y, w, h)})
                return emotions
            else:
                emotion = result['dominant_emotion']
                confidence = result['emotion'][emotion]
                region = result.get('region', None)
                if region:
                    x, y, w, h = region['x'], region['y'], region['w'], region['h']
                else:
                    x, y, w, h = None, None, None, None
                return [{'emotion': emotion, 'confidence': confidence, 'box': (x, y, w, h)}]
        except Exception as e:
            print(f"감정 분석 중 오류 발생: {e}")
            return []
    
    def analyze_multiple_images(self, image_paths):
        results = {}
        for image_path in image_paths:
            if os.path.exists(image_path):
                print(f"분석 중: {image_path}")
                emotions = self.analyze_emotion(image_path)
                results[image_path] = emotions
            else:
                print(f"파일을 찾을 수 없습니다: {image_path}")
                results[image_path] = []
        return results
    
    def visualize_results(self, results, save_path=None, show_plot=True):
        if not results:
            print("시각화할 결과가 없습니다.")
            return
        
        n_images = len(results)
        cols = 2
        rows = n_images
        fig, axes = plt.subplots(rows, cols, figsize=(20, 8 * rows))
        if n_images == 1:
            axes = axes.reshape(1, -1)
        
        plt.subplots_adjust(wspace=0.1)
        
        for idx, (image_path, emotions) in enumerate(results.items()):
            if idx >= rows:
                break
            
            img = cv2.imread(image_path)
            if img is not None:
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                if emotions:
                    for i, emotion_data in enumerate(emotions):
                        box = emotion_data.get('box', (None, None, None, None))
                        x, y, w, h = box
                        if None not in box:
                            cv2.rectangle(img_rgb, (x, y), (x + w, y + h), (0, 255, 0), 2)
                            label = f"{i+1}: {emotion_data['emotion']}"
                            cv2.putText(img_rgb, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX,
                                        0.8, (255, 0, 0), 2)
                axes[idx, 0].imshow(img_rgb)
                axes[idx, 0].axis('off')
            else:
                axes[idx, 0].text(0.5, 0.5, f"이미지를 로드할 수 없습니다:\n{os.path.basename(image_path)}",
                                  ha='center', va='center', transform=axes[idx, 0].transAxes)
                axes[idx, 0].axis('off')
            
            axes[idx, 1].axis('off')
            if emotions:
                emotion_text = f"result :\n\n"
                for i, emotion_data in enumerate(emotions):
                    emotion_text += f"face {i+1}: {emotion_data['emotion']} ({emotion_data['confidence']:.1f}%)\n"
                axes[idx, 1].text(0.1, 0.9, emotion_text, transform=axes[idx, 1].transAxes,
                                  fontsize=12, va='top',
                                  bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.8))
            else:
                axes[idx, 1].text(0.5, 0.5, "얼굴을 감지할 수 없습니다.",
                                  ha='center', va='center', transform=axes[idx, 1].transAxes,
                                  fontsize=12, bbox=dict(boxstyle='round,pad=0.5', facecolor='lightcoral', alpha=0.8))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"시각화 결과가 저장되었습니다: {save_path}")
        
        if show_plot:
            plt.show()
        else:
            plt.close(fig)

    def print_summary(self, results):
        print("\n" + "="*50)
        print("감정 분석 결과 요약")
        print("="*50)
        
        total_faces = 0
        emotion_counts = {emotion: 0 for emotion in self.emotions}
        
        for image_path, emotions in results.items():
            print(f"\n📸 {os.path.basename(image_path)}:")
            if emotions:
                for i, emotion_data in enumerate(emotions):
                    emotion = emotion_data['emotion']
                    confidence = emotion_data['confidence']
                    print(f"  얼굴 {i+1}: {emotion} ({confidence:.1f}%)")
                    emotion_counts[emotion] += 1
                    total_faces += 1
            else:
                print("  ❌ 얼굴을 감지할 수 없습니다.")
        
        print(f"\n📊 전체 통계:")
        print(f"  총 감지된 얼굴 수: {total_faces}")
        if total_faces > 0:
            print("  감정 분포:")
            for emotion in self.emotions:
                if emotion_counts[emotion] > 0:
                    percentage = (emotion_counts[emotion] / total_faces) * 100
                    print(f"    {emotion}: {emotion_counts[emotion]}명 ({percentage:.1f}%)")


def main():
    analyzer = EmotionAnalyzer()
    test_images = ['input.jpg']
    existing_images = [img for img in test_images if os.path.exists(img)]
    
    if not existing_images:
        print("분석할 이미지 파일이 없습니다.")
        sample_image = np.random.randint(0, 255, (300, 400, 3), dtype=np.uint8)
        cv2.imwrite('sample_face.jpg', sample_image)
        existing_images = ['sample_face.jpg']
    
    print(f"분석할 이미지: {existing_images}")
    results = analyzer.analyze_multiple_images(existing_images)
    analyzer.visualize_results(results, save_path="emotion_result.png", show_plot=False)
    analyzer.print_summary(results)


if __name__ == "__main__":
    main()
