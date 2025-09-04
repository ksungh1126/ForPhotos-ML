"""커맨드라인 인터페이스.

인자 파싱 → 감정 분석 → 시각화 → 이모지 합성 순으로 전체 파이프라인을 실행합니다.
이모지 겹침 방지 옵션 추가.
"""

from __future__ import annotations
import os
import argparse
import numpy as np
import cv2
from typing import List

from .analyzer import EmotionAnalyzer
from .config import AppConfig
from .emoji import add_emotion_emojis
from .visualize import visualize_results


def positive_float(x: str) -> float:
    try:
        v = float(x)
        if v < 0:
            raise ValueError
        return v
    except Exception:
        raise argparse.ArgumentTypeError("0 이상의 실수를 입력하세요.")


def main() -> None:
    p = argparse.ArgumentParser(description="Emotion – HSemotion+MTCNN")
    p.add_argument("--images", nargs="+", help="입력 이미지 경로(여러 장 가능)")
    p.add_argument("--device", default="cpu", choices=["cpu", "cuda"], help="모델 실행 디바이스")
    p.add_argument("--model", default="enet_b2_8", help="HSEmotion 모델명")
    p.add_argument("--conf-min", type=positive_float, default=0.0, help="감정 신뢰도(0~1) 임계값")
    p.add_argument("--emoji-dir", default="./examples/emojis", help="이모지 PNG 디렉터리")
    p.add_argument("--panel-out", default=None, help="패널 저장 파일 경로(.png)")
    p.add_argument("--overlay-out-dir", default=".", help="이모지 합성 결과 저장 디렉터리")
    p.add_argument("--show", action="store_true", help="패널을 화면에 표시")
    p.add_argument("--no-overlap-avoid", action="store_true", help="이모지 겹침 방지 기능 비활성화")
    p.add_argument("--emoji-size-scale", type=positive_float, default=0.6, help="얼굴 대비 이모지 크기 비율")
    p.add_argument("--emoji-y-offset", type=positive_float, default=0.15, help="얼굴 위쪽 오프셋 비율")

    args = p.parse_args()

    # 입력 중 존재하는 파일만 필터링, 없으면 더미 생성
    existing: List[str] = [p for p in args.images if os.path.exists(p)] if args.images else []
    if not existing:
        dummy = np.random.randint(0, 255, (300, 400, 3), dtype=np.uint8)
        cv2.imwrite("sample.jpg", dummy)
        existing = ["sample.jpg"]

    # 설정/매핑
    cfg = AppConfig(emoji_dir=args.emoji_dir)
    emoji_map = cfg.build_emoji_map()

    # 1) 감정 분석
    analyzer = EmotionAnalyzer(device=args.device, model_name=args.model)
    results = analyzer.analyze_multiple_images(existing, conf_min=args.conf_min)

    # 2) 패널 저장/표시
    visualize_results(results, save_path=args.panel_out, show_plot=args.show)

    # 3) 이모지 오버레이 저장
    os.makedirs(args.overlay_out_dir, exist_ok=True)
    for img_path, emos in results.items():
        base = os.path.splitext(os.path.basename(img_path))[0]
        out_path = os.path.join(args.overlay_out_dir, f"{base}_emoji.png")
        add_emotion_emojis(
            img_path, 
            emos, 
            out_path=out_path, 
            emoji_map=emoji_map,
            size_scale=args.emoji_size_scale,
            y_offset_ratio=args.emoji_y_offset,
            avoid_overlap=not args.no_overlap_avoid
        )
        print(f"이모지 오버레이 저장: {out_path}")


if __name__ == "__main__":
    main()