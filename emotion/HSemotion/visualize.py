"""분석 결과 시각화 패널 생성.

- `visualize_results`: 원본 이미지(좌) + 박스/라벨, 텍스트 요약(우) 패널을 생성/저장
"""

from __future__ import annotations
import os
import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from typing import Dict, List


def visualize_results(results: Dict[str, List[dict]], save_path: str | None = None, show_plot: bool = False) -> None:
    """입력 이미지별 감정 분석 결과를 패널로 저장/표시"""
    if not results:
        print("시각화할 결과가 없습니다.")
        return
    n = len(results)
    fig, axes = plt.subplots(n, 2, figsize=(20, 8 * n))
    if n == 1:
        axes = np.array([axes])

    for i, (path, emos) in enumerate(results.items()):
        bgr = cv2.imread(path)
        img = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB) if bgr is not None else None
        if img is None:
            axes[i, 0].text(
                0.5,
                0.5,
                f"이미지 열기 실패\n{os.path.basename(path)}",
                ha="center",
                va="center",
                transform=axes[i, 0].transAxes,
            )
            axes[i, 0].axis("off")
        else:
            for j, e in enumerate(emos):
                x, y, w, h = e["box"]
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(
                    img,
                    f"{j+1}:{e['emotion']}({e['confidence']:.1f}%)",
                    (x, y - 8),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 0, 0),
                    2,
                )
            axes[i, 0].imshow(img)
            axes[i, 0].axis("off")

        axes[i, 1].axis("off")
        if emos:
            txt = "result:\n\n" + "\n".join(
                [f"face {k+1}: {e['emotion']} ({e['confidence']:.1f}%)" for k, e in enumerate(emos)]
            )
            axes[i, 1].text(
                0.1,
                0.9,
                txt,
                transform=axes[i, 1].transAxes,
                fontsize=12,
                va="top",
                bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.85),
            )
        else:
            axes[i, 1].text(
                0.5,
                0.5,
                "얼굴을 감지할 수 없습니다.",
                ha="center",
                va="center",
                transform=axes[i, 1].transAxes,
                fontsize=12,
                bbox=dict(boxstyle="round,pad=0.5", facecolor="lightcoral", alpha=0.85),
            )

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"시각화 저장: {save_path}")
    if show_plot:
        plt.show()
    else:
        plt.close(fig)