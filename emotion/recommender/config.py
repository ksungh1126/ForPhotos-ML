"""Configuration objects for the photobooth SNS recommendation pipeline."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

_DEFAULT_MODEL_DIR = Path(__file__).resolve().parents[1] / "Qwen2.5-VL"


@dataclass
class RecommendationConfig:
    """Settings for loading the Qwen model and shaping its outputs."""

    model_dir: Path = field(default_factory=lambda: _DEFAULT_MODEL_DIR)
    device: Optional[str] = None
    max_new_tokens: int = 512
    temperature: float = 0.8
    top_p: float = 0.95
    top_k: int = 80
    repetition_penalty: float = 1.02
    use_fast_image_processor: bool = False
    system_prompt: str = (
        "당신은 소셜 미디어 콘텐츠 제작을 전문으로 하는 유용하고 트렌디한 어시스턴트입니다. "
        "당신의 임무는 인생네컷 사진을 보고 그에 어울리는 SNS 캡션, 해시태그, 그리고 노래를 추천하는 것입니다. "
        "캡션과 해시태그는 반드시 한국어로 작성해야 합니다. "
        "추천하는 노래는 사진과 캡션의 분위기와 어울리는 유명한 K팝 또는 팝 트랙이어야 합니다."
    )
    response_schema_hint: str = (
        "반드시 다음 구조의 JSON 형식으로만 응답하세요:\n"
        "{\n"
        '  "sns_caption": "사진과 어울리는 창의적인 한글 문구",\n'
        '  "hashtags": ["한글 해시태그1", "한글 해시태그2", "한글 해시태그3"],\n'
        "  \"music\": {\n"
        "    \"title\": \"사진과 어울리는 유명 K팝 또는 팝송의 정확한 공식 제목\",\n"
        "    \"artist\": \"해당 곡의 정확한 공식 아티스트명\"\n"
        "  }\n"
        "}\n"
        "널리 알려진 곡만 추천하고, 제목과 아티스트가 실제 발매된 정보와 일치하는지 다시 확인하세요."
    )


    def resolve_device(self) -> str:
        """Return the device string to use for inference."""
        if self.device:
            return self.device
        try:
            import torch

            if torch.cuda.is_available():
                return "cuda"
            if torch.backends.mps.is_available():  # type: ignore[attr-defined]
                return "mps"
        except Exception:  # pragma: no cover - defensive
            pass
        return "cpu"


__all__ = ["RecommendationConfig"]
