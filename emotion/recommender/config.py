"""Configuration objects for the photobooth SNS recommendation pipeline."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

_DEFAULT_VLM_MODEL_NAME = "Qwen/Qwen2.5-VL-7B-Instruct"
_DEFAULT_MUSIC_DB_PATH = Path(__file__).resolve().with_name("music_db.json")
_DEFAULT_MUSIC_EMBEDDER = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


@dataclass
class RecommendationConfig:
    """Settings for loading the Qwen model and shaping its outputs."""

    model_dir: str = field(default_factory=lambda: _DEFAULT_VLM_MODEL_NAME)
    device: Optional[str] = None
    max_new_tokens: int = 512
    temperature: float = 0.8
    top_p: float = 0.95
    top_k: int = 80
    repetition_penalty: float = 1.02
    use_fast_image_processor: bool = False
    system_prompt: str = (
        "You are a fashionable social media assistant who turns photobooth images into viral-ready posts. "
        "First craft a Korean SNS caption and hashtags that match the mood, then—on a separate step—recommend a fitting song."
    )
    caption_user_prompt: str = (
        "Look at the attached photobooth image. Based on the people, poses, facial expressions, and overall vibe, "
        "suggest a concise yet punchy SNS caption plus 3-5 natural and popular hashtags."
    )
    caption_response_schema_hint: str = (
        "Respond strictly with JSON in the following structure:\n"
        "{\n"
        '  "sns_caption": "caption text",\n'
        '  "hashtags": ["hashtag 1", "hashtag 2", "hashtag 3"]\n'
        "}\n"
        "Do not include the # symbol inside the list items; return pure Korean words."
    )
    music_kb_path: Optional[str] = field(default_factory=lambda: str(_DEFAULT_MUSIC_DB_PATH))
    music_kb_top_k: int = 5
    music_embedder_model: Optional[str] = field(default_factory=lambda: _DEFAULT_MUSIC_EMBEDDER)
    music_rag_intro: str = (
        "Ground your answer in the verified songs below. Prefer one of them if it matches the vibe, "
        "or explain briefly why another famous track is a better fit."
    )

    def _detect_device(self) -> str:
        """Return the device string to use for inference."""
        try:
            import torch

            if torch.cuda.is_available():
                return "cuda"
            if torch.backends.mps.is_available():  # type: ignore[attr-defined]
                return "mps"
        except Exception:  # pragma: no cover - defensive
            pass
        return "cpu"

    def resolve_device(self) -> str:
        if self.device:
            return self.device
        return self._detect_device()

__all__ = ["RecommendationConfig"]
