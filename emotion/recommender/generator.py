"""Core generation utilities wrapping a local Qwen vision-language model (revamped)."""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from PIL import Image
from transformers import AutoModelForVision2Seq, AutoProcessor

from .config import RecommendationConfig


@dataclass
class RecommendationRequest:
    """Inputs required to build a recommendation."""

    image_path: Path
    user_hint: Optional[str] = None

    def resolve_image(self) -> Image.Image:
        image = Image.open(self.image_path)
        if image.mode != "RGB":
            image = image.convert("RGB")
        return image


@dataclass
class RecommendationResult:
    """Structured representation of the model output."""

    caption: str
    song_title: str
    song_artist: str
    raw_text: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "sns_caption": self.caption,
            "music": {"title": self.song_title, "artist": self.song_artist},
            "raw_text": self.raw_text,
        }


class RecommendationEngine:
    """High level wrapper that handles prompting and decoding."""

    def __init__(self, config: Optional[RecommendationConfig] = None) -> None:
        self.config = config or RecommendationConfig()
        self._model = None
        self._processor = None
        self._device = self.config.resolve_device()

    @property
    def processor(self) -> AutoProcessor:
        if self._processor is None:
            processor_kwargs = {"trust_remote_code": True}
            processor_kwargs["use_fast"] = self.config.use_fast_image_processor
            self._processor = AutoProcessor.from_pretrained(
                self.config.model_dir,
                **processor_kwargs,
            )
        return self._processor

    @property
    def model(self) -> AutoModelForVision2Seq:
        if self._model is None:
            torch_dtype = None
            if self._device == "cuda":
                import torch

                torch_dtype = torch.float16
            self._model = AutoModelForVision2Seq.from_pretrained(
                self.config.model_dir,
                torch_dtype=torch_dtype,
                trust_remote_code=True,
            )
            self._model.to(self._device)
            self._model.eval()
        return self._model

    def build_prompt(self, request: RecommendationRequest) -> str:
        hint_block = ""
        if request.user_hint:
            hint_block = (
                "추가적인 사진 촬영 맥락: " + request.user_hint.strip()
            )
        user_instructions = (
           "첨부된 인생네컷 사진을 분석해주세요. 사진 속 인물, 포즈, 표정, 그리고 전반적인 분위기를 바탕으로 SNS 캡션, 해시태그, 그리고 음악 추천을 생성해주세요."
        )
        prompt = user_instructions + self.config.response_schema_hint
        if hint_block:
            prompt += "\n" + hint_block
        return prompt

    def _extract_json(self, text: str) -> Optional[Dict[str, Any]]:
        candidate = None
        match = re.search(r"\{(?:[^{}]|\{[^{}]*\})*\}", text, re.DOTALL)
        if match:
            candidate = match.group(0)
        else:
            bracket_match = re.search(r"\[[^\]]*\]", text, re.DOTALL)
            if bracket_match:
                candidate = bracket_match.group(0)
        if not candidate:
            return None
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            return None

    def _parse_result(self, raw_text: str) -> RecommendationResult:
        data = self._extract_json(raw_text) or {}
        caption = str(data.get("sns_caption") or data.get("caption") or "").strip()
        hashtags_field = data.get("hashtags", [])
        hashtags: List[str]
        if isinstance(hashtags_field, str):
            hashtags = [tag.strip() for tag in hashtags_field.split() if tag.strip()]
        elif isinstance(hashtags_field, list):
            hashtags = [str(tag).strip() for tag in hashtags_field if str(tag).strip()]
        else:
            hashtags = []
        if hashtags:
            hashtag_block = " ".join(
                tag if tag.startswith("#") else f"#{tag}" for tag in hashtags
            )
            caption = f"{caption}\n{hashtag_block}" if caption else hashtag_block
        music = data.get("music") or data.get("song") or {}
        song_title = str(music.get("title") or "").strip() if isinstance(music, dict) else ""
        song_artist = (
            str(music.get("artist") or music.get("singer") or "").strip()
            if isinstance(music, dict)
            else ""
        )
        return RecommendationResult(
            caption=caption,
            song_title=song_title,
            song_artist=song_artist,
            raw_text=raw_text.strip(),
        )

    def generate(self, request: RecommendationRequest) -> RecommendationResult:
        image = request.resolve_image()
        messages = [
            {"role": "system", "content": [{"type": "text", "text": self.config.system_prompt}]},
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": self.build_prompt(request)},
                ],
            },
        ]
        chat_prompt = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.processor(
            text=[chat_prompt],
            images=[image],
            return_tensors="pt",
        )
        model_inputs = {}
        for key, value in inputs.items():
            if hasattr(value, "to"):
                model_inputs[key] = value.to(self._device)
            else:
                model_inputs[key] = value
        generate_kwargs = {
            "max_new_tokens": self.config.max_new_tokens,
            "temperature": self.config.temperature,
            "top_p": self.config.top_p,
            "top_k": self.config.top_k,
            "repetition_penalty": self.config.repetition_penalty,
            "do_sample": self.config.temperature > 0,
        }
        output_ids = self.model.generate(**model_inputs, **generate_kwargs)
        prompt_length = model_inputs["input_ids"].shape[-1]
        trimmed = output_ids[:, prompt_length:]
        generated_text = self.processor.batch_decode(trimmed, skip_special_tokens=True)[0]
        return self._parse_result(generated_text)


__all__ = [
    "RecommendationEngine",
    "RecommendationRequest",
    "RecommendationResult",
]
