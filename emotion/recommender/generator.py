"""Core generation utilities wrapping a local Qwen vision-language model (revamped)."""
from __future__ import annotations

import json
import re
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image
from transformers import AutoModelForVision2Seq, AutoProcessor

from .config import RecommendationConfig
from .knowledge import MusicKnowledgeBase, SongEntry


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
    hashtags: List[str]
    song_title: str
    song_artist: str
    caption_raw: str
    music_raw: str
    music_candidates: List[Dict[str, Any]]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "sns_caption": self.caption,
            "hashtags": self.hashtags,
            "music": {"title": self.song_title, "artist": self.song_artist},
            "raw_text": {
                "caption": self.caption_raw,
                "music": self.music_raw,
            },
            "music_candidates": self.music_candidates,
        }


class RecommendationEngine:
    """High level wrapper that handles prompting and decoding."""

    def __init__(self, config: Optional[RecommendationConfig] = None) -> None:
        self.config = config or RecommendationConfig()
        self._model = None
        self._processor = None
        self._device = self.config.resolve_device()
        self._music_kb: Optional[MusicKnowledgeBase] = None
        if self.config.music_kb_path:
            try:
                self._music_kb = MusicKnowledgeBase(
                    self.config.music_kb_path,
                    self.config.music_kb_top_k,
                    embedder_name=self.config.music_embedder_model,
                )
            except (FileNotFoundError, ValueError) as exc:
                warnings.warn(
                    f"Failed to load music knowledge base '{self.config.music_kb_path}': {exc}",
                    RuntimeWarning,
                )
                self._music_kb = None

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

    def _build_caption_prompt(self, request: RecommendationRequest) -> str:
        parts = [self.config.caption_user_prompt]
        if request.user_hint:
            parts.append("Photographer hint: " + request.user_hint.strip())
        parts.append(self.config.caption_response_schema_hint)
        return "\n".join(parts)

    def _generate_with_prompt(self, image: Image.Image, prompt: str) -> str:
        messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": self.config.system_prompt}],
            },
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": prompt},
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
        return generated_text.strip()

    def _search_music_entries(
        self,
        caption: str,
        hashtags: List[str],
        request: RecommendationRequest,
    ) -> List[SongEntry]:
        if not self._music_kb or not self._music_kb.loaded:
            return []
        query_parts = [caption]
        if hashtags:
            query_parts.append(" ".join(hashtags))
            query_parts.extend(hashtags)
        if request.user_hint:
            query_parts.append(request.user_hint)
        query = " ".join(part for part in query_parts if part).strip()
        return self._music_kb.retrieve(query)

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

    def _parse_caption_result(self, raw_text: str) -> Tuple[str, List[str]]:
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
        normalized_hashtags = [
            tag[1:].strip() if tag.startswith("#") else tag for tag in hashtags
        ]
        return caption, [tag for tag in normalized_hashtags if tag]

    def generate(self, request: RecommendationRequest) -> RecommendationResult:
        image = request.resolve_image()
        caption_prompt = self._build_caption_prompt(request)
        caption_raw = self._generate_with_prompt(image, caption_prompt)
        caption, hashtags = self._parse_caption_result(caption_raw)

        kb_entries = self._search_music_entries(caption, hashtags, request)
        song_title = ""
        song_artist = ""
        music_raw = ""
        if kb_entries:
            top_entry = kb_entries[0]
            song_title = top_entry.title
            song_artist = top_entry.artist
            try:
                music_raw = json.dumps(top_entry.as_dict(), ensure_ascii=False)
            except Exception:
                music_raw = f"{top_entry.title} - {top_entry.artist}"

        return RecommendationResult(
            caption=caption,
            hashtags=hashtags,
            song_title=song_title,
            song_artist=song_artist,
            caption_raw=caption_raw,
            music_raw=music_raw,
            music_candidates=[entry.as_dict() for entry in kb_entries],
        )


__all__ = [
    "RecommendationEngine",
    "RecommendationRequest",
    "RecommendationResult",
]
