"""Command line interface for generating SNS recommendations from an image."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

from .config import RecommendationConfig
from .generator import RecommendationEngine, RecommendationRequest


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate SNS caption, hashtags, and song recommendations from an image."
    )
    parser.add_argument("image", type=Path, help="Path to the photobooth-style image file")
    parser.add_argument(
        "--hint",
        type=str,
        default=None,
        help="Optional photographer note to guide the tone or context",
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        default=None,
        help="Override the default vision-language model (HF repo id or local path)",
    )
    parser.add_argument(
        "--music-kb",
        type=str,
        default=None,
        help="Optional path to a JSON/CSV knowledge base for factual song metadata",
    )
    parser.add_argument(
        "--music-embedder",
        type=str,
        default=None,
        help="SentenceTransformer model id for semantic music retrieval (set empty to disable)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device override (e.g. cuda, cpu, mps)",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=None,
        help="Maximum number of new tokens to generate",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="Sampling temperature; set 0 for deterministic output",
    )
    parser.add_argument(
        "--use-fast-processor",
        action="store_true",
        help="Use the fast image processor implementation (requires recent PyTorch)",
    )
    parser.add_argument(
        "--music-kb-top-k",
        type=int,
        default=None,
        help="Number of top matches to surface from the music knowledge base",
    )
    return parser


def run_cli(args: Optional[argparse.Namespace] = None) -> int:
    parser = build_arg_parser()
    parsed = parser.parse_args(args=args)

    config_kwargs = {}
    if parsed.model_dir:
        config_kwargs["model_dir"] = parsed.model_dir
    if parsed.device:
        config_kwargs["device"] = parsed.device
    if parsed.max_new_tokens is not None:
        config_kwargs["max_new_tokens"] = parsed.max_new_tokens
    if parsed.temperature is not None:
        config_kwargs["temperature"] = parsed.temperature
    if parsed.use_fast_processor:
        config_kwargs["use_fast_image_processor"] = True
    if parsed.music_kb:
        config_kwargs["music_kb_path"] = parsed.music_kb
    if parsed.music_embedder is not None:
        config_kwargs["music_embedder_model"] = parsed.music_embedder or None
    if parsed.music_kb_top_k is not None:
        config_kwargs["music_kb_top_k"] = parsed.music_kb_top_k

    engine = RecommendationEngine(RecommendationConfig(**config_kwargs))
    request = RecommendationRequest(image_path=parsed.image, user_hint=parsed.hint)
    result = engine.generate(request)
    print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(run_cli())
