"""Simple retrieval-augmented lookup for factual music metadata."""
from __future__ import annotations

import csv
import json
import re
import warnings
from difflib import SequenceMatcher
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

try:
    import numpy as np
except ImportError:  # pragma: no cover
    np = None  # type: ignore[assignment]

try:
    from sentence_transformers import SentenceTransformer  # type: ignore
except ImportError:  # pragma: no cover
    SentenceTransformer = None  # type: ignore[assignment]


_TOKEN_PATTERN = re.compile(r"[A-Za-z0-9가-힣']+")
_KOR_TO_ENG_KEYWORDS: Dict[str, List[str]] = {
    "사랑": ["love", "romantic"],
    "연애": ["love", "romantic"],
    "커플": ["love", "romantic", "couple"],
    "데이트": ["date", "romantic"],
    "감성": ["emotional", "sentimental"],
    "감동": ["emotional"],
    "행복": ["happy", "bright"],
    "즐거움": ["happy", "bright"],
    "추억": ["nostalgic", "memory"],
    "우정": ["friendship"],
    "여행": ["travel", "adventure"],
    "휴가": ["vacation", "travel"],
    "여름": ["summer", "sunny"],
    "겨울": ["winter", "snow"],
    "봄": ["spring", "fresh"],
    "가을": ["autumn", "fall"],
    "바다": ["beach", "ocean"],
    "하늘": ["sky"],
    "비": ["rain", "rainy"],
    "눈": ["snow", "winter"],
    "밤": ["night"],
    "노을": ["sunset"],
    "파티": ["party", "dance"],
    "생일": ["birthday", "celebration"],
    "청량": ["fresh", "bright"],
    "힐링": ["healing", "calm"],
    "차분": ["calm"],
    "슬픔": ["sad", "melancholy"],
    "따뜻": ["warm", "cozy"],
    "포근": ["warm", "cozy"],
    "시크": ["chic"],
    "에너지": ["energetic"],
    "댄스": ["dance"],
    "발라드": ["ballad"],
    "록": ["rock"],
    "락": ["rock"],
    "팝": ["pop"],
    "레트로": ["retro"],
    "귀여": ["cute"],
    "섹시": ["sexy"],
    "힙": ["hip-hop", "hiphop"],
    "힙합": ["hip-hop", "hiphop"],
    "도시": ["city"],
    "자연": ["nature"],
    "꽃": ["flower"],
    "피크닉": ["picnic"],
    "캠핑": ["camping"],
    "운동": ["workout", "sporty"],
    "필름": ["vintage", "retro"],
    "빈티": ["vintage", "retro"],
    "카페": ["cafe"],
    "추상": ["abstract"],
    "몽환": ["dreamy"],
    "요정": ["dreamy", "fairy"],
    "분위기": ["vibe", "mood"],
    "열정": ["passion", "energetic"],
    "시원": ["refreshing", "cool"],
    "여신": ["goddess", "elegant"],
    "우울": ["sad", "melancholy"],
    "차가": ["cool", "chic"],
    "클래식": ["classical"],
    "재즈": ["jazz"],
    "소울": ["soul"],
    "알앤비": ["r&b"],
    "인디": ["indie"],
    "밝": ["bright"],
    "어두": ["dark"],
    "드라마": ["drama"],
}


def _tokenize(text: str) -> List[str]:
    return [token.lower() for token in _TOKEN_PATTERN.findall(text)]


def _augment_query_tokens(tokens: List[str]) -> List[str]:
    augmented: List[str] = []
    for token in tokens:
        augmented.append(token)
        for kor, engs in _KOR_TO_ENG_KEYWORDS.items():
            if kor in token:
                augmented.extend(eng.lower() for eng in engs)
        if token in _KOR_TO_ENG_KEYWORDS:
            augmented.extend(eng.lower() for eng in _KOR_TO_ENG_KEYWORDS[token])
    return augmented


def _jaccard_score(query_tokens: Iterable[str], doc_tokens: Iterable[str]) -> float:
    query_set = set(query_tokens)
    doc_set = set(doc_tokens)
    if not query_set or not doc_set:
        return 0.0
    intersection = query_set & doc_set
    union = query_set | doc_set
    return len(intersection) / len(union)


@dataclass
class SongEntry:
    title: str
    artist: str
    extra: Dict[str, Any]

    def to_context_line(self) -> str:
        pieces = [f"title: {self.title}", f"artist: {self.artist}"]
        description = self.extra.get("description") or self.extra.get("notes") or ""
        if description:
            pieces.append(f"note: {description}")
        keywords = self.extra.get("keywords")
        if isinstance(keywords, list):
            keyword_str = ", ".join(str(k) for k in keywords if k)
            if keyword_str:
                pieces.append(f"keywords: {keyword_str}")
        elif isinstance(keywords, str) and keywords.strip():
            pieces.append(f"keywords: {keywords.strip()}")
        return " | ".join(pieces)

    def as_dict(self) -> Dict[str, Any]:
        payload = {"title": self.title, "artist": self.artist}
        payload.update(self.extra)
        return payload


class MusicKnowledgeBase:
    """Lightweight knowledge base for grounding music responses."""

    def __init__(
        self,
        path: Optional[str],
        top_k: int,
        *,
        embedder_name: Optional[str] = None,
    ) -> None:
        self._entries: List[SongEntry] = []
        self.top_k = max(1, top_k)
        self._embedder_name = embedder_name
        self._embedder: Optional[SentenceTransformer] = None  # type: ignore[assignment]
        self._entry_embeddings: Optional[np.ndarray] = None  # type: ignore[assignment]
        if path:
            self._load(path)
            self._maybe_prepare_embeddings()

    @property
    def loaded(self) -> bool:
        return bool(self._entries)

    def _load(self, path: str) -> None:
        kb_path = Path(path)
        if not kb_path.exists():
            raise FileNotFoundError(f"Music knowledge base not found: {path}")
        suffix = kb_path.suffix.lower()
        if suffix in {".json", ".jsonl"}:
            self._load_json(kb_path, json_lines=(suffix == ".jsonl"))
        elif suffix in {".csv", ".tsv"}:
            delimiter = "\t" if suffix == ".tsv" else ","
            self._load_csv(kb_path, delimiter)
        else:
            raise ValueError(
                f"Unsupported knowledge base format '{suffix}'. Use JSON, JSONL, CSV, or TSV."
            )

    def _load_json(self, path: Path, json_lines: bool) -> None:
        entries: List[Dict[str, Any]]
        if json_lines:
            entries = []
            with path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    line = line.strip()
                    if not line:
                        continue
                    entries.append(json.loads(line))
        else:
            with path.open("r", encoding="utf-8") as handle:
                data = json.load(handle)
            if isinstance(data, dict):
                entries = list(data.values())
            elif isinstance(data, list):
                entries = data
            else:
                raise ValueError("JSON knowledge base must be an object or array.")
        self._entries = [self._normalize_entry(item) for item in entries if item]

    def _load_csv(self, path: Path, delimiter: str) -> None:
        entries: List[Dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as handle:
            reader = csv.DictReader(handle, delimiter=delimiter)
            for row in reader:
                entries.append(row)
        self._entries = [self._normalize_entry(item) for item in entries if item]

    def _normalize_entry(self, raw: Dict[str, Any]) -> SongEntry:
        title = str(raw.get("title") or raw.get("song") or "").strip()
        artist = str(raw.get("artist") or raw.get("singer") or "").strip()
        if not title or not artist:
            raise ValueError("Every knowledge base entry must include 'title' and 'artist'.")
        extra = dict(raw)
        extra.pop("title", None)
        extra.pop("song", None)
        extra.pop("artist", None)
        extra.pop("singer", None)
        return SongEntry(title=title, artist=artist, extra=extra)

    def _maybe_prepare_embeddings(self) -> None:
        if not self._embedder_name:
            return
        if SentenceTransformer is None or np is None:
            warnings.warn(
                "sentence-transformers or numpy not available; semantic music retrieval disabled.",
                RuntimeWarning,
            )
            return
        try:
            self._embedder = SentenceTransformer(self._embedder_name)
        except Exception as exc:
            warnings.warn(
                f"Failed to load music embedder '{self._embedder_name}': {exc}",
                RuntimeWarning,
            )
            self._embedder = None
            return
        try:
            texts = [self._entry_to_text(entry) for entry in self._entries]
            embeddings = self._embedder.encode(
                texts,
                convert_to_numpy=True,
                normalize_embeddings=True,
            )
            self._entry_embeddings = embeddings
        except Exception as exc:
            warnings.warn(
                f"Failed to compute music embeddings: {exc}",
                RuntimeWarning,
            )
            self._entry_embeddings = None

    def _entry_to_text(self, entry: SongEntry) -> str:
        fragments: List[str] = [entry.title, entry.artist]
        keywords = entry.extra.get("keywords")
        if isinstance(keywords, list):
            fragments.extend(str(k) for k in keywords if k)
        elif isinstance(keywords, str):
            fragments.append(keywords)
        description = entry.extra.get("description") or entry.extra.get("notes")
        if isinstance(description, str):
            fragments.append(description)
        return " | ".join(fragment for fragment in fragments if fragment)

    def _string_similarity(self, query: str, entry: SongEntry) -> float:
        if not query.strip():
            return 0.0
        try:
            entry_text = self._entry_to_text(entry)
            return SequenceMatcher(None, query.lower(), entry_text.lower()).ratio()
        except Exception:
            return 0.0

    def _embed_query(self, query: str) -> Optional[np.ndarray]:
        if not query.strip() or not self._embedder or self._entry_embeddings is None:
            return None
        try:
            embedding = self._embedder.encode(
                [query],
                convert_to_numpy=True,
                normalize_embeddings=True,
            )[0]
            return embedding
        except Exception as exc:
            warnings.warn(
                f"Failed to embed music query: {exc}",
                RuntimeWarning,
            )
            return None

    def retrieve(self, query: str) -> List[SongEntry]:
        if not self._entries or not query.strip():
            return []
        semantic_results = self._semantic_retrieve(query)
        if semantic_results:
            return semantic_results
        token_results = self._token_retrieve(query)
        if token_results:
            return token_results
        # ultimate fallback: first top_k entries
        return self._entries[: self.top_k]

    def _semantic_retrieve(self, query: str) -> List[SongEntry]:
        if self._entry_embeddings is None:
            return []
        query_embedding = self._embed_query(query)
        if query_embedding is None:
            return []
        similarities = np.dot(self._entry_embeddings, query_embedding)  # type: ignore[arg-type]
        ranked_indices = np.argsort(-similarities)[: self.top_k]
        ranked: List[Tuple[float, SongEntry]] = []
        for idx in ranked_indices:
            score = float(similarities[idx])
            ranked.append((score, self._entries[int(idx)]))
        positives = [entry for score, entry in ranked if score > 0]
        if positives:
            return positives[: self.top_k]
        if ranked:
            ranked.sort(key=lambda item: item[0], reverse=True)
            return [entry for _, entry in ranked[: self.top_k]]
        return []

    def _token_retrieve(self, query: str) -> List[SongEntry]:
        query_tokens = _augment_query_tokens(_tokenize(query))
        scored: List[Tuple[float, SongEntry]] = []
        for entry in self._entries:
            document_tokens = _tokenize(entry.title) + _tokenize(entry.artist)
            description = entry.extra.get("description") or entry.extra.get("notes") or ""
            if isinstance(description, str):
                document_tokens.extend(_tokenize(description))
            keywords = entry.extra.get("keywords")
            if isinstance(keywords, list):
                for keyword in keywords:
                    document_tokens.extend(_tokenize(str(keyword)))
            elif isinstance(keywords, str):
                document_tokens.extend(_tokenize(keywords))
            score = _jaccard_score(query_tokens, document_tokens)
            scored.append((score, entry))
        positives = [item for item in scored if item[0] > 0]
        if positives:
            positives.sort(key=lambda item: item[0], reverse=True)
            return [entry for _, entry in positives[: self.top_k]]
        scored.sort(key=lambda item: item[0], reverse=True)
        if scored:
            similarity_rank: List[Tuple[float, SongEntry]] = [
                (self._string_similarity(query, entry), entry) for _, entry in scored
            ]
            similarity_rank.sort(key=lambda item: item[0], reverse=True)
            filtered = [entry for score, entry in similarity_rank[: self.top_k] if score > 0]
            if filtered:
                return filtered
        return [entry for _, entry in scored[: self.top_k]]

    def render_context(self, entries: List[SongEntry]) -> str:
        if not entries:
            return ""
        lines = [entry.to_context_line() for entry in entries]
        return "\n".join(f"- {line}" for line in lines)
