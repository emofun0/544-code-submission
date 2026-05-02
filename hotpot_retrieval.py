#!/usr/bin/env python3
"""Utilities for building and querying a HotpotQA fullwiki Chroma index."""

from __future__ import annotations

import bz2
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_COLLECTION_NAME = "hotpot_fullwiki_abstracts"
DEFAULT_ENTITY_MODEL = "urchade/gliner_medium-v2.1"
DEFAULT_ENTITY_LABELS = (
    "person",
    "organization",
    "location",
    "creative work",
    "film",
    "book",
    "album",
    "song",
    "band",
    "event",
)
TITLE_SUFFIXES = (
    "(film)",
    "(TV series)",
    "(band)",
    "(album)",
    "(book)",
    "(novel)",
    "(song)",
)
LEADING_QUESTION_WORDS = {
    "a",
    "an",
    "and",
    "are",
    "did",
    "do",
    "does",
    "in",
    "is",
    "of",
    "the",
    "these",
    "this",
    "what",
    "when",
    "where",
    "which",
    "who",
    "whom",
    "whose",
    "were",
}


def strip_html_tags(text: str) -> str:
    text = re.sub(r"<[^>]+>", "", text)
    text = text.replace("&amp;", "&").replace("&quot;", '"').replace("&#39;", "'")
    return re.sub(r"\s+", " ", text).strip()


def normalize_paragraph(sentences: list[str]) -> str:
    paragraph = " ".join(str(sentence).strip() for sentence in sentences if str(sentence).strip())
    return strip_html_tags(paragraph)


def normalize_sentences(sentences: list[str]) -> list[str]:
    return [strip_html_tags(str(sentence).strip()) for sentence in sentences if strip_html_tags(str(sentence).strip())]


def iter_wiki_paragraph_records(source_dir: Path) -> Iterable[dict[str, Any]]:
    for subdir in sorted(path for path in source_dir.iterdir() if path.is_dir()):
        for bz2_path in sorted(subdir.glob("*.bz2")):
            with bz2.open(bz2_path, "rt", encoding="utf-8") as handle:
                for line in handle:
                    if not line.strip():
                        continue
                    page = json.loads(line)
                    title = str(page.get("title", "")).strip()
                    article_id = page.get("id")
                    raw_text = page.get("text", [])
                    if raw_text and all(isinstance(item, str) for item in raw_text):
                        paragraphs = [raw_text]
                    else:
                        paragraphs = raw_text
                    for paragraph_index, sentences in enumerate(paragraphs):
                        if not isinstance(sentences, list):
                            continue
                        clean_sentences = normalize_sentences(sentences)
                        paragraph_text = " ".join(clean_sentences).strip()
                        if not paragraph_text:
                            continue
                        yield {
                            "chunk_id": f"{title}::{paragraph_index}",
                            "article_id": article_id,
                            "title": title,
                            "paragraph_index": paragraph_index,
                            "sentences": clean_sentences,
                            "text": paragraph_text,
                        }


class EmbeddingBackend:
    def __init__(self, model_name: str, cache_dir: Path | None = None) -> None:
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name
        self.model = SentenceTransformer(model_name, cache_folder=str(cache_dir) if cache_dir else None)

    def encode(self, texts: list[str], batch_size: int) -> list[list[float]]:
        vectors = self.model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return vectors.tolist()


@dataclass
class RetrievedChunk:
    chunk_id: str
    title: str
    paragraph_index: int
    sentences: list[str]
    text: str
    distance: float | None
    retrieval_source: str = "vector"
    query_mention: str | None = None


def choose_torch_device(requested: str) -> str:
    if requested == "cpu":
        return "cpu"
    if requested == "cuda":
        return "cuda"
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:  # noqa: BLE001
        return "cpu"


def clean_title_candidate(text: str) -> str:
    candidate = re.sub(r"\s+", " ", text).strip(" \t\r\n\"'“”‘’.,;:!?()[]{}")
    candidate = re.sub(r"'s$", "", candidate)
    parts = candidate.split()
    while parts and parts[0].lower() in LEADING_QUESTION_WORDS:
        parts.pop(0)
    while parts and parts[-1].lower() in LEADING_QUESTION_WORDS:
        parts.pop()
    return " ".join(parts).strip()


def add_unique(items: list[str], item: str) -> None:
    if item and item not in items:
        items.append(item)


def extract_quoted_title_candidates(question: str) -> list[str]:
    candidates: list[str] = []
    for match in re.finditer(r'"([^"]+)"|“([^”]+)”|‘([^’]+)’', question):
        add_unique(candidates, clean_title_candidate(next(group for group in match.groups() if group)))
    return candidates


def extract_rule_title_candidates(question: str) -> list[str]:
    candidates = extract_quoted_title_candidates(question)

    token = r"(?:[A-Z][A-Za-z0-9'&./-]*|[A-Z]|\d{4}(?:/\d{2})?|\d{4}\s*S/S|S/S)"
    for match in re.finditer(rf"\b{token}(?:\s+{token})*", question):
        candidate = clean_title_candidate(match.group(0))
        if len(candidate) >= 2 and not candidate.isdigit():
            add_unique(candidates, candidate)
    return candidates


def is_useful_title_candidate(candidate: str) -> bool:
    parts = candidate.split()
    if not parts:
        return False
    if len(parts) == 1 and not candidate.isupper() and not re.search(r"\d|/", candidate):
        return False
    return True


def title_id_variants(candidate: str, include_suffixes: bool) -> list[str]:
    variants = [candidate]
    if include_suffixes and "(" not in candidate:
        variants.extend(f"{candidate} {suffix}" for suffix in TITLE_SUFFIXES)
    return [f"{variant}::0" for variant in variants]


class EntityTitleExtractor:
    def __init__(
        self,
        model_name: str = DEFAULT_ENTITY_MODEL,
        cache_dir: Path | None = None,
        device: str = "auto",
        threshold: float = 0.35,
    ) -> None:
        self.model_name = model_name
        self.cache_dir = cache_dir
        self.device = choose_torch_device(device)
        self.threshold = threshold
        self._model = None

    def _load_model(self):
        if self._model is None:
            try:
                from gliner import GLiNER
            except ImportError as exc:
                raise ImportError("GLiNER is required for entity title retrieval. Install it with `pip install gliner`.") from exc
            self._model = GLiNER.from_pretrained(
                self.model_name,
                cache_dir=str(self.cache_dir) if self.cache_dir else None,
                map_location=self.device,
            )
            if hasattr(self._model, "to"):
                self._model.to(self.device)
        return self._model

    def extract(self, question: str) -> list[str]:
        candidates = extract_rule_title_candidates(question)
        model = self._load_model()
        entities = model.predict_entities(
            question,
            labels=list(DEFAULT_ENTITY_LABELS),
            threshold=self.threshold,
            flat_ner=True,
        )
        for entity in entities:
            add_unique(candidates, clean_title_candidate(str(entity.get("text", ""))))
        return candidates


class ChromaWikiRetriever:
    def __init__(
        self,
        chroma_dir: Path,
        embedding_model: str = DEFAULT_EMBEDDING_MODEL,
        cache_dir: Path | None = None,
        collection_name: str = DEFAULT_COLLECTION_NAME,
        use_entity_title_retrieval: bool = True,
        entity_model: str = DEFAULT_ENTITY_MODEL,
        entity_cache_dir: Path | None = None,
        entity_device: str = "auto",
        entity_threshold: float = 0.35,
    ) -> None:
        import chromadb

        self.embedding_backend = EmbeddingBackend(model_name=embedding_model, cache_dir=cache_dir)
        self.client = chromadb.PersistentClient(path=str(chroma_dir))
        self.collection = self.client.get_collection(name=collection_name)
        self.entity_title_extractor = (
            EntityTitleExtractor(
                model_name=entity_model,
                cache_dir=entity_cache_dir,
                device=entity_device,
                threshold=entity_threshold,
            )
            if use_entity_title_retrieval
            else None
        )
        self._title_cache: dict[str, list[RetrievedChunk]] = {}

    def _chunks_from_chroma_result(
        self,
        ids: list[str],
        documents: list[str],
        metadatas: list[dict[str, Any] | None],
        distances: list[float | None] | None,
        source: str,
        mentions_by_id: dict[str, str] | None = None,
    ) -> list[RetrievedChunk]:
        chunks = []
        if distances is None:
            distances = [None] * len(ids)
        for chunk_id, text, metadata, distance in zip(ids, documents, metadatas, distances):
            chunk_metadata = metadata or {}
            chunks.append(
                RetrievedChunk(
                    chunk_id=str(chunk_id),
                    title=str(chunk_metadata.get("title", "")),
                    paragraph_index=int(chunk_metadata.get("paragraph_index", 0)),
                    sentences=json.loads(str(chunk_metadata.get("sentences_json", "[]"))),
                    text=str(text),
                    distance=float(distance) if distance is not None else None,
                    retrieval_source=source,
                    query_mention=mentions_by_id.get(str(chunk_id)) if mentions_by_id else None,
                )
            )
        return chunks

    def query_vector(self, question: str, top_k: int) -> list[RetrievedChunk]:
        embedding = self.embedding_backend.encode([question], batch_size=1)[0]
        result = self.collection.query(
            query_embeddings=[embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )
        documents = result.get("documents", [[]])[0]
        metadatas = result.get("metadatas", [[]])[0]
        distances = result.get("distances", [[]])[0]
        ids = result.get("ids", [[]])[0]
        return self._chunks_from_chroma_result(ids, documents, metadatas, distances, source="vector")

    def query_title_mentions(self, question: str, max_results: int) -> list[RetrievedChunk]:
        if self.entity_title_extractor is None:
            return []
        if question in self._title_cache:
            return self._title_cache[question][:max_results]

        quoted_candidates = set(extract_quoted_title_candidates(question))
        candidates = [
            candidate
            for candidate in self.entity_title_extractor.extract(question)
            if is_useful_title_candidate(candidate)
        ]
        requested_ids: list[str] = []
        mentions_by_id: dict[str, str] = {}
        for candidate in candidates:
            for chunk_id in title_id_variants(candidate, include_suffixes=candidate in quoted_candidates):
                if chunk_id not in mentions_by_id:
                    requested_ids.append(chunk_id)
                    mentions_by_id[chunk_id] = candidate
        if not requested_ids:
            self._title_cache[question] = []
            return []

        result = self.collection.get(ids=requested_ids, include=["documents", "metadatas"])
        found = {
            str(chunk_id): (str(document), metadata)
            for chunk_id, document, metadata in zip(
                result.get("ids", []),
                result.get("documents", []),
                result.get("metadatas", []),
            )
        }
        exact_matched_mentions = {
            mention
            for chunk_id, mention in mentions_by_id.items()
            if chunk_id == f"{mention}::0" and chunk_id in found
        }
        ordered_ids = [
            chunk_id
            for chunk_id in requested_ids
            if chunk_id in found
            and (
                mentions_by_id[chunk_id] not in exact_matched_mentions
                or chunk_id == f"{mentions_by_id[chunk_id]}::0"
            )
        ]
        documents = [found[chunk_id][0] for chunk_id in ordered_ids]
        metadatas = [found[chunk_id][1] for chunk_id in ordered_ids]
        chunks = self._chunks_from_chroma_result(
            ordered_ids,
            documents,
            metadatas,
            distances=None,
            source="title",
            mentions_by_id=mentions_by_id,
        )
        deduped = []
        seen_titles = set()
        for chunk in chunks:
            if chunk.title in seen_titles:
                continue
            seen_titles.add(chunk.title)
            deduped.append(chunk)
        self._title_cache[question] = deduped
        return deduped[:max_results]

    def query(self, question: str, top_k: int) -> list[RetrievedChunk]:
        title_chunks = self.query_title_mentions(question, max_results=top_k)
        vector_chunks = self.query_vector(question, top_k=top_k + len(title_chunks))
        chunks = []
        seen_ids = set()
        for chunk in [*title_chunks, *vector_chunks]:
            if chunk.chunk_id in seen_ids:
                continue
            seen_ids.add(chunk.chunk_id)
            chunks.append(chunk)
            if len(chunks) >= top_k:
                break
        return chunks
