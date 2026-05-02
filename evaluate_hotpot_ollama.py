#!/usr/bin/env python3
"""Evaluate one local Ollama model on HotpotQA with multiple prompting methods."""

from __future__ import annotations

import argparse
import json
import re
import string
import sys
import time
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import requests
from datasets import load_from_disk
from tqdm import tqdm

from hotpot_retrieval import (
    DEFAULT_COLLECTION_NAME,
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_ENTITY_MODEL,
    ChromaWikiRetriever,
)


OLLAMA_URL = "http://127.0.0.1:11434/api/generate"
DEFAULT_NLI_MODEL = "cross-encoder/nli-distilroberta-base"
DEFAULT_RAG_K_VALUES = (2, 5, 10)


@dataclass(frozen=True)
class MethodSpec:
    name: str
    prompt_style: str
    rag_k: int | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate one Ollama model on HotpotQA.")
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=Path("data/fullwiki"),
        help="Path created by datasets.save_to_disk (default: data/fullwiki)",
    )
    parser.add_argument("--split", default="validation", help="Dataset split name")
    parser.add_argument("--model", required=True, help="One Ollama model name")
    parser.add_argument("--start-index", type=int, default=0, help="Start index")
    parser.add_argument(
        "--num-samples",
        type=int,
        default=30,
        help="How many examples to run",
    )
    parser.add_argument(
        "--rag-k-values",
        default="2,5,10",
        help="Comma-separated top-k values for RAG (default: 2,5,10)",
    )
    parser.add_argument(
        "--nli-model",
        default=DEFAULT_NLI_MODEL,
        help=f"Hugging Face NLI model name (default: {DEFAULT_NLI_MODEL})",
    )
    parser.add_argument(
        "--nli-cache-dir",
        type=Path,
        default=Path(".hf_cache"),
        help="Directory used to cache the NLI model locally",
    )
    parser.add_argument(
        "--nli-device",
        default="auto",
        choices=("cpu", "cuda", "auto"),
        help="Device for NLI model (default: auto)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results"),
        help="Output directory",
    )
    parser.add_argument(
        "--state-file",
        type=Path,
        default=None,
        help="Persistent state file (default: <output-dir>/<model>_progress_state.json)",
    )
    parser.add_argument(
        "--reset-state",
        action="store_true",
        help="Reset persistent state before this run",
    )
    parser.add_argument(
        "--request-timeout",
        type=int,
        default=600,
        help="Per-request timeout seconds",
    )
    parser.add_argument(
        "--wiki-chroma-dir",
        type=Path,
        default=Path("data/wiki/chroma"),
        help="Persistent Chroma directory for fullwiki retrieval (default: data/wiki/chroma).",
    )
    parser.add_argument(
        "--wiki-collection-name",
        default=DEFAULT_COLLECTION_NAME,
        help=f"Chroma collection name (default: {DEFAULT_COLLECTION_NAME})",
    )
    parser.add_argument(
        "--wiki-embedding-model",
        default=DEFAULT_EMBEDDING_MODEL,
        help=f"SentenceTransformer model for querying the Chroma index (default: {DEFAULT_EMBEDDING_MODEL})",
    )
    parser.add_argument(
        "--wiki-embedding-cache-dir",
        type=Path,
        default=Path(".hf_cache"),
        help="Cache dir for retrieval embedding model downloads",
    )
    parser.add_argument(
        "--disable-entity-title-retrieval",
        action="store_true",
        help="Disable GLiNER entity/title exact-match retrieval before vector retrieval",
    )
    parser.add_argument(
        "--entity-model",
        default=DEFAULT_ENTITY_MODEL,
        help=f"GLiNER model for entity/title extraction (default: {DEFAULT_ENTITY_MODEL})",
    )
    parser.add_argument(
        "--entity-cache-dir",
        type=Path,
        default=Path(".hf_cache"),
        help="Cache dir for the GLiNER entity model",
    )
    parser.add_argument(
        "--entity-device",
        default="auto",
        choices=("cpu", "cuda", "auto"),
        help="Device for the GLiNER entity model (default: auto)",
    )
    parser.add_argument(
        "--entity-threshold",
        type=float,
        default=0.35,
        help="GLiNER entity extraction threshold (default: 0.35)",
    )
    return parser.parse_args()


def parse_rag_k_values(raw: str) -> list[int]:
    values = []
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        value = int(item)
        if value <= 0:
            raise ValueError(f"RAG k must be positive, got {value}")
        values.append(value)
    if not values:
        return list(DEFAULT_RAG_K_VALUES)
    return sorted(set(values))


def slugify(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", text).strip("_") or "model"


def normalize_answer(s: str) -> str:
    def remove_articles(text: str) -> str:
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text: str) -> str:
        return " ".join(text.split())

    def remove_punc(text: str) -> str:
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text: str) -> str:
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def f1_score(prediction: str, ground_truth: str) -> tuple[float, float, float]:
    pred_tokens = normalize_answer(prediction).split()
    truth_tokens = normalize_answer(ground_truth).split()
    common = Counter(pred_tokens) & Counter(truth_tokens)
    num_same = sum(common.values())
    if len(pred_tokens) == 0 or len(truth_tokens) == 0:
        empty_match = float(pred_tokens == truth_tokens)
        return empty_match, empty_match, empty_match
    if num_same == 0:
        return 0.0, 0.0, 0.0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1, precision, recall


def exact_match_score(prediction: str, ground_truth: str) -> float:
    return float(normalize_answer(prediction) == normalize_answer(ground_truth))


def update_answer(answer_metrics: dict[str, float], prediction: str, gold: str) -> tuple[float, float, float, float]:
    em = exact_match_score(prediction, gold)
    f1, prec, recall = f1_score(prediction, gold)
    answer_metrics["em"] += em
    answer_metrics["f1"] += f1
    answer_metrics["prec"] += prec
    answer_metrics["recall"] += recall
    return em, prec, recall, f1


def update_support(sp_metrics: dict[str, float], prediction: list[list[Any]], gold: list[list[Any]]) -> tuple[float, float, float, float]:
    predicted = {(str(x[0]), int(x[1])) for x in prediction if isinstance(x, (list, tuple)) and len(x) == 2}
    gold_set = {(str(x[0]), int(x[1])) for x in gold if isinstance(x, (list, tuple)) and len(x) == 2}
    tp = sum(1 for item in predicted if item in gold_set)
    fp = sum(1 for item in predicted if item not in gold_set)
    fn = sum(1 for item in gold_set if item not in predicted)

    prec = tp / (tp + fp) if tp + fp > 0 else 0.0
    recall = tp / (tp + fn) if tp + fn > 0 else 0.0
    f1 = 2 * prec * recall / (prec + recall) if (prec + recall) > 0 else 0.0
    em = float(fp + fn == 0)

    sp_metrics["em"] += em
    sp_metrics["f1"] += f1
    sp_metrics["prec"] += prec
    sp_metrics["recall"] += recall
    return em, prec, recall, f1


def build_chunks_from_wiki_hits(hits: list[Any]) -> list[dict[str, Any]]:
    return [
        {
            "title": hit.title,
            "sentences": hit.sentences,
        }
        for hit in hits
    ]


def render_context(chunks: list[dict[str, Any]], include_indices: bool) -> str:
    blocks = []
    for chunk in chunks:
        if include_indices:
            lines = [f"[{idx}] {sent}" for idx, sent in enumerate(chunk["sentences"])]
        else:
            lines = chunk["sentences"]
        body = "\n".join(lines).strip()
        if body:
            blocks.append(f"Title: {chunk['title']}\n{body}")
        else:
            blocks.append(f"Title: {chunk['title']}")
    return "\n\n".join(blocks)


def prepare_context_for_method(
    question: str,
    method: MethodSpec,
    wiki_retriever: ChromaWikiRetriever,
) -> dict[str, Any]:
    if method.rag_k is None:
        raise ValueError(f"Method {method.name} is not a RAG method")

    hits = wiki_retriever.query(question=question, top_k=method.rag_k)
    selected = build_chunks_from_wiki_hits(hits)
    retrieval = [
        {
            "chunk_id": hit.chunk_id,
            "title": hit.title,
                "paragraph_index": hit.paragraph_index,
                "distance": hit.distance,
                "source": hit.retrieval_source,
                "query_mention": hit.query_mention,
            }
            for hit in hits
        ]
    fallback_pairs = selected[: min(2, len(selected))]
    return {
        "chunks": selected,
        "prompt_context": render_context(selected, include_indices=True),
        "nli_fallback_premise": render_context(fallback_pairs, include_indices=False),
        "retrieval": retrieval,
    }


def build_prompt(question: str, context_text: str, prompt_style: str) -> str:
    if prompt_style == "naive":
        return (
            'Return JSON only with keys "answer" and "supporting_facts".\n'
            '{"answer": "<short answer>", "supporting_facts": [["Title A", 0], ["Title B", 3]]}\n\n'
            f"Question:\n{question}\n\n"
            f"Context:\n{context_text}\n"
        )

    if prompt_style == "grounded":
        return (
            "You are solving a HotpotQA-style multi-hop question.\n"
            "Given the question and context documents, return JSON only with this exact schema:\n"
            '{"reasoning_steps": [{"step": "brief factual step", "supporting_facts": [["Title A", 0]]}], "answer": "<short answer>", "supporting_facts": [["Title A", 0], ["Title B", 3]]}\n'
            "Rules:\n"
            "- Use only the provided context.\n"
            "- reasoning_steps should be a short ordered chain of factual hops.\n"
            "- Each reasoning step must include the supporting_facts it directly relies on.\n"
            "- supporting_facts must be a list of [title, sentence_index] pairs.\n"
            "- If uncertain, provide the best concise answer and minimal supporting facts.\n\n"
            f"Question:\n{question}\n\n"
            f"Context:\n{context_text}\n"
        )

    if prompt_style == "cot":
        return (
            "You are solving a HotpotQA-style multi-hop question.\n"
            "Reason in short evidence-grounded steps: identify the most relevant sentences, connect them, then answer.\n"
            "Return JSON only with this exact schema:\n"
            '{"reasoning_steps": [{"step": "step 1", "supporting_facts": [["Title A", 0]]}, {"step": "step 2", "supporting_facts": [["Title B", 3]]}], "answer": "<short answer>", "supporting_facts": [["Title A", 0], ["Title B", 3]]}\n'
            "Rules:\n"
            "- Use only the provided context.\n"
            "- reasoning_steps should be brief and directly tied to the context.\n"
            "- Each reasoning step must include the supporting_facts it directly relies on.\n"
            "- supporting_facts must be a list of [title, sentence_index] pairs.\n"
            "- answer must be concise.\n\n"
            f"Question:\n{question}\n\n"
            f"Context:\n{context_text}\n"
        )

    raise ValueError(f"Unknown prompt style: {prompt_style}")


def json_object_candidates(text: str) -> list[str]:
    stripped = text.strip()
    if not stripped:
        return []
    candidates = [stripped]
    starts = [idx for idx, ch in enumerate(text) if ch == "{"]
    seen = set(candidates)
    for start in starts:
        depth = 0
        for idx in range(start, len(text)):
            ch = text[idx]
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    candidate = text[start : idx + 1].strip()
                    if candidate and candidate not in seen:
                        candidates.append(candidate)
                        seen.add(candidate)
                    break
    return candidates


def strip_title_prefix(title: str) -> str:
    return re.sub(r"^\s*Title:\s*", "", title).strip()


def salvage_json_fields(text: str) -> dict[str, Any] | None:
    stripped = text.strip()
    if not stripped:
        return None

    obj: dict[str, Any] = {}

    answer_match = re.search(r'"answer"\s*:\s*(true|false|null|"((?:\\.|[^"\\])*)")', stripped, flags=re.DOTALL)
    if answer_match:
        raw_answer = answer_match.group(1)
        if raw_answer == "true":
            obj["answer"] = True
        elif raw_answer == "false":
            obj["answer"] = False
        elif raw_answer == "null":
            obj["answer"] = ""
        else:
            try:
                obj["answer"] = json.loads(raw_answer)
            except json.JSONDecodeError:
                obj["answer"] = answer_match.group(2)

    for key in ("supporting_facts", "reasoning_steps"):
        key_match = re.search(rf'"{key}"\s*:\s*\[', stripped)
        if not key_match:
            continue
        start = key_match.end() - 1
        depth = 0
        in_string = False
        escaped = False
        end = None
        for idx in range(start, len(stripped)):
            ch = stripped[idx]
            if in_string:
                if escaped:
                    escaped = False
                elif ch == "\\":
                    escaped = True
                elif ch == '"':
                    in_string = False
                continue
            if ch == '"':
                in_string = True
            elif ch == "[":
                depth += 1
            elif ch == "]":
                depth -= 1
                if depth == 0:
                    end = idx + 1
                    break
        if end is None:
            continue
        try:
            obj[key] = json.loads(stripped[start:end])
        except json.JSONDecodeError:
            continue

    return obj or None


def extract_json_object(text: str) -> dict[str, Any] | None:
    for candidate in json_object_candidates(text):
        try:
            obj = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            return obj
    return salvage_json_fields(text)


def normalize_supporting_facts(raw_supporting_facts: Any) -> list[list[Any]]:
    if not isinstance(raw_supporting_facts, list):
        return []
    normalized = []
    for item in raw_supporting_facts:
        title: Any = None
        sent_id: Any = None
        if isinstance(item, (list, tuple)) and len(item) >= 2:
            title, sent_id = item[0], item[1]
        elif isinstance(item, dict):
            title = item.get("title") or item.get("document") or item.get("page")
            sent_id = item.get("sent_id")
            if sent_id is None:
                sent_id = item.get("sentence_id")
            if sent_id is None:
                sent_id = item.get("index")
        if title is None or sent_id is None:
            continue
        try:
            title_text = strip_title_prefix(str(title))
            if not title_text:
                continue
            normalized.append([title_text, int(sent_id)])
        except (TypeError, ValueError):
            continue
    return normalized


def normalize_reasoning_steps(raw_reasoning_steps: Any) -> list[dict[str, Any]]:
    if not isinstance(raw_reasoning_steps, list):
        return []
    normalized = []
    pending = list(raw_reasoning_steps)
    while pending:
        item = pending.pop(0)
        if isinstance(item, str):
            step_text = item.strip()
            if step_text and step_text.lower() != "answer":
                normalized.append({"step": step_text, "supporting_facts": []})
            continue
        if not isinstance(item, dict):
            continue
        step_text = str(item.get("step") or item.get("text") or "").strip()
        if step_text and step_text.lower() != "answer":
            normalized.append(
                {
                    "step": step_text,
                    "supporting_facts": normalize_supporting_facts(item.get("supporting_facts", [])),
                }
            )
        next_step = item.get("next_step")
        if isinstance(next_step, list):
            pending = next_step + pending
        elif isinstance(next_step, (dict, str)):
            pending.insert(0, next_step)
    return normalized


def query_ollama(model: str, prompt: str, timeout: int) -> tuple[str, list[list[Any]], list[dict[str, Any]], str]:
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "format": "json",
        "options": {"temperature": 0},
    }
    resp = requests.post(OLLAMA_URL, json=payload, timeout=timeout)
    resp.raise_for_status()
    raw = resp.json().get("response", "")
    obj = extract_json_object(raw)
    if obj is None:
        return raw.strip().splitlines()[0] if raw.strip() else "", [], [], raw
    answer = str(obj.get("answer", "")).strip()
    supporting_facts = normalize_supporting_facts(obj.get("supporting_facts", []))
    reasoning_steps = normalize_reasoning_steps(obj.get("reasoning_steps", []))
    return answer, supporting_facts, reasoning_steps, raw


class NLIPredictor:
    def __init__(self, model_name: str, device: str, cache_dir: Path) -> None:
        try:
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
        except ImportError as exc:  # pragma: no cover - depends on environment
            raise RuntimeError(
                "NLI requires torch and transformers. Install them in the active environment first."
            ) from exc

        if device == "auto":
            resolved_device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            resolved_device = device

        self.torch = torch
        self.device = torch.device(resolved_device)
        self.model_name = model_name
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        print(f"Loading NLI model: {model_name} on {resolved_device}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=str(self.cache_dir))
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            cache_dir=str(self.cache_dir),
        )
        self.model.to(self.device)
        self.model.eval()

    @staticmethod
    def canonical_label(label: str, idx: int) -> str:
        lowered = label.lower().replace("_", " ").strip()
        if "entail" in lowered:
            return "entailment"
        if "contrad" in lowered:
            return "contradiction"
        if "neutral" in lowered:
            return "neutral"
        fallback = {
            0: "contradiction",
            1: "neutral",
            2: "entailment",
        }
        return fallback.get(idx, lowered or "neutral")

    def predict(self, premise: str, hypothesis: str) -> str:
        encoded = self.tokenizer(
            premise,
            hypothesis,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        )
        encoded = {key: value.to(self.device) for key, value in encoded.items()}
        with self.torch.no_grad():
            logits = self.model(**encoded).logits[0]
        label_idx = int(logits.argmax().item())
        raw_label = str(self.model.config.id2label.get(label_idx, label_idx))
        return self.canonical_label(raw_label, label_idx)


def cleanup_statement(text: str) -> str:
    cleaned = re.sub(r"\s+", " ", text).strip()
    if not cleaned:
        return "Unknown."
    if cleaned[-1] not in ".!?":
        cleaned += "."
    return cleaned[0].upper() + cleaned[1:]


def build_nli_hypothesis(question: str, answer: str) -> str:
    clean_question = question.replace('"', "'").strip().rstrip("?")
    clean_answer = (answer or "").replace('"', "'").strip() or "unknown"
    return cleanup_statement(f'The answer to the question "{clean_question}" is "{clean_answer}"')


def build_nli_premise(
    chunks: list[dict[str, Any]],
    predicted_supporting_facts: list[list[Any]],
    fallback_premise: str,
) -> str:
    sentence_lookup = {
        chunk["title"]: chunk["sentences"]
        for chunk in chunks
    }
    evidence_lines = []
    seen = set()
    for item in predicted_supporting_facts:
        if not isinstance(item, (list, tuple)) or len(item) != 2:
            continue
        title, sent_id = str(item[0]), item[1]
        try:
            sent_idx = int(sent_id)
        except (TypeError, ValueError):
            continue
        sentences = sentence_lookup.get(title)
        if not sentences or not (0 <= sent_idx < len(sentences)):
            continue
        key = (title, sent_idx)
        if key in seen:
            continue
        seen.add(key)
        evidence_lines.append(f"{title}: {sentences[sent_idx]}")

    if evidence_lines:
        return "\n".join(evidence_lines)
    return fallback_premise


def evaluate_reasoning_steps_with_nli(
    reasoning_steps: list[dict[str, Any]],
    chunks: list[dict[str, Any]],
    overall_supporting_facts: list[list[Any]],
    fallback_premise: str,
    nli_predictor: NLIPredictor,
) -> list[dict[str, Any]]:
    results = []
    overall_premise = build_nli_premise(
        chunks=chunks,
        predicted_supporting_facts=overall_supporting_facts,
        fallback_premise=fallback_premise,
    )
    for step in reasoning_steps:
        step_supporting_facts = step.get("supporting_facts", []) or overall_supporting_facts
        premise = build_nli_premise(
            chunks=chunks,
            predicted_supporting_facts=step_supporting_facts,
            fallback_premise=overall_premise,
        )
        hypothesis = cleanup_statement(str(step.get("step", "")))
        label = nli_predictor.predict(premise=premise, hypothesis=hypothesis)
        results.append(
            {
                "step": str(step.get("step", "")),
                "supporting_facts": step_supporting_facts,
                "nli_premise": premise,
                "nli_hypothesis": hypothesis,
                "nli_label": label,
            }
        )
    return results


def init_score_dict() -> dict[str, float]:
    return {"em": 0.0, "f1": 0.0, "prec": 0.0, "recall": 0.0}


def evaluate_method(
    model: str,
    method: MethodSpec,
    dataset,
    start_index: int,
    num_samples: int,
    timeout: int,
    nli_predictor: NLIPredictor,
    wiki_retriever: ChromaWikiRetriever,
) -> dict[str, Any]:
    answer_metrics = init_score_dict()
    sp_metrics = init_score_dict()
    joint_metrics = init_score_dict()
    nli_counts = {"entailment": 0, "contradiction": 0, "neutral": 0}
    step_nli_counts = {"entailment": 0, "contradiction": 0, "neutral": 0}
    total_reasoning_steps = 0
    samples_with_reasoning_steps = 0
    all_steps_entailed_samples = 0
    generation_times = []
    per_item_times = []
    failures = 0
    predictions = {}

    end_index = min(start_index + num_samples, len(dataset))
    iterator = range(start_index, end_index)
    pbar = tqdm(
        total=(end_index - start_index),
        desc=f"{model} | {method.name}",
        leave=True,
        dynamic_ncols=True,
        mininterval=0.1,
        maxinterval=0.5,
        ascii=True,
        file=sys.stdout,
    )

    for idx in iterator:
        item_start = time.time()
        ex = dataset[idx]
        qid = ex["id"]
        context_pack = prepare_context_for_method(ex["question"], method, wiki_retriever)
        prompt = build_prompt(ex["question"], context_pack["prompt_context"], method.prompt_style)
        gold_answer = ex["answer"]
        gold_sp = [
            [title, sent_id]
            for title, sent_id in zip(ex["supporting_facts"]["title"], ex["supporting_facts"]["sent_id"])
        ]

        generation_start = time.time()
        try:
            pred_answer, pred_sp, pred_reasoning_steps, raw = query_ollama(model, prompt, timeout=timeout)
        except Exception as err:  # noqa: BLE001
            failures += 1
            pred_answer, pred_sp, pred_reasoning_steps, raw = "", [], [], f"ERROR: {err}"
        generation_dt = time.time() - generation_start
        generation_times.append(generation_dt)

        ans_em, ans_prec, ans_recall, ans_f1 = update_answer(answer_metrics, pred_answer, gold_answer)
        sp_em, sp_prec, sp_recall, sp_f1 = update_support(sp_metrics, pred_sp, gold_sp)

        joint_prec = ans_prec * sp_prec
        joint_recall = ans_recall * sp_recall
        joint_f1 = 2 * joint_prec * joint_recall / (joint_prec + joint_recall) if (joint_prec + joint_recall) > 0 else 0.0
        joint_em = ans_em * sp_em
        joint_metrics["em"] += joint_em
        joint_metrics["f1"] += joint_f1
        joint_metrics["prec"] += joint_prec
        joint_metrics["recall"] += joint_recall

        nli_premise = build_nli_premise(
            chunks=context_pack["chunks"],
            predicted_supporting_facts=pred_sp,
            fallback_premise=context_pack["nli_fallback_premise"],
        )
        nli_hypothesis = build_nli_hypothesis(ex["question"], pred_answer)
        nli_label = nli_predictor.predict(
            premise=nli_premise,
            hypothesis=nli_hypothesis,
        )
        nli_counts[nli_label] += 1
        step_nli_results = evaluate_reasoning_steps_with_nli(
            reasoning_steps=pred_reasoning_steps,
            chunks=context_pack["chunks"],
            overall_supporting_facts=pred_sp,
            fallback_premise=context_pack["nli_fallback_premise"],
            nli_predictor=nli_predictor,
        )
        if step_nli_results:
            samples_with_reasoning_steps += 1
        if step_nli_results and all(item["nli_label"] == "entailment" for item in step_nli_results):
            all_steps_entailed_samples += 1
        total_reasoning_steps += len(step_nli_results)
        for step_result in step_nli_results:
            step_nli_counts[step_result["nli_label"]] += 1

        item_dt = time.time() - item_start
        per_item_times.append(item_dt)
        predictions[qid] = {
            "question": ex["question"],
            "gold_answer": gold_answer,
            "gold_supporting_facts": gold_sp,
            "pred_answer": pred_answer,
            "pred_supporting_facts": pred_sp,
            "pred_reasoning_steps": pred_reasoning_steps,
            "nli_label": nli_label,
            "nli_premise": nli_premise,
            "nli_hypothesis": nli_hypothesis,
            "step_nli": step_nli_results,
            "retrieved_context_titles": [entry["title"] for entry in context_pack["retrieval"]],
            "retrieval": context_pack["retrieval"],
            "generation_sec": generation_dt,
            "item_runtime_sec": item_dt,
            "raw": raw,
        }
        pbar.update(1)
        pbar.set_postfix_str(f"indices {idx + 1}/{len(dataset)}")

    pbar.close()

    n = max(1, end_index - start_index)
    step_denominator = max(1, total_reasoning_steps)
    result = {
        "model": model,
        "method": method.name,
        "prompt_style": method.prompt_style,
        "rag_k": method.rag_k,
        "num_samples": end_index - start_index,
        "start_index": start_index,
        "end_index": end_index,
        "failures": failures,
        "avg_generation_sec": sum(generation_times) / n,
        "avg_item_runtime_sec": sum(per_item_times) / n,
        "total_runtime_sec": sum(per_item_times),
        "metrics": {
            "ans_em": answer_metrics["em"] / n,
            "ans_f1": answer_metrics["f1"] / n,
            "sup_em": sp_metrics["em"] / n,
            "sup_f1": sp_metrics["f1"] / n,
            "joint_em": joint_metrics["em"] / n,
            "joint_f1": joint_metrics["f1"] / n,
            "nli_entailment_rate": nli_counts["entailment"] / n,
            "nli_contradiction_rate": nli_counts["contradiction"] / n,
            "nli_neutral_rate": nli_counts["neutral"] / n,
            "avg_reasoning_steps": total_reasoning_steps / n,
            "samples_with_reasoning_steps_rate": samples_with_reasoning_steps / n,
            "all_steps_entailed_rate": all_steps_entailed_samples / n,
            "step_nli_entailment_rate": step_nli_counts["entailment"] / step_denominator if total_reasoning_steps else 0.0,
            "step_nli_contradiction_rate": step_nli_counts["contradiction"] / step_denominator if total_reasoning_steps else 0.0,
            "step_nli_neutral_rate": step_nli_counts["neutral"] / step_denominator if total_reasoning_steps else 0.0,
        },
        "metric_sums": {
            "ans_em": answer_metrics["em"],
            "ans_f1": answer_metrics["f1"],
            "sup_em": sp_metrics["em"],
            "sup_f1": sp_metrics["f1"],
            "joint_em": joint_metrics["em"],
            "joint_f1": joint_metrics["f1"],
        },
        "nli_counts": nli_counts,
        "reasoning_steps": {
            "total_steps": total_reasoning_steps,
            "samples_with_steps": samples_with_reasoning_steps,
            "all_steps_entailed_samples": all_steps_entailed_samples,
            "nli_counts": step_nli_counts,
        },
        "predictions": predictions,
    }
    return result


def _ensure_method_state_defaults(method_state: dict[str, Any]) -> dict[str, Any]:
    method_state.setdefault("total_reasoning_steps", 0)
    method_state.setdefault("samples_with_reasoning_steps", 0)
    method_state.setdefault("all_steps_entailed_samples", 0)
    method_state.setdefault("step_nli_entailment", 0)
    method_state.setdefault("step_nli_contradiction", 0)
    method_state.setdefault("step_nli_neutral", 0)
    return method_state


def load_state(state_file: Path, reset: bool, dataset_dir: Path, split: str, dataset_size: int, nli_model: str) -> dict[str, Any]:
    if reset or (not state_file.exists()):
        return {
            "meta": {
                "dataset_dir": str(dataset_dir),
                "split": split,
                "dataset_size": dataset_size,
                "nli_model": nli_model,
            },
            "runs": [],
            "models": {},
        }
    with state_file.open("r", encoding="utf-8") as f:
        state = json.load(f)
    for model_state in state.get("models", {}).values():
        for method_state in model_state.get("methods", {}).values():
            _ensure_method_state_defaults(method_state)
    return state


def merge_into_state(
    state: dict[str, Any],
    model: str,
    run_results: list[dict[str, Any]],
    start_index: int,
    end_index: int,
) -> dict[str, Any]:
    state["runs"].append(
        {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "model": model,
            "start_index": start_index,
            "end_index": end_index,
            "num_samples": end_index - start_index,
            "methods": [result["method"] for result in run_results],
        }
    )

    model_state = state["models"].setdefault(model, {"methods": {}})
    for result in run_results:
        method_state = _ensure_method_state_defaults(
            model_state["methods"].setdefault(
                result["method"],
                {
                    "prompt_style": result["prompt_style"],
                    "rag_k": result["rag_k"],
                    "total_samples": 0,
                    "total_runtime_sec": 0.0,
                    "failures": 0,
                    "sum_ans_em": 0.0,
                    "sum_ans_f1": 0.0,
                    "sum_sup_em": 0.0,
                    "sum_sup_f1": 0.0,
                    "sum_joint_em": 0.0,
                    "sum_joint_f1": 0.0,
                    "nli_entailment": 0,
                    "nli_contradiction": 0,
                    "nli_neutral": 0,
                    "total_reasoning_steps": 0,
                    "samples_with_reasoning_steps": 0,
                    "all_steps_entailed_samples": 0,
                    "step_nli_entailment": 0,
                    "step_nli_contradiction": 0,
                    "step_nli_neutral": 0,
                },
            )
        )
        method_state["total_samples"] += result["num_samples"]
        method_state["total_runtime_sec"] += result["total_runtime_sec"]
        method_state["failures"] += result["failures"]
        sums = result["metric_sums"]
        method_state["sum_ans_em"] += sums["ans_em"]
        method_state["sum_ans_f1"] += sums["ans_f1"]
        method_state["sum_sup_em"] += sums["sup_em"]
        method_state["sum_sup_f1"] += sums["sup_f1"]
        method_state["sum_joint_em"] += sums["joint_em"]
        method_state["sum_joint_f1"] += sums["joint_f1"]
        method_state["nli_entailment"] += result["nli_counts"]["entailment"]
        method_state["nli_contradiction"] += result["nli_counts"]["contradiction"]
        method_state["nli_neutral"] += result["nli_counts"]["neutral"]
        reasoning_state = result["reasoning_steps"]
        method_state["total_reasoning_steps"] += reasoning_state["total_steps"]
        method_state["samples_with_reasoning_steps"] += reasoning_state["samples_with_steps"]
        method_state["all_steps_entailed_samples"] += reasoning_state["all_steps_entailed_samples"]
        method_state["step_nli_entailment"] += reasoning_state["nli_counts"]["entailment"]
        method_state["step_nli_contradiction"] += reasoning_state["nli_counts"]["contradiction"]
        method_state["step_nli_neutral"] += reasoning_state["nli_counts"]["neutral"]
    return state


def format_method_table(results: list[dict[str, Any]]) -> list[str]:
    lines = [
        "| Method | Samples | Avg sec/sample | Ans EM | Ans F1 | Sup EM | Sup F1 | Joint EM | Joint F1 | Ans Entail | Ans Contradict | Ans Neutral | Avg steps | Step Entail | Step Contradict | Step Neutral |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for result in results:
        metrics = result["metrics"]
        lines.append(
            f"| {result['method']} | {result['num_samples']} | {result['avg_item_runtime_sec']:.2f} | "
            f"{metrics['ans_em'] * 100:.2f} | {metrics['ans_f1'] * 100:.2f} | "
            f"{metrics['sup_em'] * 100:.2f} | {metrics['sup_f1'] * 100:.2f} | "
            f"{metrics['joint_em'] * 100:.2f} | {metrics['joint_f1'] * 100:.2f} | "
            f"{metrics['nli_entailment_rate'] * 100:.2f} | {metrics['nli_contradiction_rate'] * 100:.2f} | "
            f"{metrics['nli_neutral_rate'] * 100:.2f} | {metrics['avg_reasoning_steps']:.2f} | "
            f"{metrics['step_nli_entailment_rate'] * 100:.2f} | {metrics['step_nli_contradiction_rate'] * 100:.2f} | "
            f"{metrics['step_nli_neutral_rate'] * 100:.2f} |"
        )
    return lines


def format_cumulative_table(state: dict[str, Any], model: str, method_names: set[str]) -> list[str]:
    lines = [
        "| Method | Total samples | Avg sec/sample | Ans EM | Ans F1 | Sup EM | Sup F1 | Joint EM | Joint F1 | Ans Entail | Ans Contradict | Ans Neutral | Avg steps | Step Entail | Step Contradict | Step Neutral |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    model_state = state["models"].get(model, {})
    for method_name, method_state in model_state.get("methods", {}).items():
        if method_name not in method_names:
            continue
        n = max(1, method_state["total_samples"])
        total_steps = method_state["total_reasoning_steps"]
        step_n = max(1, total_steps)
        lines.append(
            f"| {method_name} | {method_state['total_samples']} | {method_state['total_runtime_sec'] / n:.2f} | "
            f"{(method_state['sum_ans_em'] / n) * 100:.2f} | {(method_state['sum_ans_f1'] / n) * 100:.2f} | "
            f"{(method_state['sum_sup_em'] / n) * 100:.2f} | {(method_state['sum_sup_f1'] / n) * 100:.2f} | "
            f"{(method_state['sum_joint_em'] / n) * 100:.2f} | {(method_state['sum_joint_f1'] / n) * 100:.2f} | "
            f"{(method_state['nli_entailment'] / n) * 100:.2f} | {(method_state['nli_contradiction'] / n) * 100:.2f} | "
            f"{(method_state['nli_neutral'] / n) * 100:.2f} | {(method_state['total_reasoning_steps'] / n):.2f} | "
            f"{(method_state['step_nli_entailment'] / step_n) * 100:.2f} | {(method_state['step_nli_contradiction'] / step_n) * 100:.2f} | "
            f"{(method_state['step_nli_neutral'] / step_n) * 100:.2f} |"
        )
    return lines


def append_history_log(history_log_path: Path, lines: list[str]) -> None:
    with history_log_path.open("a", encoding="utf-8") as f:
        f.write("\n".join(lines))
        f.write("\n\n")


def write_report(
    model: str,
    run_results: list[dict[str, Any]],
    state: dict[str, Any],
    output_dir: Path,
    state_file: Path,
    next_start_index: int,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    model_slug = slugify(model)
    json_path = output_dir / f"{model_slug}_report_latest.json"
    md_path = output_dir / f"{model_slug}_report_latest.md"
    history_log_path = output_dir / f"{model_slug}_history.log"

    payload = {
        "model": model,
        "current_run": run_results,
        "state": state,
        "next_start_index": next_start_index,
    }
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    with state_file.open("w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)

    lines = [
        "# HotpotQA Ollama Evaluation Report",
        "",
        f"- Model: `{model}`",
        f"- NLI model: `{state['meta']['nli_model']}`",
        f"- Generated at: {datetime.now().isoformat(timespec='seconds')}",
        "",
        "## Current Run",
        "",
        *format_method_table(run_results),
        "",
        "## Cumulative (Across Runs)",
        "",
        *format_cumulative_table(state, model, {result["method"] for result in run_results}),
    ]

    dataset_size = state["meta"]["dataset_size"]
    if next_start_index < dataset_size:
        lines += ["", f"Resume next run with: `--start-index {next_start_index}`"]
    else:
        lines += ["", "All indices completed for sequential chunking."]

    with md_path.open("w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    history_lines = [
        f"[{datetime.now().isoformat(timespec='seconds')}] model={model} start={run_results[0]['start_index']} end={run_results[0]['end_index']}",
        *format_method_table(run_results),
    ]
    append_history_log(history_log_path, history_lines)

    print(f"Saved latest report JSON: {json_path}")
    print(f"Saved latest report MD:   {md_path}")
    print(f"Saved history log:        {history_log_path}")
    print(f"Saved state:              {state_file}")
    if next_start_index < dataset_size:
        print(f"Next start-index: {next_start_index}")
    else:
        print("No remaining indices to continue.")
    print("\n".join(lines))


def build_method_specs(rag_k_values: list[int]) -> list[MethodSpec]:
    return [
        MethodSpec(name=f"rag_k{k}", prompt_style="grounded", rag_k=k)
        for k in rag_k_values
    ]


def main() -> None:
    args = parse_args()
    rag_k_values = parse_rag_k_values(args.rag_k_values)
    method_specs = build_method_specs(rag_k_values)

    dataset = load_from_disk(str(args.dataset_dir))[args.split]
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    state_file = args.state_file or (output_dir / f"{slugify(args.model)}_progress_state.json")
    state = load_state(
        state_file=state_file,
        reset=args.reset_state,
        dataset_dir=args.dataset_dir,
        split=args.split,
        dataset_size=len(dataset),
        nli_model=args.nli_model,
    )
    end_index = min(args.start_index + args.num_samples, len(dataset))

    print(f"Evaluating model: {args.model}")
    print(f"Dataset: {args.dataset_dir} [{args.split}]")
    print(f"Sample range: {args.start_index}..{end_index - 1}")
    print(f"Methods: {', '.join(method.name for method in method_specs)}")

    wiki_retriever = ChromaWikiRetriever(
        chroma_dir=args.wiki_chroma_dir,
        embedding_model=args.wiki_embedding_model,
        cache_dir=args.wiki_embedding_cache_dir,
        collection_name=args.wiki_collection_name,
        use_entity_title_retrieval=not args.disable_entity_title_retrieval,
        entity_model=args.entity_model,
        entity_cache_dir=args.entity_cache_dir,
        entity_device=args.entity_device,
        entity_threshold=args.entity_threshold,
    )
    print(f"Fullwiki retriever: {args.wiki_chroma_dir} [{args.wiki_collection_name}]")
    print(
        "Entity/title retrieval: "
        + (
            "disabled"
            if args.disable_entity_title_retrieval
            else f"enabled [{args.entity_model}, device={args.entity_device}]"
        )
    )

    nli_predictor = NLIPredictor(
        model_name=args.nli_model,
        device=args.nli_device,
        cache_dir=args.nli_cache_dir,
    )

    run_results = []
    for method in method_specs:
        print(f"\n=== Running method: {method.name} ===")
        result = evaluate_method(
            model=args.model,
            method=method,
            dataset=dataset,
            start_index=args.start_index,
            num_samples=args.num_samples,
            timeout=args.request_timeout,
            nli_predictor=nli_predictor,
            wiki_retriever=wiki_retriever,
        )
        run_results.append(result)

    state = merge_into_state(
        state=state,
        model=args.model,
        run_results=run_results,
        start_index=args.start_index,
        end_index=end_index,
    )
    write_report(
        model=args.model,
        run_results=run_results,
        state=state,
        output_dir=output_dir,
        state_file=state_file,
        next_start_index=end_index,
    )


if __name__ == "__main__":
    main()
