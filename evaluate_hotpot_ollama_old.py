#!/usr/bin/env python3
"""Evaluate one local Ollama model on HotpotQA with multiple prompting methods."""

from __future__ import annotations

import argparse
import json
import math
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
        default=Path("data/distractor"),
        help="Path created by datasets.save_to_disk (default: data/distractor)",
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
        help="Device for NLI model (default: cpu)",
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


def build_context_chunks(context: dict[str, Any]) -> list[dict[str, Any]]:
    titles = context.get("title", [])
    all_sents = context.get("sentences", [])
    chunks = []
    for title, sents in zip(titles, all_sents):
        clean_sents = [str(sent).strip() for sent in sents if str(sent).strip()]
        chunks.append(
            {
                "title": str(title).strip(),
                "sentences": clean_sents,
            }
        )
    return chunks


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


def retrieval_query_text(question: str) -> str:
    return question.strip()


def score_chunks_with_tfidf(question: str, chunks: list[dict[str, Any]]) -> list[float]:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import linear_kernel

    docs = [
        f"{chunk['title']} {chunk['title']} {' '.join(chunk['sentences'])}".strip()
        for chunk in chunks
    ]
    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
    matrix = vectorizer.fit_transform(docs + [retrieval_query_text(question)])
    scores = linear_kernel(matrix[:-1], matrix[-1]).ravel()
    return [float(score) for score in scores]


def tokenize_for_retrieval(text: str) -> list[str]:
    return re.findall(r"[A-Za-z0-9]+", text.lower())


def score_chunks_fallback(question: str, chunks: list[dict[str, Any]]) -> list[float]:
    query_tokens = tokenize_for_retrieval(question)
    query_counts = Counter(query_tokens)
    scores = []
    for chunk in chunks:
        doc_tokens = tokenize_for_retrieval(f"{chunk['title']} {' '.join(chunk['sentences'])}")
        doc_counts = Counter(doc_tokens)
        title_tokens = set(tokenize_for_retrieval(chunk["title"]))
        score = 0.0
        for token, q_count in query_counts.items():
            tf = doc_counts.get(token, 0)
            if tf == 0:
                continue
            score += (1.0 + math.log1p(tf)) * q_count
            if token in title_tokens:
                score += 0.5 * q_count
        scores.append(score)
    return scores


def rank_context_chunks(question: str, chunks: list[dict[str, Any]]) -> list[tuple[dict[str, Any], float]]:
    if len(chunks) <= 1:
        return [(chunk, 1.0) for chunk in chunks]
    try:
        scores = score_chunks_with_tfidf(question, chunks)
    except Exception:  # noqa: BLE001
        scores = score_chunks_fallback(question, chunks)
    order = sorted(range(len(chunks)), key=lambda idx: (scores[idx], -idx), reverse=True)
    return [(chunks[idx], float(scores[idx])) for idx in order]


def prepare_context_for_method(question: str, context: dict[str, Any], method: MethodSpec) -> dict[str, Any]:
    chunks = build_context_chunks(context)
    ranked = rank_context_chunks(question, chunks)
    if method.rag_k is None:
        selected = chunks
        retrieval = [{"title": chunk["title"], "score": None} for chunk in chunks]
    else:
        top_k = min(method.rag_k, len(ranked))
        selected_pairs = ranked[:top_k]
        selected = [chunk for chunk, _ in selected_pairs]
        retrieval = [{"title": chunk["title"], "score": score} for chunk, score in selected_pairs]

    fallback_pairs = ranked[: min(2, len(ranked))]
    fallback_chunks = [chunk for chunk, _ in fallback_pairs]
    return {
        "chunks": selected,
        "prompt_context": render_context(selected, include_indices=True),
        "nli_fallback_premise": render_context(fallback_chunks, include_indices=False),
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
            '{"answer": "<short answer>", "supporting_facts": [["Title A", 0], ["Title B", 3]]}\n'
            "Rules:\n"
            "- Use only the provided context.\n"
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
            '{"reasoning_steps": ["step 1", "step 2"], "answer": "<short answer>", "supporting_facts": [["Title A", 0], ["Title B", 3]]}\n'
            "Rules:\n"
            "- Use only the provided context.\n"
            "- reasoning_steps should be brief and directly tied to the context.\n"
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


def extract_json_object(text: str) -> dict[str, Any] | None:
    for candidate in json_object_candidates(text):
        try:
            obj = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            return obj
    return None


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
            normalized.append([str(title), int(sent_id)])
        except (TypeError, ValueError):
            continue
    return normalized


def query_ollama(model: str, prompt: str, timeout: int) -> tuple[str, list[list[Any]], str]:
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
        return raw.strip().splitlines()[0] if raw.strip() else "", [], raw
    answer = str(obj.get("answer", "")).strip()
    supporting_facts = normalize_supporting_facts(obj.get("supporting_facts", []))
    return answer, supporting_facts, raw


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


def replace_first_wh_word(question: str, answer: str) -> str | None:
    patterns = [
        r"\bwho\b",
        r"\bwhere\b",
        r"\bwhen\b",
        r"\bwhich\b",
        r"\bhow many\b",
        r"\bhow much\b",
        r"\bhow old\b",
    ]
    for pattern in patterns:
        if re.search(pattern, question, flags=re.IGNORECASE):
            return cleanup_statement(re.sub(pattern, answer, question, count=1, flags=re.IGNORECASE))
    return None


def build_nli_hypothesis(question: str, answer: str) -> str:
    clean_question = question.replace('"', "'").strip().rstrip("?")
    clean_answer = (answer or "").replace('"', "'").strip() or "unknown"
    normalized_answer = normalize_answer(clean_answer)

    if not clean_question:
        return cleanup_statement(clean_answer)

    older_match = re.match(r"^Who is older, (.+?) or (.+)$", clean_question, flags=re.IGNORECASE)
    if older_match:
        first, second = older_match.group(1), older_match.group(2)
        other = second if normalize_answer(clean_answer) == normalize_answer(first) else first
        return cleanup_statement(f"{clean_answer} is older than {other}")

    aux_and_match = re.match(
        r"^(Is|Are|Was|Were) (.+?) and (.+?) "
        r"((?:of|from|in|on|at|located|both|the same|based|part|older|younger).+)$",
        clean_question,
        flags=re.IGNORECASE,
    )
    if aux_and_match and normalized_answer in {"yes", "no"}:
        aux, left_subject, right_subject, predicate = aux_and_match.groups()
        statement = f"{left_subject} and {right_subject} {aux.lower()} {predicate}"
        if normalized_answer == "no":
            statement = f"It is false that {statement}"
        return cleanup_statement(statement)

    aux_match = re.match(r"^(Is|Are|Was|Were) (.+)$", clean_question, flags=re.IGNORECASE)
    if aux_match and normalized_answer in {"yes", "no"}:
        aux, predicate = aux_match.groups()
        statement = f"{predicate} {aux.lower()}"
        if normalized_answer == "no":
            statement = f"It is false that {statement}"
        return cleanup_statement(statement)

    mid_question_match = re.search(r"\b(what [^?]+|who|where|when|which|how many [^?]+|how much [^?]+)\b$", clean_question, flags=re.IGNORECASE)
    if mid_question_match and not clean_question.lower().startswith("what "):
        start, end = mid_question_match.span()
        statement = clean_question[:start] + clean_answer + clean_question[end:]
        return cleanup_statement(statement)

    what_has_match = re.match(r"^What (.+?) has (.+)$", clean_question, flags=re.IGNORECASE)
    if what_has_match:
        noun_phrase, remainder = what_has_match.groups()
        return cleanup_statement(f"The {noun_phrase} that has {remainder} is {clean_answer}")

    what_be_match = re.match(r"^What (.+?) (was|is|are|were) (.+)$", clean_question, flags=re.IGNORECASE)
    if what_be_match:
        noun_phrase, be_verb, remainder = what_be_match.groups()
        return cleanup_statement(f"The {noun_phrase} {remainder} {be_verb} {clean_answer}")

    replaced = replace_first_wh_word(clean_question, clean_answer)
    if replaced is not None:
        return replaced

    if clean_question.lower().startswith(("what ", "which ")):
        return cleanup_statement(f"The {clean_question.split(' ', 1)[1]} is {clean_answer}")

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
) -> dict[str, Any]:
    answer_metrics = init_score_dict()
    sp_metrics = init_score_dict()
    joint_metrics = init_score_dict()
    nli_counts = {"entailment": 0, "contradiction": 0, "neutral": 0}
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
        context_pack = prepare_context_for_method(ex["question"], ex["context"], method)
        prompt = build_prompt(ex["question"], context_pack["prompt_context"], method.prompt_style)
        gold_answer = ex["answer"]
        gold_sp = [
            [title, sent_id]
            for title, sent_id in zip(ex["supporting_facts"]["title"], ex["supporting_facts"]["sent_id"])
        ]

        generation_start = time.time()
        try:
            pred_answer, pred_sp, raw = query_ollama(model, prompt, timeout=timeout)
        except Exception as err:  # noqa: BLE001
            failures += 1
            pred_answer, pred_sp, raw = "", [], f"ERROR: {err}"
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

        item_dt = time.time() - item_start
        per_item_times.append(item_dt)
        predictions[qid] = {
            "question": ex["question"],
            "gold_answer": gold_answer,
            "gold_supporting_facts": gold_sp,
            "pred_answer": pred_answer,
            "pred_supporting_facts": pred_sp,
            "nli_label": nli_label,
            "nli_premise": nli_premise,
            "nli_hypothesis": nli_hypothesis,
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
        "predictions": predictions,
    }
    return result


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
        return json.load(f)


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
        method_state = model_state["methods"].setdefault(
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
            },
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
    return state


def format_method_table(results: list[dict[str, Any]]) -> list[str]:
    lines = [
        "| Method | Samples | Avg sec/sample | Ans EM | Ans F1 | Sup EM | Sup F1 | Joint EM | Joint F1 | Entail | Contradict | Neutral |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for result in results:
        metrics = result["metrics"]
        lines.append(
            f"| {result['method']} | {result['num_samples']} | {result['avg_item_runtime_sec']:.2f} | "
            f"{metrics['ans_em'] * 100:.2f} | {metrics['ans_f1'] * 100:.2f} | "
            f"{metrics['sup_em'] * 100:.2f} | {metrics['sup_f1'] * 100:.2f} | "
            f"{metrics['joint_em'] * 100:.2f} | {metrics['joint_f1'] * 100:.2f} | "
            f"{metrics['nli_entailment_rate'] * 100:.2f} | {metrics['nli_contradiction_rate'] * 100:.2f} | "
            f"{metrics['nli_neutral_rate'] * 100:.2f} |"
        )
    return lines


def format_cumulative_table(state: dict[str, Any], model: str) -> list[str]:
    lines = [
        "| Method | Total samples | Avg sec/sample | Ans EM | Ans F1 | Sup EM | Sup F1 | Joint EM | Joint F1 | Entail | Contradict | Neutral |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    model_state = state["models"].get(model, {})
    for method_name, method_state in model_state.get("methods", {}).items():
        n = max(1, method_state["total_samples"])
        lines.append(
            f"| {method_name} | {method_state['total_samples']} | {method_state['total_runtime_sec'] / n:.2f} | "
            f"{(method_state['sum_ans_em'] / n) * 100:.2f} | {(method_state['sum_ans_f1'] / n) * 100:.2f} | "
            f"{(method_state['sum_sup_em'] / n) * 100:.2f} | {(method_state['sum_sup_f1'] / n) * 100:.2f} | "
            f"{(method_state['sum_joint_em'] / n) * 100:.2f} | {(method_state['sum_joint_f1'] / n) * 100:.2f} | "
            f"{(method_state['nli_entailment'] / n) * 100:.2f} | {(method_state['nli_contradiction'] / n) * 100:.2f} | "
            f"{(method_state['nli_neutral'] / n) * 100:.2f} |"
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
        *format_cumulative_table(state, model),
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
    methods = [
        MethodSpec(name="naive", prompt_style="naive"),
        MethodSpec(name="grounded", prompt_style="grounded"),
        MethodSpec(name="cot", prompt_style="cot"),
    ]
    methods.extend(
        MethodSpec(name=f"rag_k{k}", prompt_style="grounded", rag_k=k)
        for k in rag_k_values
    )
    return methods


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
