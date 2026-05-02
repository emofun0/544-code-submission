import os
import re
import json
import time
import random
import string
from collections import Counter

from datasets import load_dataset, DatasetDict, Dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm.auto import tqdm
from transformers import pipeline
from transformers.pipelines.pt_utils import KeyDataset
from openai import OpenAI


# =========================
# Config
# =========================
SEED = 544
MODEL_NAME = "gpt-5.4-mini"
MAX_EXAMPLES = 200
OUTPUT_DIR = "gpt_outputs_nli"
REQUEST_SLEEP = 1.5
MAX_RETRIES = 10
MAX_CONTEXT_SENTENCES = 50
SELF_CONSISTENCY_N = 3
NLI_MODEL_NAME = "roberta-large-mnli"
NLI_BATCH_SIZE = 16
NLI_ENTAILMENT_THRESHOLD = 0.35

random.seed(SEED)
os.environ["PYTHONHASHSEED"] = str(SEED)

# insert api key here
client = OpenAI(api_key="", max_retries=8, timeout=60.0)
os.makedirs(OUTPUT_DIR, exist_ok=True)


# =========================
# Load dataset
# =========================
raw_dataset = load_dataset("hotpot_qa", "distractor")
dataset = DatasetDict({
    "validation": raw_dataset["validation"].select(range(MAX_EXAMPLES))
})
val_set = dataset["validation"]


# =========================
# Context helpers
# =========================
def _coerce_to_text(value):
    if isinstance(value, str):
        return value
    if value is None:
        return ""
    if isinstance(value, (list, tuple)):
        return "\n".join(_coerce_to_text(v) for v in value)
    if isinstance(value, dict):
        return json.dumps(value, ensure_ascii=False)
    return str(value)


def _extract_titles_and_sentences(example):
    ctx = example["context"]

    if isinstance(ctx, dict):
        titles = ctx.get("title", [])
        sent_lists = ctx.get("sentences", ctx.get("sents", []))
        return titles, sent_lists

    if isinstance(ctx, (list, tuple)) and len(ctx) == 2:
        return ctx[0], ctx[1]

    return [], []


def build_context(example, max_sentences=MAX_CONTEXT_SENTENCES):
    context_lines = []
    count = 0

    titles, sent_lists = _extract_titles_and_sentences(example)

    for title, sents in zip(titles, sent_lists):
        for sent in sents:
            sent = _coerce_to_text(sent).strip()
            if sent:
                context_lines.append(f"{title}: {sent}")
                count += 1
            if count >= max_sentences:
                return "\n".join(context_lines)

    return "\n".join(context_lines)


def _example_sentence_candidates(example):
    titles, sent_lists = _extract_titles_and_sentences(example)
    candidates = []
    for title, sents in zip(titles, sent_lists):
        for sid, sent in enumerate(sents):
            sent_text = _coerce_to_text(sent).strip()
            if sent_text:
                candidates.append((f"{_coerce_to_text(title)}: {sent_text}", _coerce_to_text(title), sid))
    return candidates


# =========================
# Prompting strategies
# =========================
def prompt_baseline(q, c):
    return f"""Answer the question based on the context.

Context:
{c}

Question: {q}
Answer:"""


def prompt_grounded(q, c):
    return f"""Answer using ONLY the context. If unsupported, say \"NOT ENOUGH INFORMATION\".

Context:
{c}

Question: {q}
Answer:"""


def prompt_cot(q, c):
    return f"""Answer step by step.

Context:
{c}

Question: {q}
Reasoning:
Answer:"""


def prompt_cot_citation(q, c):
    return f"""Answer step by step and cite evidence from the context text.

Context:
{c}

Question: {q}
Reasoning (with citations):
Answer:"""


# =========================
# GPT call / generation
# =========================
def call_gpt(prompt, max_retries=MAX_RETRIES):
    for attempt in range(max_retries):
        try:
            response = client.responses.create(
                model=MODEL_NAME,
                input=prompt,
            )
            return response.output_text.strip()
        except Exception as e:
            err = str(e)

            if "503" in err or "overloaded" in err or "service_unavailable" in err:
                wait_time = min(120, 5 * (2 ** attempt))
                print(f"Server overloaded. Waiting {wait_time}s before retry...")
                time.sleep(wait_time)
                continue

            if "429" in err or "Rate limit" in err:
                wait_time = min(120, 2 ** attempt)
                print(f"Rate limited. Waiting {wait_time}s before retry...")
                time.sleep(wait_time)
                continue

            raise

    raise RuntimeError("Max retries exceeded.")


def extract_answer(text):
    text = _coerce_to_text(text)
    match = re.search(r"Answer:\s*(.*)", text, flags=re.IGNORECASE | re.DOTALL)
    if match:
        return match.group(1).strip()
    return text.strip()


def generate_answer(prompt):
    return call_gpt(prompt)


def self_consistency(prompt, n=SELF_CONSISTENCY_N):
    answers = []
    full_outputs = []
    for _ in range(n):
        out = generate_answer(prompt)
        full_outputs.append(out)
        answers.append(extract_answer(out))
        time.sleep(REQUEST_SLEEP)

    winner = Counter(answers).most_common(1)[0][0]
    for out in full_outputs:
        if extract_answer(out) == winner:
            return out
    return full_outputs[0]


def self_verify(q, c, draft, max_rounds=2):
    revised = draft

    for _ in range(max_rounds):
        verify_prompt = f"""Revise the reasoning to remove unsupported or contradictory steps.
Use ONLY the provided context.
If the answer is unsupported, answer: NOT ENOUGH INFORMATION.

Context:
{c}

Question: {q}

Current Draft:
{revised}

Return a corrected response in this format:
Reasoning: ...
Answer: ..."""
        revised = generate_answer(verify_prompt)
        time.sleep(REQUEST_SLEEP)

    return revised


# =========================
# Retrieval
# =========================
def build_corpus(examples, show_progress=True):
    corpus = []
    iterator = tqdm(examples, desc="Building retrieval corpus") if show_progress else examples
    for ex in iterator:
        text = build_context(ex)
        if text.strip():
            corpus.append(text)
    return corpus


corpus = build_corpus(val_set)
vectorizer = TfidfVectorizer(token_pattern=r"(?u)\b\w+\b").fit(corpus)
vectors = vectorizer.transform(corpus)


def retrieve_within_example(example, query, k=5):
    candidates = _example_sentence_candidates(example)
    if not candidates:
        return ""

    texts = [c[0] for c in candidates]
    local_vectorizer = TfidfVectorizer(token_pattern=r"(?u)\b\w+\b")
    local_vectors = local_vectorizer.fit_transform(texts)
    q_vec = local_vectorizer.transform([_coerce_to_text(query)])
    scores = cosine_similarity(q_vec, local_vectors)[0]
    topk_idx = scores.argsort()[-k:][::-1]
    return "\n".join(texts[i] for i in topk_idx if scores[i] > 0)


# =========================
# NLI metrics
# =========================
NLI_DEVICE = 0 if os.environ.get("CUDA_VISIBLE_DEVICES", "") != "" else -1
try:
    nli = pipeline("text-classification", model=NLI_MODEL_NAME, top_k=None, device=NLI_DEVICE)
except Exception:
    nli = pipeline("text-classification", model=NLI_MODEL_NAME, top_k=None, device=-1)


def _truncate_context_for_nli(context, max_chars=2200):
    context = _coerce_to_text(context)
    if len(context) <= max_chars:
        return context
    return context[-max_chars:]


def _normalize_nli_scores(output_item):
    if isinstance(output_item, list):
        scores = output_item
    else:
        scores = [output_item]
    return {item["label"].upper(): item["score"] for item in scores}


def _get_label_prob(score_map, target):
    target = target.upper()
    for label, prob in score_map.items():
        if target in label:
            return prob

    if target == "CONTRADICTION":
        return score_map.get("LABEL_0", 0.0)
    if target == "NEUTRAL":
        return score_map.get("LABEL_1", 0.0)
    if target == "ENTAILMENT":
        return score_map.get("LABEL_2", 0.0)
    return 0.0


def reasoning_sentences(text):
    text = _coerce_to_text(text)
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []

    reasoning_match = re.search(r"Reasoning:\s*(.*?)\s*Answer:", text, flags=re.IGNORECASE | re.DOTALL)
    if reasoning_match:
        reasoning_text = reasoning_match.group(1).strip()
    else:
        reasoning_text = re.sub(r"Answer:\s*.*$", "", text, flags=re.IGNORECASE | re.DOTALL).strip()

    if not reasoning_text:
        return []

    pieces = re.split(r"(?<=[.!?])\s+|\n+", reasoning_text)
    return [p.strip(" -•\t") for p in pieces if p.strip(" -•\t")]


def _batch_nli_score_maps(context, sentences, batch_size=NLI_BATCH_SIZE):
    clean = [_coerce_to_text(s).strip() for s in sentences if _coerce_to_text(s).strip()]
    if not clean:
        return []

    safe_context = _truncate_context_for_nli(context)
    pair_inputs = [f"{safe_context} </s> {s}" for s in clean]
    pair_dataset = Dataset.from_dict({"inputs": pair_inputs})

    raw_outputs = list(
        nli(
            KeyDataset(pair_dataset, "inputs"),
            truncation=True,
            max_length=512,
            batch_size=batch_size,
        )
    )
    return [_normalize_nli_scores(out) for out in raw_outputs]


def sentence_level_nli(context, sentences, entailment_threshold=NLI_ENTAILMENT_THRESHOLD):
    clean = [_coerce_to_text(s).strip() for s in sentences if _coerce_to_text(s).strip()]
    maps = _batch_nli_score_maps(context, clean)
    rows = []

    for sent, score_map in zip(clean, maps):
        entail_prob = _get_label_prob(score_map, "ENTAILMENT")
        neutral_prob = _get_label_prob(score_map, "NEUTRAL")
        contradiction_prob = _get_label_prob(score_map, "CONTRADICTION")

        if contradiction_prob >= max(entail_prob, neutral_prob):
            label = "contradiction"
        elif entail_prob >= entailment_threshold:
            label = "entailed"
        else:
            label = "unsupported"

        rows.append({
            "sentence": sent,
            "entailment": entail_prob,
            "neutral": neutral_prob,
            "contradiction": contradiction_prob,
            "label": label,
        })

    return rows


def hallucination_metrics(context, generated_text, entailment_threshold=NLI_ENTAILMENT_THRESHOLD):
    steps = reasoning_sentences(generated_text)
    rows = sentence_level_nli(context, steps, entailment_threshold=entailment_threshold)

    if not rows:
        return {
            "step_count": 0,
            "entailed_steps": 0,
            "unsupported_steps": 0,
            "contradicted_steps": 0,
            "unsupported_rate": 0.0,
            "contradiction_rate": 0.0,
            "hallucination_flag": 0,
            "sentence_nli": [],
        }

    entailed_steps = sum(1 for r in rows if r["label"] == "entailed")
    unsupported_steps = sum(1 for r in rows if r["label"] == "unsupported")
    contradicted_steps = sum(1 for r in rows if r["label"] == "contradiction")
    total = len(rows)

    return {
        "step_count": total,
        "entailed_steps": entailed_steps,
        "unsupported_steps": unsupported_steps,
        "contradicted_steps": contradicted_steps,
        "unsupported_rate": unsupported_steps / total,
        "contradiction_rate": contradicted_steps / total,
        "hallucination_flag": 1 if contradicted_steps > 0 else 0,
        "sentence_nli": rows,
    }


# =========================
# Official-style answer metrics
# =========================
YES_NO_NOANSWER = {"yes", "no", "noanswer"}


def normalize_answer(s):
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def remove_punctuation(text):
        return "".join(ch for ch in text if ch not in string.punctuation)

    def lowercase(text):
        return text.lower()

    return " ".join(remove_articles(remove_punctuation(lowercase(_coerce_to_text(s)))).split())


def answer_f1_components(prediction, ground_truth):
    normalized_prediction = normalize_answer(prediction)
    normalized_ground_truth = normalize_answer(ground_truth)

    if normalized_prediction in YES_NO_NOANSWER and normalized_prediction != normalized_ground_truth:
        return 0.0, 0.0, 0.0
    if normalized_ground_truth in YES_NO_NOANSWER and normalized_prediction != normalized_ground_truth:
        return 0.0, 0.0, 0.0

    pred_tokens = normalized_prediction.split()
    gold_tokens = normalized_ground_truth.split()

    if len(pred_tokens) == 0 and len(gold_tokens) == 0:
        return 1.0, 1.0, 1.0
    if len(pred_tokens) == 0 or len(gold_tokens) == 0:
        return 0.0, 0.0, 0.0

    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0, 0.0, 0.0

    precision = num_same / len(pred_tokens)
    recall = num_same / len(gold_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1, precision, recall


def exact_match_score(prediction, ground_truth):
    return float(normalize_answer(prediction) == normalize_answer(ground_truth))


def score_answer(prediction, gold_answer):
    em = exact_match_score(prediction, gold_answer)
    f1, prec, recall = answer_f1_components(prediction, gold_answer)
    return em, f1, prec, recall


# =========================
# Supporting fact helpers
# =========================
def _gold_sp_pairs(example):
    sf = example.get("supporting_facts", {})

    if isinstance(sf, dict):
        titles = sf.get("title", [])
        sent_ids = sf.get("sent_id", [])
        return set((t, int(sid)) for t, sid in zip(titles, sent_ids))

    if isinstance(sf, list):
        pairs = set()
        for item in sf:
            if isinstance(item, (list, tuple)) and len(item) >= 2:
                pairs.add((_coerce_to_text(item[0]), int(item[1])))
        return pairs

    return set()


def _token_set(text):
    return set(normalize_answer(text).split())


def _sp_overlap_score(sentence_text, query_tokens):
    sent_tokens = _token_set(sentence_text)
    if not sent_tokens or not query_tokens:
        return 0.0
    overlap = len(sent_tokens & query_tokens)
    if overlap == 0:
        return 0.0
    return (overlap / len(sent_tokens)) + (overlap / len(query_tokens))


def predict_supporting_facts(example, question, generated_text, max_facts=2):
    titles, sent_lists = _extract_titles_and_sentences(example)
    answer_text = extract_answer(generated_text)
    query_tokens = _token_set(f"{question} {answer_text} {generated_text}")

    scored = []
    for title, sents in zip(titles, sent_lists):
        title_text = _coerce_to_text(title)
        for sid, sent in enumerate(sents):
            sent_text = _coerce_to_text(sent)
            score = _sp_overlap_score(sent_text, query_tokens)
            if score > 0:
                scored.append((score, title_text, sid))

    scored.sort(key=lambda x: x[0], reverse=True)
    selected = []
    seen = set()

    for _, title, sid in scored:
        pair = (title, sid)
        if pair not in seen:
            selected.append([title, sid])
            seen.add(pair)
        if len(selected) >= max_facts:
            break

    return selected


def score_supporting_facts(prediction_sp, gold_sp):
    pred_sp = set((_coerce_to_text(t), int(sid)) for t, sid in prediction_sp)
    gold_sp = set((_coerce_to_text(t), int(sid)) for t, sid in gold_sp)

    tp = len(pred_sp & gold_sp)
    fp = len(pred_sp - gold_sp)
    fn = len(gold_sp - pred_sp)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    em = 1.0 if (fp + fn) == 0 else 0.0
    return em, f1, precision, recall


def evaluate_official_predictions(prediction, dataset):
    answer_pred = prediction.get("answer", {})
    sp_pred = prediction.get("sp", {})
    items = list(dataset)
    n = len(items)

    metrics = {
        "em": 0.0, "f1": 0.0, "prec": 0.0, "recall": 0.0,
        "sp_em": 0.0, "sp_f1": 0.0, "sp_prec": 0.0, "sp_recall": 0.0,
        "joint_em": 0.0, "joint_f1": 0.0, "joint_prec": 0.0, "joint_recall": 0.0,
    }

    for ex in items:
        cur_id = _coerce_to_text(ex.get("_id", ex.get("id", "")))
        ans = answer_pred.get(cur_id, "")
        sp = sp_pred.get(cur_id, [])

        ans_em, ans_f1, ans_prec, ans_recall = score_answer(ans, ex.get("answer", ""))
        sp_em, sp_f1, sp_prec, sp_recall = score_supporting_facts(sp, _gold_sp_pairs(ex))

        joint_prec = ans_prec * sp_prec
        joint_recall = ans_recall * sp_recall
        joint_f1 = (2 * joint_prec * joint_recall / (joint_prec + joint_recall)) if (joint_prec + joint_recall) > 0 else 0.0
        joint_em = ans_em * sp_em

        metrics["em"] += ans_em
        metrics["f1"] += ans_f1
        metrics["prec"] += ans_prec
        metrics["recall"] += ans_recall
        metrics["sp_em"] += sp_em
        metrics["sp_f1"] += sp_f1
        metrics["sp_prec"] += sp_prec
        metrics["sp_recall"] += sp_recall
        metrics["joint_em"] += joint_em
        metrics["joint_f1"] += joint_f1
        metrics["joint_prec"] += joint_prec
        metrics["joint_recall"] += joint_recall

    for k in metrics:
        metrics[k] /= n

    return {
        "EM": metrics["em"],
        "F1": metrics["f1"],
        "Precision": metrics["prec"],
        "Recall": metrics["recall"],
        "SupportingFactEM": metrics["sp_em"],
        "SupportingFactF1": metrics["sp_f1"],
        "SupportingFactPrecision": metrics["sp_prec"],
        "SupportingFactRecall": metrics["sp_recall"],
        "JointEM": metrics["joint_em"],
        "JointF1": metrics["joint_f1"],
        "JointPrecision": metrics["joint_prec"],
        "JointRecall": metrics["joint_recall"],
    }


# =========================
# Save/load helpers
# =========================
def load_existing_predictions(save_name):
    out_path = os.path.join(OUTPUT_DIR, f"{save_name}.json")
    if os.path.exists(out_path):
        with open(out_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        print(f"Loaded existing progress from {out_path}")
        return data
    return {"answer": {}, "sp": {}, "meta": {}}


def save_prediction_file(save_name, prediction):
    out_path = os.path.join(OUTPUT_DIR, f"{save_name}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(prediction, f, ensure_ascii=False, indent=2)
    return out_path


# =========================
# Evaluation runner with NLI
# =========================
def _build_context_for_mode(example, question, mode, k):
    if mode in ("rag", "rag_cot"):
        local_ctx = retrieve_within_example(example, question, k)
        if local_ctx.strip():
            return local_ctx
    return build_context(example)


def _build_prompt(mode, q, c):
    if mode == "baseline":
        return prompt_baseline(q, c)
    if mode == "grounded":
        return prompt_grounded(q, c)
    if mode == "cot":
        return prompt_cot(q, c)
    if mode == "cot_cite":
        return prompt_cot_citation(q, c)
    if mode == "rag":
        return prompt_baseline(q, c)
    if mode == "rag_cot":
        return prompt_cot(q, c)
    raise ValueError(f"Unknown mode: {mode}")


def evaluate(dataset, mode="baseline", k=5, save_name=None, entailment_threshold=NLI_ENTAILMENT_THRESHOLD):
    items = list(dataset)
    prediction = load_existing_predictions(save_name) if save_name else {"answer": {}, "sp": {}, "meta": {}}
    prediction.setdefault("answer", {})
    prediction.setdefault("sp", {})
    prediction.setdefault("meta", {})

    for idx, ex in enumerate(tqdm(items, desc=f"Evaluating {mode}")):
        qid = _coerce_to_text(ex.get("_id", ex.get("id", "")))

        if qid in prediction["answer"] and qid in prediction["sp"] and qid in prediction["meta"]:
            print(f"Skipping completed example {idx + 1}/{len(items)}: {qid}")
            continue

        question = _coerce_to_text(ex.get("question", ""))
        context = _build_context_for_mode(ex, question, mode, k)

        if mode == "cot_sc":
            pred_text = self_consistency(prompt_cot(question, context), n=SELF_CONSISTENCY_N)
        elif mode == "verify":
            draft = generate_answer(prompt_cot(question, context))
            time.sleep(REQUEST_SLEEP)
            pred_text = self_verify(question, context, draft)
        else:
            prompt = _build_prompt(mode, question, context)
            pred_text = generate_answer(prompt)

        answer = extract_answer(pred_text)
        sp = predict_supporting_facts(ex, question, pred_text, max_facts=2)
        nli_metrics = hallucination_metrics(context, pred_text, entailment_threshold=entailment_threshold)

        prediction["answer"][qid] = answer
        prediction["sp"][qid] = sp
        prediction["meta"][qid] = {
            "question": question,
            "raw_output": pred_text,
            "mode": mode,
            "nli": nli_metrics,
        }

        if save_name:
            out_path = save_prediction_file(save_name, prediction)
            if (idx + 1) % 10 == 0:
                print(f"Saved progress to {out_path}")

        time.sleep(REQUEST_SLEEP)

    official = evaluate_official_predictions(prediction, items)

    all_meta = list(prediction["meta"].values())
    total = len(all_meta) if all_meta else 1
    avg_step_count = sum(m["nli"]["step_count"] for m in all_meta) / total
    avg_unsupported_rate = sum(m["nli"]["unsupported_rate"] for m in all_meta) / total
    avg_contradiction_rate = sum(m["nli"]["contradiction_rate"] for m in all_meta) / total
    hallucination_rate = sum(m["nli"]["hallucination_flag"] for m in all_meta) / total

    results = {
        **official,
        "AvgReasoningSteps": avg_step_count,
        "AvgUnsupportedRate": avg_unsupported_rate,
        "AvgContradictionRate": avg_contradiction_rate,
        "HallucinationRate": hallucination_rate,
    }

    if save_name:
        metrics_path = os.path.join(OUTPUT_DIR, f"{save_name}_metrics.json")
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"Saved metrics to {metrics_path}")

    return results


# =========================
# Run experiments
# =========================
if __name__ == "__main__":
    print("Baseline:", evaluate(val_set, mode="baseline", save_name="baseline_gpt54mini_nli_200"))
    print("Grounded:", evaluate(val_set, mode="grounded", save_name="grounded_gpt54mini_nli_200"))
    print("CoT:", evaluate(val_set, mode="cot", save_name="cot_gpt54mini_nli_200"))
    print("CoT + SC:", evaluate(val_set, mode="cot_sc", save_name="cot_sc_gpt54mini_nli_200"))
    print("CoT + Citation:", evaluate(val_set, mode="cot_cite", save_name="cot_cite_gpt54mini_nli_200"))
    print("RAG (k=2):", evaluate(val_set, mode="rag", k=2, save_name="rag2_gpt54mini_nli_200"))
    print("RAG (k=5):", evaluate(val_set, mode="rag", k=5, save_name="rag5_gpt54mini_nli_200"))
    print("RAG + CoT (k=2):", evaluate(val_set, mode="rag_cot", k=2, save_name="rag_cot2_gpt54mini_nli_200"))
    print("RAG + CoT (k=5):", evaluate(val_set, mode="rag_cot", k=5, save_name="rag_cot5_gpt54mini_nli_200"))
    print("Self-Verify:", evaluate(val_set, mode="verify", save_name="verify_gpt54mini_nli_200"))
