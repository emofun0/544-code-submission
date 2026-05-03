import json
import re
import string
from collections import Counter


def normalize_answer(s):
    if s is None:
        return ""
    s = str(s).lower()

    prefixes = [
        "the answer is",
        "answer:",
        "the series is",
        "it was",
        "it is",
        "they are",
        "he is",
        "she is",
    ]
    for p in prefixes:
        if s.startswith(p):
            s = s[len(p):].strip()

    s = re.sub(f"[{re.escape(string.punctuation)}]", " ", s)
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = " ".join(s.split())
    return s


def extract_yes_no(pred):
    p = str(pred).lower().strip()

    if p.startswith("yes"):
        return "yes"
    if p.startswith("no"):
        return "no"

    yes_match = re.search(r"\byes\b", p)
    no_match = re.search(r"\bno\b", p)

    if yes_match and not no_match:
        return "yes"
    if no_match and not yes_match:
        return "no"

    return None


def extract_short_answer(pred, gold=None):
    p = str(pred).strip()

    yn = extract_yes_no(p)
    if yn is not None:
        return yn

    patterns = [
        r"^the answer is\s+",
        r"^answer:\s*",
        r"^it was\s+",
        r"^it is\s+",
        r"^they are\s+",
        r"^he is\s+",
        r"^she is\s+",
        r"^the series is\s+",
        r"^the director is\s+",
    ]
    lowered = p.lower()
    for pat in patterns:
        lowered = re.sub(pat, "", lowered).strip()

    lowered = re.split(r"[.!?\n]", lowered)[0].strip()

    if gold:
        gold_norm = normalize_answer(gold)
        pred_norm = normalize_answer(p)
        if gold_norm and gold_norm in pred_norm:
            return gold

    return lowered


def exact_match_score(pred, gold):
    pred_yn = extract_yes_no(pred)
    gold_norm = normalize_answer(gold)

    if gold_norm in {"yes", "no"} and pred_yn is not None:
        return int(pred_yn == gold_norm)

    pred_norm = normalize_answer(pred)

    if pred_norm == gold_norm:
        return 1

    if gold_norm and gold_norm in pred_norm:
        return 1

    return 0


def f1_score(pred, gold):
    short_pred = extract_short_answer(pred, gold)

    pred_tokens = normalize_answer(short_pred).split()
    gold_tokens = normalize_answer(gold).split()

    if len(pred_tokens) == 0 or len(gold_tokens) == 0:
        return float(pred_tokens == gold_tokens)

    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_same = sum(common.values())

    if num_same == 0:
        return 0.0

    precision = num_same / len(pred_tokens)
    recall = num_same / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


# -------- Supporting facts evaluation --------

def canonicalize_supporting_facts(sf):
    """
    Convert supporting facts into a set of (title, sent_id) tuples.

    Accepts either:
    1. {"title": [...], "sent_id": [...]}
    2. [["Title", 0], ["Another Title", 1]]
    3. [{"title": "...", "sent_id": 0}, ...]
    """
    result = set()

    if sf is None:
        return result

    if isinstance(sf, dict):
        titles = sf.get("title", [])
        sent_ids = sf.get("sent_id", [])
        for t, s in zip(titles, sent_ids):
            result.add((str(t).strip(), int(s)))
        return result

    if isinstance(sf, list):
        for item in sf:
            if isinstance(item, (list, tuple)) and len(item) == 2:
                result.add((str(item[0]).strip(), int(item[1])))
            elif isinstance(item, dict):
                if "title" in item and "sent_id" in item:
                    result.add((str(item["title"]).strip(), int(item["sent_id"])))
        return result

    return result


def supporting_fact_metrics(pred_sf, gold_sf):
    pred_set = canonicalize_supporting_facts(pred_sf)
    gold_set = canonicalize_supporting_facts(gold_sf)

    if len(pred_set) == 0 and len(gold_set) == 0:
        return 1.0, 1.0
    if len(pred_set) == 0 or len(gold_set) == 0:
        return 0.0, 0.0

    common = pred_set & gold_set
    num_same = len(common)

    em = 1.0 if pred_set == gold_set else 0.0

    if num_same == 0:
        return em, 0.0

    precision = num_same / len(pred_set)
    recall = num_same / len(gold_set)
    f1 = 2 * precision * recall / (precision + recall)
    return em, f1


def evaluate(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    ans_em_total = 0.0
    ans_f1_total = 0.0

    sup_em_total = 0.0
    sup_f1_total = 0.0
    joint_em_total = 0.0
    joint_f1_total = 0.0

    has_support_metrics = True

    for ex in data:
        pred = ex["prediction"]
        gold = ex["gold_answer"]

        ans_em = exact_match_score(pred, gold)
        ans_f1 = f1_score(pred, gold)

        ans_em_total += ans_em
        ans_f1_total += ans_f1

        gold_sf = ex.get("supporting_facts")
        pred_sf = ex.get("pred_supporting_facts")

        if gold_sf is None or pred_sf is None:
            has_support_metrics = False
            continue

        sup_em, sup_f1 = supporting_fact_metrics(pred_sf, gold_sf)
        sup_em_total += sup_em
        sup_f1_total += sup_f1

        joint_em_total += ans_em * sup_em
        joint_f1_total += ans_f1 * sup_f1

    n = len(data)

    print(f"Total: {n}")
    print(f"Ans EM: {ans_em_total / n:.4f}")
    print(f"Ans F1: {ans_f1_total / n:.4f}")

    if has_support_metrics:
        print(f"Sup EM: {sup_em_total / n:.4f}")
        print(f"Sup F1: {sup_f1_total / n:.4f}")
        print(f"Joint EM: {joint_em_total / n:.4f}")
        print(f"Joint F1: {joint_f1_total / n:.4f}")
    else:
        print("Sup EM: N/A (pred_supporting_facts missing)")
        print("Sup F1: N/A (pred_supporting_facts missing)")
        print("Joint EM: N/A (pred_supporting_facts missing)")
        print("Joint F1: N/A (pred_supporting_facts missing)")


if __name__ == "__main__":
    evaluate("outputs/baseline_grounded_200.json")
