import os
import json
import re
from openai import OpenAI
from data_loader import load_hotpot, format_context

# insert api key here
client = OpenAI(api_key="")


def build_direct_prompt(question):
    return f"""You are answering a HotpotQA-style question.

Return your output as valid JSON only.
Do not include markdown fences, explanations, or any extra text.

Required JSON schema:
{{
  "answer": "short final answer",
  "supporting_facts": []
}}

Question:
{question}
"""


def build_grounded_prompt(question, context):
    return f"""You are answering a HotpotQA-style question.

Given the question and context, return your output as valid JSON only.
Do not include markdown fences, explanations, or any extra text.

Required JSON schema:
{{
  "answer": "short final answer",
  "supporting_facts": [
    ["Document Title 1", 0],
    ["Document Title 2", 3]
  ]
}}

Rules:
1. Answer using only the provided context.
2. "answer" must be a short answer only.
3. "supporting_facts" must be a list of [title, sentence_id].
4. Use only titles and sentence ids that appear in the provided context.
5. If the answer cannot be inferred from the context, answer "insufficient evidence".
6. Output JSON only.

Question:
{question}

Context:
{context}
"""


def extract_json_block(text):
    text = text.strip()

    try:
        return json.loads(text)
    except Exception:
        pass

    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        candidate = match.group(0)
        try:
            return json.loads(candidate)
        except Exception:
            pass

    raise ValueError(f"Model output is not valid JSON:\n{text}")


def normalize_supporting_facts(sf):
    normalized = []

    if not isinstance(sf, list):
        return normalized

    for item in sf:
        if isinstance(item, (list, tuple)) and len(item) == 2:
            title, sent_id = item
            try:
                normalized.append([str(title).strip(), int(sent_id)])
            except Exception:
                continue
        elif isinstance(item, dict):
            if "title" in item and "sent_id" in item:
                try:
                    normalized.append([str(item["title"]).strip(), int(item["sent_id"])])
                except Exception:
                    continue

    seen = set()
    deduped = []
    for title, sent_id in normalized:
        key = (title, sent_id)
        if key not in seen:
            seen.add(key)
            deduped.append([title, sent_id])

    return deduped


def real_generate_json(prompt):
    response = client.chat.completions.create(
        model="gpt-5.4-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )

    raw = response.choices[0].message.content.strip()
    obj = extract_json_block(raw)

    answer = str(obj.get("answer", "")).strip()
    pred_sf = normalize_supporting_facts(obj.get("supporting_facts", []))

    return {
        "answer": answer,
        "pred_supporting_facts": pred_sf,
        "raw_output": raw
    }


def run(input_path, output_path, max_examples=200):
    data = load_hotpot(input_path)
    results = []

    for i, ex in enumerate(data[:max_examples], 1):
        question = ex["question"]
        gold_answer = ex["answer"]
        gold_supporting_facts = ex.get("supporting_facts", [])

        if isDirect:
            prompt = build_direct_prompt(question)
        else:
            context = format_context(ex, max_docs=None)
            prompt = build_grounded_prompt(question, context)

        try:
            pred = real_generate_json(prompt)
            answer = pred["answer"]
            pred_sf = pred["pred_supporting_facts"]
            raw_output = pred["raw_output"]
        except Exception as e:
            answer = ""
            pred_sf = []
            raw_output = f"ERROR: {str(e)}"

        results.append({
            "id": ex["id"],
            "question": question,
            "gold_answer": gold_answer,
            "supporting_facts": gold_supporting_facts,
            "prediction": answer,
            "pred_supporting_facts": pred_sf,
            "prompt": prompt,
            "raw_output": raw_output
        })

        print(f"[{i}/{min(len(data), max_examples)}] done: {ex['id']}")

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"Saved {len(results)} predictions to {output_path}")


if __name__ == "__main__":
    isDirect = False
    run(
        input_path="data/hotpot_dev_distractor_200.json",
        output_path="outputs/baseline_grounded_200.json",
        max_examples=200
    )
