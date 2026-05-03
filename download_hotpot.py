from datasets import load_dataset
import json
import os

os.makedirs("data", exist_ok=True)

## Load the HotpotQA dataset and save a subset to a local JSON file
ds = load_dataset("hotpotqa/hotpot_qa", "distractor", split="validation[:200]")

data = [dict(x) for x in ds]

with open("data/hotpot_dev_distractor_200.json", "w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False, indent=2)

print(f"saved {len(data)} examples to data/hotpot_dev_distractor_200.json")
print(data[0].keys())
print(data[0])
