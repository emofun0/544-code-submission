import json

def load_hotpot(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def format_context(example, max_docs=None):
    titles = example["context"]["title"]
    sentences = example["context"]["sentences"]

    blocks = []
    num_docs = len(titles) if max_docs is None else min(len(titles), max_docs)

    for i in range(num_docs):
        title = titles[i]
        sents = sentences[i]
        joined = " ".join(sents)
        blocks.append(f"[{title}] {joined}")

    return "\n\n".join(blocks)

if __name__ == "__main__":
    data = load_hotpot("data/hotpot_dev_distractor_200.json")
    ex = data[0]

    print("QUESTION:")
    print(ex["question"])
    print("\nANSWER:")
    print(ex["answer"])
    print("\nFORMATTED CONTEXT:")
    print(format_context(ex, max_docs=200))
