# CSCI 544 Code Submission

### Spring 2026

### Team member: Guanqi Wang, Shaolin Tan, Zihan Xie, Minghui Yang, Yipeng Wang

# Dataset Preparation

### TBDTBD

# Running Instructions

## Local ollama models

First download ollama and local LLMs.

## Scripts

- `build_hotpot_wiki_index.py` – build fullwiki Chroma index
- `evaluate_hotpot_ollama.py` – fullwiki (RAG retrieval)
- `evaluate_hotpot_ollama_old.py` – distractor (TF-IDF retrieval)

### 1. Build index
```bash
python build_hotpot_wiki_index.py \
  --source-dir /path/to/wiki_abstracts \
  --chroma-dir data/wiki/chroma \
  --reset
```

### 2. Run eval
```bash
python evaluate_hotpot_ollama.py \
  --model {ollama_model_name} \
  --dataset-dir data/fullwiki \
  --num-samples 200
```

Distractor (old)
```bash
python evaluate_hotpot_ollama_old.py \
  --model {ollama_model_name} \
  --dataset-dir data/distractor \
  --num-samples 200
```
