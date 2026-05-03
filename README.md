# CSCI 544 Code Submission

### Spring 2026

### Team member: Guanqi Wang, Shaolin Tan, Zihan Xie, Minghui Yang, Yipeng Wang


# Dataset Preparation

We use the HotpotQA dataset, which supports multi-hop reasoning.

Two evaluation settings:

1. Fullwiki Setting
Requires retrieving relevant documents from Wikipedia
Used with RAG pipeline
2. Distractor Setting
Each question comes with candidate paragraphs
Retrieval is simplified (TF-IDF)


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
## Local Qwen Baseline

### Model Setup
We use:

```
Qwen/Qwen2.5-7B-Instruct
```

### Run Notebook
```
jupyter notebook HotpotQA_Qwen2.5-7B_baseline.ipynb
```
## Running Llama 3.3 8B on first 200 distractor sample
Preferably run it on Google Colab:
1. Open file with Google Colab.
2. Select GPU(preferably A100) and click Run All.

Running in local machine:
```
jupyter notebook llama3.3_8B_distractor200.ipynb
```

## GPT-5.4-MINI BASELINE

### The pipeline includes:
Dataset preparation
Prompt-based QA (Direct & Grounded)
Evaluation (EM / F1 / Supporting Facts)

### File Structure:
```bash
├── download_hotpot.py 
├── data_loader.py 
├── run_baseline.py 
├── eval_qa.py 
├── data/ 
│     └── hotpot_dev_distractor_200.json 
├── outputs/
│      └── baseline_grounded_200.json
```

### Components:
### 1. download_hotpot.py
Downloads the HotpotQA dataset (distractor setting) and saves 200 samples:
```
python download_hotpot.py
```
Output:
```
data/hotpot_dev_distractor_200.json
```

### 2. data_loader.py
Responsible for:
Loading dataset
Converting structured context into plain text
Example transformation:
[Title] sentence1 sentence2 ...
This is required for LLM input.

### 3. run_baseline.py
Main script for generating predictions using GPT-5.4-mini.
Supports:
Direct QA (no context)
Grounded QA (with context)

### 4. eval_qa.py
Evaluates predictions using:
Answer EM
Answer F1
Supporting Facts EM/F1
Joint EM/F1

### Setup:
Install dependencies:
pip install openai datasets
Set API key:
insert api_key in run_baseline.py

### How to Run:

### Step 1: Download dataset
```
python download_hotpot.py
```

### Step 2: Run baseline
```
python run_baseline.py
```
Output:
```
outputs/baseline_grounded_200.json
```

### Step 3: Evaluate
```
python eval_qa.py
```
Example output:
Total: 200
Ans EM: 0.45
Ans F1: 0.62
Sup EM: 0.30
Sup F1: 0.40
Joint EM: 0.20
Joint F1: 0.28

### Modes:
In run_baseline.py:
isDirect = False
True → Direct QA
False → Grounded QA

### Notes
Model: GPT-5.4-mini
Temperature: 0
Context is fully provided (no retrieval)
Supporting facts are predicted and evaluated

### Limitations
No retrieval (uses full context → noisy)
LLM output may require JSON parsing fixes
API latency and cost

### Future Improvements
Add TF-IDF / BM25 retrieval
Add CoT prompting
Improve supporting fact extraction
