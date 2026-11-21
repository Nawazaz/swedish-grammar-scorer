---
language: sv
license: apache-2.0
library_name: transformers
pipeline_tag: text-classification
base_model: AI-Sweden-Models/ModernBERT-base
datasets:
- Nawazaz/swedish-grammar-data
tags:
- grammaticality
- grammatical-error-detection
- swedish
---

# Swedish grammaticality scorer

Scores Swedish text from 0 (many grammatical errors) to 1 (grammatically correct).
It is a fine-tune of `AI-Sweden-Models/ModernBERT-base` (~150M parameters) with a single output logit. The score is `sigmoid(logit)`.

Code, training details and paper: [github.com/Nawazaz/swedish-grammar-scorer](https://github.com/Nawazaz/swedish-grammar-scorer)

## Usage

```python
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

model_id = "Nawazaz/swedish-grammar-scorer"
tok = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForSequenceClassification.from_pretrained(model_id).eval()

texts = ["Förra året reste vi till Japan.", "Förra året vi reste till Japan."]
enc = tok(texts, padding=True, truncation=True, max_length=256, return_tensors="pt")
with torch.inference_mode():
    scores = torch.sigmoid(model(**enc).logits.squeeze(-1))
print(scores.tolist())  # the first (correct) sentence scores higher
```

## Training

- **Data:** [`Nawazaz/swedish-grammar-data`](https://huggingface.co/datasets/Nawazaz/swedish-grammar-data), 121k texts: real learner errors from DaLAJ-GED, Swedish Wikipedia with synthetic errors added, and the task's starter data.
- **Objective:** binary cross-entropy against soft target scores, with mean pooling.
- **Settings:** 2 epochs, learning rate 3e-5, batch size 32, max length 256, seed 42. Training took 23 minutes on an NVIDIA T4.

## Results

| | Spearman | ROC-AUC |
|---|---|---|
| Starter test set (120 texts) | 0.819 | 0.959 |
| Validation split (3.7k texts) | 0.837 | 0.959 |
| Baseline (frozen ModernBERT-large + SVR), starter test | 0.528 | 0.783 |

## Limitations

- It misses wrong inflected forms that are still real words, such as strong-verb present tense (*ägar*), comparatives (*mer korta*), participles (*tillåtit*) and imperatives (*Ringa mig*).
- Inputs are truncated at 256 tokens.
- The training text is mostly formal, so informal Swedish may score lower than it should.
- Scores reflect an error-count scheme, not human ratings. Use them for ranking texts rather than as absolute grades.
