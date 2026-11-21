"""Score Swedish texts for grammatical correctness (0 = poor, 1 = good).

    python predict.py --input test.jsonl --output scores.jsonl
"""

from __future__ import annotations

import argparse
import json

import torch

import grammar_scorer.compat  # noqa: F401  (Python 3.14 fix; must come before transformers)
from transformers import AutoModelForSequenceClassification, AutoTokenizer

MODEL_ID = "Nawazaz/swedish-grammar-scorer"  # Hugging Face Hub id (or a local folder)
MAX_LENGTH = 256


def load(model_id: str, device: str):
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    dtype = torch.float16 if device == "cuda" else torch.float32
    model = AutoModelForSequenceClassification.from_pretrained(model_id, dtype=dtype)
    return tokenizer, model.to(device).eval()


@torch.inference_mode()
def score(texts: list[str], tokenizer, model, batch_size: int) -> list[float]:
    """Batches texts of similar length together so little time goes to padding."""
    device = model.device
    lengths = [len(ids) for ids in tokenizer(texts, truncation=True, max_length=MAX_LENGTH)["input_ids"]]
    order = sorted(range(len(texts)), key=lengths.__getitem__)
    scores = [0.0] * len(texts)
    for start in range(0, len(order), batch_size):
        idx = order[start:start + batch_size]
        enc = tokenizer([texts[i] for i in idx], padding=True, truncation=True,
                        max_length=MAX_LENGTH, return_tensors="pt")
        logits = model(input_ids=enc["input_ids"].to(device),
                       attention_mask=enc["attention_mask"].to(device)).logits
        for i, s in zip(idx, torch.sigmoid(logits.float().squeeze(-1)).tolist()):
            scores[i] = s
    return scores


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", required=True, help="JSONL with one {'id', 'text'} per line")
    parser.add_argument("--output", required=True, help="JSONL to write one {'id', 'score'} per line")
    parser.add_argument("--model", default=MODEL_ID, help="Hub model id or local folder")
    parser.add_argument("--batch-size", type=int, default=None)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size = args.batch_size or (128 if device == "cuda" else 16)

    with open(args.input, encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    tokenizer, model = load(args.model, device)
    scores = score([r["text"] for r in rows], tokenizer, model, batch_size)

    with open(args.output, "w", encoding="utf-8") as f:
        for r, s in zip(rows, scores):
            f.write(json.dumps({"id": r["id"], "score": min(1.0, max(0.0, s))}) + "\n")


if __name__ == "__main__":
    main()
