"""Fine-tune a Swedish encoder to predict a grammaticality score in [0, 1].

The model is a sequence classifier with one output logit. It is trained with
binary cross-entropy against the soft target score, so sigmoid(logit) is the
predicted score and is always inside [0, 1].

    python train.py --data data/processed --output-dir models/scorer
    python train.py --data <hf-user>/<dataset> --push <hf-user>/<model>
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import torch
from datasets import load_dataset

import grammar_scorer.compat  # noqa: F401  (Python 3.14 fix; must come before transformers)
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score
from transformers import (AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding,
                          Trainer, TrainingArguments, set_seed)

SEED = 42
BASE_MODEL = "AI-Sweden-Models/ModernBERT-base"
MAX_LENGTH = 256


class SoftBCETrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        logits = model(**inputs).logits.squeeze(-1)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logits.float(), labels.float())
        return (loss, {"logits": logits}) if return_outputs else loss


def compute_metrics(eval_pred):
    logits, scores = eval_pred
    pred = 1 / (1 + np.exp(-np.asarray(logits).reshape(-1)))
    return {
        "spearman": spearmanr(scores, pred).correlation,
        "roc_auc": roc_auc_score(np.asarray(scores) > 0.7, pred),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", default="data/processed", help="local folder with train/validation.jsonl, or a Hub dataset id")
    parser.add_argument("--base-model", default=BASE_MODEL)
    parser.add_argument("--output-dir", default="models/scorer")
    parser.add_argument("--epochs", type=float, default=2)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-train-samples", type=int, help="subsample for quick experiments")
    parser.add_argument("--push", help="Hugging Face model repo id to publish to")
    args = parser.parse_args()

    set_seed(SEED)  # python, numpy and torch

    if os.path.isdir(args.data):
        data = load_dataset("json", data_files={s: f"{args.data}/{s}.jsonl" for s in ("train", "validation")})
    else:
        data = load_dataset(args.data)
    if args.max_train_samples:
        data["train"] = data["train"].select(range(min(args.max_train_samples, len(data["train"]))))

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.base_model, num_labels=1, classifier_pooling="mean", reference_compile=False)

    def encode(batch):
        enc = tokenizer(batch["text"], truncation=True, max_length=MAX_LENGTH)
        enc["labels"] = batch["score"]
        return enc

    data = data.map(encode, batched=True, remove_columns=data["train"].column_names)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        seed=SEED,
        data_seed=SEED,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size * 2,
        warmup_ratio=0.06,
        weight_decay=0.01,
        lr_scheduler_type="linear",
        fp16=torch.cuda.is_available(),
        group_by_length=True,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="spearman",
        logging_steps=100,
        report_to="none",
    )
    trainer = SoftBCETrainer(
        model=model,
        args=training_args,
        train_dataset=data["train"],
        eval_dataset=data["validation"],
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=compute_metrics,
    )
    trainer.train()
    print("validation:", trainer.evaluate())

    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"Saved model to {args.output_dir}")
    if args.push:
        model.push_to_hub(args.push)
        tokenizer.push_to_hub(args.push)
        print(f"Pushed to https://huggingface.co/{args.push}")


if __name__ == "__main__":
    main()
