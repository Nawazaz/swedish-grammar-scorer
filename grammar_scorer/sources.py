"""Loaders for the raw text sources used to build the training data."""

from __future__ import annotations

import json
import re
from collections.abc import Iterator
from pathlib import Path

DALAJ_REPO = "sbx/superlim-2"
DALAJ_REVISION = "1b998c272be50352e967bd8b45d28fe145e9064e"  # pinned commit
WIKI_REPO = "wikimedia/wikipedia"
WIKI_CONFIG = "20231101.sv"

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-ZÅÄÖ])")
_BAD_CHARS = re.compile(r"[|=\[\]{}<>#*_~^\\/@]|\d{4,}|https?:")


def read_jsonl(path: str | Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(rows: list[dict], path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_dalaj(split: str) -> list[dict]:
    """DaLAJ-GED (Volodina et al., 2023): learner sentences with one real error
    each, plus their corrected versions. Returns [{'text', 'correct'}]."""
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(
        repo_id=DALAJ_REPO,
        repo_type="dataset",
        filename=f"data/dalaj-ged-superlim/dalaj-ged-superlim_{split}.jsonl",
        revision=DALAJ_REVISION,
    )
    rows = []
    for r in read_jsonl(path):
        label = r["label"]
        correct = label in ("correct", 0)
        rows.append({"text": " ".join(r["sentence"].split()), "correct": correct})
    return rows


def is_clean_sentence(s: str) -> bool:
    words = s.split()
    return (
        5 <= len(words) <= 40
        and s[0].isupper()
        and s[-1] in ".!?"
        and not _BAD_CHARS.search(s)
        and s.count("(") == s.count(")")
        and sum(w[0].isupper() for w in words) / len(words) < 0.4  # skip name lists
    )


def wikipedia_passages(n: int, seed: int, max_sentences: int = 3) -> Iterator[str]:
    """Yield n short passages (1-3 consecutive clean sentences) from Swedish
    Wikipedia. Bot-generated species stubs (Lsjbot) and short articles are
    skipped, since they are repetitive and not representative of Swedish."""
    import random

    from datasets import load_dataset

    rng = random.Random(seed)
    stream = load_dataset(WIKI_REPO, WIKI_CONFIG, split="train", streaming=True)
    stream = stream.shuffle(seed=seed, buffer_size=10_000)
    produced = 0
    for article in stream:
        text = article["text"]
        if len(text) < 2000 or "beskrevs av" in text:
            continue
        paragraphs = [p for p in text.split("\n") if len(p) > 120]
        if not paragraphs:
            continue
        sentences = [s for s in _SENTENCE_SPLIT.split(rng.choice(paragraphs))]
        k = rng.randint(1, max_sentences)
        start = rng.randrange(max(1, len(sentences) - k + 1))
        window = sentences[start:start + k]
        if window and all(is_clean_sentence(s) for s in window):
            yield " ".join(window)
            produced += 1
            if produced >= n:
                return
