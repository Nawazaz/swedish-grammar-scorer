"""Build the training data: real learner errors (DaLAJ) + synthetic errors
injected into clean Swedish (Wikipedia and DaLAJ corrections) + the starter data.

    python build_dataset.py --out data/processed            # build locally
    python build_dataset.py --push <hf-user>/<dataset-name> # build and publish
"""

from __future__ import annotations

import argparse
import random
from collections import Counter

from grammar_scorer.corruptions import CLEAN_SCORE, FACTORS, Corrupter, Error, score_for
from grammar_scorer.sources import load_dalaj, read_jsonl, wikipedia_passages, write_jsonl

SEED = 42
DALAJ_ERROR = Error("learner_error", "real learner error (DaLAJ)", FACTORS["agreement"])
N_ERRORS_WEIGHTS = {1: 0.5, 2: 0.3, 3: 0.2}  # how many errors a corrupted copy gets


def example(text, errors, source, group):
    score = score_for(errors) if errors else CLEAN_SCORE
    return {
        "text": text,
        "score": score,
        "label": "good" if not errors else "poor",
        "error": "; ".join(e.description for e in errors) or None,
        "source": source,
        "group": group,
    }


def add_corrupted_copy(rows, text, source, group, corrupter, rng, extra_errors=()):
    n = rng.choices(list(N_ERRORS_WEIGHTS), weights=list(N_ERRORS_WEIGHTS.values()))[0]
    result = corrupter.corrupt(text, n, rng)
    if result:
        corrupted, errors = result
        rows.append(example(corrupted, list(extra_errors) + errors, source, group))


def build(args) -> dict[str, list[dict]]:
    rng = random.Random(SEED)
    rows: list[dict] = []
    group = 0

    clean_texts: list[tuple[str, str]] = []  # (text, source)
    if not args.no_wikipedia:
        clean_texts += [(t, "wikipedia") for t in wikipedia_passages(args.wiki_passages, SEED)]

    if not args.no_dalaj:
        dalaj = load_dalaj("train") + load_dalaj("dev")
        correct = sorted({r["text"] for r in dalaj if r["correct"]})  # corrections repeat a lot
        clean_texts += [(t, "dalaj") for t in correct]
        incorrect = sorted({r["text"] for r in dalaj if not r["correct"]} - set(correct))
    else:
        incorrect = []

    vocabulary = sorted({w for t, _ in clean_texts for w in t.split() if w.isalpha() and w.islower() and len(w) > 4})
    corrupter = Corrupter(vocabulary=vocabulary)

    for text, source in clean_texts:
        group += 1
        rows.append(example(text, [], source, group))
        add_corrupted_copy(rows, text, source, group, corrupter, rng)

    # Real learner errors. A quarter also get synthetic errors on top, so the
    # model sees real errors at several severity levels.
    for text in incorrect:
        group += 1
        rows.append(example(text, [DALAJ_ERROR], "dalaj", group))
        if rng.random() < 0.25:
            add_corrupted_copy(rows, text, "dalaj+synthetic", group, corrupter, rng, [DALAJ_ERROR])

    # The organisers' toy data keeps the label scale anchored to theirs.
    for r in read_jsonl("data/starter/train.jsonl"):
        rows.append({"text": r["text"], "score": r["score"], "label": r["label"], "error": r["error"],
                     "source": "starter", "group": f"starter-{r['group']}"})

    # Split by group so a text and its corrupted copy never end up in different splits.
    groups = sorted({str(r["group"]) for r in rows})
    rng.shuffle(groups)
    val_groups = set(groups[: int(len(groups) * args.val_fraction)])
    splits = {"train": [], "validation": []}
    for r in rows:
        r["group"] = str(r["group"])
        splits["validation" if r["group"] in val_groups else "train"].append(r)
    rng.shuffle(splits["train"])
    return splits


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", default="data/processed")
    parser.add_argument("--push", help="Hugging Face dataset repo id to publish to")
    parser.add_argument("--wiki-passages", type=int, default=30_000)
    parser.add_argument("--val-fraction", type=float, default=0.03)
    parser.add_argument("--no-wikipedia", action="store_true", help="skip Wikipedia (offline testing)")
    parser.add_argument("--no-dalaj", action="store_true", help="skip DaLAJ (offline testing)")
    args = parser.parse_args()

    splits = build(args)
    for name, rows in splits.items():
        write_jsonl(rows, f"{args.out}/{name}.jsonl")
        stats = Counter((r["source"], r["label"]) for r in rows)
        print(f"{name}: {len(rows)} rows  " + ", ".join(f"{s}/{l}={n}" for (s, l), n in sorted(stats.items())))

    if args.push:
        from datasets import load_dataset

        files = {name: f"{args.out}/{name}.jsonl" for name in splits}
        load_dataset("json", data_files=files).push_to_hub(args.push)
        print(f"Pushed to https://huggingface.co/datasets/{args.push}")


if __name__ == "__main__":
    main()
