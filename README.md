# Swedish grammaticality scorer

Scores Swedish text from 0 (poor) to 1 (grammatically correct).
A Swedish ModernBERT-base encoder (~150M parameters) is fine-tuned on real learner errors plus synthetic errors.

- Model: [`Nawazaz/swedish-grammar-scorer`](https://huggingface.co/Nawazaz/swedish-grammar-scorer)
- Data: [`Nawazaz/swedish-grammar-data`](https://huggingface.co/datasets/Nawazaz/swedish-grammar-data)
- Paper: [`paper/paper.md`](paper/paper.md)

## Predict

```bash
pip install -r requirements.txt
python predict.py --input test.jsonl --output scores.jsonl
python evaluate.py          # starter test set: Spearman, ROC-AUC, samples/min
```

`predict.py` downloads the model from the Hub on first run. Use `--model path/to/folder` for a local model.

## Reproduce from scratch

On your own machine, one command runs everything. It uses the full data on a GPU and a 15k-example subset on CPU:

```bash
python -m pip install -r requirements.txt
python run_local.py --quick             # 5-10 min test run
python run_local.py --hf-user <name>    # real run, publishes data and model
```

On a free T4 GPU, use [`colab_reproduce.ipynb`](colab_reproduce.ipynb). Or run each step yourself:

```bash
pip install -r requirements.txt
python build_dataset.py --out data/processed            # add --push <user>/<dataset> to publish
python train.py --data data/processed --output-dir models/scorer   # add --push <user>/<model>
python predict.py --model models/scorer --input data/starter/test.jsonl --output scores.jsonl
python -m pytest tests
```

All randomness uses a fixed seed (42). Dependency versions are pinned in `requirements.txt`. DaLAJ is pinned to a Hub commit. Wikipedia uses the dated `20231101.sv` dump. The exact data used for the published model is the published dataset, so `python train.py --data Nawazaz/swedish-grammar-data` retrains on identical data.

## How it works

| File | Purpose |
|---|---|
| `grammar_scorer/corruptions.py` | Rule-based Swedish error generators and the scoring scheme |
| `grammar_scorer/sources.py` | Loads DaLAJ and Swedish Wikipedia, filters clean sentences |
| `build_dataset.py` | Combines sources, injects errors, labels, splits by group |
| `train.py` | Fine-tunes the encoder with soft-label BCE |
| `predict.py` | Required submission interface; length-sorted batched inference |
| `evaluate.py` | Organisers' evaluation script, pointed at `data/starter/` |

**Data.** Three sources:
1. DaLAJ-GED (SuperLim 2.0): learner sentences with one real error each, plus their corrections.
2. Clean passages from Swedish Wikipedia (bot-written species stubs removed) and DaLAJ corrections, each paired with a copy that has 1–3 synthetic errors.
3. The 478 starter training examples.

**Labels.** Clean text scores 0.95. Each error multiplies the score by a factor between 0.25 and 0.5, depending on its type. One error gives about 0.35–0.45 and two errors about 0.15–0.2, which matches the starter data.

**Synthetic errors.** Gender (`en/ett`), adjective agreement, definiteness, plural forms, verb forms, V2 word order, negation placement, prepositions, pronoun case (`de/dem`), split compounds, missing and duplicated words, scrambled spans, and out-of-place words.

## Licence and attribution

Code is MIT. DaLAJ-GED is © Språkbanken Text (CC BY 4.0). Wikipedia text is CC BY-SA 4.0, so the published dataset is CC BY-SA 4.0.
The Hugging Face model and dataset cards are in `hub/`, and `python upload_cards.py` uploads them.
