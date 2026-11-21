# Grading Swedish grammaticality with a fine-tuned ModernBERT

Nawaz Azim Pawaskar — Blekinge Institute of Technology

Code: github.com/Nawazaz/swedish-grammar-scorer · Model: `Nawazaz/swedish-grammar-scorer` · Data: `Nawazaz/swedish-grammar-data`

## Approach

I fine-tune `AI-Sweden-Models/ModernBERT-base` (~150M parameters), a ModernBERT further pre-trained on Scandinavian text. A single output logit goes through a sigmoid, so every score is in [0, 1]. Training uses binary cross-entropy against soft target scores, with mean pooling over tokens.

I chose a fine-tuned encoder over the frozen-embedding baseline, because mean-pooled frozen embeddings mostly encode topic, not grammar. Fine-tuning lets the model learn agreement and word-order cues.
I chose it over a decoder (e.g. perplexity from a small LLM), because perplexity also penalises rare but correct words and is slower to compute. I chose the base model over ModernBERT-large because it is about 3× faster, and speed is graded.

## Data

The dataset (`Nawazaz/swedish-grammar-data`) has 120,985 training rows and 3,743 validation rows, from three sources:

- **Real learner errors:** DaLAJ-GED from SuperLim 2.0 (train + dev splits; the test split is unused). Each sentence has one real error and comes with its correction.
- **Synthetic errors:** 30,000 clean passages of 1–3 sentences from Swedish Wikipedia, plus the unique DaLAJ corrections. Bot-written species stubs and noisy lines are filtered out. Each clean text is paired with a copy that has 1–3 rule-based errors: gender (en/ett), adjective agreement, definiteness, plural form, verb form, V2 word order, negation placement, preposition, pronoun case (de/dem), split compound, missing or duplicated word, scrambled span, or out-of-place word. A quarter of the DaLAJ errors get one extra synthetic error, so real errors appear at several severity levels.
- **Starter data:** the 478 training examples.

**Labels.** Clean text gets 0.95. Each error multiplies the score by a type-specific factor (0.25–0.5). The factors are chosen to match the starter scores: about 0.4 for one error, 0.2 for two, and under 0.1 for three. A clean text and its corrupted copy share a group ID, and the train/validation split is done by group, so no pair is split across them.

## Experiments

Metrics are from `evaluate.py` on the 120-example starter test set. Training used seed 42, learning rate 3e-5, batch size 32, and a maximum length of 256 tokens.

| Run | Val. Spearman | Test Spearman | Test ROC-AUC |
|---|---|---|---|
| Baseline: ModernBERT-large embeddings + SVR (starter) | – | 0.528 | 0.783 |
| CPU test run: 2k examples, 1 epoch | 0.271 | 0.359 | 0.645 |
| Full data, after epoch 1 | 0.811 | – | – |
| **Full data, 2 epochs (submitted)** | **0.837** | **0.819** | **0.959** |

**What did not work.** A small CPU-only run (2,000 examples) was far worse than the baseline, so the amount of training data clearly matters.
At inference, ModernBERT's `torch.compile` code path compiled the model on the first batch. That made GPU scoring slower than CPU scoring (179 vs 359 samples/min). Disabling compilation in `predict.py` raised GPU speed to 529 samples/min.

## Results

The submitted model reaches **Spearman 0.819 and ROC-AUC 0.959** on the starter test set, against 0.528 and 0.783 for the baseline.

Speed is measured with `evaluate.py` and includes Python start-up and model loading:
- **529 samples/min** on a Google Colab NVIDIA T4 GPU (fp16).
- **About 528 samples/min** on my laptop CPU (Windows, Python 3.14). This was measured with the same architecture during a test run.

With only 120 test texts, most of the time is model loading. Throughput on a larger test set will be much higher.

Training took 23 minutes on the T4.

## Limitations

- The synthetic errors come from heuristics without a POS tagger, so some "corrupted" texts are still correct (e.g. a preposition swap that is also valid). This adds label noise.
- Errors that need meaning or context are hard to catch: wrong word choice, calques, and tense consistency across sentences.
- Inputs are truncated at 256 tokens, so errors late in long texts are not seen.
- Most clean training text is formal Wikipedia prose. Informal or spoken-style Swedish may be scored lower than it should be.
- The label scale follows my own error-count scheme, not human ratings. The model ranks texts well (high Spearman), but its absolute scores may not match a human grader's.

TODO: add 1–2 real test examples the model misjudged.

## References

- Volodina, E., Mohammed, Y. A., Berdicevskis, A., Bouma, G., & Öhman, J. (2023). DaLAJ-GED – a dataset for Grammatical Error Detection tasks on Swedish.
- Warner, B., et al. (2024). Smarter, Better, Faster, Longer: A Modern Bidirectional Encoder (ModernBERT). arXiv:2412.13663.
- AI Sweden. ModernBERT-base (Scandinavian continued pre-training). Hugging Face: `AI-Sweden-Models/ModernBERT-base`.
