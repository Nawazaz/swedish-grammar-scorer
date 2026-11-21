# Grading Swedish grammaticality with a fine-tuned ModernBERT

Nawaz Azim Pawaskar — Blekinge Institute of Technology

## Approach

I fine-tune `AI-Sweden-Models/ModernBERT-base` (~150M parameters) as a regressor. One output logit goes through a sigmoid, so every score is in [0, 1]. Training uses binary cross-entropy against soft targets.
I chose a fine-tuned encoder over the frozen-embedding baseline, because mean-pooled embeddings mostly encode topic, not grammar. Fine-tuning lets the model attend to agreement and word order.
I chose it over a decoder (e.g. perplexity from a small LLM) because perplexity also penalises rare but correct words, and over ModernBERT-large because the base model is about 3× faster, and speed is graded.

## Data

Published as `Nawazaz/swedish-grammar-data` (TODO rows: train / validation).

- **Real errors:** DaLAJ-GED from SuperLim 2.0 (train + dev). These are learner sentences with one error each, plus their corrected versions. The DaLAJ test split is not used for training.
- **Synthetic errors:** clean passages of 1–3 sentences from Swedish Wikipedia, plus DaLAJ corrections. Bot-written species stubs and noisy lines are filtered out. Each clean text is paired with a copy that has 1–3 rule-based errors: gender, agreement, definiteness, plural, verb form, V2, negation placement, preposition, pronoun case, split compound, missing or duplicated word, scrambled span, or out-of-place word. A quarter of the DaLAJ errors get an extra synthetic error.
- **Starter data:** the 478 training examples.
- **Labels:** clean text gets 0.95. Each error multiplies the score by a type-specific factor (0.25–0.5), chosen to match the starter scores (about 0.4 for one error, 0.2 for two). A clean text and its corrupted copy share a group ID, and the train/validation split is done by group.

## Experiments

All runs evaluated with `evaluate.py` on the starter test set (120 examples) and on my validation split.

| Model / data | Spearman | ROC-AUC |
|---|---|---|
| Baseline: ModernBERT-large embeddings + SVR (starter) | 0.528 | 0.783 |
| ModernBERT-base, DaLAJ + starter only | TODO | TODO |
| ModernBERT-base, synthetic + starter only | TODO | TODO |
| ModernBERT-base, all data (submitted) | TODO | TODO |
| (optional) ModernBERT-large, all data | TODO | TODO |

TODO: one sentence on what did not work, e.g. MSE loss vs BCE, or the effect of removing word-salad errors.

## Results

Submitted model on the starter test set: Spearman TODO, ROC-AUC TODO.
Speed with `evaluate.py`, model loading included: TODO samples/min on a Colab T4 GPU, TODO samples/min on CPU (TODO CPU model).

## Limitations

- The synthetic errors are heuristic and lack a POS tagger, so a few "corrupted" texts are still correct (e.g. a preposition swap that is also valid). This adds label noise.
- Errors that need meaning or context are hard: wrong word choice, calques ("ta en beslut"), and tense consistency across sentences. TODO: add one or two real misjudged examples from the test set.
- Inputs are truncated at 256 tokens, so errors late in long texts are invisible.
- Wikipedia is formal. Informal, spoken-style Swedish may be scored lower than it should be.

## References

- Volodina et al. (2023). DaLAJ-GED – a dataset for Grammatical Error Detection tasks on Swedish.
- Warner et al. (2024). ModernBERT. arXiv:2412.13663.
- AI Sweden. ModernBERT-base (Scandinavian continued pre-training), Hugging Face.
