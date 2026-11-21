---
language: sv
license: cc-by-sa-4.0
task_categories:
- text-classification
pretty_name: Swedish grammaticality scores
size_categories:
- 100K<n<1M
---

# Swedish grammaticality scores

Swedish texts labelled with a grammaticality score from 0 (many errors) to 1 (correct). It was built to train [`Nawazaz/swedish-grammar-scorer`](https://huggingface.co/Nawazaz/swedish-grammar-scorer).
Code to rebuild it: [github.com/Nawazaz/swedish-grammar-scorer](https://github.com/Nawazaz/swedish-grammar-scorer) (`build_dataset.py`, seed 42).

## Splits

| Split | Rows |
|---|---|
| train | 120,985 |
| validation | 3,743 |

The split is done by `group`, so a clean text and its corrupted copy are always in the same split.

## Fields

- `text`: the Swedish text (1–3 sentences).
- `score`: float in [0, 1].
- `label`: `good` or `poor`.
- `error`: description of each error, or null for clean text.
- `source`: `dalaj`, `dalaj+synthetic`, `wikipedia` or `starter`.
- `group`: shared by a clean text and its corrupted copy.

## Sources

- **DaLAJ-GED (SuperLim 2.0), train + dev:** learner sentences with one real error each, plus their corrections. Volodina et al. (2023), Språkbanken Text, CC BY 4.0. DOI: 10.23695/kxvz-tx42.
- **Swedish Wikipedia (`wikimedia/wikipedia`, `20231101.sv`):** 30,000 clean passages, with bot-written stubs filtered out. CC BY-SA 4.0.
- **Task starter data:** 478 examples.

## Labelling

Clean text gets 0.95. Clean texts (Wikipedia and DaLAJ corrections) are paired with copies that have 1–3 rule-based errors. The error types are gender, agreement, definiteness, plural form, verb form, V2 word order, negation placement, preposition, pronoun case, split compound, missing or duplicated word, scrambled span, and out-of-place word.
Each error multiplies the score by a type-specific factor between 0.25 and 0.5, so one error gives about 0.4 and two errors about 0.2. Real DaLAJ errors count as one error, and a quarter of them get an extra synthetic error.

The synthetic rules are heuristics without a POS tagger, so a small share of "corrupted" texts may still be correct.

## Licence

CC BY-SA 4.0, because the dataset contains Wikipedia text (CC BY-SA 4.0) and DaLAJ-GED (CC BY 4.0).
