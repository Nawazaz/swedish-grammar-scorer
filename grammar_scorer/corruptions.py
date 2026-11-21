"""Rule-based generators of common Swedish grammatical errors.

Each corruption takes a list of tokens and a random generator, and returns
(new_tokens, error) or None when the rule does not apply to the text.
A token is a [prefix, core, suffix] triple so that punctuation such as the
final period stays in place when words are moved around.

The rules are deliberately simple heuristics without a POS tagger. They are
noisy, but cover the error types seen in learner Swedish: gender, agreement,
definiteness, verb forms, V2 word order, negation placement, prepositions,
pronoun case, split compounds and plain word salad.
"""

from __future__ import annotations

import math
import random
import re
from dataclasses import dataclass
from typing import Callable

Token = list[str]  # [prefix, core, suffix]

_TOKEN_RE = re.compile(r"^(\W*)(.*?)(\W*)$", re.UNICODE)


@dataclass(frozen=True)
class Error:
    kind: str
    description: str
    factor: float  # multiplicative score penalty: lower = more severe


# Score of a clean text, and the penalty factor per error type. One error
# gives roughly 0.35-0.45 and two errors roughly 0.15-0.2, which matches the
# scoring of the starter data.
CLEAN_SCORE = 0.95
MIN_SCORE = 0.02
FACTORS = {
    "verb_form": 0.37,
    "v2_word_order": 0.37,
    "adverb_placement": 0.42,
    "gender": 0.43,
    "agreement": 0.43,
    "definiteness": 0.43,
    "plural_form": 0.43,
    "pronoun_case": 0.45,
    "missing_word": 0.42,
    "preposition": 0.48,
    "compound_split": 0.48,
    "duplicated_word": 0.5,
    "wrong_word": 0.5,
    "word_salad": 0.25,
}


def score_for(errors: list[Error]) -> float:
    """Combine error penalties into one score in [MIN_SCORE, CLEAN_SCORE]."""
    score = CLEAN_SCORE * math.prod(e.factor for e in errors)
    return round(max(MIN_SCORE, score), 3)


# --------------------------------------------------------------------------
# Tokenisation helpers
# --------------------------------------------------------------------------

def tokenize(text: str) -> list[Token]:
    return [list(_TOKEN_RE.match(w).groups()) for w in text.split()]


def detokenize(tokens: list[Token]) -> str:
    return " ".join("".join(t) for t in tokens if "".join(t))


def _low(tok: Token) -> str:
    return tok[1].lower()


def _match_case(new: str, old: str) -> str:
    return new[:1].upper() + new[1:] if old[:1].isupper() else new


def _is_word(tok: Token) -> bool:
    return tok[1].isalpha()


def _plain(tok: Token) -> bool:
    """A word with no punctuation attached (so no clause boundary next to it)."""
    return _is_word(tok) and not tok[0] and not tok[2]


def _with_core(tok: Token, core: str) -> Token:
    return [tok[0], _match_case(core, tok[1]), tok[2]]


# --------------------------------------------------------------------------
# Word lists
# --------------------------------------------------------------------------

SUBJECT_PRONOUNS = {"jag", "du", "han", "hon", "vi", "ni", "de", "man", "det", "den"}
MODALS = {"kan", "ska", "skall", "måste", "vill", "bör", "får", "kunde", "skulle",
          "ville", "borde", "brukar", "brukade", "behöver", "behövde", "låter", "lät"}
AUXILIARIES = {"har", "hade"}
COMMON_FINITE = {"är", "var", "har", "hade", "kan", "ska", "kommer", "kom", "blev",
                 "blir", "finns", "fanns", "fick", "får", "gick", "går", "såg", "ser",
                 "sa", "sade", "säger", "måste", "vill", "ville", "kunde", "skulle",
                 "gör", "gjorde", "tog", "tar", "stod", "står", "låg", "ligger"}
SUBORDINATORS = {"att", "som", "om", "eftersom", "när", "medan", "innan", "tills",
                 "ifall", "fast", "fastän", "trots", "då", "där", "vilket"}
SENTENCE_ADVERBS = {"inte", "aldrig", "alltid", "ofta", "också", "redan", "bara", "nog"}
PREPOSITIONS = ["i", "på", "till", "för", "av", "med", "om", "åt", "vid", "från"]
GENDER_SWAPS = {"en": "ett", "ett": "en", "ingen": "inget", "inget": "ingen",
                "min": "mitt", "mitt": "min", "din": "ditt", "ditt": "din",
                "sin": "sitt", "sitt": "sin", "vilken": "vilket", "vilket": "vilken",
                "någon": "något", "något": "någon", "denna": "detta", "detta": "denna"}
PRONOUN_SWAPS = {"de": "dem", "dem": "de", "jag": "mig", "mig": "jag", "han": "honom",
                 "honom": "han", "hon": "henne", "henne": "hon", "vi": "oss", "oss": "vi",
                 "sin": "hans", "sitt": "hans", "sina": "deras", "deras": "sina"}
QUANTIFIERS = {"två", "tre", "fyra", "fem", "sex", "sju", "åtta", "nio", "tio",
               "många", "flera", "några", "alla", "olika"}
DROPPABLE = {"att", "som", "det", "är", "har", "en", "ett", "den", "de", "sig"}
COMPOUND_HEADS = (
    "arbete", "jacka", "skola", "hus", "bil", "tid", "dag", "rum", "plats", "stad",
    "land", "bok", "väg", "resa", "system", "område", "program", "match", "tåg",
    "fråga", "kvinna", "barn", "bord", "dörr", "kort", "station", "grupp", "företag",
    "möte", "verk", "hem", "sida", "vatten", "mat", "tjänst", "kyrka", "musik", "film",
    "spel", "bana", "ledare", "krig", "parti", "klubb", "sjukhus", "ord", "lärare",
    "vård", "kommun", "politik", "regering", "säsong", "fotboll", "trädgård", "kök",
)


# --------------------------------------------------------------------------
# Corruptions
# --------------------------------------------------------------------------

Result = tuple[list[Token], Error] | None


def _err(kind: str, description: str) -> Error:
    return Error(kind, description, FACTORS[kind])


def _pick(rng: random.Random, candidates: list[int]) -> int | None:
    return rng.choice(candidates) if candidates else None


def gender(tokens: list[Token], rng: random.Random) -> Result:
    i = _pick(rng, [i for i, t in enumerate(tokens[:-1])
                    if _low(t) in GENDER_SWAPS and _is_word(tokens[i + 1])])
    if i is None:
        return None
    old = tokens[i][1]
    new = GENDER_SWAPS[old.lower()]
    out = tokens.copy()
    out[i] = _with_core(tokens[i], new)
    return out, _err("gender", f"wrong grammatical gender ('{old}' -> '{new}')")


def agreement(tokens: list[Token], rng: random.Random) -> Result:
    """'en stor bil' -> 'en stort bil', 'ett stort hus' -> 'ett stor hus'."""
    cands = [i for i in range(1, len(tokens) - 1)
             if _low(tokens[i - 1]) in {"en", "ett"} and _plain(tokens[i])
             and _is_word(tokens[i + 1]) and tokens[i][1].islower() and len(tokens[i][1]) > 2]
    i = _pick(rng, cands)
    if i is None:
        return None
    word = tokens[i][1]
    if _low(tokens[i - 1]) == "en":
        new = word + "t"
    elif word.endswith("t"):
        new = word[:-1]
    else:
        return None
    out = tokens.copy()
    out[i] = _with_core(tokens[i], new)
    return out, _err("agreement", f"agreement error ('{word}' -> '{new}')")


def definiteness(tokens: list[Token], rng: random.Random) -> Result:
    """'den gamla bron' -> 'den gamla bro'."""
    cands = [i for i in range(2, len(tokens))
             if _low(tokens[i - 2]) in {"den", "det", "de"} and tokens[i - 1][1].endswith(("a", "e"))
             and _plain(tokens[i - 1]) and _is_word(tokens[i]) and tokens[i][1].islower()
             and tokens[i][1].endswith(("n", "t", "na")) and len(tokens[i][1]) > 3]
    i = _pick(rng, cands)
    if i is None:
        return None
    word = tokens[i][1]
    new = word[:-2] if word.endswith("na") else word[:-1]
    out = tokens.copy()
    out[i] = _with_core(tokens[i], new)
    return out, _err("definiteness", f"wrong definite/indefinite form ('{word}' -> '{new}')")


def plural_form(tokens: list[Token], rng: random.Random) -> Result:
    """'två bilar' -> 'två biler'."""
    cands = [i for i in range(1, len(tokens))
             if _low(tokens[i - 1]) in QUANTIFIERS and _is_word(tokens[i])
             and tokens[i][1].islower() and tokens[i][1].endswith(("ar", "er", "or"))
             and len(tokens[i][1]) > 4]
    i = _pick(rng, cands)
    if i is None:
        return None
    word = tokens[i][1]
    new = word[:-2] + rng.choice([s for s in ("ar", "er", "or", "s") if not word.endswith(s)])
    out = tokens.copy()
    out[i] = _with_core(tokens[i], new)
    return out, _err("plural_form", f"wrong plural form ('{word}' -> '{new}')")


def verb_form(tokens: list[Token], rng: random.Random) -> Result:
    cands = []
    for i in range(1, len(tokens)):
        prev, word = _low(tokens[i - 1]), tokens[i][1]
        if not (_is_word(tokens[i]) and word.islower() and len(word) > 3):
            continue
        if prev in AUXILIARIES and word.endswith("at"):        # har arbetat -> har arbetade
            cands.append((i, word[:-2] + "ade"))
        elif prev in MODALS and word.endswith("a"):            # kan arbeta -> kan arbetar
            cands.append((i, word + "r"))
        elif prev in SUBJECT_PRONOUNS and word.endswith("ar") and word not in COMMON_FINITE:
            cands.append((i, word[:-1]))                       # jag arbetar -> jag arbeta
    if not cands:
        return None
    i, new = rng.choice(cands)
    old = tokens[i][1]
    out = tokens.copy()
    out[i] = _with_core(tokens[i], new)
    return out, _err("verb_form", f"wrong verb form ('{old}' -> '{new}')")


def v2_word_order(tokens: list[Token], rng: random.Random) -> Result:
    """'Förra året reste vi' -> 'Förra året vi reste'."""
    cands = [i for i in range(1, len(tokens) - 1)
             if _plain(tokens[i]) and _low(tokens[i + 1]) in SUBJECT_PRONOUNS
             and not tokens[i + 1][0] and not tokens[i - 1][2]
             and (_low(tokens[i]) in COMMON_FINITE or _low(tokens[i]).endswith(("ar", "er", "de", "te")))]
    i = _pick(rng, cands)
    if i is None:
        return None
    out = tokens.copy()
    verb, subj = tokens[i], tokens[i + 1]
    out[i] = [verb[0], subj[1], verb[2]]
    out[i + 1] = [subj[0], verb[1], subj[2]]
    return out, _err("v2_word_order", f"verb-second word order violated ('{verb[1]} {subj[1]}' -> '{subj[1]} {verb[1]}')")


def adverb_placement(tokens: list[Token], rng: random.Random) -> Result:
    """Main clause 'jag kommer inte' -> 'jag inte kommer';
    subordinate clause 'att jag inte kommer' -> 'att jag kommer inte'."""
    cands = []
    for i in range(1, len(tokens) - 2):
        if _low(tokens[i]) not in SUBJECT_PRONOUNS or not _plain(tokens[i + 1]):
            continue
        subordinate = _low(tokens[i - 1]) in SUBORDINATORS
        a, b = _low(tokens[i + 1]), _low(tokens[i + 2])
        if subordinate and a in SENTENCE_ADVERBS and _is_word(tokens[i + 2]):
            cands.append(i + 1)
        elif not subordinate and b in SENTENCE_ADVERBS and _is_word(tokens[i + 1]):
            cands.append(i + 1)
    i = _pick(rng, cands)
    if i is None:
        return None
    out = tokens.copy()
    x, y = tokens[i], tokens[i + 1]
    out[i] = [x[0], y[1], x[2]]
    out[i + 1] = [y[0], x[1], y[2]]
    return out, _err("adverb_placement", f"adverb/negation placement ('{x[1]} {y[1]}' -> '{y[1]} {x[1]}')")


def preposition(tokens: list[Token], rng: random.Random) -> Result:
    i = _pick(rng, [i for i, t in enumerate(tokens) if _low(t) in PREPOSITIONS])
    if i is None:
        return None
    old = tokens[i][1]
    new = rng.choice([p for p in PREPOSITIONS if p != old.lower()])
    out = tokens.copy()
    out[i] = _with_core(tokens[i], new)
    return out, _err("preposition", f"wrong or missing preposition ('{old}' -> '{new}')")


def pronoun_case(tokens: list[Token], rng: random.Random) -> Result:
    i = _pick(rng, [i for i, t in enumerate(tokens) if _low(t) in PRONOUN_SWAPS])
    if i is None:
        return None
    old = tokens[i][1]
    new = PRONOUN_SWAPS[old.lower()]
    out = tokens.copy()
    out[i] = _with_core(tokens[i], new)
    return out, _err("pronoun_case", f"wrong pronoun form ('{old}' -> '{new}')")


def compound_split(tokens: list[Token], rng: random.Random) -> Result:
    """'vinterjackan' -> 'vinter jackan'."""
    cands = []
    for i, t in enumerate(tokens):
        word = t[1]
        if not (word.isalpha() and word.islower() and len(word) >= 7):
            continue
        for cut in range(3, len(word) - 2):
            if word[cut:].startswith(COMPOUND_HEADS):
                cands.append((i, cut))
                break
    if not cands:
        return None
    i, cut = rng.choice(cands)
    word = tokens[i][1]
    out = tokens[:i] + [[tokens[i][0], word[:cut], ""], ["", word[cut:], tokens[i][2]]] + tokens[i + 1:]
    return out, _err("compound_split", f"compound written as two words ('{word}' -> '{word[:cut]} {word[cut:]}')")


def missing_word(tokens: list[Token], rng: random.Random) -> Result:
    i = _pick(rng, [i for i in range(1, len(tokens)) if _plain(tokens[i]) and _low(tokens[i]) in DROPPABLE])
    if i is None:
        return None
    return tokens[:i] + tokens[i + 1:], _err("missing_word", f"missing word ('{tokens[i][1]}')")


def duplicated_word(tokens: list[Token], rng: random.Random) -> Result:
    i = _pick(rng, [i for i in range(len(tokens)) if _plain(tokens[i]) and len(tokens[i][1]) < 6])
    if i is None:
        return None
    return tokens[:i + 1] + [list(tokens[i])] + tokens[i + 1:], _err("duplicated_word", f"duplicated word ('{tokens[i][1]}')")


def word_salad(tokens: list[Token], rng: random.Random) -> Result:
    """Shuffle the words of a span of 3-5 tokens (keeps punctuation in place)."""
    if len(tokens) < 6:
        return None
    n = rng.randint(3, min(5, len(tokens) - 2))
    start = rng.randint(1, len(tokens) - n - 1)
    cores = [t[1] for t in tokens[start:start + n]]
    shuffled = cores.copy()
    for _ in range(5):
        rng.shuffle(shuffled)
        if shuffled != cores:
            break
    else:
        return None
    out = tokens.copy()
    for k, core in enumerate(shuffled):
        t = tokens[start + k]
        out[start + k] = [t[0], core, t[2]]
    return out, _err("word_salad", "scrambled word order")


def make_wrong_word(vocabulary: list[str]) -> Callable[[list[Token], random.Random], Result]:
    """Replace a content word with a random word from elsewhere in the corpus,
    which gives fluent-looking but incoherent text."""

    def wrong_word(tokens: list[Token], rng: random.Random) -> Result:
        i = _pick(rng, [i for i in range(1, len(tokens))
                        if _is_word(tokens[i]) and tokens[i][1].islower() and len(tokens[i][1]) > 4])
        if i is None or not vocabulary:
            return None
        new = rng.choice(vocabulary)
        out = tokens.copy()
        out[i] = [tokens[i][0], new, tokens[i][2]]
        return out, _err("wrong_word", f"nonsensical word choice ('{tokens[i][1]}' -> '{new}')")

    return wrong_word


GRAMMAR_CORRUPTIONS = [gender, agreement, definiteness, plural_form, verb_form,
                       v2_word_order, adverb_placement, preposition, pronoun_case,
                       compound_split, missing_word]
NOISE_CORRUPTIONS = [duplicated_word, word_salad]


class Corrupter:
    """Applies a given number of random corruptions to a text."""

    def __init__(self, vocabulary: list[str] | None = None, noise_prob: float = 0.15):
        self.grammar = list(GRAMMAR_CORRUPTIONS)
        self.noise = list(NOISE_CORRUPTIONS)
        if vocabulary:
            self.noise.append(make_wrong_word(vocabulary))
        self.noise_prob = noise_prob

    def corrupt(self, text: str, n_errors: int, rng: random.Random) -> tuple[str, list[Error]] | None:
        tokens = tokenize(text)
        errors: list[Error] = []
        used: set[str] = set()  # one error per type, so a second edit cannot undo the first
        for _ in range(n_errors):
            pool = self.noise if rng.random() < self.noise_prob else self.grammar
            for fn in rng.sample(pool, len(pool)) + rng.sample(self.grammar, len(self.grammar)):
                result = fn(tokens, rng)
                if result is not None and result[1].kind not in used:
                    tokens, error = result
                    errors.append(error)
                    used.add(error.kind)
                    break
        corrupted = detokenize(tokens)
        if not errors or corrupted == text:
            return None
        return corrupted, errors
