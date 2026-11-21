import random

from grammar_scorer import corruptions as c


def run(fn, text, seed=0):
    result = fn(c.tokenize(text), random.Random(seed))
    assert result is not None, f"{fn.__name__} did not apply to: {text}"
    tokens, error = result
    return c.detokenize(tokens), error


def test_tokenize_roundtrip():
    text = "Hej, vad heter du? Jag heter Anna."
    assert c.detokenize(c.tokenize(text)) == text


def test_v2_word_order():
    assert run(c.v2_word_order, "Förra året reste vi till Japan.")[0] == "Förra året vi reste till Japan."


def test_gender_keeps_case():
    assert run(c.gender, "Ett barn satt på bänken.")[0] == "En barn satt på bänken."


def test_negation_in_subordinate_clause():
    out, _ = run(c.adverb_placement, "Hon sa att hon inte kommer i kväll.")
    assert out == "Hon sa att hon kommer inte i kväll."


def test_compound_split():
    assert run(c.compound_split, "Hon köpte en ny vinterjacka.")[0] == "Hon köpte en ny vinter jacka."


def test_definiteness():
    assert run(c.definiteness, "Den gamla bron renoverades.")[0] == "Den gamla bro renoverades."


def test_verb_form_after_modal():
    assert run(c.verb_form, "Vi kan arbeta hemma.")[0] == "Vi kan arbetar hemma."


def test_not_applicable_returns_none():
    assert c.plural_form(c.tokenize("Hej."), random.Random(0)) is None


def test_scores_fall_with_more_errors():
    one = c.score_for([c.Error("gender", "", c.FACTORS["gender"])])
    two = c.score_for([c.Error("gender", "", c.FACTORS["gender"])] * 2)
    assert c.CLEAN_SCORE > one > two >= c.MIN_SCORE


def test_corrupter_changes_text_and_is_seeded():
    text = "Trots att vädret var dåligt kom nästan alla till festen, och vi dansade till midnatt."
    a = c.Corrupter().corrupt(text, 2, random.Random(1))
    b = c.Corrupter().corrupt(text, 2, random.Random(1))
    assert a == b and a[0] != text and len(a[1]) == 2


def test_compile_patch_is_a_no_op():
    import torch
    from grammar_scorer import compat

    original = torch.compile
    try:
        compat.patch_torch_compile(force=True)

        @torch.compile(dynamic=True)
        def f(x):
            return x + 1

        assert f(1) == 2 and torch.compile(f) is f
    finally:
        torch.compile = original
