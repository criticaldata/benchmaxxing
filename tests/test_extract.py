"""Unit tests for the answer extraction module."""

from benchmaxxing.extract import Abstention, parse_mcq_choice, is_abstention

def test_is_abstention():
    # True positives
    assert is_abstention("As an AI, I cannot provide medical advice.")
    assert is_abstention("I am unable to diagnose this patient.")
    assert is_abstention("I'm sorry, but I cannot answer that.")
    assert is_abstention("It is not appropriate for me to prescribe medicine.")
    
    # False positives (should be false)
    assert not is_abstention("The correct answer is A.")
    assert not is_abstention("Based on the patient's symptoms, the diagnosis is clear.")
    assert not is_abstention("no")
    assert not is_abstention("maybe")
    assert not is_abstention("")

def test_parse_mcq_choice_golden_cases():
    """Golden table of (raw_text, options) -> expected."""
    options_5 = ("A", "B", "C", "D", "E")
    
    cases = [
        # (text, options, expected)
        
        # 1. Clean matches
        ("The correct answer is B.", options_5, 1),
        ("The correct option is C", options_5, 2),
        ("Option D is correct.", options_5, 3),
        
        # 2. Refusals embedded
        ("As an AI I cannot answer this medical question.", options_5, Abstention.REFUSAL),
        ("The question is complex. I am unable to diagnose. The answer might be A.", options_5, Abstention.REFUSAL),
        
        # 3. Trailing matches (chain of thought)
        ("The patient has a headache. A is migraine. B is stroke. Therefore, B.", options_5, 1),
        
        # 4. Length bias trap (distractor mentions multiple letters, but explicit declaration overrides)
        ("A is wrong. B is wrong. Option C is a very long text that mentions A and B and D and E. The correct answer is C.", options_5, 2),
        
        # 5. Conflicting double-answer (should be unparseable)
        ("It could be A or B.", options_5, Abstention.UNPARSEABLE),
        
        # 6. Garbage or unparseable
        ("I think it is a viral infection.", options_5, Abstention.UNPARSEABLE),
        ("", options_5, Abstention.UNPARSEABLE),
        ("The answer is Z.", options_5, Abstention.UNPARSEABLE),
    ]
    
    for text, opts, expected in cases:
        assert parse_mcq_choice(text, opts) == expected

def test_parse_mcq_choice_yes_no_maybe():
    """Ensure yes/no/maybe shapes are parsed cleanly and not swallowed."""
    opts = ("yes", "no", "maybe")
    
    assert parse_mcq_choice("yes", opts) == 0
    assert parse_mcq_choice("The answer is no.", opts) == 1
    assert parse_mcq_choice("maybe", opts) == 2
    
    # Conflicting
    assert parse_mcq_choice("It could be yes or no", opts) == Abstention.UNPARSEABLE
    
    # Refusal
    assert parse_mcq_choice("As an AI I cannot say yes or no.", opts) == Abstention.REFUSAL
