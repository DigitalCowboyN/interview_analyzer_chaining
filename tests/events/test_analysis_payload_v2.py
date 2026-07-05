from src.events.aggregates import Sentence


def make_sentence():
    s = Sentence("11111111-1111-1111-1111-111111111111")
    s.create(interview_id="22222222-2222-2222-2222-222222222222", index=0, text="Hi.")
    return s


def test_generate_analysis_carries_v2_fields():
    s = make_sentence()
    event = s.generate_analysis(
        model="claude-haiku-4-5-20251001",
        model_version="m4.2",
        classification={"purpose": "Statement"},
        dimension_confidences={"purpose": 0.85},
        flags={"function_type_disagreement": "spacy=interrogative llm=declarative"},
        provider="anthropic",
    )
    assert event.data["dimension_confidences"] == {"purpose": 0.85}
    assert event.data["provider"] == "anthropic"
    assert event.data["flags"]["function_type_disagreement"].startswith("spacy=")
    assert s.dimension_confidences["purpose"] == 0.85
    assert s.analysis_provider == "anthropic"


def test_v1_events_still_apply():
    s = make_sentence()
    event = s.generate_analysis(model="gpt", model_version="1.0", classification={"purpose": "Q"})
    assert event.data["dimension_confidences"] == {}
    assert event.data["provider"] is None
    replayed = Sentence("11111111-1111-1111-1111-111111111111")
    replayed.load_from_history(s.get_uncommitted_events())
    assert replayed.classification == {"purpose": "Q"}
    assert replayed.dimension_confidences == {}
    assert replayed.analysis_provider is None


def test_replay_reconstructs_v2_state():
    s = make_sentence()
    s.generate_analysis(
        model="m", model_version="m4.2", classification={"purpose": "Q"},
        dimension_confidences={"purpose": 0.7}, flags={"x": "y"}, provider="claude_code",
    )
    replayed = Sentence("11111111-1111-1111-1111-111111111111")
    replayed.load_from_history(s.get_uncommitted_events())
    assert replayed.dimension_confidences == {"purpose": 0.7}
    assert replayed.flags == {"x": "y"}
    assert replayed.analysis_provider == "claude_code"


def test_regenerate_preserves_v2_state():
    """AnalysisRegenerated does not carry v2 fields, but must not clobber them —
    they persist from the prior AnalysisGenerated (regen implies a prior gen)."""
    from src.events.sentence_events import EditorType  # noqa: F401 (import parity)

    s = make_sentence()
    s.generate_analysis(
        model="m", model_version="m4.2", classification={"purpose": "Q"},
        dimension_confidences={"purpose": 0.7}, flags={"x": "y"}, provider="anthropic",
    )
    # Simulate a regeneration event applied on top.
    from src.events.envelope import AggregateType, EventEnvelope

    regen = EventEnvelope(
        event_type="AnalysisRegenerated",
        aggregate_type=AggregateType.SENTENCE,
        aggregate_id=s.aggregate_id,
        version=s.version + 1,
        data={"model": "m2", "reason": "retune", "classification": {"purpose": "Statement"}},
    )
    s.apply_event(regen)
    assert s.classification == {"purpose": "Statement"}  # regen updated
    assert s.dimension_confidences == {"purpose": 0.7}  # v2 state preserved
    assert s.analysis_provider == "anthropic"
