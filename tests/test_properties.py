"""
Property-based tests for the Donizo Truth Engine using Hypothesis.

Tests properties that must ALWAYS hold:
  - All prices are integers (cents) — never floats
  - Bias list never exceeds 5 entries
  - Same inputs always produce same hash (determinism)
  - Median always returns an integer
  - Circuit breaker prevents learning from anomalies
  - Decay always reduces or preserves bias magnitude
  - Accepted human price equals final price
  - Non-HUMAN events never trigger HUMAN_OVERRIDE_ACCEPTED flag
"""
import json
import os
import tempfile

import pytest
from hypothesis import given, settings, assume, HealthCheck
from hypothesis import strategies as st

from donizo_engine.engine import TruthEngine, _median_int, run_engine
from donizo_engine.models import (
    OUTCOME_NONE,
    OUTCOME_QUOTE_ACCEPTED,
    OUTCOME_QUOTE_REJECTED,
    DECAY_THRESHOLD_SECONDS,
    SOURCE_HISTORIC,
    SOURCE_HUMAN,
    SOURCE_SUPPLIER,
    Event,
    ItemState,
    RulesState,
)
from donizo_engine.state import compute_state_hash, save_state


# -----------------------------------------------------------------------
# Strategies
# -----------------------------------------------------------------------
sources = st.sampled_from([SOURCE_HISTORIC, SOURCE_SUPPLIER, SOURCE_HUMAN])
outcomes = st.sampled_from([OUTCOME_NONE, OUTCOME_QUOTE_ACCEPTED, OUTCOME_QUOTE_REJECTED])
item_ids = st.sampled_from(["item_a", "item_b", "item_c"])


@st.composite
def event_strategy(draw, idx=0):
    source = draw(sources)
    outcome = OUTCOME_NONE
    if source == SOURCE_HUMAN:
        outcome = draw(outcomes)
    return Event(
        event_id=f"prop-{draw(st.integers(min_value=0, max_value=999_999_999))}",
        timestamp=draw(st.integers(min_value=1_000_000_000, max_value=2_000_000_000)),
        item_id=draw(item_ids),
        source=source,
        price_cents=draw(st.integers(min_value=1, max_value=1_000_000)),
        outcome=outcome,
    )


def unique_event_lists(min_size=1, max_size=50):
    """Generate lists of events with unique event_ids."""
    return st.lists(
        event_strategy(), min_size=min_size, max_size=max_size,
        unique_by=lambda e: e.event_id,
    )


# -----------------------------------------------------------------------
# Property 1: All output prices are integers
# -----------------------------------------------------------------------
class TestPropertyIntegerOnly:
    @given(events=unique_event_lists())
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_output_prices_are_int(self, events):
        engine = TruthEngine(RulesState())
        for event in sorted(events, key=lambda e: e.timestamp):
            record = engine.process_event(event)
            assert isinstance(record.final_price_cents, int), (
                f"final_price_cents is {type(record.final_price_cents).__name__}, not int"
            )
            assert isinstance(record.bias_applied_cents, int), (
                f"bias_applied_cents is {type(record.bias_applied_cents).__name__}, not int"
            )


# -----------------------------------------------------------------------
# Property 2: Bias list never exceeds 5
# -----------------------------------------------------------------------
class TestPropertyBiasListSize:
    @given(events=unique_event_lists())
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_bias_list_max_5(self, events):
        engine = TruthEngine(RulesState())
        for event in sorted(events, key=lambda e: e.timestamp):
            engine.process_event(event)
        for item_id, item_state in engine.state.items.items():
            assert len(item_state.accepted_human_deltas_cents) <= 5, (
                f"Item {item_id} has {len(item_state.accepted_human_deltas_cents)} deltas"
            )


# -----------------------------------------------------------------------
# Property 3: Determinism — same inputs yield identical hash
# -----------------------------------------------------------------------
class TestPropertyDeterminism:
    @given(events=unique_event_lists(max_size=30))
    @settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow])
    def test_same_input_same_hash(self, events):
        sorted_events = sorted(events, key=lambda e: e.timestamp)

        engine1 = TruthEngine(RulesState())
        for event in sorted_events:
            engine1.process_event(event)
        hash1 = compute_state_hash(engine1.state)

        engine2 = TruthEngine(RulesState())
        for event in sorted_events:
            engine2.process_event(event)
        hash2 = compute_state_hash(engine2.state)

        assert hash1 == hash2


# -----------------------------------------------------------------------
# Property 4: Median is always an integer
# -----------------------------------------------------------------------
class TestPropertyMedian:
    @given(values=st.lists(st.integers(min_value=-10000, max_value=10000), min_size=1, max_size=10))
    @settings(max_examples=200)
    def test_median_is_int(self, values):
        result = _median_int(values)
        assert isinstance(result, int)


# -----------------------------------------------------------------------
# Property 5: Circuit breaker prevents learning from anomalies
# -----------------------------------------------------------------------
class TestPropertyCircuitBreaker:
    @given(
        supplier_price=st.integers(min_value=100, max_value=100000),
        multiplier=st.integers(min_value=151, max_value=500),
    )
    @settings(max_examples=100)
    def test_anomaly_blocks_learning(self, supplier_price, multiplier):
        """If human price > 150% of supplier, no learning occurs."""
        engine = TruthEngine(RulesState())
        human_price = (supplier_price * multiplier) // 100 + 1
        engine.process_event(Event(
            event_id="s1", timestamp=1000, item_id="x",
            source=SOURCE_SUPPLIER, price_cents=supplier_price,
        ))
        record = engine.process_event(Event(
            event_id="h1", timestamp=1100, item_id="x",
            source=SOURCE_HUMAN, price_cents=human_price,
            outcome=OUTCOME_QUOTE_ACCEPTED,
        ))
        assert "ANOMALY_REJECTED" in record.flags
        assert "x" not in engine.state.items


# -----------------------------------------------------------------------
# Property 6: Decay always reduces or preserves bias magnitude
# -----------------------------------------------------------------------
class TestPropertyDecay:
    @given(
        bias=st.integers(min_value=-10000, max_value=10000),
        gap_days=st.integers(min_value=8, max_value=365),
    )
    @settings(max_examples=100)
    def test_decay_reduces_bias(self, bias, gap_days):
        """After >7 days, the applied bias should be floor(bias/2)."""
        engine = TruthEngine(RulesState())
        engine.state.items["item_a"] = ItemState(
            bias_cents=bias,
            last_updated_ts=1000,
            accepted_human_deltas_cents=[bias] if bias != 0 else [],
        )
        # Add a historic price so we get USED_HISTORIC_PLUS_BIAS
        engine.process_event(Event(
            event_id="h1", timestamp=1000, item_id="item_a",
            source=SOURCE_HISTORIC, price_cents=5000,
        ))
        # Query after decay gap
        ts_after = 1000 + gap_days * 86400
        record = engine.process_event(Event(
            event_id="q1", timestamp=ts_after, item_id="item_a",
            source=SOURCE_HISTORIC, price_cents=5000,
        ))
        expected = bias // 2
        assert record.bias_applied_cents == expected


# -----------------------------------------------------------------------
# Property 7: Accepted human price == final price (no circuit breaker)
# -----------------------------------------------------------------------
class TestPropertyHumanAccepted:
    @given(
        supplier_price=st.integers(min_value=1000, max_value=100000),
        markup_pct=st.integers(min_value=1, max_value=49),
    )
    @settings(max_examples=100)
    def test_accepted_human_is_final_price(self, supplier_price, markup_pct):
        """When human accepted and NOT anomaly, final_price == human_price."""
        engine = TruthEngine(RulesState())
        # Keep human price under 150% of supplier to avoid circuit breaker
        human_price = supplier_price + (supplier_price * markup_pct) // 100
        engine.process_event(Event(
            event_id="s1", timestamp=1000, item_id="x",
            source=SOURCE_SUPPLIER, price_cents=supplier_price,
        ))
        record = engine.process_event(Event(
            event_id="h1", timestamp=1100, item_id="x",
            source=SOURCE_HUMAN, price_cents=human_price,
            outcome=OUTCOME_QUOTE_ACCEPTED,
        ))
        assert "ANOMALY_REJECTED" not in record.flags
        assert record.final_price_cents == human_price
        assert record.decision == "USED_HUMAN"


# -----------------------------------------------------------------------
# Property 8: Non-HUMAN events never trigger human flags
# -----------------------------------------------------------------------
class TestPropertyNonHumanFlags:
    @given(events=unique_event_lists(max_size=30))
    @settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow])
    def test_non_human_no_human_flags(self, events):
        engine = TruthEngine(RulesState())
        for event in sorted(events, key=lambda e: e.timestamp):
            record = engine.process_event(event)
            if event.source != SOURCE_HUMAN:
                assert "HUMAN_OVERRIDE_ACCEPTED" not in record.flags
                assert "HUMAN_REJECTED" not in record.flags
                assert "ANOMALY_REJECTED" not in record.flags


# -----------------------------------------------------------------------
# Property 9: Audit record hash matches current state hash
# -----------------------------------------------------------------------
class TestPropertyAuditHashConsistency:
    @given(events=unique_event_lists(max_size=20))
    @settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow])
    def test_audit_hash_matches_state(self, events):
        engine = TruthEngine(RulesState())
        for event in sorted(events, key=lambda e: e.timestamp):
            record = engine.process_event(event)
            # After each event, the rules_hash in audit should match current state
            assert record.rules_hash == engine.state.state_hash
