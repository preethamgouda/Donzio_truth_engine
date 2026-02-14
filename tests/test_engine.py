"""
Unit tests for each rule in the Donizo Truth Engine.
"""
import json
import os
import tempfile
import itertools

import pytest

from donizo_engine.engine import TruthEngine, _median_int
from donizo_engine.models import (
    OUTCOME_QUOTE_ACCEPTED,
    OUTCOME_QUOTE_REJECTED,
    OUTCOME_NONE,
    SOURCE_HISTORIC,
    SOURCE_HUMAN,
    SOURCE_SUPPLIER,
    Event,
    ItemState,
    RulesState,
)
from donizo_engine.state import compute_state_hash


# -----------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------
_counter = itertools.count(1)


def _event(
    item_id="copper_pipe_15mm",
    source=SOURCE_SUPPLIER,
    price=1200,
    ts=1700000000,
    outcome=OUTCOME_NONE,
    event_id=None,
) -> Event:
    return Event(
        event_id=event_id or f"test-{next(_counter)}",
        timestamp=ts,
        item_id=item_id,
        source=source,
        price_cents=price,
        outcome=outcome,
    )


def _engine(state=None) -> TruthEngine:
    return TruthEngine(state or RulesState())


# -----------------------------------------------------------------------
# Test: Median helper
# -----------------------------------------------------------------------
class TestMedian:
    def test_empty(self):
        assert _median_int([]) == 0

    def test_single(self):
        assert _median_int([300]) == 300

    def test_odd(self):
        assert _median_int([100, 200, 300]) == 200

    def test_even(self):
        # median([100,200]) = 150.0 → int(150.0) = 150
        assert _median_int([100, 200]) == 150

    def test_five(self):
        assert _median_int([300, 250, 400, 200, 350]) == 300

    def test_negative_delta(self):
        assert _median_int([-100, -50, 0]) == -50


# -----------------------------------------------------------------------
# Test: Rule A — Candidate Selection
# -----------------------------------------------------------------------
class TestRuleA:
    def test_supplier_eligible_within_1h(self):
        engine = _engine()
        # Supplier event at ts=1000
        engine.process_event(_event(source=SOURCE_SUPPLIER, price=1000, ts=1000))
        # Query within 1 hour
        rec = engine.process_event(_event(source=SOURCE_HISTORIC, price=900, ts=2000))
        # Supplier should still be candidates
        assert rec.inputs_seen["supplier_cents"] == 1000

    def test_supplier_expired_after_1h(self):
        engine = _engine()
        engine.process_event(_event(source=SOURCE_SUPPLIER, price=1000, ts=1000))
        # Query after 1 hour + 1 second
        rec = engine.process_event(_event(source=SOURCE_HISTORIC, price=900, ts=4601))
        # Supplier should be expired → fallback to historic
        assert rec.decision == "USED_HISTORIC_PLUS_BIAS"

    def test_historic_always_eligible(self):
        engine = _engine()
        rec = engine.process_event(_event(source=SOURCE_HISTORIC, price=900, ts=1000))
        assert rec.inputs_seen["historic_cents"] == 900

    def test_human_only_when_source_human(self):
        engine = _engine()
        rec = engine.process_event(_event(source=SOURCE_SUPPLIER, price=1000, ts=1000))
        assert rec.inputs_seen["human_cents"] is None


# -----------------------------------------------------------------------
# Test: Rule B — Decision Tree
# -----------------------------------------------------------------------
class TestRuleB:
    def test_human_accepted_uses_human_price(self):
        engine = _engine()
        # Add supplier first
        engine.process_event(_event(source=SOURCE_SUPPLIER, price=1000, ts=1000))
        rec = engine.process_event(_event(
            source=SOURCE_HUMAN, price=1500, ts=1100,
            outcome=OUTCOME_QUOTE_ACCEPTED,
        ))
        assert rec.final_price_cents == 1500
        assert rec.decision == "USED_HUMAN"
        assert "HUMAN_OVERRIDE_ACCEPTED" in rec.flags

    def test_human_rejected_falls_back(self):
        engine = _engine()
        engine.process_event(_event(source=SOURCE_SUPPLIER, price=1000, ts=1000))
        rec = engine.process_event(_event(
            source=SOURCE_HUMAN, price=1200, ts=1100,
            outcome=OUTCOME_QUOTE_REJECTED,
        ))
        assert rec.decision == "USED_SUPPLIER_PLUS_BIAS"
        assert "HUMAN_REJECTED" in rec.flags
        assert rec.final_price_cents == 1000  # bias is 0

    def test_standard_query_uses_supplier_plus_bias(self):
        engine = _engine()
        # Set up bias for item
        engine.state.items["copper_pipe_15mm"] = ItemState(
            bias_cents=200, last_updated_ts=999,
            accepted_human_deltas_cents=[200],
        )
        engine.process_event(_event(source=SOURCE_SUPPLIER, price=1000, ts=1000))
        rec = engine.process_event(_event(source=SOURCE_HISTORIC, price=900, ts=1050))
        # Supplier is fresh → supplier + bias
        assert rec.final_price_cents == 1200
        assert rec.decision == "USED_SUPPLIER_PLUS_BIAS"

    def test_standard_query_falls_back_to_historic(self):
        engine = _engine()
        rec = engine.process_event(_event(source=SOURCE_HISTORIC, price=900, ts=1000))
        assert rec.decision == "USED_HISTORIC_PLUS_BIAS"
        assert rec.final_price_cents == 900

    def test_no_data_fallback(self):
        engine = _engine()
        # Supplier event for item A, but query for item B
        engine.process_event(_event(
            item_id="item_a", source=SOURCE_SUPPLIER, price=1000, ts=1000,
        ))
        rec = engine.process_event(_event(
            item_id="item_b", source=SOURCE_SUPPLIER, price=500, ts=1000,
        ))
        # item_b now has supplier data
        assert rec.decision == "USED_SUPPLIER_PLUS_BIAS"


# -----------------------------------------------------------------------
# Test: Rule C — Learning (Bias Update)
# -----------------------------------------------------------------------
class TestRuleC:
    def test_bias_updates_on_accept(self):
        engine = _engine()
        engine.process_event(_event(source=SOURCE_SUPPLIER, price=1000, ts=1000))
        engine.process_event(_event(
            source=SOURCE_HUMAN, price=1300, ts=1100,
            outcome=OUTCOME_QUOTE_ACCEPTED,
        ))
        item = engine.state.items["copper_pipe_15mm"]
        assert item.accepted_human_deltas_cents == [300]
        assert item.bias_cents == 300

    def test_bias_is_median_of_deltas(self):
        engine = _engine()
        deltas = [300, 250, 400, 200, 350]
        for i, delta in enumerate(deltas):
            ts = 1000 + i * 200
            engine.process_event(_event(source=SOURCE_SUPPLIER, price=1000, ts=ts))
            engine.process_event(_event(
                source=SOURCE_HUMAN, price=1000 + delta, ts=ts + 50,
                outcome=OUTCOME_QUOTE_ACCEPTED,
            ))
        item = engine.state.items["copper_pipe_15mm"]
        assert len(item.accepted_human_deltas_cents) == 5
        assert item.bias_cents == 300  # median of [300,250,400,200,350]

    def test_rolling_window_keeps_last_5(self):
        engine = _engine()
        # Use supplier=5000 to avoid circuit breaker (human prices stay below 150%)
        for i in range(7):
            ts = 1000 + i * 200
            engine.process_event(_event(source=SOURCE_SUPPLIER, price=5000, ts=ts))
            engine.process_event(_event(
                source=SOURCE_HUMAN, price=5000 + (i + 1) * 100, ts=ts + 50,
                outcome=OUTCOME_QUOTE_ACCEPTED,
            ))
        item = engine.state.items["copper_pipe_15mm"]
        assert len(item.accepted_human_deltas_cents) == 5
        # Deltas: 100,200,300,400,500,600,700 → last 5 = [300,400,500,600,700]
        assert item.accepted_human_deltas_cents == [300, 400, 500, 600, 700]

    def test_no_learning_on_reject(self):
        engine = _engine()
        engine.process_event(_event(source=SOURCE_SUPPLIER, price=1000, ts=1000))
        engine.process_event(_event(
            source=SOURCE_HUMAN, price=1300, ts=1100,
            outcome=OUTCOME_QUOTE_REJECTED,
        ))
        assert "copper_pipe_15mm" not in engine.state.items


# -----------------------------------------------------------------------
# Test: Rule D — Time Decay
# -----------------------------------------------------------------------
class TestRuleD:
    def test_no_decay_within_7_days(self):
        engine = _engine()
        engine.state.items["copper_pipe_15mm"] = ItemState(
            bias_cents=300, last_updated_ts=1000,
            accepted_human_deltas_cents=[300],
        )
        engine.process_event(_event(source=SOURCE_SUPPLIER, price=1000, ts=1000))
        rec = engine.process_event(_event(
            source=SOURCE_HISTORIC, price=900,
            ts=1000 + 604799,  # just under 7 days
        ))
        assert rec.bias_applied_cents == 300

    def test_decay_after_7_days(self):
        engine = _engine()
        engine.state.items["copper_pipe_15mm"] = ItemState(
            bias_cents=300, last_updated_ts=1000,
            accepted_human_deltas_cents=[300],
        )
        engine.process_event(_event(source=SOURCE_SUPPLIER, price=1000, ts=1000))
        rec = engine.process_event(_event(
            source=SOURCE_HISTORIC, price=900,
            ts=1000 + 604801,  # just over 7 days
        ))
        # 300 // 2 = 150
        assert rec.bias_applied_cents == 150


# -----------------------------------------------------------------------
# Test: Rule E — Circuit Breaker
# -----------------------------------------------------------------------
class TestRuleE:
    def test_anomaly_rejected(self):
        engine = _engine()
        engine.process_event(_event(source=SOURCE_SUPPLIER, price=1000, ts=1000))
        rec = engine.process_event(_event(
            source=SOURCE_HUMAN, price=1600, ts=1100,
            outcome=OUTCOME_QUOTE_ACCEPTED,
        ))
        # 1600 > 1000 * 1.5 → anomaly
        assert "ANOMALY_REJECTED" in rec.flags
        assert rec.decision == "USED_SUPPLIER_PLUS_BIAS"
        # No learning
        assert "copper_pipe_15mm" not in engine.state.items

    def test_borderline_not_anomaly(self):
        engine = _engine()
        engine.process_event(_event(source=SOURCE_SUPPLIER, price=1000, ts=1000))
        rec = engine.process_event(_event(
            source=SOURCE_HUMAN, price=1500, ts=1100,
            outcome=OUTCOME_QUOTE_ACCEPTED,
        ))
        # 1500 is exactly 150% — NOT over, so not anomaly
        assert "ANOMALY_REJECTED" not in rec.flags
        assert rec.decision == "USED_HUMAN"

    def test_anomaly_does_not_learn(self):
        engine = _engine()
        engine.process_event(_event(source=SOURCE_SUPPLIER, price=1000, ts=1000))
        engine.process_event(_event(
            source=SOURCE_HUMAN, price=2000, ts=1100,
            outcome=OUTCOME_QUOTE_ACCEPTED,
        ))
        # No item state should be created
        assert "copper_pipe_15mm" not in engine.state.items


# -----------------------------------------------------------------------
# Test: State Hash
# -----------------------------------------------------------------------
class TestStateHash:
    def test_hash_changes_on_mutation(self):
        state1 = RulesState()
        h1 = compute_state_hash(state1)
        state1.items["x"] = ItemState(bias_cents=100, last_updated_ts=1)
        h2 = compute_state_hash(state1)
        assert h1 != h2

    def test_hash_deterministic(self):
        state = RulesState()
        state.items["a"] = ItemState(bias_cents=100, last_updated_ts=1)
        state.items["b"] = ItemState(bias_cents=200, last_updated_ts=2)
        h1 = compute_state_hash(state)
        h2 = compute_state_hash(state)
        assert h1 == h2


# -----------------------------------------------------------------------
# Test: Input Validation
# -----------------------------------------------------------------------
class TestInputValidation:
    def test_duplicate_event_id_raises(self):
        engine = _engine()
        engine.process_event(_event(event_id="dup-1", source=SOURCE_HISTORIC, price=100, ts=1000))
        with pytest.raises(ValueError, match="Duplicate event_id"):
            engine.process_event(_event(event_id="dup-1", source=SOURCE_HISTORIC, price=200, ts=2000))

    def test_negative_price_raises(self):
        engine = _engine()
        with pytest.raises(ValueError, match="Negative price_cents"):
            engine.process_event(_event(source=SOURCE_SUPPLIER, price=-100, ts=1000))

    def test_non_human_with_outcome_raises(self):
        engine = _engine()
        with pytest.raises(ValueError, match="Non-HUMAN event"):
            engine.process_event(Event(
                event_id="bad-1",
                timestamp=1000,
                item_id="x",
                source=SOURCE_SUPPLIER,
                price_cents=100,
                outcome=OUTCOME_QUOTE_ACCEPTED,
            ))

    def test_zero_price_is_valid(self):
        engine = _engine()
        rec = engine.process_event(_event(source=SOURCE_HISTORIC, price=0, ts=1000))
        assert rec.final_price_cents == 0

    def test_float_price_raises(self):
        with pytest.raises(TypeError, match="price_cents must be int"):
            Event(
                event_id="bad-2", timestamp=1000, item_id="x",
                source=SOURCE_SUPPLIER, price_cents=12.50,  # type: ignore
            )

    def test_invalid_source_raises(self):
        with pytest.raises(ValueError, match="Invalid source"):
            Event(
                event_id="bad-3", timestamp=1000, item_id="x",
                source="INVALID", price_cents=100,
            )
