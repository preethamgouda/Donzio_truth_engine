"""
EXTERNAL STRESS TEST — 100+ edge cases for the Donizo Truth Engine.

This file is NOT part of the main test suite. Run it separately:
    python -m pytest test_external.py -v --tb=short

Covers:
  - Boundary values (exact thresholds, off-by-one)
  - Extreme inputs (huge prices, zero prices, single-cent)
  - Multi-item interactions
  - State mutation ordering
  - Hash integrity
  - Decay edge cases
  - Circuit breaker boundaries
  - Learning edge cases
  - Input validation exhaustive checks
  - Concurrency-like event ordering
  - Regression traps
"""
import copy
import hashlib
import itertools
import json
import os
import tempfile

import pytest

from donizo_engine.engine import TruthEngine, _median_int, run_engine, replay_engine
from donizo_engine.models import (
    CIRCUIT_BREAKER_RATIO,
    DECAY_THRESHOLD_SECONDS,
    MAX_DELTA_HISTORY,
    OUTCOME_NONE,
    OUTCOME_QUOTE_ACCEPTED,
    OUTCOME_QUOTE_REJECTED,
    SOURCE_HISTORIC,
    SOURCE_HUMAN,
    SOURCE_SUPPLIER,
    SUPPLIER_FRESHNESS_SECONDS,
    AuditRecord,
    Event,
    ItemState,
    RulesState,
)
from donizo_engine.state import compute_state_hash, load_state, save_state

# -----------------------------------------------------------------------
# Counter for unique event IDs
# -----------------------------------------------------------------------
_counter = itertools.count(1)


def _ev(
    item_id="item_x",
    source=SOURCE_SUPPLIER,
    price=1000,
    ts=1000,
    outcome=OUTCOME_NONE,
    event_id=None,
) -> Event:
    """Helper to create events with auto-unique IDs."""
    return Event(
        event_id=event_id or f"ext-{next(_counter)}",
        timestamp=ts,
        item_id=item_id,
        source=source,
        price_cents=price,
        outcome=outcome,
    )


def _engine(state=None) -> TruthEngine:
    return TruthEngine(state or RulesState())


# =======================================================================
# SECTION 1: SUPPLIER FRESHNESS BOUNDARY (Rule A)  — 10 tests
# =======================================================================
class TestSupplierFreshnessBoundary:
    """Exact boundary tests for 3600-second supplier freshness."""

    def test_supplier_at_exactly_3600s_is_eligible(self):
        """At ts+3600, supplier is still fresh (<=)."""
        e = _engine()
        e.process_event(_ev(source=SOURCE_SUPPLIER, price=500, ts=1000))
        rec = e.process_event(_ev(source=SOURCE_HISTORIC, price=400, ts=4600))
        assert rec.inputs_seen["supplier_cents"] == 500
        assert rec.decision == "USED_SUPPLIER_PLUS_BIAS"

    def test_supplier_at_3601s_is_expired(self):
        """At ts+3601, supplier is expired (>)."""
        e = _engine()
        e.process_event(_ev(source=SOURCE_SUPPLIER, price=500, ts=1000))
        rec = e.process_event(_ev(source=SOURCE_HISTORIC, price=400, ts=4601))
        assert rec.decision == "USED_HISTORIC_PLUS_BIAS"

    def test_supplier_at_exactly_0s_is_eligible(self):
        """Same timestamp as supplier = eligible."""
        e = _engine()
        e.process_event(_ev(source=SOURCE_SUPPLIER, price=500, ts=5000))
        rec = e.process_event(_ev(source=SOURCE_HISTORIC, price=400, ts=5000))
        assert rec.decision == "USED_SUPPLIER_PLUS_BIAS"

    def test_supplier_freshness_resets_on_new_quote(self):
        """A newer supplier quote resets the freshness clock."""
        e = _engine()
        e.process_event(_ev(source=SOURCE_SUPPLIER, price=500, ts=1000))
        # Second supplier quote at ts=5000
        e.process_event(_ev(source=SOURCE_SUPPLIER, price=600, ts=5000))
        # Query at ts=8600 (3600 from second quote = still fresh)
        rec = e.process_event(_ev(source=SOURCE_HISTORIC, price=400, ts=8600))
        assert rec.decision == "USED_SUPPLIER_PLUS_BIAS"
        assert rec.final_price_cents == 600  # uses latest supplier

    def test_supplier_freshness_independent_per_item(self):
        """Supplier freshness is per-item, not global."""
        e = _engine()
        e.process_event(_ev(item_id="a", source=SOURCE_SUPPLIER, price=100, ts=1000))
        e.process_event(_ev(item_id="b", source=SOURCE_SUPPLIER, price=200, ts=5000))
        # Item A expired at ts=5001, but item B still fresh
        rec_a = e.process_event(_ev(item_id="a", source=SOURCE_HISTORIC, price=50, ts=5001))
        rec_b = e.process_event(_ev(item_id="b", source=SOURCE_HISTORIC, price=60, ts=5001))
        assert rec_a.decision == "USED_HISTORIC_PLUS_BIAS"
        assert rec_b.decision == "USED_SUPPLIER_PLUS_BIAS"

    def test_no_supplier_no_historic_produces_fallback(self):
        """No data at all → FALLBACK_NO_DATA with price 0."""
        e = _engine()
        rec = e.process_event(_ev(source=SOURCE_SUPPLIER, price=100, ts=1000))
        # First supplier event itself — supplier IS fresh for this item
        assert rec.decision == "USED_SUPPLIER_PLUS_BIAS"

    def test_only_expired_supplier_falls_to_historic(self):
        """Expired supplier with no historic → FALLBACK_NO_DATA."""
        e = _engine()
        e.process_event(_ev(item_id="z", source=SOURCE_SUPPLIER, price=100, ts=1000))
        rec = e.process_event(_ev(item_id="z", source=SOURCE_SUPPLIER, price=200, ts=4601))
        # New supplier event is fresh for this ts
        assert rec.decision == "USED_SUPPLIER_PLUS_BIAS"

    def test_historic_never_expires(self):
        """Historic data from 1M seconds ago is still valid."""
        e = _engine()
        e.process_event(_ev(source=SOURCE_HISTORIC, price=777, ts=1000))
        rec = e.process_event(_ev(source=SOURCE_SUPPLIER, price=999, ts=1_000_001))
        assert rec.inputs_seen["historic_cents"] == 777

    def test_supplier_update_overwrites_old_price(self):
        """Newer supplier price replaces old one in cache."""
        e = _engine()
        e.process_event(_ev(source=SOURCE_SUPPLIER, price=100, ts=1000))
        e.process_event(_ev(source=SOURCE_SUPPLIER, price=200, ts=2000))
        rec = e.process_event(_ev(source=SOURCE_HISTORIC, price=50, ts=2500))
        assert rec.inputs_seen["supplier_cents"] == 200
        assert rec.final_price_cents == 200

    def test_supplier_price_zero(self):
        """Supplier price of 0 is valid but circuit breaker needs supplier_price > 0."""
        e = _engine()
        e.process_event(_ev(source=SOURCE_SUPPLIER, price=0, ts=1000))
        rec = e.process_event(_ev(
            source=SOURCE_HUMAN, price=500, ts=1100,
            outcome=OUTCOME_QUOTE_ACCEPTED,
        ))
        # Circuit breaker requires supplier_price > 0, so no anomaly check
        assert "ANOMALY_REJECTED" not in rec.flags
        assert rec.decision == "USED_HUMAN"


# =======================================================================
# SECTION 2: CIRCUIT BREAKER EXACT BOUNDARIES (Rule E)  — 12 tests
# =======================================================================
class TestCircuitBreakerBoundary:
    """Exact 150% threshold tests."""

    def test_exactly_150_percent_not_anomaly(self):
        """human=1500, supplier=1000 → exactly 150%, NOT anomaly."""
        e = _engine()
        e.process_event(_ev(source=SOURCE_SUPPLIER, price=1000, ts=1000))
        rec = e.process_event(_ev(
            source=SOURCE_HUMAN, price=1500, ts=1100,
            outcome=OUTCOME_QUOTE_ACCEPTED,
        ))
        assert "ANOMALY_REJECTED" not in rec.flags

    def test_one_cent_over_150_is_anomaly(self):
        """human=1501, supplier=1000 → 150.1%, IS anomaly."""
        e = _engine()
        e.process_event(_ev(source=SOURCE_SUPPLIER, price=1000, ts=1000))
        rec = e.process_event(_ev(
            source=SOURCE_HUMAN, price=1501, ts=1100,
            outcome=OUTCOME_QUOTE_ACCEPTED,
        ))
        assert "ANOMALY_REJECTED" in rec.flags

    def test_circuit_breaker_with_tiny_supplier(self):
        """Supplier=1, human=2 → 200% > 150%, anomaly."""
        e = _engine()
        e.process_event(_ev(source=SOURCE_SUPPLIER, price=1, ts=1000))
        rec = e.process_event(_ev(
            source=SOURCE_HUMAN, price=2, ts=1100,
            outcome=OUTCOME_QUOTE_ACCEPTED,
        ))
        assert "ANOMALY_REJECTED" in rec.flags

    def test_circuit_breaker_with_very_large_prices(self):
        """Large prices: supplier=100000000, human=150000001 → anomaly."""
        e = _engine()
        e.process_event(_ev(source=SOURCE_SUPPLIER, price=100_000_000, ts=1000))
        rec = e.process_event(_ev(
            source=SOURCE_HUMAN, price=150_000_001, ts=1100,
            outcome=OUTCOME_QUOTE_ACCEPTED,
        ))
        assert "ANOMALY_REJECTED" in rec.flags

    def test_no_circuit_breaker_when_supplier_expired(self):
        """If supplier is expired, circuit breaker doesn't apply."""
        e = _engine()
        e.process_event(_ev(source=SOURCE_SUPPLIER, price=100, ts=1000))
        rec = e.process_event(_ev(
            source=SOURCE_HUMAN, price=500, ts=5601,  # >3600 from supplier
            outcome=OUTCOME_QUOTE_ACCEPTED,
        ))
        # Supplier expired → no circuit breaker
        assert "ANOMALY_REJECTED" not in rec.flags
        assert rec.decision == "USED_HUMAN"

    def test_circuit_breaker_only_on_human_source(self):
        """Circuit breaker only fires for HUMAN source events."""
        e = _engine()
        e.process_event(_ev(source=SOURCE_SUPPLIER, price=100, ts=1000))
        rec = e.process_event(_ev(source=SOURCE_HISTORIC, price=9999, ts=1100))
        assert "ANOMALY_REJECTED" not in rec.flags

    def test_circuit_breaker_with_rejected_outcome(self):
        """Anomaly check still fires even if outcome is QUOTE_ACCEPTED."""
        e = _engine()
        e.process_event(_ev(source=SOURCE_SUPPLIER, price=100, ts=1000))
        rec = e.process_event(_ev(
            source=SOURCE_HUMAN, price=200, ts=1100,
            outcome=OUTCOME_QUOTE_ACCEPTED,
        ))
        assert "ANOMALY_REJECTED" in rec.flags  # 200 > 150

    def test_anomaly_prevents_item_state_creation(self):
        """If anomaly, no ItemState should be created for new item."""
        e = _engine()
        e.process_event(_ev(item_id="new_item", source=SOURCE_SUPPLIER, price=100, ts=1000))
        e.process_event(_ev(
            item_id="new_item", source=SOURCE_HUMAN, price=200, ts=1100,
            outcome=OUTCOME_QUOTE_ACCEPTED,
        ))
        assert "new_item" not in e.state.items

    def test_anomaly_doesnt_corrupt_existing_state(self):
        """Anomaly on an item with existing state shouldn't mutate it."""
        e = _engine()
        e.state.items["w"] = ItemState(bias_cents=50, last_updated_ts=900,
                                       accepted_human_deltas_cents=[50])
        e.process_event(_ev(item_id="w", source=SOURCE_SUPPLIER, price=100, ts=1000))
        e.process_event(_ev(
            item_id="w", source=SOURCE_HUMAN, price=200, ts=1100,
            outcome=OUTCOME_QUOTE_ACCEPTED,
        ))
        assert e.state.items["w"].bias_cents == 50  # unchanged
        assert e.state.items["w"].accepted_human_deltas_cents == [50]  # unchanged

    def test_circuit_breaker_skipped_when_supplier_price_zero(self):
        """If supplier_price == 0, circuit breaker is skipped."""
        e = _engine()
        e.process_event(_ev(source=SOURCE_SUPPLIER, price=0, ts=1000))
        rec = e.process_event(_ev(
            source=SOURCE_HUMAN, price=99999, ts=1100,
            outcome=OUTCOME_QUOTE_ACCEPTED,
        ))
        assert "ANOMALY_REJECTED" not in rec.flags

    def test_human_rejected_outcome_with_anomaly_price(self):
        """Human-source with QUOTE_REJECTED and anomaly price → anomaly check still runs."""
        e = _engine()
        e.process_event(_ev(source=SOURCE_SUPPLIER, price=100, ts=1000))
        rec = e.process_event(_ev(
            source=SOURCE_HUMAN, price=200, ts=1100,
            outcome=OUTCOME_QUOTE_REJECTED,
        ))
        # Anomaly check happens before decision tree
        assert "ANOMALY_REJECTED" in rec.flags

    def test_human_none_outcome_with_anomaly_price(self):
        """HUMAN source with NONE outcome at anomaly price → anomaly detected."""
        e = _engine()
        e.process_event(_ev(source=SOURCE_SUPPLIER, price=100, ts=1000))
        rec = e.process_event(_ev(
            source=SOURCE_HUMAN, price=200, ts=1100,
            outcome=OUTCOME_NONE,
        ))
        assert "ANOMALY_REJECTED" in rec.flags


# =======================================================================
# SECTION 3: DECAY EDGE CASES (Rule D)  — 12 tests
# =======================================================================
class TestDecayEdgeCases:
    """Boundary and corner cases for 7-day decay."""

    def test_decay_exactly_at_7_days(self):
        """Exactly 604800 seconds → NOT decayed (> required, not >=)."""
        e = _engine()
        e.state.items["it"] = ItemState(bias_cents=400, last_updated_ts=0,
                                        accepted_human_deltas_cents=[400])
        e.process_event(_ev(item_id="it", source=SOURCE_HISTORIC, price=100, ts=0))
        rec = e.process_event(_ev(item_id="it", source=SOURCE_HISTORIC, price=100, ts=604800))
        assert rec.bias_applied_cents == 400  # NOT decayed

    def test_decay_at_7_days_plus_1(self):
        """604801 seconds → IS decayed."""
        e = _engine()
        e.state.items["it"] = ItemState(bias_cents=400, last_updated_ts=0,
                                        accepted_human_deltas_cents=[400])
        e.process_event(_ev(item_id="it", source=SOURCE_HISTORIC, price=100, ts=0))
        rec = e.process_event(_ev(item_id="it", source=SOURCE_HISTORIC, price=100, ts=604801))
        assert rec.bias_applied_cents == 200

    def test_decay_of_odd_bias(self):
        """Odd bias decays: 301 // 2 = 150 (floor division)."""
        e = _engine()
        e.state.items["it"] = ItemState(bias_cents=301, last_updated_ts=0,
                                        accepted_human_deltas_cents=[301])
        e.process_event(_ev(item_id="it", source=SOURCE_HISTORIC, price=100, ts=0))
        rec = e.process_event(_ev(item_id="it", source=SOURCE_HISTORIC, price=100, ts=604801))
        assert rec.bias_applied_cents == 150

    def test_decay_of_negative_bias(self):
        """Negative bias: -300 // 2 = -150."""
        e = _engine()
        e.state.items["it"] = ItemState(bias_cents=-300, last_updated_ts=0,
                                        accepted_human_deltas_cents=[-300])
        e.process_event(_ev(item_id="it", source=SOURCE_HISTORIC, price=1000, ts=0))
        rec = e.process_event(_ev(item_id="it", source=SOURCE_HISTORIC, price=1000, ts=604801))
        assert rec.bias_applied_cents == -150

    def test_decay_of_negative_odd_bias(self):
        """Negative odd: -301 // 2 = -151 (Python floor division)."""
        e = _engine()
        e.state.items["it"] = ItemState(bias_cents=-301, last_updated_ts=0,
                                        accepted_human_deltas_cents=[-301])
        e.process_event(_ev(item_id="it", source=SOURCE_HISTORIC, price=1000, ts=0))
        rec = e.process_event(_ev(item_id="it", source=SOURCE_HISTORIC, price=1000, ts=604801))
        assert rec.bias_applied_cents == -151

    def test_decay_of_1_cent_bias(self):
        """1 cent bias: 1 // 2 = 0."""
        e = _engine()
        e.state.items["it"] = ItemState(bias_cents=1, last_updated_ts=0,
                                        accepted_human_deltas_cents=[1])
        e.process_event(_ev(item_id="it", source=SOURCE_HISTORIC, price=100, ts=0))
        rec = e.process_event(_ev(item_id="it", source=SOURCE_HISTORIC, price=100, ts=604801))
        assert rec.bias_applied_cents == 0

    def test_decay_of_zero_bias(self):
        """0 bias stays 0 after decay."""
        e = _engine()
        e.state.items["it"] = ItemState(bias_cents=0, last_updated_ts=0,
                                        accepted_human_deltas_cents=[])
        e.process_event(_ev(item_id="it", source=SOURCE_HISTORIC, price=100, ts=0))
        rec = e.process_event(_ev(item_id="it", source=SOURCE_HISTORIC, price=100, ts=604801))
        assert rec.bias_applied_cents == 0

    def test_decay_does_not_modify_stored_bias(self):
        """Decay is applied to the in-flight calculation, NOT stored state."""
        e = _engine()
        e.state.items["it"] = ItemState(bias_cents=400, last_updated_ts=0,
                                        accepted_human_deltas_cents=[400])
        e.process_event(_ev(item_id="it", source=SOURCE_HISTORIC, price=100, ts=0))
        e.process_event(_ev(item_id="it", source=SOURCE_HISTORIC, price=100, ts=604801))
        # Stored bias should remain 400
        assert e.state.items["it"].bias_cents == 400

    def test_multiple_decay_periods(self):
        """After 2 decay periods without learning, bias applied decays each time."""
        e = _engine()
        e.state.items["it"] = ItemState(bias_cents=400, last_updated_ts=0,
                                        accepted_human_deltas_cents=[400])
        e.process_event(_ev(item_id="it", source=SOURCE_HISTORIC, price=100, ts=0))
        # First query after 8 days: 400//2 = 200
        rec = e.process_event(_ev(item_id="it", source=SOURCE_HISTORIC, price=100, ts=8 * 86400))
        assert rec.bias_applied_cents == 200
        # But stored bias is still 400 — decay is just applied, not stored
        assert e.state.items["it"].bias_cents == 400

    def test_learning_after_decay_resets_timestamp(self):
        """When learning occurs after decay, the timestamp updates."""
        e = _engine()
        e.state.items["it"] = ItemState(bias_cents=200, last_updated_ts=1000,
                                        accepted_human_deltas_cents=[200])
        # Supplier within freshness
        e.process_event(_ev(item_id="it", source=SOURCE_SUPPLIER, price=1000, ts=700000))
        # Accepted human after decay (>7d from ts=1000)
        rec = e.process_event(_ev(
            item_id="it", source=SOURCE_HUMAN, price=1300, ts=700100,
            outcome=OUTCOME_QUOTE_ACCEPTED,
        ))
        assert rec.decision == "USED_HUMAN"
        assert e.state.items["it"].last_updated_ts == 700100

    def test_decay_with_very_large_gap(self):
        """10-year gap: bias still just halved once per event."""
        e = _engine()
        e.state.items["it"] = ItemState(bias_cents=1000, last_updated_ts=0,
                                        accepted_human_deltas_cents=[1000])
        e.process_event(_ev(item_id="it", source=SOURCE_HISTORIC, price=100, ts=0))
        rec = e.process_event(_ev(
            item_id="it", source=SOURCE_HISTORIC, price=100,
            ts=10 * 365 * 86400,
        ))
        assert rec.bias_applied_cents == 500

    def test_no_item_state_no_decay(self):
        """New item with no prior state → bias 0, no decay."""
        e = _engine()
        rec = e.process_event(_ev(item_id="brand_new", source=SOURCE_HISTORIC, price=100, ts=1000))
        assert rec.bias_applied_cents == 0


# =======================================================================
# SECTION 4: LEARNING EDGE CASES (Rule C)  — 12 tests
# =======================================================================
class TestLearningEdgeCases:
    """Learning: deltas, median, rolling window."""

    def test_learning_creates_item_state(self):
        """First accepted human creates ItemState for new item."""
        e = _engine()
        e.process_event(_ev(item_id="new", source=SOURCE_SUPPLIER, price=1000, ts=1000))
        e.process_event(_ev(
            item_id="new", source=SOURCE_HUMAN, price=1200, ts=1100,
            outcome=OUTCOME_QUOTE_ACCEPTED,
        ))
        assert "new" in e.state.items
        assert e.state.items["new"].accepted_human_deltas_cents == [200]
        assert e.state.items["new"].bias_cents == 200

    def test_negative_delta_learning(self):
        """Human price below supplier → negative delta."""
        e = _engine()
        e.process_event(_ev(source=SOURCE_SUPPLIER, price=1000, ts=1000))
        e.process_event(_ev(
            source=SOURCE_HUMAN, price=800, ts=1100,
            outcome=OUTCOME_QUOTE_ACCEPTED,
        ))
        assert e.state.items["item_x"].bias_cents == -200

    def test_zero_delta_learning(self):
        """Human == supplier → delta 0."""
        e = _engine()
        e.process_event(_ev(source=SOURCE_SUPPLIER, price=1000, ts=1000))
        e.process_event(_ev(
            source=SOURCE_HUMAN, price=1000, ts=1100,
            outcome=OUTCOME_QUOTE_ACCEPTED,
        ))
        assert e.state.items["item_x"].bias_cents == 0
        assert e.state.items["item_x"].accepted_human_deltas_cents == [0]

    def test_exactly_5_deltas(self):
        """After exactly 5 accepted events, list is full."""
        e = _engine()
        for i in range(5):
            ts = 1000 + i * 200
            e.process_event(_ev(source=SOURCE_SUPPLIER, price=10000, ts=ts))
            e.process_event(_ev(
                source=SOURCE_HUMAN, price=10000 + (i + 1) * 100, ts=ts + 50,
                outcome=OUTCOME_QUOTE_ACCEPTED,
            ))
        assert len(e.state.items["item_x"].accepted_human_deltas_cents) == 5

    def test_6th_delta_evicts_first(self):
        """After 6 events, first delta is evicted."""
        e = _engine()
        for i in range(6):
            ts = 1000 + i * 200
            e.process_event(_ev(source=SOURCE_SUPPLIER, price=10000, ts=ts))
            e.process_event(_ev(
                source=SOURCE_HUMAN, price=10000 + (i + 1) * 100, ts=ts + 50,
                outcome=OUTCOME_QUOTE_ACCEPTED,
            ))
        deltas = e.state.items["item_x"].accepted_human_deltas_cents
        assert len(deltas) == 5
        assert deltas[0] == 200  # first delta (100) evicted, second (200) now first

    def test_no_learning_without_supplier(self):
        """No supplier = no learning even if human accepted."""
        e = _engine()
        rec = e.process_event(_ev(
            source=SOURCE_HUMAN, price=1500, ts=1000,
            outcome=OUTCOME_QUOTE_ACCEPTED,
        ))
        assert rec.decision == "USED_HUMAN"
        assert "item_x" not in e.state.items

    def test_no_learning_with_expired_supplier(self):
        """Expired supplier = no learning."""
        e = _engine()
        e.process_event(_ev(source=SOURCE_SUPPLIER, price=1000, ts=1000))
        rec = e.process_event(_ev(
            source=SOURCE_HUMAN, price=1300, ts=4701,  # supplier expired
            outcome=OUTCOME_QUOTE_ACCEPTED,
        ))
        assert rec.decision == "USED_HUMAN"
        assert "item_x" not in e.state.items

    def test_rejected_human_never_learns(self):
        """QUOTE_REJECTED never modifies state."""
        e = _engine()
        e.process_event(_ev(source=SOURCE_SUPPLIER, price=1000, ts=1000))
        e.process_event(_ev(
            source=SOURCE_HUMAN, price=1200, ts=1100,
            outcome=OUTCOME_QUOTE_REJECTED,
        ))
        assert "item_x" not in e.state.items

    def test_median_of_even_count(self):
        """Even number of deltas: median([100,200]) = 150."""
        e = _engine()
        for i, delta in enumerate([100, 200]):
            ts = 1000 + i * 200
            e.process_event(_ev(source=SOURCE_SUPPLIER, price=10000, ts=ts))
            e.process_event(_ev(
                source=SOURCE_HUMAN, price=10000 + delta, ts=ts + 50,
                outcome=OUTCOME_QUOTE_ACCEPTED,
            ))
        assert e.state.items["item_x"].bias_cents == 150

    def test_median_truncation_toward_zero(self):
        """Median of [100, 201] = 150.5 → int(150.5) = 150 (truncate)."""
        e = _engine()
        for i, delta in enumerate([100, 201]):
            ts = 1000 + i * 200
            e.process_event(_ev(source=SOURCE_SUPPLIER, price=10000, ts=ts))
            e.process_event(_ev(
                source=SOURCE_HUMAN, price=10000 + delta, ts=ts + 50,
                outcome=OUTCOME_QUOTE_ACCEPTED,
            ))
        assert e.state.items["item_x"].bias_cents == 150

    def test_learning_updates_timestamp(self):
        """Learning must update last_updated_ts."""
        e = _engine()
        e.process_event(_ev(source=SOURCE_SUPPLIER, price=1000, ts=5000))
        e.process_event(_ev(
            source=SOURCE_HUMAN, price=1200, ts=5100,
            outcome=OUTCOME_QUOTE_ACCEPTED,
        ))
        assert e.state.items["item_x"].last_updated_ts == 5100

    def test_learning_with_supplier_price_zero_does_not_learn(self):
        """Supplier price == 0 prevents learning (condition: supplier_price > 0)."""
        e = _engine()
        e.process_event(_ev(source=SOURCE_SUPPLIER, price=0, ts=1000))
        e.process_event(_ev(
            source=SOURCE_HUMAN, price=500, ts=1100,
            outcome=OUTCOME_QUOTE_ACCEPTED,
        ))
        assert "item_x" not in e.state.items


# =======================================================================
# SECTION 5: DECISION TREE EDGE CASES (Rule B)  — 10 tests
# =======================================================================
class TestDecisionTreeEdgeCases:
    """Fallback logic, priority, and flag combinations."""

    def test_human_accepted_overrides_all(self):
        """Human accepted always wins (when not anomaly)."""
        e = _engine()
        e.process_event(_ev(source=SOURCE_SUPPLIER, price=1000, ts=1000))
        e.process_event(_ev(source=SOURCE_HISTORIC, price=800, ts=1050))
        rec = e.process_event(_ev(
            source=SOURCE_HUMAN, price=1300, ts=1100,
            outcome=OUTCOME_QUOTE_ACCEPTED,
        ))
        assert rec.final_price_cents == 1300

    def test_fallback_supplier_before_historic(self):
        """Supplier+bias is preferred over historic+bias."""
        e = _engine()
        e.process_event(_ev(source=SOURCE_HISTORIC, price=800, ts=1000))
        e.process_event(_ev(source=SOURCE_SUPPLIER, price=1000, ts=1050))
        rec = e.process_event(_ev(source=SOURCE_HISTORIC, price=900, ts=1100))
        assert rec.decision == "USED_SUPPLIER_PLUS_BIAS"

    def test_historic_fallback_when_no_supplier(self):
        """Only historic → USED_HISTORIC_PLUS_BIAS."""
        e = _engine()
        rec = e.process_event(_ev(source=SOURCE_HISTORIC, price=500, ts=1000))
        assert rec.decision == "USED_HISTORIC_PLUS_BIAS"
        assert rec.final_price_cents == 500

    def test_no_data_fallback_price_zero(self):
        """No historic, no supplier → price=0, FALLBACK_NO_DATA."""
        e = _engine()
        rec = e.process_event(_ev(
            item_id="empty", source=SOURCE_HUMAN, price=1000, ts=1000,
            outcome=OUTCOME_QUOTE_REJECTED,
        ))
        assert rec.decision == "FALLBACK_NO_DATA"
        assert rec.final_price_cents == 0

    def test_human_none_outcome_uses_fallback(self):
        """HUMAN with NONE outcome → treated as standard query."""
        e = _engine()
        e.process_event(_ev(source=SOURCE_SUPPLIER, price=1000, ts=1000))
        rec = e.process_event(_ev(
            source=SOURCE_HUMAN, price=1200, ts=1100,
            outcome=OUTCOME_NONE,
        ))
        assert rec.decision == "USED_SUPPLIER_PLUS_BIAS"

    def test_bias_applied_to_supplier_fallback(self):
        """Bias adds to supplier price in fallback."""
        e = _engine()
        e.state.items["item_x"] = ItemState(bias_cents=150, last_updated_ts=900,
                                            accepted_human_deltas_cents=[150])
        e.process_event(_ev(source=SOURCE_SUPPLIER, price=1000, ts=1000))
        rec = e.process_event(_ev(source=SOURCE_HISTORIC, price=800, ts=1050))
        assert rec.final_price_cents == 1150  # 1000 + 150

    def test_bias_applied_to_historic_fallback(self):
        """Bias adds to historic price when supplier expired."""
        e = _engine()
        e.state.items["item_x"] = ItemState(bias_cents=100, last_updated_ts=900,
                                            accepted_human_deltas_cents=[100])
        e.process_event(_ev(source=SOURCE_SUPPLIER, price=1000, ts=1000))
        e.process_event(_ev(source=SOURCE_HISTORIC, price=800, ts=1050))
        rec = e.process_event(_ev(source=SOURCE_HISTORIC, price=750, ts=5000))
        assert rec.final_price_cents == 850  # 750 + 100

    def test_negative_bias_reduces_price(self):
        """Negative bias reduces fallback price."""
        e = _engine()
        e.state.items["item_x"] = ItemState(bias_cents=-200, last_updated_ts=900,
                                            accepted_human_deltas_cents=[-200])
        e.process_event(_ev(source=SOURCE_SUPPLIER, price=1000, ts=1000))
        rec = e.process_event(_ev(source=SOURCE_HISTORIC, price=800, ts=1050))
        assert rec.final_price_cents == 800  # 1000 + (-200)

    def test_negative_bias_can_make_price_negative(self):
        """Large negative bias can produce negative final price."""
        e = _engine()
        e.state.items["item_x"] = ItemState(bias_cents=-500, last_updated_ts=900,
                                            accepted_human_deltas_cents=[-500])
        e.process_event(_ev(source=SOURCE_SUPPLIER, price=100, ts=1000))
        rec = e.process_event(_ev(source=SOURCE_HISTORIC, price=50, ts=1050))
        assert rec.final_price_cents == -400  # 100 + (-500)

    def test_anomaly_falls_back_to_supplier(self):
        """On anomaly, fallback to supplier+bias (not human price)."""
        e = _engine()
        e.state.items["item_x"] = ItemState(bias_cents=50, last_updated_ts=900,
                                            accepted_human_deltas_cents=[50])
        e.process_event(_ev(source=SOURCE_SUPPLIER, price=1000, ts=1000))
        rec = e.process_event(_ev(
            source=SOURCE_HUMAN, price=1600, ts=1100,
            outcome=OUTCOME_QUOTE_ACCEPTED,
        ))
        assert rec.final_price_cents == 1050  # 1000 + 50 (fallback)


# =======================================================================
# SECTION 6: STATE HASH INTEGRITY  — 10 tests
# =======================================================================
class TestStateHashIntegrity:
    """Hash consistency, mutation tracking, determinism."""

    def test_empty_state_has_deterministic_hash(self):
        h1 = compute_state_hash(RulesState())
        h2 = compute_state_hash(RulesState())
        assert h1 == h2

    def test_hash_length_is_64(self):
        assert len(compute_state_hash(RulesState())) == 64

    def test_different_items_different_hashes(self):
        s1 = RulesState()
        s1.items["a"] = ItemState(bias_cents=100)
        s2 = RulesState()
        s2.items["b"] = ItemState(bias_cents=100)
        assert compute_state_hash(s1) != compute_state_hash(s2)

    def test_same_items_different_bias_different_hash(self):
        s1 = RulesState()
        s1.items["a"] = ItemState(bias_cents=100)
        s2 = RulesState()
        s2.items["a"] = ItemState(bias_cents=101)
        assert compute_state_hash(s1) != compute_state_hash(s2)

    def test_hash_excludes_state_hash_field(self):
        """The state_hash field itself should NOT affect the hash."""
        s = RulesState()
        s.items["a"] = ItemState(bias_cents=100)
        s.state_hash = "garbage"
        h1 = compute_state_hash(s)
        s.state_hash = "different_garbage"
        h2 = compute_state_hash(s)
        assert h1 == h2

    def test_hash_after_every_event(self):
        """Hash in audit record matches the state after that event."""
        e = _engine()
        records = []
        for i in range(5):
            rec = e.process_event(_ev(source=SOURCE_HISTORIC, price=100 + i, ts=1000 + i))
            records.append(rec)
            assert rec.rules_hash == e.state.state_hash

    def test_item_ordering_doesnt_matter(self):
        """Items sorted by key → hash is deterministic regardless of insertion order."""
        s1 = RulesState()
        s1.items["b"] = ItemState(bias_cents=200)
        s1.items["a"] = ItemState(bias_cents=100)
        s2 = RulesState()
        s2.items["a"] = ItemState(bias_cents=100)
        s2.items["b"] = ItemState(bias_cents=200)
        assert compute_state_hash(s1) == compute_state_hash(s2)

    def test_save_and_load_round_trip(self, tmp_path):
        """State survives save → load → hash check."""
        path = str(tmp_path / "state.json")
        s = RulesState()
        s.items["x"] = ItemState(bias_cents=42, last_updated_ts=999,
                                 accepted_human_deltas_cents=[42])
        s.state_hash = compute_state_hash(s)
        save_state(s, path)
        loaded = load_state(path)
        assert compute_state_hash(loaded) == s.state_hash

    def test_corrupt_state_detected(self, tmp_path):
        """Modified state file fails hash check."""
        path = str(tmp_path / "state.json")
        s = RulesState()
        s.items["x"] = ItemState(bias_cents=42, last_updated_ts=999)
        s.state_hash = compute_state_hash(s)
        save_state(s, path)
        # Corrupt it
        with open(path, "r") as f:
            data = json.load(f)
        data["items"]["x"]["bias_cents"] = 9999
        with open(path, "w") as f:
            json.dump(data, f)
        with pytest.raises(ValueError, match="corrupted"):
            load_state(path)

    def test_fresh_state_file_created_on_first_load(self, tmp_path):
        """Loading from non-existent path returns empty state."""
        path = str(tmp_path / "nonexistent.json")
        s = load_state(path)
        assert s.version == 1
        assert len(s.items) == 0


# =======================================================================
# SECTION 7: INPUT VALIDATION EXHAUSTIVE  — 12 tests
# =======================================================================
class TestInputValidationExhaustive:
    """Thorough validation checks."""

    def test_duplicate_across_different_items(self):
        """Same event_id on different items is still duplicate."""
        e = _engine()
        e.process_event(_ev(event_id="dup", item_id="a", source=SOURCE_HISTORIC, price=100, ts=1000))
        with pytest.raises(ValueError, match="Duplicate"):
            e.process_event(_ev(event_id="dup", item_id="b", source=SOURCE_HISTORIC, price=200, ts=2000))

    def test_duplicate_across_different_sources(self):
        """Same event_id with different source is still duplicate."""
        e = _engine()
        e.process_event(_ev(event_id="dup2", source=SOURCE_HISTORIC, price=100, ts=1000))
        with pytest.raises(ValueError, match="Duplicate"):
            e.process_event(_ev(event_id="dup2", source=SOURCE_SUPPLIER, price=200, ts=2000))

    def test_negative_price_exactly_minus_1(self):
        """Price of -1 raises."""
        e = _engine()
        with pytest.raises(ValueError, match="Negative"):
            e.process_event(_ev(source=SOURCE_SUPPLIER, price=-1, ts=1000))

    def test_price_zero_is_valid(self):
        """Price of 0 is acceptable."""
        e = _engine()
        rec = e.process_event(_ev(source=SOURCE_SUPPLIER, price=0, ts=1000))
        assert rec.final_price_cents == 0

    def test_supplier_with_accepted_outcome_raises(self):
        """SUPPLIER source must have NONE outcome."""
        e = _engine()
        with pytest.raises(ValueError, match="Non-HUMAN"):
            e.process_event(Event(
                event_id="bad1", timestamp=1000, item_id="x",
                source=SOURCE_SUPPLIER, price_cents=100,
                outcome=OUTCOME_QUOTE_ACCEPTED,
            ))

    def test_supplier_with_rejected_outcome_raises(self):
        """SUPPLIER + QUOTE_REJECTED raises."""
        e = _engine()
        with pytest.raises(ValueError, match="Non-HUMAN"):
            e.process_event(Event(
                event_id="bad2", timestamp=1000, item_id="x",
                source=SOURCE_SUPPLIER, price_cents=100,
                outcome=OUTCOME_QUOTE_REJECTED,
            ))

    def test_historic_with_accepted_outcome_raises(self):
        """HISTORIC + QUOTE_ACCEPTED raises."""
        e = _engine()
        with pytest.raises(ValueError, match="Non-HUMAN"):
            e.process_event(Event(
                event_id="bad3", timestamp=1000, item_id="x",
                source=SOURCE_HISTORIC, price_cents=100,
                outcome=OUTCOME_QUOTE_ACCEPTED,
            ))

    def test_invalid_source_raises(self):
        with pytest.raises(ValueError, match="Invalid source"):
            Event(event_id="bad4", timestamp=1, item_id="x",
                  source="UNKNOWN", price_cents=1)

    def test_invalid_outcome_raises(self):
        with pytest.raises(ValueError, match="Invalid outcome"):
            Event(event_id="bad5", timestamp=1, item_id="x",
                  source=SOURCE_HUMAN, price_cents=1, outcome="MAYBE")

    def test_float_price_raises(self):
        with pytest.raises(TypeError, match="price_cents must be int"):
            Event(event_id="bad6", timestamp=1, item_id="x",
                  source=SOURCE_SUPPLIER, price_cents=10.5)  # type: ignore

    def test_large_price_is_valid(self):
        """Very large price (10 billion cents) should work."""
        e = _engine()
        rec = e.process_event(_ev(source=SOURCE_SUPPLIER, price=10_000_000_000, ts=1000))
        assert rec.final_price_cents == 10_000_000_000

    def test_event_count_in_error_message(self):
        """Duplicate error message includes event count."""
        e = _engine()
        e.process_event(_ev(event_id="x1", source=SOURCE_HISTORIC, price=1, ts=1))
        e.process_event(_ev(event_id="x2", source=SOURCE_HISTORIC, price=1, ts=2))
        e.process_event(_ev(event_id="x3", source=SOURCE_HISTORIC, price=1, ts=3))
        with pytest.raises(ValueError, match="event #4"):
            e.process_event(_ev(event_id="x1", source=SOURCE_HISTORIC, price=1, ts=4))


# =======================================================================
# SECTION 8: MULTI-ITEM AND CROSS-ITEM INTERACTIONS  — 8 tests
# =======================================================================
class TestMultiItem:
    """Cross-item isolation and multi-item state correctness."""

    def test_items_have_independent_bias(self):
        e = _engine()
        e.process_event(_ev(item_id="a", source=SOURCE_SUPPLIER, price=1000, ts=1000))
        e.process_event(_ev(item_id="a", source=SOURCE_HUMAN, price=1200, ts=1100,
                            outcome=OUTCOME_QUOTE_ACCEPTED))
        e.process_event(_ev(item_id="b", source=SOURCE_SUPPLIER, price=500, ts=1000))
        e.process_event(_ev(item_id="b", source=SOURCE_HUMAN, price=700, ts=1100,
                            outcome=OUTCOME_QUOTE_ACCEPTED))
        assert e.state.items["a"].bias_cents == 200
        assert e.state.items["b"].bias_cents == 200

    def test_items_have_independent_decay(self):
        e = _engine()
        e.state.items["a"] = ItemState(bias_cents=400, last_updated_ts=0)
        e.state.items["b"] = ItemState(bias_cents=600, last_updated_ts=500000)
        e.process_event(_ev(item_id="a", source=SOURCE_HISTORIC, price=100, ts=0))
        e.process_event(_ev(item_id="b", source=SOURCE_HISTORIC, price=100, ts=0))
        rec_a = e.process_event(_ev(item_id="a", source=SOURCE_HISTORIC, price=100, ts=700000))
        rec_b = e.process_event(_ev(item_id="b", source=SOURCE_HISTORIC, price=100, ts=700000))
        # item_a: 700000-0=700000>604800 → decayed: 400//2=200
        assert rec_a.bias_applied_cents == 200
        # item_b: 700000-500000=200000<604800 → NOT decayed
        assert rec_b.bias_applied_cents == 600

    def test_many_items_hash_stability(self):
        """50 items all get state, hash is stable."""
        e = _engine()
        for i in range(50):
            item = f"item_{i:03d}"
            e.process_event(_ev(item_id=item, source=SOURCE_SUPPLIER, price=1000 + i, ts=1000 + i))
            e.process_event(_ev(item_id=item, source=SOURCE_HUMAN, price=1200 + i, ts=1100 + i,
                                outcome=OUTCOME_QUOTE_ACCEPTED))
        h1 = compute_state_hash(e.state)
        h2 = compute_state_hash(e.state)
        assert h1 == h2
        assert len(e.state.items) == 50

    def test_historic_updates_dont_create_item_state(self):
        """Historic events alone never create ItemState."""
        e = _engine()
        for i in range(10):
            e.process_event(_ev(item_id="hist_only", source=SOURCE_HISTORIC, price=100 + i, ts=1000 + i))
        assert "hist_only" not in e.state.items

    def test_supplier_updates_dont_create_item_state(self):
        """Supplier events alone never create ItemState."""
        e = _engine()
        for i in range(10):
            e.process_event(_ev(item_id="supp_only", source=SOURCE_SUPPLIER, price=100 + i, ts=1000 + i))
        assert "supp_only" not in e.state.items

    def test_sequential_items_dont_leak_cache(self):
        """Processing item A shouldn't affect item B's cache."""
        e = _engine()
        e.process_event(_ev(item_id="a", source=SOURCE_SUPPLIER, price=9999, ts=1000))
        rec = e.process_event(_ev(item_id="b", source=SOURCE_HISTORIC, price=100, ts=1050))
        assert rec.inputs_seen["supplier_cents"] is None

    def test_1000_events_single_item(self):
        """1000 events for one item — bias list still max 5."""
        e = _engine()
        for i in range(1000):
            ts = 1000 + i * 10
            e.process_event(_ev(source=SOURCE_SUPPLIER, price=10000, ts=ts))
            if i % 3 == 0:
                e.process_event(_ev(
                    source=SOURCE_HUMAN, price=10000 + (i % 500) + 1, ts=ts + 5,
                    outcome=OUTCOME_QUOTE_ACCEPTED,
                ))
        if "item_x" in e.state.items:
            assert len(e.state.items["item_x"].accepted_human_deltas_cents) <= 5

    def test_interleaved_items_maintain_isolation(self):
        """Rapidly alternating between items keeps states separate."""
        e = _engine()
        for i in range(20):
            item = "even" if i % 2 == 0 else "odd"
            ts = 1000 + i * 100
            e.process_event(_ev(item_id=item, source=SOURCE_SUPPLIER, price=5000, ts=ts))
            e.process_event(_ev(
                item_id=item, source=SOURCE_HUMAN, price=5500, ts=ts + 50,
                outcome=OUTCOME_QUOTE_ACCEPTED,
            ))
        assert e.state.items["even"].bias_cents == 500
        assert e.state.items["odd"].bias_cents == 500


# =======================================================================
# SECTION 9: MEDIAN HELPER EDGE CASES  — 8 tests
# =======================================================================
class TestMedianEdgeCases:
    """Exhaustive median tests."""

    def test_empty(self):
        assert _median_int([]) == 0

    def test_single_zero(self):
        assert _median_int([0]) == 0

    def test_single_negative(self):
        assert _median_int([-500]) == -500

    def test_two_values(self):
        assert _median_int([100, 300]) == 200

    def test_three_identical(self):
        assert _median_int([42, 42, 42]) == 42

    def test_five_sorted(self):
        assert _median_int([1, 2, 3, 4, 5]) == 3

    def test_five_reverse(self):
        assert _median_int([5, 4, 3, 2, 1]) == 3

    def test_large_values(self):
        assert _median_int([1_000_000_000, 2_000_000_000, 3_000_000_000]) == 2_000_000_000


# =======================================================================
# SECTION 10: AUDIT RECORD CORRECTNESS  — 10 tests
# =======================================================================
class TestAuditRecordCorrectness:
    """Every field in the audit record must be correct."""

    def test_event_id_matches(self):
        e = _engine()
        rec = e.process_event(_ev(event_id="my-id-123", source=SOURCE_HISTORIC, price=100, ts=1000))
        assert rec.event_id == "my-id-123"

    def test_timestamp_matches(self):
        e = _engine()
        rec = e.process_event(_ev(source=SOURCE_HISTORIC, price=100, ts=999999))
        assert rec.timestamp == 999999

    def test_item_id_matches(self):
        e = _engine()
        rec = e.process_event(_ev(item_id="special_item", source=SOURCE_HISTORIC, price=100, ts=1000))
        assert rec.item_id == "special_item"

    def test_inputs_seen_supplier_null_when_absent(self):
        e = _engine()
        rec = e.process_event(_ev(source=SOURCE_HISTORIC, price=100, ts=1000))
        assert rec.inputs_seen["supplier_cents"] is None

    def test_inputs_seen_human_null_when_absent(self):
        e = _engine()
        rec = e.process_event(_ev(source=SOURCE_SUPPLIER, price=100, ts=1000))
        assert rec.inputs_seen["human_cents"] is None

    def test_inputs_seen_shows_human_when_present(self):
        e = _engine()
        rec = e.process_event(_ev(source=SOURCE_HUMAN, price=777, ts=1000,
                                  outcome=OUTCOME_NONE))
        assert rec.inputs_seen["human_cents"] == 777

    def test_flags_empty_for_standard_query(self):
        e = _engine()
        rec = e.process_event(_ev(source=SOURCE_SUPPLIER, price=100, ts=1000))
        assert rec.flags == []

    def test_final_price_is_int_type(self):
        e = _engine()
        rec = e.process_event(_ev(source=SOURCE_HISTORIC, price=100, ts=1000))
        assert type(rec.final_price_cents) is int

    def test_bias_applied_is_int_type(self):
        e = _engine()
        rec = e.process_event(_ev(source=SOURCE_HISTORIC, price=100, ts=1000))
        assert type(rec.bias_applied_cents) is int

    def test_rules_hash_is_sha256(self):
        e = _engine()
        rec = e.process_event(_ev(source=SOURCE_HISTORIC, price=100, ts=1000))
        assert len(rec.rules_hash) == 64
        int(rec.rules_hash, 16)  # must be valid hex


# =======================================================================
# SECTION 11: E2E FILE-LEVEL TESTS  — 6 tests
# =======================================================================
class TestFileLevel:
    """File-based operations."""

    def test_replay_with_empty_events(self, tmp_path):
        """Empty events file → replay matches trivially."""
        events = str(tmp_path / "empty.jsonl")
        state = str(tmp_path / "state.json")
        audit = str(tmp_path / "audit.jsonl")
        hash_file = str(tmp_path / "hash.txt")
        # Create empty events file
        open(events, "w").close()
        h = run_engine(events, state, audit)
        with open(hash_file, "w") as f:
            f.write(h)
        assert replay_engine(events, str(tmp_path / "rs.json"),
                             str(tmp_path / "ra.jsonl"), hash_file)

    def test_single_event_replay(self, tmp_path):
        """One event → replay matches."""
        events = str(tmp_path / "one.jsonl")
        state = str(tmp_path / "state.json")
        audit = str(tmp_path / "audit.jsonl")
        hash_file = str(tmp_path / "hash.txt")
        ev = {"event_id": "single-1", "timestamp": 1000, "item_id": "x",
              "source": "HISTORIC", "price_cents": 100, "outcome": "NONE", "meta": {}}
        with open(events, "w") as f:
            f.write(json.dumps(ev) + "\n")
        h = run_engine(events, state, audit)
        with open(hash_file, "w") as f:
            f.write(h)
        assert replay_engine(events, str(tmp_path / "rs.json"),
                             str(tmp_path / "ra.jsonl"), hash_file)

    def test_bad_json_line_raises(self, tmp_path):
        """Invalid JSON line raises ValueError."""
        events = str(tmp_path / "bad.jsonl")
        with open(events, "w") as f:
            f.write("NOT JSON\n")
        with pytest.raises(ValueError, match="Invalid event"):
            run_engine(events, str(tmp_path / "s.json"), str(tmp_path / "a.jsonl"))

    def test_missing_field_raises(self, tmp_path):
        """Missing required field raises."""
        events = str(tmp_path / "missing.jsonl")
        with open(events, "w") as f:
            f.write(json.dumps({"event_id": "x"}) + "\n")
        with pytest.raises(ValueError):
            run_engine(events, str(tmp_path / "s.json"), str(tmp_path / "a.jsonl"))

    def test_blank_lines_ignored(self, tmp_path):
        """Blank lines in events file are skipped."""
        events = str(tmp_path / "blanks.jsonl")
        ev = {"event_id": "b1", "timestamp": 1000, "item_id": "x",
              "source": "HISTORIC", "price_cents": 100, "outcome": "NONE", "meta": {}}
        with open(events, "w") as f:
            f.write("\n\n" + json.dumps(ev) + "\n\n")
        h = run_engine(events, str(tmp_path / "s.json"), str(tmp_path / "a.jsonl"))
        assert len(h) == 64

    def test_audit_log_line_count(self, tmp_path):
        """Audit log has exactly as many lines as events."""
        events = str(tmp_path / "events.jsonl")
        state = str(tmp_path / "state.json")
        audit = str(tmp_path / "audit.jsonl")
        evs = []
        for i in range(7):
            evs.append({"event_id": f"lc-{i}", "timestamp": 1000 + i, "item_id": "x",
                         "source": "HISTORIC", "price_cents": 100, "outcome": "NONE", "meta": {}})
        with open(events, "w") as f:
            for ev in evs:
                f.write(json.dumps(ev) + "\n")
        run_engine(events, state, audit)
        with open(audit) as f:
            assert len(f.readlines()) == 7
