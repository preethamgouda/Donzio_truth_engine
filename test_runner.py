"""
Interactive Test Runner for the Donizo Truth Engine.

A web-based UI that lets judges create and run test cases
without writing code. Select a scenario, tweak values, hit Run.

Usage:
    python test_runner.py
    # Open http://localhost:5050
"""
from __future__ import annotations

import traceback
import uuid
from dataclasses import asdict
from typing import Any, Dict, List

from flask import Flask, jsonify, render_template, request

from donizo_engine.engine import TruthEngine
from donizo_engine.models import (
    CIRCUIT_BREAKER_RATIO,
    DECAY_THRESHOLD_SECONDS,
    OUTCOME_NONE,
    OUTCOME_QUOTE_ACCEPTED,
    OUTCOME_QUOTE_REJECTED,
    SOURCE_HISTORIC,
    SOURCE_HUMAN,
    SOURCE_SUPPLIER,
    SUPPLIER_FRESHNESS_SECONDS,
    Event,
    ItemState,
    RulesState,
)
from donizo_engine.state import compute_state_hash

app = Flask(__name__)

# -----------------------------------------------------------------------
# Scenario definitions
# -----------------------------------------------------------------------
SCENARIOS: List[Dict[str, Any]] = [
    {
        "id": "supplier_freshness",
        "name": "Supplier Freshness (Rule A)",
        "icon": "⏱️",
        "description": "Test whether a supplier quote is still fresh or expired based on the time offset from the original quote.",
        "fields": [
            {"name": "supplier_price", "label": "Supplier Price (cents)", "type": "number", "default": 1000, "help": "The supplier quote price"},
            {"name": "historic_price", "label": "Historic Price (cents)", "type": "number", "default": 800, "help": "Historic reference price"},
            {"name": "time_offset", "label": "Time Offset (seconds)", "type": "number", "default": 3600, "help": f"Seconds after supplier quote. Threshold = {SUPPLIER_FRESHNESS_SECONDS}s"},
            {"name": "expected_decision", "label": "Expected Decision", "type": "select", "default": "USED_SUPPLIER_PLUS_BIAS",
             "options": ["USED_SUPPLIER_PLUS_BIAS", "USED_HISTORIC_PLUS_BIAS"]},
        ],
    },
    {
        "id": "circuit_breaker",
        "name": "Circuit Breaker (Rule E)",
        "icon": "🛡️",
        "description": f"Test anomaly detection. Human price >{CIRCUIT_BREAKER_RATIO}% of supplier price triggers the circuit breaker.",
        "fields": [
            {"name": "supplier_price", "label": "Supplier Price (cents)", "type": "number", "default": 1000},
            {"name": "human_price", "label": "Human Price (cents)", "type": "number", "default": 1501, "help": f"Circuit breaker fires when human > {CIRCUIT_BREAKER_RATIO}% of supplier"},
            {"name": "outcome", "label": "Outcome", "type": "select", "default": "QUOTE_ACCEPTED",
             "options": ["QUOTE_ACCEPTED", "QUOTE_REJECTED", "NONE"]},
            {"name": "expect_anomaly", "label": "Expect Anomaly?", "type": "select", "default": "yes", "options": ["yes", "no"]},
        ],
    },
    {
        "id": "time_decay",
        "name": "Time Decay (Rule D)",
        "icon": "📉",
        "description": f"Test bias decay after >{DECAY_THRESHOLD_SECONDS}s (7 days). Bias is halved (integer division).",
        "fields": [
            {"name": "initial_bias", "label": "Initial Bias (cents)", "type": "number", "default": 400},
            {"name": "historic_price", "label": "Historic Price (cents)", "type": "number", "default": 100},
            {"name": "time_gap", "label": "Time Gap (seconds)", "type": "number", "default": 604801, "help": f"Gap from last update. Threshold = {DECAY_THRESHOLD_SECONDS}s (7 days)"},
            {"name": "expected_bias", "label": "Expected Applied Bias", "type": "number", "default": 200},
        ],
    },
    {
        "id": "learning",
        "name": "Learning / Bias Update (Rule C)",
        "icon": "🧠",
        "description": "Test that accepted human deltas accumulate and bias = median of last 5 deltas.",
        "fields": [
            {"name": "supplier_price", "label": "Supplier Price (cents)", "type": "number", "default": 1000},
            {"name": "human_prices", "label": "Human Prices (comma-separated cents)", "type": "text", "default": "1100,1200,1300",
             "help": "Each human price produces a delta = human - supplier. Bias = median(last 5 deltas)."},
            {"name": "expected_bias", "label": "Expected Final Bias", "type": "number", "default": 200},
        ],
    },
    {
        "id": "decision_tree",
        "name": "Decision Tree Priority (Rule B)",
        "icon": "🌳",
        "description": "Test the fallback priority: Human Accepted → Supplier+Bias → Historic+Bias → No Data.",
        "fields": [
            {"name": "has_supplier", "label": "Has Supplier?", "type": "select", "default": "yes", "options": ["yes", "no"]},
            {"name": "supplier_price", "label": "Supplier Price (cents)", "type": "number", "default": 1000},
            {"name": "has_historic", "label": "Has Historic?", "type": "select", "default": "yes", "options": ["yes", "no"]},
            {"name": "historic_price", "label": "Historic Price (cents)", "type": "number", "default": 800},
            {"name": "has_human", "label": "Has Human?", "type": "select", "default": "no", "options": ["yes", "no"]},
            {"name": "human_price", "label": "Human Price (cents)", "type": "number", "default": 1200},
            {"name": "human_outcome", "label": "Human Outcome", "type": "select", "default": "QUOTE_ACCEPTED",
             "options": ["QUOTE_ACCEPTED", "QUOTE_REJECTED", "NONE"]},
            {"name": "expected_decision", "label": "Expected Decision", "type": "select", "default": "USED_SUPPLIER_PLUS_BIAS",
             "options": ["USED_HUMAN", "USED_SUPPLIER_PLUS_BIAS", "USED_HISTORIC_PLUS_BIAS", "FALLBACK_NO_DATA"]},
        ],
    },
    {
        "id": "input_validation",
        "name": "Input Validation",
        "icon": "🚫",
        "description": "Test that invalid inputs are properly rejected.",
        "fields": [
            {"name": "test_type", "label": "Validation Test", "type": "select", "default": "duplicate_event_id",
             "options": ["duplicate_event_id", "negative_price", "non_human_with_outcome", "invalid_source"]},
            {"name": "expect_error", "label": "Expect Error?", "type": "select", "default": "yes", "options": ["yes", "no"]},
        ],
    },
    {
        "id": "custom_events",
        "name": "Custom Event Sequence",
        "icon": "🔧",
        "description": "Build a custom sequence of events, run them through the engine, and check any output field.",
        "fields": [],  # Handled by the custom event builder UI
    },
]


# -----------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------
def _uid() -> str:
    return str(uuid.uuid4())


def _audit_to_dict(rec: Any) -> Dict[str, Any]:
    """Convert AuditRecord to a JSON-safe dict."""
    d = asdict(rec)
    return d


# -----------------------------------------------------------------------
# Scenario runners
# -----------------------------------------------------------------------
def _run_supplier_freshness(params: Dict) -> Dict[str, Any]:
    supplier_price = int(params["supplier_price"])
    historic_price = int(params["historic_price"])
    time_offset = int(params["time_offset"])
    expected = params["expected_decision"]

    engine = TruthEngine(RulesState())
    base_ts = 10000
    engine.process_event(Event(
        event_id=_uid(), timestamp=base_ts, item_id="test_item",
        source=SOURCE_SUPPLIER, price_cents=supplier_price,
    ))
    rec = engine.process_event(Event(
        event_id=_uid(), timestamp=base_ts + time_offset, item_id="test_item",
        source=SOURCE_HISTORIC, price_cents=historic_price,
    ))

    passed = rec.decision == expected
    return {
        "passed": passed,
        "actual_decision": rec.decision,
        "expected_decision": expected,
        "final_price": rec.final_price_cents,
        "bias_applied": rec.bias_applied_cents,
        "audit": _audit_to_dict(rec),
        "explanation": (
            f"Supplier quote at t={base_ts}, query at t={base_ts + time_offset} "
            f"(offset={time_offset}s, threshold={SUPPLIER_FRESHNESS_SECONDS}s). "
            f"{'Supplier still fresh ✅' if time_offset <= SUPPLIER_FRESHNESS_SECONDS else 'Supplier expired ⏰'}"
        ),
    }


def _run_circuit_breaker(params: Dict) -> Dict[str, Any]:
    supplier_price = int(params["supplier_price"])
    human_price = int(params["human_price"])
    outcome = params["outcome"]
    expect_anomaly = params["expect_anomaly"] == "yes"

    engine = TruthEngine(RulesState())
    engine.process_event(Event(
        event_id=_uid(), timestamp=1000, item_id="test_item",
        source=SOURCE_SUPPLIER, price_cents=supplier_price,
    ))
    rec = engine.process_event(Event(
        event_id=_uid(), timestamp=1100, item_id="test_item",
        source=SOURCE_HUMAN, price_cents=human_price, outcome=outcome,
    ))

    is_anomaly = "ANOMALY_REJECTED" in rec.flags
    passed = is_anomaly == expect_anomaly

    pct = (human_price * 100 // supplier_price) if supplier_price > 0 else 0
    return {
        "passed": passed,
        "is_anomaly": is_anomaly,
        "expected_anomaly": expect_anomaly,
        "human_pct_of_supplier": pct,
        "threshold_pct": CIRCUIT_BREAKER_RATIO,
        "final_price": rec.final_price_cents,
        "decision": rec.decision,
        "flags": rec.flags,
        "audit": _audit_to_dict(rec),
        "explanation": (
            f"Human={human_price}¢, Supplier={supplier_price}¢ → "
            f"{pct}% of supplier (threshold={CIRCUIT_BREAKER_RATIO}%). "
            f"{'ANOMALY ⚠️' if is_anomaly else 'Normal ✅'}"
        ),
    }


def _run_time_decay(params: Dict) -> Dict[str, Any]:
    initial_bias = int(params["initial_bias"])
    historic_price = int(params["historic_price"])
    time_gap = int(params["time_gap"])
    expected_bias = int(params["expected_bias"])

    state = RulesState()
    state.items["test_item"] = ItemState(
        bias_cents=initial_bias, last_updated_ts=0,
        accepted_human_deltas_cents=[initial_bias] if initial_bias else [],
    )
    engine = TruthEngine(state)
    # First event to establish the historic cache
    engine.process_event(Event(
        event_id=_uid(), timestamp=0, item_id="test_item",
        source=SOURCE_HISTORIC, price_cents=historic_price,
    ))
    rec = engine.process_event(Event(
        event_id=_uid(), timestamp=time_gap, item_id="test_item",
        source=SOURCE_HISTORIC, price_cents=historic_price,
    ))

    passed = rec.bias_applied_cents == expected_bias
    decayed = time_gap > DECAY_THRESHOLD_SECONDS
    return {
        "passed": passed,
        "actual_bias": rec.bias_applied_cents,
        "expected_bias": expected_bias,
        "initial_bias": initial_bias,
        "time_gap_s": time_gap,
        "threshold_s": DECAY_THRESHOLD_SECONDS,
        "was_decayed": decayed,
        "final_price": rec.final_price_cents,
        "audit": _audit_to_dict(rec),
        "explanation": (
            f"Bias={initial_bias}¢, gap={time_gap}s (threshold={DECAY_THRESHOLD_SECONDS}s). "
            f"{'Decayed: {initial_bias}//2={initial_bias // 2}¢' if decayed else 'Not decayed — within 7 days'}"
        ),
    }


def _run_learning(params: Dict) -> Dict[str, Any]:
    supplier_price = int(params["supplier_price"])
    human_prices_str = params["human_prices"]
    human_prices = [int(p.strip()) for p in human_prices_str.split(",") if p.strip()]
    expected_bias = int(params["expected_bias"])

    engine = TruthEngine(RulesState())
    records = []
    for i, hp in enumerate(human_prices):
        ts = 1000 + i * 200
        engine.process_event(Event(
            event_id=_uid(), timestamp=ts, item_id="test_item",
            source=SOURCE_SUPPLIER, price_cents=supplier_price,
        ))
        rec = engine.process_event(Event(
            event_id=_uid(), timestamp=ts + 50, item_id="test_item",
            source=SOURCE_HUMAN, price_cents=hp, outcome=OUTCOME_QUOTE_ACCEPTED,
        ))
        records.append(rec)

    item_state = engine.state.items.get("test_item")
    actual_bias = item_state.bias_cents if item_state else 0
    deltas = item_state.accepted_human_deltas_cents if item_state else []
    passed = actual_bias == expected_bias

    return {
        "passed": passed,
        "actual_bias": actual_bias,
        "expected_bias": expected_bias,
        "deltas": deltas,
        "supplier_price": supplier_price,
        "human_prices": human_prices,
        "num_events": len(human_prices),
        "audit": _audit_to_dict(records[-1]) if records else None,
        "explanation": (
            f"Deltas = {[hp - supplier_price for hp in human_prices]}. "
            f"Last 5 = {deltas}. Median = {actual_bias}¢."
        ),
    }


def _run_decision_tree(params: Dict) -> Dict[str, Any]:
    has_supplier = params["has_supplier"] == "yes"
    supplier_price = int(params["supplier_price"])
    has_historic = params["has_historic"] == "yes"
    historic_price = int(params["historic_price"])
    has_human = params["has_human"] == "yes"
    human_price = int(params["human_price"])
    human_outcome = params["human_outcome"]
    expected = params["expected_decision"]

    engine = TruthEngine(RulesState())
    ts = 1000
    if has_historic:
        engine.process_event(Event(
            event_id=_uid(), timestamp=ts, item_id="test_item",
            source=SOURCE_HISTORIC, price_cents=historic_price,
        ))
        ts += 50
    if has_supplier:
        engine.process_event(Event(
            event_id=_uid(), timestamp=ts, item_id="test_item",
            source=SOURCE_SUPPLIER, price_cents=supplier_price,
        ))
        ts += 50

    if has_human:
        rec = engine.process_event(Event(
            event_id=_uid(), timestamp=ts, item_id="test_item",
            source=SOURCE_HUMAN, price_cents=human_price, outcome=human_outcome,
        ))
    else:
        # Standard query via historic
        rec = engine.process_event(Event(
            event_id=_uid(), timestamp=ts, item_id="test_item",
            source=SOURCE_HISTORIC, price_cents=historic_price,
        ))

    passed = rec.decision == expected
    return {
        "passed": passed,
        "actual_decision": rec.decision,
        "expected_decision": expected,
        "final_price": rec.final_price_cents,
        "flags": rec.flags,
        "audit": _audit_to_dict(rec),
        "explanation": (
            f"Sources: {'Supplier(' + str(supplier_price) + '¢)' if has_supplier else '—'}, "
            f"{'Historic(' + str(historic_price) + '¢)' if has_historic else '—'}, "
            f"{'Human(' + str(human_price) + '¢, ' + human_outcome + ')' if has_human else '—'}. "
            f"Decision: {rec.decision}"
        ),
    }


def _run_input_validation(params: Dict) -> Dict[str, Any]:
    test_type = params["test_type"]
    expect_error = params["expect_error"] == "yes"

    engine = TruthEngine(RulesState())
    error_msg = None
    try:
        if test_type == "duplicate_event_id":
            engine.process_event(Event(
                event_id="dup-test", timestamp=1000, item_id="x",
                source=SOURCE_HISTORIC, price_cents=100,
            ))
            engine.process_event(Event(
                event_id="dup-test", timestamp=2000, item_id="x",
                source=SOURCE_HISTORIC, price_cents=200,
            ))
        elif test_type == "negative_price":
            engine.process_event(Event(
                event_id=_uid(), timestamp=1000, item_id="x",
                source=SOURCE_SUPPLIER, price_cents=-100,
            ))
        elif test_type == "non_human_with_outcome":
            engine.process_event(Event(
                event_id=_uid(), timestamp=1000, item_id="x",
                source=SOURCE_SUPPLIER, price_cents=100,
                outcome=OUTCOME_QUOTE_ACCEPTED,
            ))
        elif test_type == "invalid_source":
            Event(
                event_id=_uid(), timestamp=1000, item_id="x",
                source="UNKNOWN", price_cents=100,
            )
    except (ValueError, TypeError) as exc:
        error_msg = str(exc)

    got_error = error_msg is not None
    passed = got_error == expect_error
    return {
        "passed": passed,
        "got_error": got_error,
        "expected_error": expect_error,
        "error_message": error_msg,
        "test_type": test_type,
        "explanation": (
            f"Test '{test_type}': "
            f"{'Error raised ✅' if got_error else 'No error'} — "
            f"{'as expected' if passed else 'UNEXPECTED!'}"
        ),
    }


def _run_custom_events(params: Dict) -> Dict[str, Any]:
    events_data = params.get("events", [])
    assertions = params.get("assertions", [])

    if not events_data:
        return {"passed": False, "error": "No events provided"}

    engine = TruthEngine(RulesState())
    records = []
    error_msg = None

    try:
        for i, ev_data in enumerate(events_data):
            event = Event(
                event_id=ev_data.get("event_id", _uid()),
                timestamp=int(ev_data.get("timestamp", 1000 + i * 100)),
                item_id=ev_data.get("item_id", "test_item"),
                source=ev_data.get("source", SOURCE_HISTORIC),
                price_cents=int(ev_data.get("price_cents", 100)),
                outcome=ev_data.get("outcome", OUTCOME_NONE),
            )
            rec = engine.process_event(event)
            records.append(_audit_to_dict(rec))
    except (ValueError, TypeError) as exc:
        error_msg = str(exc)

    # Check assertions
    assertion_results = []
    all_passed = True
    for assertion in assertions:
        event_idx = int(assertion.get("event_index", -1))
        field = assertion.get("field", "")
        expected = assertion.get("expected", "")
        operator = assertion.get("operator", "==")

        if event_idx < 0:
            event_idx = len(records) - 1
        if event_idx >= len(records):
            assertion_results.append({
                "passed": False, "field": field,
                "reason": f"Event index {event_idx} out of range",
            })
            all_passed = False
            continue

        rec = records[event_idx]
        # Navigate nested fields like "inputs_seen.supplier_cents"
        actual = rec
        for key in field.split("."):
            if isinstance(actual, dict):
                actual = actual.get(key)
            else:
                actual = None
                break

        # Type-coerce expected for comparison
        if isinstance(actual, int):
            try:
                expected = int(expected)
            except (ValueError, TypeError):
                pass
        elif isinstance(actual, bool):
            expected = expected in ("true", "True", "1", True)

        if operator == "==":
            ok = actual == expected
        elif operator == "!=":
            ok = actual != expected
        elif operator == "contains":
            ok = expected in str(actual)
        elif operator == "not_contains":
            ok = expected not in str(actual)
        else:
            ok = actual == expected

        if not ok:
            all_passed = False
        assertion_results.append({
            "passed": ok, "field": field, "operator": operator,
            "expected": expected, "actual": actual,
            "event_index": event_idx,
        })

    return {
        "passed": all_passed and error_msg is None,
        "error": error_msg,
        "records": records,
        "assertion_results": assertion_results,
        "total_events_processed": len(records),
        "state_hash": engine.state.state_hash,
    }


RUNNERS = {
    "supplier_freshness": _run_supplier_freshness,
    "circuit_breaker": _run_circuit_breaker,
    "time_decay": _run_time_decay,
    "learning": _run_learning,
    "decision_tree": _run_decision_tree,
    "input_validation": _run_input_validation,
    "custom_events": _run_custom_events,
}


# -----------------------------------------------------------------------
# Routes
# -----------------------------------------------------------------------
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/scenarios")
def get_scenarios():
    return jsonify(SCENARIOS)


@app.route("/api/run-test", methods=["POST"])
def run_test():
    data = request.get_json()
    scenario_id = data.get("scenario")
    params = data.get("params", {})

    runner = RUNNERS.get(scenario_id)
    if not runner:
        return jsonify({"error": f"Unknown scenario: {scenario_id}"}), 400

    try:
        result = runner(params)
        return jsonify(result)
    except Exception as exc:
        return jsonify({
            "passed": False,
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }), 200


# -----------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------
if __name__ == "__main__":
    print("\n  🧪 Donizo Test Runner → http://localhost:5050\n")
    app.run(host="0.0.0.0", port=5050, debug=True)
