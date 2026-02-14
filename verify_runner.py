"""Quick API verification for the test runner."""
import json
import requests
import sys

base = "http://127.0.0.1:5050"

tests = [
    ("Scenarios loaded", "GET", "/api/scenarios", None, lambda r: len(r.json()) == 7),
    ("Supplier Freshness (3600s=PASS)", "POST", "/api/run-test", {
        "scenario": "supplier_freshness",
        "params": {"supplier_price": 1000, "historic_price": 800, "time_offset": 3600,
                   "expected_decision": "USED_SUPPLIER_PLUS_BIAS"}
    }, lambda r: r.json()["passed"]),
    ("Supplier Freshness (3601s=expired)", "POST", "/api/run-test", {
        "scenario": "supplier_freshness",
        "params": {"supplier_price": 1000, "historic_price": 800, "time_offset": 3601,
                   "expected_decision": "USED_HISTORIC_PLUS_BIAS"}
    }, lambda r: r.json()["passed"]),
    ("Circuit Breaker (1501=anomaly)", "POST", "/api/run-test", {
        "scenario": "circuit_breaker",
        "params": {"supplier_price": 1000, "human_price": 1501,
                   "outcome": "QUOTE_ACCEPTED", "expect_anomaly": "yes"}
    }, lambda r: r.json()["passed"]),
    ("Circuit Breaker (1500=ok)", "POST", "/api/run-test", {
        "scenario": "circuit_breaker",
        "params": {"supplier_price": 1000, "human_price": 1500,
                   "outcome": "QUOTE_ACCEPTED", "expect_anomaly": "no"}
    }, lambda r: r.json()["passed"]),
    ("Time Decay (604801s)", "POST", "/api/run-test", {
        "scenario": "time_decay",
        "params": {"initial_bias": 400, "historic_price": 100,
                   "time_gap": 604801, "expected_bias": 200}
    }, lambda r: r.json()["passed"]),
    ("Learning (3 deltas)", "POST", "/api/run-test", {
        "scenario": "learning",
        "params": {"supplier_price": 1000, "human_prices": "1100,1200,1300",
                   "expected_bias": 200}
    }, lambda r: r.json()["passed"]),
    ("Decision Tree", "POST", "/api/run-test", {
        "scenario": "decision_tree",
        "params": {"has_supplier": "yes", "supplier_price": 1000,
                   "has_historic": "yes", "historic_price": 800,
                   "has_human": "no", "human_price": 1200,
                   "human_outcome": "QUOTE_ACCEPTED",
                   "expected_decision": "USED_SUPPLIER_PLUS_BIAS"}
    }, lambda r: r.json()["passed"]),
    ("Input Validation (dup)", "POST", "/api/run-test", {
        "scenario": "input_validation",
        "params": {"test_type": "duplicate_event_id", "expect_error": "yes"}
    }, lambda r: r.json()["passed"]),
    ("Custom Events", "POST", "/api/run-test", {
        "scenario": "custom_events",
        "params": {
            "events": [
                {"source": "SUPPLIER", "price_cents": 1000, "timestamp": 1000},
                {"source": "HUMAN", "price_cents": 1200, "timestamp": 1100,
                 "outcome": "QUOTE_ACCEPTED"},
            ],
            "assertions": [
                {"event_index": 1, "field": "decision", "operator": "==",
                 "expected": "USED_HUMAN"},
                {"event_index": 1, "field": "final_price_cents", "operator": "==",
                 "expected": 1200},
            ],
        }
    }, lambda r: r.json()["passed"]),
    ("Frontend HTML", "GET", "/", None,
     lambda r: "Donizo Test Runner" in r.text and len(r.text) > 5000),
]

print("=" * 60)
ok = 0
for name, method, path, body, check in tests:
    try:
        if method == "GET":
            r = requests.get(base + path)
        else:
            r = requests.post(base + path, json=body)
        passed = check(r)
        status = "PASS" if passed else "FAIL"
        detail = ""
        if not passed and method == "POST":
            detail = f" | {r.text[:120]}"
    except Exception as exc:
        status = "ERR"
        detail = f" | {exc}"
        passed = False
    print(f"  {'✅' if passed else '❌'} [{status}] {name}{detail}")
    if passed:
        ok += 1

print(f"\n  {ok}/{len(tests)} passed")
print("=" * 60)
if ok != len(tests):
    sys.exit(1)
