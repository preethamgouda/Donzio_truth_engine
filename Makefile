.PHONY: install test generate run replay clean

PYTHON ?= python

# Install dependencies
install:
	$(PYTHON) -m pip install -r requirements.txt

# Run all tests
test:
	$(PYTHON) -m pytest tests/ -v --tb=short

# Generate synthetic test data (1000+ events)
generate:
	$(PYTHON) -m donizo_engine generate --output events.jsonl --count 1000 --seed 42

# Process events
run:
	$(PYTHON) -m donizo_engine run --events events.jsonl --state rules_state.json --audit audit_log.jsonl

# Replay and verify
replay:
	$(PYTHON) -m donizo_engine replay --events events.jsonl --state rules_state.json --audit audit_log.jsonl --verify expected_hash.txt

# Full pipeline: generate → run → save hash → replay
e2e: generate run
	$(PYTHON) -c "import json; s=json.load(open('rules_state.json')); f=open('expected_hash.txt','w'); f.write(s['state_hash']); f.close(); print('Hash saved:', s['state_hash'])"
	$(MAKE) replay

# Clean generated files
clean:
	rm -f events.jsonl rules_state.json audit_log.jsonl expected_hash.txt
	rm -rf __pycache__ donizo_engine/__pycache__ tests/__pycache__ .pytest_cache
