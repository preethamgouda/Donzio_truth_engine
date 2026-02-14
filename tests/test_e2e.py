"""
End-to-end integration test.

Runs the full pipeline: generate events → process → save hash → replay verify.
"""
import json
import os
import tempfile

import pytest

from donizo_engine.engine import run_engine, replay_engine
from donizo_engine.generate_events import generate_events


class TestE2E:
    """Full pipeline integration tests."""

    def test_full_pipeline_1000_events(self, tmp_path):
        """Generate 1000 events → run → replay → hash matches."""
        events = str(tmp_path / "events.jsonl")
        state = str(tmp_path / "state.json")
        audit = str(tmp_path / "audit.jsonl")
        hash_file = str(tmp_path / "expected.txt")

        # Generate
        generate_events(events, count=1000, seed=42)
        assert os.path.exists(events)
        with open(events) as f:
            lines = f.readlines()
        assert len(lines) >= 1000

        # Run
        final_hash = run_engine(events, state, audit)
        assert len(final_hash) == 64  # SHA-256 hex

        # Verify state file
        with open(state) as f:
            state_data = json.load(f)
        assert state_data["state_hash"] == final_hash
        assert state_data["version"] == 1
        assert len(state_data["items"]) > 0

        # Verify audit log
        with open(audit) as f:
            audit_lines = f.readlines()
        assert len(audit_lines) >= 1000

        # Check all audit records are valid JSON
        for line in audit_lines:
            rec = json.loads(line)
            assert "event_id" in rec
            assert isinstance(rec["final_price_cents"], int)

        # Save hash and replay
        with open(hash_file, "w") as f:
            f.write(final_hash)

        replay_state = str(tmp_path / "replay_state.json")
        replay_audit = str(tmp_path / "replay_audit.jsonl")
        assert replay_engine(events, replay_state, replay_audit, hash_file)

        # Verify replay state matches original
        with open(replay_state) as f:
            replay_data = json.load(f)
        assert replay_data == state_data

    def test_different_seeds_produce_different_events(self, tmp_path):
        """Different seeds → different events → different hashes."""
        e1 = str(tmp_path / "events1.jsonl")
        e2 = str(tmp_path / "events2.jsonl")

        generate_events(e1, count=100, seed=1)
        generate_events(e2, count=100, seed=2)

        s1 = str(tmp_path / "s1.json")
        s2 = str(tmp_path / "s2.json")
        a1 = str(tmp_path / "a1.jsonl")
        a2 = str(tmp_path / "a2.jsonl")

        h1 = run_engine(e1, s1, a1)
        h2 = run_engine(e2, s2, a2)

        assert h1 != h2

    def test_same_seed_same_events(self, tmp_path):
        """Same seed → identical events → identical hash."""
        e1 = str(tmp_path / "events1.jsonl")
        e2 = str(tmp_path / "events2.jsonl")

        generate_events(e1, count=100, seed=42)
        generate_events(e2, count=100, seed=42)

        with open(e1) as f1, open(e2) as f2:
            assert f1.read() == f2.read()

    def test_audit_log_has_all_required_fields(self, tmp_path):
        """Every audit record must have all spec-required fields."""
        events = str(tmp_path / "events.jsonl")
        state = str(tmp_path / "state.json")
        audit = str(tmp_path / "audit.jsonl")

        generate_events(events, count=100, seed=99)
        run_engine(events, state, audit)

        required = {
            "event_id", "timestamp", "item_id", "inputs_seen",
            "final_price_cents", "decision", "bias_applied_cents",
            "flags", "rules_hash",
        }
        with open(audit) as f:
            for line in f:
                rec = json.loads(line)
                assert required.issubset(rec.keys()), (
                    f"Missing fields: {required - rec.keys()}"
                )

    def test_no_float_prices_in_output(self, tmp_path):
        """No floating-point numbers in any monetary fields."""
        events = str(tmp_path / "events.jsonl")
        state = str(tmp_path / "state.json")
        audit = str(tmp_path / "audit.jsonl")

        generate_events(events, count=500, seed=123)
        run_engine(events, state, audit)

        with open(audit) as f:
            for line in f:
                rec = json.loads(line)
                assert isinstance(rec["final_price_cents"], int)
                assert isinstance(rec["bias_applied_cents"], int)
                for k, v in rec["inputs_seen"].items():
                    if v is not None:
                        assert isinstance(v, int), f"{k} is {type(v).__name__}"
