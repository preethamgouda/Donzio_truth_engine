"""
Determinism tests — verify that the same input always produces the same hash.
"""
import json
import os
import tempfile

import pytest

from donizo_engine.engine import run_engine
from donizo_engine.models import RulesState
from donizo_engine.state import compute_state_hash, save_state


@pytest.fixture
def sample_events_file(tmp_path):
    """Create a small deterministic events file."""
    events = [
        {
            "event_id": "e1",
            "timestamp": 1700000000,
            "item_id": "copper_pipe_15mm",
            "source": "HISTORIC",
            "price_cents": 1000,
            "outcome": "NONE",
            "meta": {},
        },
        {
            "event_id": "e2",
            "timestamp": 1700000100,
            "item_id": "copper_pipe_15mm",
            "source": "SUPPLIER",
            "price_cents": 1200,
            "outcome": "NONE",
            "meta": {"supplier": "point_p"},
        },
        {
            "event_id": "e3",
            "timestamp": 1700000200,
            "item_id": "copper_pipe_15mm",
            "source": "HUMAN",
            "price_cents": 1500,
            "outcome": "QUOTE_ACCEPTED",
            "meta": {},
        },
        {
            "event_id": "e4",
            "timestamp": 1700000300,
            "item_id": "copper_pipe_15mm",
            "source": "SUPPLIER",
            "price_cents": 1250,
            "outcome": "NONE",
            "meta": {"supplier": "point_p"},
        },
        {
            "event_id": "e5",
            "timestamp": 1700000400,
            "item_id": "copper_pipe_15mm",
            "source": "HUMAN",
            "price_cents": 1400,
            "outcome": "QUOTE_ACCEPTED",
            "meta": {},
        },
    ]
    path = tmp_path / "test_events.jsonl"
    with open(path, "w") as f:
        for e in events:
            f.write(json.dumps(e) + "\n")
    return str(path)


class TestDeterminism:
    def test_same_events_same_hash(self, sample_events_file, tmp_path):
        """Process the same events twice → identical hashes."""
        state1 = str(tmp_path / "state1.json")
        audit1 = str(tmp_path / "audit1.jsonl")
        hash1 = run_engine(sample_events_file, state1, audit1)

        state2 = str(tmp_path / "state2.json")
        audit2 = str(tmp_path / "audit2.jsonl")
        hash2 = run_engine(sample_events_file, state2, audit2)

        assert hash1 == hash2

    def test_audit_logs_identical(self, sample_events_file, tmp_path):
        """Audit logs from two runs are byte-identical."""
        state1 = str(tmp_path / "state1.json")
        audit1 = str(tmp_path / "audit1.jsonl")
        run_engine(sample_events_file, state1, audit1)

        state2 = str(tmp_path / "state2.json")
        audit2 = str(tmp_path / "audit2.jsonl")
        run_engine(sample_events_file, state2, audit2)

        with open(audit1) as f1, open(audit2) as f2:
            assert f1.read() == f2.read()

    def test_state_files_identical(self, sample_events_file, tmp_path):
        """State files from two runs are identical."""
        state1 = str(tmp_path / "state1.json")
        audit1 = str(tmp_path / "audit1.jsonl")
        run_engine(sample_events_file, state1, audit1)

        state2 = str(tmp_path / "state2.json")
        audit2 = str(tmp_path / "audit2.jsonl")
        run_engine(sample_events_file, state2, audit2)

        with open(state1) as f1, open(state2) as f2:
            assert json.load(f1) == json.load(f2)

    def test_replay_matches(self, sample_events_file, tmp_path):
        """Run → save hash → replay → hashes match."""
        from donizo_engine.engine import replay_engine

        state_path = str(tmp_path / "state.json")
        audit_path = str(tmp_path / "audit.jsonl")
        final_hash = run_engine(sample_events_file, state_path, audit_path)

        # Write expected hash
        hash_path = str(tmp_path / "expected_hash.txt")
        with open(hash_path, "w") as f:
            f.write(final_hash)

        # Replay
        replay_state = str(tmp_path / "replay_state.json")
        replay_audit = str(tmp_path / "replay_audit.jsonl")
        assert replay_engine(
            sample_events_file, replay_state, replay_audit, hash_path
        )

    def test_replay_fails_on_wrong_hash(self, sample_events_file, tmp_path):
        """Replay with wrong hash → should fail."""
        from donizo_engine.engine import replay_engine

        state_path = str(tmp_path / "state.json")
        audit_path = str(tmp_path / "audit.jsonl")
        run_engine(sample_events_file, state_path, audit_path)

        hash_path = str(tmp_path / "wrong_hash.txt")
        with open(hash_path, "w") as f:
            f.write("0000000000000000000000000000000000000000000000000000000000000000")

        replay_state = str(tmp_path / "replay_state.json")
        replay_audit = str(tmp_path / "replay_audit.jsonl")
        assert not replay_engine(
            sample_events_file, replay_state, replay_audit, hash_path
        )
