"""
State management for the Donizo Truth Engine.

Handles loading, saving, and SHA-256 hashing of rules_state.json.
The hash is computed over a canonical JSON representation (sorted keys,
no whitespace, state_hash field excluded) to guarantee determinism.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Dict

from donizo_engine.models import RulesState


def compute_state_hash(state: RulesState) -> str:
    """Compute SHA-256 of the canonical state JSON (excluding state_hash)."""
    d = state.to_dict()
    d.pop("state_hash", None)
    canonical = json.dumps(d, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def load_state(path: str) -> RulesState:
    """Load state from disk; return empty state if file does not exist."""
    p = Path(path)
    if not p.exists():
        state = RulesState()
        state.state_hash = compute_state_hash(state)
        return state

    with open(p, "r", encoding="utf-8") as f:
        data: Dict[str, Any] = json.load(f)

    state = RulesState.from_dict(data)

    # Validate hash on load
    expected = compute_state_hash(state)
    if state.state_hash and state.state_hash != expected:
        raise ValueError(
            f"State file is corrupted! "
            f"Expected hash {expected}, got {state.state_hash}"
        )

    return state


def save_state(state: RulesState, path: str) -> str:
    """Recompute hash, persist to disk, and return the hash."""
    state.state_hash = compute_state_hash(state)
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)

    with open(p, "w", encoding="utf-8") as f:
        json.dump(state.to_dict(), f, indent=2, sort_keys=True)

    return state.state_hash
