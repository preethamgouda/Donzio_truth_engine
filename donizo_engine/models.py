"""
Data models for the Donizo Truth Engine.

All monetary values are in integer CENTS — no floating-point math is allowed.
"""
from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SUPPLIER_FRESHNESS_SECONDS: int = 3600        # 1 hour
DECAY_THRESHOLD_SECONDS: int = 604800         # 7 days
MAX_DELTA_HISTORY: int = 5
CIRCUIT_BREAKER_RATIO: int = 150              # 150 % of supplier price


# ---------------------------------------------------------------------------
# Source / Outcome enums (string-based for JSON compat)
# ---------------------------------------------------------------------------
SOURCE_HISTORIC = "HISTORIC"
SOURCE_SUPPLIER = "SUPPLIER"
SOURCE_HUMAN    = "HUMAN"

OUTCOME_NONE            = "NONE"
OUTCOME_QUOTE_ACCEPTED  = "QUOTE_ACCEPTED"
OUTCOME_QUOTE_REJECTED  = "QUOTE_REJECTED"


# ---------------------------------------------------------------------------
# Event (input)
# ---------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class Event:
    """A single price event from events.jsonl."""

    event_id:    str
    timestamp:   int
    item_id:     str
    source:      str                          # HISTORIC | SUPPLIER | HUMAN
    price_cents: int
    outcome:     str   = OUTCOME_NONE         # NONE | QUOTE_ACCEPTED | QUOTE_REJECTED
    meta:        Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.price_cents, int):
            raise TypeError(
                f"price_cents must be int (cents), got {type(self.price_cents).__name__}"
            )
        if self.source not in (SOURCE_HISTORIC, SOURCE_SUPPLIER, SOURCE_HUMAN):
            raise ValueError(f"Invalid source: {self.source!r}")
        if self.outcome not in (OUTCOME_NONE, OUTCOME_QUOTE_ACCEPTED, OUTCOME_QUOTE_REJECTED):
            raise ValueError(f"Invalid outcome: {self.outcome!r}")

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "Event":
        return Event(
            event_id=d["event_id"],
            timestamp=int(d["timestamp"]),
            item_id=d["item_id"],
            source=d["source"],
            price_cents=int(d["price_cents"]),
            outcome=d.get("outcome", OUTCOME_NONE),
            meta=d.get("meta", {}),
        )

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)


# ---------------------------------------------------------------------------
# ItemState (per-item learned state)
# ---------------------------------------------------------------------------
@dataclass(slots=True)
class ItemState:
    """Per-item learned state stored inside rules_state.json."""

    bias_cents: int = 0
    last_updated_ts: int = 0
    accepted_human_deltas_cents: List[int] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "bias_cents": self.bias_cents,
            "last_updated_ts": self.last_updated_ts,
            "accepted_human_deltas_cents": list(self.accepted_human_deltas_cents),
        }

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "ItemState":
        return ItemState(
            bias_cents=int(d["bias_cents"]),
            last_updated_ts=int(d["last_updated_ts"]),
            accepted_human_deltas_cents=[int(x) for x in d.get("accepted_human_deltas_cents", [])],
        )


# ---------------------------------------------------------------------------
# RulesState (the brain)
# ---------------------------------------------------------------------------
@dataclass(slots=True)
class RulesState:
    """The persistent state file (rules_state.json)."""

    version: int = 1
    items: Dict[str, ItemState] = field(default_factory=dict)
    state_hash: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "version": self.version,
            "items": {k: v.to_dict() for k, v in sorted(self.items.items())},
            "state_hash": self.state_hash,
        }

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "RulesState":
        return RulesState(
            version=int(d.get("version", 1)),
            items={k: ItemState.from_dict(v) for k, v in d.get("items", {}).items()},
            state_hash=d.get("state_hash", ""),
        )


# ---------------------------------------------------------------------------
# AuditRecord (output)
# ---------------------------------------------------------------------------
@dataclass(slots=True)
class AuditRecord:
    """A single record appended to audit_log.jsonl."""

    event_id:           str
    timestamp:          int
    item_id:            str
    inputs_seen:        Dict[str, Optional[int]]
    final_price_cents:  int
    decision:           str
    bias_applied_cents: int
    flags:              List[str]
    rules_hash:         str

    def __post_init__(self) -> None:
        if not isinstance(self.final_price_cents, int):
            raise TypeError(
                f"final_price_cents must be int (cents), got {type(self.final_price_cents).__name__}"
            )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_id": self.event_id,
            "timestamp": self.timestamp,
            "item_id": self.item_id,
            "inputs_seen": self.inputs_seen,
            "final_price_cents": self.final_price_cents,
            "decision": self.decision,
            "bias_applied_cents": self.bias_applied_cents,
            "flags": self.flags,
            "rules_hash": self.rules_hash,
        }
