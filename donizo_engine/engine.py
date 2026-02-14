"""
The Donizo Truth Engine — Core Algorithm ("The Judge").

Implements Rules A–E exactly as specified:
  A. Candidate Selection
  B. Decision Tree
  C. Learning (Bias Update)
  D. Time Decay
  E. Circuit Breaker (Anti-Hallucination)

ALL monetary values are integers (cents).  No floating-point math.
"""
from __future__ import annotations

import json
import logging
import statistics
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

from donizo_engine.models import (
    CIRCUIT_BREAKER_RATIO,
    DECAY_THRESHOLD_SECONDS,
    MAX_DELTA_HISTORY,
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
from donizo_engine.state import compute_state_hash

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Per-item price cache (tracks latest supplier/historic prices + timestamps)
# ---------------------------------------------------------------------------
@dataclass(slots=True)
class PriceEntry:
    """Cached latest price for a source/item pair."""
    price_cents: int = 0
    timestamp: int = 0


@dataclass(slots=True)
class ItemPriceCache:
    """Tracks the latest supplier and historic prices for one item."""
    supplier: Optional[PriceEntry] = None
    historic: Optional[PriceEntry] = None


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------
class TruthEngine:
    """
    Deterministic pricing engine.

    Usage:
        engine = TruthEngine(state)
        for event in events:
            audit = engine.process_event(event)
            # write audit ...
        # engine.state is updated in-place
    """

    def __init__(self, state: RulesState) -> None:
        self.state: RulesState = state
        # Index of latest prices seen so far (not persisted — rebuilt each run)
        self._prices: Dict[str, ItemPriceCache] = {}
        self._seen_event_ids: Set[str] = set()
        self._event_count: int = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def process_event(self, event: Event) -> AuditRecord:
        """Process a single event and return the audit record."""
        self._event_count += 1

        # ---- Input validation ----
        if event.event_id in self._seen_event_ids:
            raise ValueError(
                f"Duplicate event_id: {event.event_id!r} "
                f"(event #{self._event_count})"
            )
        self._seen_event_ids.add(event.event_id)

        if event.price_cents < 0:
            raise ValueError(
                f"Negative price_cents ({event.price_cents}) "
                f"in event {event.event_id!r}"
            )

        if event.source != SOURCE_HUMAN and event.outcome != "NONE":
            raise ValueError(
                f"Non-HUMAN event {event.event_id!r} has "
                f"outcome={event.outcome!r} (must be 'NONE')"
            )

        cache = self._prices.setdefault(event.item_id, ItemPriceCache())
        item_state = self.state.items.get(event.item_id)

        # ---- Update price cache from this event ----
        if event.source == SOURCE_SUPPLIER:
            cache.supplier = PriceEntry(event.price_cents, event.timestamp)
            logger.debug(
                "Supplier update: item=%s price=%d ts=%d",
                event.item_id, event.price_cents, event.timestamp,
            )
        elif event.source == SOURCE_HISTORIC:
            cache.historic = PriceEntry(event.price_cents, event.timestamp)
            logger.debug(
                "Historic update: item=%s price=%d ts=%d",
                event.item_id, event.price_cents, event.timestamp,
            )

        # ---- Build inputs_seen ----
        inputs_seen = self._build_inputs_seen(cache, event)

        # ---- Determine eligible candidates (Rule A) ----
        supplier_eligible, supplier_price = self._supplier_eligible(cache, event.timestamp)
        historic_eligible, historic_price = self._historic_eligible(cache)
        human_eligible = event.source == SOURCE_HUMAN

        # ---- Rule D: Decay before applying bias ----
        bias_cents = 0
        if item_state is not None:
            bias_cents = item_state.bias_cents
            age = event.timestamp - item_state.last_updated_ts
            if age > DECAY_THRESHOLD_SECONDS:
                old_bias = bias_cents
                bias_cents = bias_cents // 2  # Floor division
                logger.info(
                    "Decay applied: item=%s bias %d→%d (age=%ds)",
                    event.item_id, old_bias, bias_cents, age,
                )

        # ---- Rule E: Circuit Breaker ----
        flags: List[str] = []
        anomaly = False
        if human_eligible and supplier_eligible and supplier_price > 0:
            # human price > 150% of supplier price → anomaly
            if event.price_cents * 100 > supplier_price * CIRCUIT_BREAKER_RATIO:
                anomaly = True
                flags.append("ANOMALY_REJECTED")
                logger.warning(
                    "Circuit breaker: item=%s human=%d supplier=%d (%.1f%%)",
                    event.item_id, event.price_cents, supplier_price,
                    event.price_cents / supplier_price * 100,
                )

        # ---- Rule B: Decision Tree ----
        final_price: int
        decision: str

        if human_eligible and not anomaly:
            if event.outcome == OUTCOME_QUOTE_ACCEPTED:
                # Ground truth
                final_price = event.price_cents
                decision = "USED_HUMAN"
                flags.append("HUMAN_OVERRIDE_ACCEPTED")
                logger.info(
                    "Human accepted: item=%s price=%d",
                    event.item_id, event.price_cents,
                )

                # ---- Rule C: Learning ----
                if supplier_eligible and supplier_price > 0:
                    delta = event.price_cents - supplier_price
                    if item_state is None:
                        item_state = ItemState()
                        self.state.items[event.item_id] = item_state
                    item_state.accepted_human_deltas_cents.append(delta)
                    # Keep only the last MAX_DELTA_HISTORY
                    if len(item_state.accepted_human_deltas_cents) > MAX_DELTA_HISTORY:
                        item_state.accepted_human_deltas_cents = (
                            item_state.accepted_human_deltas_cents[-MAX_DELTA_HISTORY:]
                        )
                    item_state.bias_cents = _median_int(item_state.accepted_human_deltas_cents)
                    item_state.last_updated_ts = event.timestamp
                    bias_cents = item_state.bias_cents
                    logger.info(
                        "Bias updated: item=%s delta=%d bias=%d deltas=%s",
                        event.item_id, delta, item_state.bias_cents,
                        item_state.accepted_human_deltas_cents,
                    )

            elif event.outcome == OUTCOME_QUOTE_REJECTED:
                # Do NOT use human price — fallback
                flags.append("HUMAN_REJECTED")
                final_price, decision = self._fallback(
                    supplier_eligible, supplier_price,
                    historic_eligible, historic_price,
                    bias_cents,
                )
                logger.info(
                    "Human rejected: item=%s fallback=%s price=%d",
                    event.item_id, decision, final_price,
                )
            else:
                # HUMAN source with NONE outcome — treat as standard query
                final_price, decision = self._fallback(
                    supplier_eligible, supplier_price,
                    historic_eligible, historic_price,
                    bias_cents,
                )
        elif anomaly:
            # Circuit breaker triggered — treat as rejected, don't learn
            final_price, decision = self._fallback(
                supplier_eligible, supplier_price,
                historic_eligible, historic_price,
                bias_cents,
            )
        else:
            # Non-HUMAN (standard query)
            final_price, decision = self._fallback(
                supplier_eligible, supplier_price,
                historic_eligible, historic_price,
                bias_cents,
            )

        # Recompute state hash after potential mutation
        self.state.state_hash = compute_state_hash(self.state)

        record = AuditRecord(
            event_id=event.event_id,
            timestamp=event.timestamp,
            item_id=event.item_id,
            inputs_seen=inputs_seen,
            final_price_cents=final_price,
            decision=decision,
            bias_applied_cents=bias_cents,
            flags=flags,
            rules_hash=self.state.state_hash,
        )

        logger.debug(
            "Event #%d processed: id=%s decision=%s price=%d bias=%d",
            self._event_count, event.event_id, decision, final_price, bias_cents,
        )

        return record

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _supplier_eligible(
        cache: ItemPriceCache, current_ts: int
    ) -> Tuple[bool, int]:
        if cache.supplier is None:
            return False, 0
        if current_ts - cache.supplier.timestamp <= SUPPLIER_FRESHNESS_SECONDS:
            return True, cache.supplier.price_cents
        return False, 0

    @staticmethod
    def _historic_eligible(cache: ItemPriceCache) -> Tuple[bool, int]:
        if cache.historic is None:
            return False, 0
        return True, cache.historic.price_cents

    @staticmethod
    def _build_inputs_seen(
        cache: ItemPriceCache, event: Event
    ) -> Dict[str, Optional[int]]:
        return {
            "historic_cents": cache.historic.price_cents if cache.historic else None,
            "supplier_cents": cache.supplier.price_cents if cache.supplier else None,
            "human_cents": event.price_cents if event.source == SOURCE_HUMAN else None,
        }

    @staticmethod
    def _fallback(
        supplier_eligible: bool,
        supplier_price: int,
        historic_eligible: bool,
        historic_price: int,
        bias_cents: int,
    ) -> Tuple[int, str]:
        """Fallback logic: supplier+bias → historic+bias → error."""
        if supplier_eligible:
            return supplier_price + bias_cents, "USED_SUPPLIER_PLUS_BIAS"
        if historic_eligible:
            return historic_price + bias_cents, "USED_HISTORIC_PLUS_BIAS"
        return 0, "FALLBACK_NO_DATA"


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------
def _median_int(values: List[int]) -> int:
    """Compute median and return as integer (truncated toward zero)."""
    if not values:
        return 0
    med = statistics.median(values)
    return int(med)


# ---------------------------------------------------------------------------
# High-level runners
# ---------------------------------------------------------------------------
def run_engine(
    events_path: str,
    state_path: str,
    audit_path: str,
) -> str:
    """
    Process all events, update state, write audit log.
    Returns the final state hash.
    """
    from donizo_engine.state import load_state, save_state

    logger.info("Loading state from %s", state_path)
    state = load_state(state_path)
    engine = TruthEngine(state)

    audit_records: List[AuditRecord] = []

    logger.info("Processing events from %s", events_path)
    with open(events_path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                event = Event.from_dict(data)
            except (json.JSONDecodeError, KeyError, ValueError, TypeError) as exc:
                raise ValueError(
                    f"Invalid event on line {line_no}: {exc}"
                ) from exc
            record = engine.process_event(event)
            audit_records.append(record)

    logger.info("Processed %d events", len(audit_records))

    # Persist
    final_hash = save_state(engine.state, state_path)
    logger.info("State saved → %s (hash=%s)", state_path, final_hash)

    # Write audit log
    p = Path(audit_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        for rec in audit_records:
            f.write(json.dumps(rec.to_dict(), sort_keys=True) + "\n")
    logger.info("Audit log saved → %s (%d records)", audit_path, len(audit_records))

    return final_hash


def replay_engine(
    events_path: str,
    state_path: str,
    audit_path: str,
    verify_hash_path: str,
) -> bool:
    """
    Run the engine from a clean state and verify the final hash matches.
    Returns True if hashes match.
    """
    from donizo_engine.state import save_state

    logger.info("Replay mode: processing from clean state")

    # Start from a CLEAN state for replay
    state = RulesState()
    state.state_hash = compute_state_hash(state)
    engine = TruthEngine(state)

    audit_records: List[AuditRecord] = []

    with open(events_path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                event = Event.from_dict(data)
            except (json.JSONDecodeError, KeyError, ValueError, TypeError) as exc:
                raise ValueError(
                    f"Invalid event on line {line_no}: {exc}"
                ) from exc
            record = engine.process_event(event)
            audit_records.append(record)

    # Compute final hash
    final_hash = compute_state_hash(engine.state)
    engine.state.state_hash = final_hash

    # Save state and audit for inspection
    save_state(engine.state, state_path)

    p = Path(audit_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        for rec in audit_records:
            f.write(json.dumps(rec.to_dict(), sort_keys=True) + "\n")

    # Verify
    with open(verify_hash_path, "r", encoding="utf-8") as f:
        expected_hash = f.read().strip()

    match = final_hash == expected_hash
    if match:
        logger.info("Replay PASSED: hash=%s", final_hash)
    else:
        logger.error(
            "Replay FAILED: expected=%s actual=%s", expected_hash, final_hash
        )

    return match
