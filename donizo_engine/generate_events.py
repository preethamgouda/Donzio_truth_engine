"""
Synthetic event generator for the Donizo Truth Engine.

Produces events.jsonl with 1000+ events that demonstrate:
  - Conflict resolution (multiple sources for same item)
  - Learning curve (accepted human overrides building bias)
  - Decay logic (timestamps >7 days apart)
  - Circuit Breaker (anomalous human prices)
"""
from __future__ import annotations

import json
import random
import uuid
from pathlib import Path
from typing import Any, Dict, List


# Item definitions with realistic base prices (cents)
ITEMS = {
    "copper_pipe_15mm":      {"base": 1200, "supplier": "point_p"},
    "pvc_pipe_32mm":         {"base": 800,  "supplier": "cedeo"},
    "steel_beam_ipn200":     {"base": 15000, "supplier": "descours"},
    "cement_bag_25kg":       {"base": 650,  "supplier": "bigmat"},
    "electrical_cable_2_5mm": {"base": 350, "supplier": "rexel"},
    "insulation_panel_100mm": {"base": 2200, "supplier": "isover"},
    "roof_tile_clay":        {"base": 180,  "supplier": "terreal"},
    "plasterboard_13mm":     {"base": 450,  "supplier": "placo"},
}

# Time constants
HOUR = 3600
DAY = 86400


def _make_event(
    rng: random.Random,
    item_id: str,
    source: str,
    price_cents: int,
    timestamp: int,
    outcome: str = "NONE",
    supplier: str = "",
) -> Dict[str, Any]:
    meta: Dict[str, Any] = {}
    if supplier:
        meta["supplier"] = supplier
    # Use seeded rng for deterministic UUIDs
    event_id = str(uuid.UUID(int=rng.getrandbits(128), version=4))
    return {
        "event_id": event_id,
        "timestamp": timestamp,
        "item_id": item_id,
        "source": source,
        "price_cents": price_cents,
        "outcome": outcome,
        "meta": meta,
    }


def generate_events(output_path: str, count: int = 1000, seed: int = 42) -> None:
    """Generate synthetic events demonstrating all engine scenarios."""
    rng = random.Random(seed)
    events: List[Dict[str, Any]] = []

    base_time = 1700000000  # ~Nov 2023

    items = list(ITEMS.keys())
    current_time = base_time

    # Phase 1: Initial historic prices for all items (~80 events)
    for item_id in items:
        base = ITEMS[item_id]["base"]
        # A few historic events per item
        for _ in range(rng.randint(5, 12)):
            noise = rng.randint(-base // 20, base // 20)
            events.append(_make_event(rng,
                item_id, "HISTORIC", base + noise, current_time
            ))
            current_time += rng.randint(60, HOUR)

    # Phase 2: Supplier quotes and standard queries (~200 events)
    for _ in range(200):
        item_id = rng.choice(items)
        base = ITEMS[item_id]["base"]
        supplier = ITEMS[item_id]["supplier"]

        # Supplier event
        supplier_noise = rng.randint(0, base // 5)
        supplier_price = base + supplier_noise
        events.append(_make_event(rng,
            item_id, "SUPPLIER", supplier_price, current_time,
            supplier=supplier,
        ))
        current_time += rng.randint(30, HOUR // 2)

        # Sometimes add a historic event too (conflict resolution)
        if rng.random() < 0.3:
            historic_noise = rng.randint(-base // 10, base // 10)
            events.append(_make_event(rng,
                item_id, "HISTORIC", base + historic_noise, current_time
            ))
            current_time += rng.randint(10, 300)

        current_time += rng.randint(60, HOUR)

    # Phase 3: Human overrides—learning curve (~250 events)
    # Focus on a few items to build up bias clearly
    learning_items = rng.sample(items, min(4, len(items)))
    for item_id in learning_items:
        base = ITEMS[item_id]["base"]
        supplier = ITEMS[item_id]["supplier"]

        for cycle in range(15):
            # Supplier quote first
            supplier_price = base + rng.randint(base // 20, base // 5)
            events.append(_make_event(rng,
                item_id, "SUPPLIER", supplier_price, current_time,
                supplier=supplier,
            ))
            current_time += rng.randint(60, 600)

            # Human override — mostly accepted, some rejected
            human_markup = rng.randint(base // 10, base // 3)
            human_price = supplier_price + human_markup
            outcome = "QUOTE_ACCEPTED" if rng.random() < 0.75 else "QUOTE_REJECTED"
            events.append(_make_event(rng,
                item_id, "HUMAN", human_price, current_time,
                outcome=outcome,
            ))
            current_time += rng.randint(300, HOUR)

    # Phase 4: Decay test—big time gap (>7 days) (~100 events)
    current_time += 8 * DAY  # Jump 8 days forward
    for _ in range(100):
        item_id = rng.choice(items)
        base = ITEMS[item_id]["base"]
        supplier = ITEMS[item_id]["supplier"]

        # Supplier event after the gap
        supplier_price = base + rng.randint(0, base // 5)
        events.append(_make_event(rng,
            item_id, "SUPPLIER", supplier_price, current_time,
            supplier=supplier,
        ))
        current_time += rng.randint(60, HOUR // 2)

        # Standard query shortly after — should use decayed bias
        events.append(_make_event(rng,
            item_id, "HISTORIC", base + rng.randint(-50, 50), current_time
        ))
        current_time += rng.randint(60, HOUR)

    # Phase 5: Circuit Breaker—anomalous human prices (~50 events)
    for _ in range(50):
        item_id = rng.choice(items)
        base = ITEMS[item_id]["base"]
        supplier = ITEMS[item_id]["supplier"]

        # Normal supplier
        supplier_price = base + rng.randint(0, base // 10)
        events.append(_make_event(rng,
            item_id, "SUPPLIER", supplier_price, current_time,
            supplier=supplier,
        ))
        current_time += rng.randint(30, 300)

        # Anomalous human price (>50% above supplier)
        anomaly_price = supplier_price * 2 + rng.randint(100, 500)
        events.append(_make_event(rng,
            item_id, "HUMAN", anomaly_price, current_time,
            outcome="QUOTE_ACCEPTED",
        ))
        current_time += rng.randint(60, HOUR)

    # Phase 6: Fill remaining events to reach count with mixed scenarios
    while len(events) < count:
        item_id = rng.choice(items)
        base = ITEMS[item_id]["base"]
        supplier = ITEMS[item_id]["supplier"]

        source = rng.choice(["HISTORIC", "SUPPLIER", "SUPPLIER", "HUMAN"])
        noise = rng.randint(-base // 10, base // 5)
        price = base + noise

        outcome = "NONE"
        if source == "HUMAN":
            outcome = rng.choice(["QUOTE_ACCEPTED", "QUOTE_REJECTED", "NONE"])

        events.append(_make_event(rng,
            item_id, source, max(1, price), current_time,
            outcome=outcome,
            supplier=supplier if source == "SUPPLIER" else "",
        ))
        current_time += rng.randint(30, HOUR)

    # Sort by timestamp for deterministic ordering
    events.sort(key=lambda e: (e["timestamp"], e["event_id"]))

    # Write
    p = Path(output_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        for ev in events:
            f.write(json.dumps(ev, sort_keys=True) + "\n")


if __name__ == "__main__":
    generate_events("events.jsonl", 1000, 42)
    print("Generated events.jsonl")
