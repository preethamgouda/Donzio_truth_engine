"""
CLI interface for the Donizo Truth Engine.

Supports three modes:
  run      — Process events, update state, write audit log.
  replay   — Process events and verify final hash matches expected.
  generate — Generate synthetic test data.
"""
from __future__ import annotations

import argparse
import logging
import sys


def _setup_logging(verbose: bool = False) -> None:
    """Configure structured logging."""
    level = logging.DEBUG if verbose else logging.INFO
    fmt = "%(asctime)s [%(levelname)-7s] %(name)s — %(message)s"
    logging.basicConfig(
        level=level,
        format=fmt,
        datefmt="%Y-%m-%d %H:%M:%S",
        stream=sys.stderr,
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="donizo_engine",
        description="Donizo Truth Engine V0.1 — The Bloomberg for Construction",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true",
        help="Enable debug-level logging",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # --- run ---
    run_p = sub.add_parser("run", help="Process events and update state")
    run_p.add_argument("--events", required=True, help="Path to events.jsonl")
    run_p.add_argument("--state", required=True, help="Path to rules_state.json")
    run_p.add_argument("--audit", required=True, help="Path to audit_log.jsonl")

    # --- replay ---
    replay_p = sub.add_parser("replay", help="Replay events and verify hash")
    replay_p.add_argument("--events", required=True, help="Path to events.jsonl")
    replay_p.add_argument("--state", required=True, help="Path to rules_state.json")
    replay_p.add_argument("--audit", required=True, help="Path to audit_log.jsonl")
    replay_p.add_argument("--verify", required=True, help="Path to expected_hash.txt")

    # --- generate ---
    gen_p = sub.add_parser("generate", help="Generate synthetic test data")
    gen_p.add_argument("--output", required=True, help="Path to output events.jsonl")
    gen_p.add_argument(
        "--count", type=int, default=1000, help="Number of events (default 1000)"
    )
    gen_p.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility"
    )

    args = parser.parse_args(argv)
    _setup_logging(verbose=args.verbose)

    logger = logging.getLogger("donizo_engine.cli")

    if args.command == "run":
        from donizo_engine.engine import run_engine

        try:
            final_hash = run_engine(args.events, args.state, args.audit)
            print(f"RUN OK — Final state hash: {final_hash}")
        except Exception as exc:
            logger.exception("Run failed")
            print(f"ERROR: {exc}", file=sys.stderr)
            sys.exit(1)

    elif args.command == "replay":
        from donizo_engine.engine import replay_engine

        try:
            ok = replay_engine(args.events, args.state, args.audit, args.verify)
            if ok:
                print("REPLAY OK: hash matches ✓")
            else:
                print("REPLAY FAILED: hash does NOT match ✗", file=sys.stderr)
                sys.exit(1)
        except Exception as exc:
            logger.exception("Replay failed")
            print(f"ERROR: {exc}", file=sys.stderr)
            sys.exit(1)

    elif args.command == "generate":
        from donizo_engine.generate_events import generate_events

        try:
            generate_events(args.output, args.count, args.seed)
            print(f"Generated {args.count} events → {args.output}")
        except Exception as exc:
            logger.exception("Generation failed")
            print(f"ERROR: {exc}", file=sys.stderr)
            sys.exit(1)


if __name__ == "__main__":
    main()
