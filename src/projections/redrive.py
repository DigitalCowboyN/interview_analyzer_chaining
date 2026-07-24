"""Re-drive CLI: replay parked (dead-lettered) projection events.

Usage: python -m src.projections.redrive [aggregate_type ...] [--max-count N]

M4.9 Task 2 made the parked-events stream append-only; M4.9 Task 4 added a
reorder buffer that catches most out-of-order deliveries before they ever
park. Parking should now be rare, so a background re-drive worker would be
YAGNI -- this is the manual recovery path an operator runs after confirming
a parked event's referent has landed (or after a historical backlog is
otherwise resolved).

For each parked event, the original event is re-dispatched through the real
handler registry (`create_handler_registry`), exactly as the live
subscription would dispatch it -- `handler.handle(event)` directly, not
`handle_with_retry`, so a still-failing event is reported rather than
re-parked or retried in a loop. Handlers MERGE (idempotent), so replaying an
already-applied event is a safe no-op.

Outcomes per parked event:
  - handler succeeds                    -> "redriven"
  - handler raises ReferentNotReadyError -> "still_parked" (run continues)
  - no handler registered for the type   -> "no_handler"

Prints a single JSON summary line to stdout, e.g.:
  {"redriven": 3, "still_parked": 1, "no_handler": 0,
   "by_aggregate": {"Interview": {...}, "Sentence": {...}}}
"""

import argparse
import asyncio
import json
import logging
from typing import Dict, List, Optional

from src.projections.bootstrap import create_handler_registry
from src.projections.handlers.registry import HandlerRegistry
from src.projections.handlers.speaker_handlers import ReferentNotReadyError
from src.projections.parked_events import ParkedEventsManager

logger = logging.getLogger(__name__)

# Aggregate types that can park events (see src/projections/config.py
# SUBSCRIPTION_CONFIG / get_parked_stream_name callers).
KNOWN_AGGREGATE_TYPES: List[str] = ["Interview", "Sentence", "Project"]

_EMPTY_COUNTS: Dict[str, int] = {"redriven": 0, "still_parked": 0, "no_handler": 0}


async def redrive_aggregate(
    aggregate_type: str,
    parked_events_manager: ParkedEventsManager,
    registry: HandlerRegistry,
    max_count: Optional[int] = None,
) -> Dict[str, int]:
    """Redrive all parked events for a single aggregate type.

    Args:
        aggregate_type: Aggregate type whose parked stream to read (e.g. "Interview").
        parked_events_manager: Source of parked events (real or fake; must expose
            get_parked_events(aggregate_type, max_count) -> List[ParkedEvent]).
        registry: Handler registry to dispatch through (real or fake; must expose
            get_handler(event_type) -> Optional[handler with async .handle(event)]).
        max_count: Optional cap on how many parked events to read.

    Returns:
        Counts for this aggregate type: {"redriven", "still_parked", "no_handler"}.
    """
    counts = dict(_EMPTY_COUNTS)

    parked_events = await parked_events_manager.get_parked_events(aggregate_type, max_count=max_count)

    for parked in parked_events:
        event = parked.original_event
        handler = registry.get_handler(event.event_type)

        if handler is None:
            counts["no_handler"] += 1
            logger.warning(
                f"No handler registered for parked event type '{event.event_type}' "
                f"(event_id={event.event_id}, aggregate={aggregate_type})"
            )
            continue

        try:
            await handler.handle(event)
            counts["redriven"] += 1
        except ReferentNotReadyError as e:
            counts["still_parked"] += 1
            logger.info(
                f"Referent still not ready for parked event {event.event_id} "
                f"(type={event.event_type}, aggregate={aggregate_type}): {e}"
            )

    return counts


async def redrive(
    aggregate_types: List[str],
    parked_events_manager: ParkedEventsManager,
    registry: HandlerRegistry,
    max_count: Optional[int] = None,
) -> Dict[str, object]:
    """Redrive parked events across one or more aggregate types.

    Returns the full JSON-summary-shaped dict: overall counts plus a
    per-aggregate-type breakdown.
    """
    totals = dict(_EMPTY_COUNTS)
    by_aggregate: Dict[str, Dict[str, int]] = {}

    for aggregate_type in aggregate_types:
        counts = await redrive_aggregate(aggregate_type, parked_events_manager, registry, max_count=max_count)
        by_aggregate[aggregate_type] = counts
        for key in totals:
            totals[key] += counts[key]

    return {**totals, "by_aggregate": by_aggregate}


def _resolve_aggregate_types(requested: List[str]) -> List[str]:
    """Map CLI input to a concrete aggregate-type list. Empty or ["all"]
    means every known parked aggregate type."""
    if not requested or requested == ["all"]:
        return list(KNOWN_AGGREGATE_TYPES)
    return list(requested)


def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Replay parked (dead-lettered) projection events once their referents have landed."
    )
    parser.add_argument(
        "aggregate_types",
        nargs="*",
        default=["all"],
        help=(
            "Aggregate type(s) to redrive, e.g. 'Interview Sentence', or 'all' "
            f"for every known type ({', '.join(KNOWN_AGGREGATE_TYPES)}). Default: all."
        ),
    )
    parser.add_argument(
        "--max-count",
        type=int,
        default=None,
        help="Maximum number of parked events to read per aggregate type (default: no limit).",
    )
    return parser.parse_args(argv)


async def main_async(argv: Optional[List[str]] = None) -> Dict[str, object]:
    args = _parse_args(argv)
    aggregate_types = _resolve_aggregate_types(args.aggregate_types)

    parked_events_manager = ParkedEventsManager()
    registry = create_handler_registry(parked_events_manager)

    summary = await redrive(aggregate_types, parked_events_manager, registry, max_count=args.max_count)
    print(json.dumps(summary))
    return summary


def main() -> None:
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
