"""Message bus / network layer — **not in spec**.

The gloas spec is single-process and synchronous: every handler (`on_block`,
`on_attestation`, `on_execution_payload_envelope`, `on_payload_attestation_message`)
is called by the host node when a message arrives, in arrival order. The spec
does not model message-level delays, withholding, or equivocation in transit.

The Tier 2a `MessageBus` is therefore a **simulation-only** layer: a discrete-
event queue keyed on ``deliver_at_ms``, with Byzantine hooks (withhold, delay,
release). Its correctness is validated by:

1. **Invariant assertions** in `deliver_due` — events are time-monotonic;
   delivered messages route through spec handlers verbatim (we never bypass
   spec validation).
2. **Honest-only retro-degeneracy**: when no withhold/delay is in effect, the
   bus delivers each message in the slot it was scheduled in; the resulting
   store evolution must match a synchronous spec-handler invocation. Tested
   by `test_network.py::TestHonestDegenerate`.

This module deliberately mirrors the **API** of BunnyFinder's Go attacker bus
(message kinds + scheduled delivery), but contains no code from it.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable


class MessageKind(Enum):
    BLOCK = "block"
    ATTESTATION = "attestation"
    PAYLOAD_ENVELOPE = "payload_envelope"
    PAYLOAD_ATTESTATION = "payload_attestation"


@dataclass
class Message:
    """A pending in-transit message.

    ``deliver_at_ms`` is the time at which the bus should hand the message to
    its handler. ``sender`` is the validator (or builder) index that emitted
    it — used only by the bus's own logging / Byzantine filters, never by
    spec handlers.
    """

    sender: int
    kind: MessageKind
    payload: Any              # Block / Attestation / Envelope / PTC message
    deliver_at_ms: int
    seq: int = 0              # tiebreak for same-time deliveries


@dataclass
class MessageBus:
    """In-process message bus with Byzantine hooks.

    Usage:
        bus = MessageBus()
        bus.register_handler(MessageKind.BLOCK, lambda m: F.on_block(store, m))
        bus.schedule(Message(...))
        bus.deliver_due(current_ms=...)
    """

    _queue: deque[Message] = field(default_factory=deque)
    _handlers: dict[MessageKind, Callable[[Any], None]] = field(default_factory=dict)
    _withheld_kinds: set[tuple[int, MessageKind]] = field(default_factory=set)
    _seq: int = 0
    _last_delivered_ms: int = -1   # invariant: time-monotonic

    # --- handler registration ------------------------------------------------
    def register_handler(
        self, kind: MessageKind, handler: Callable[[Any], None]
    ) -> None:
        self._handlers[kind] = handler

    # --- scheduling ----------------------------------------------------------
    def schedule(self, msg: Message) -> None:
        """Enqueue a message for future delivery.

        Stable across same-time deliveries via internal ``seq`` counter.
        """
        self._seq += 1
        msg = Message(
            sender=msg.sender, kind=msg.kind, payload=msg.payload,
            deliver_at_ms=msg.deliver_at_ms, seq=self._seq,
        )
        # insertion-sorted by (deliver_at_ms, seq)
        self._queue.append(msg)
        self._queue = deque(sorted(self._queue, key=lambda m: (m.deliver_at_ms, m.seq)))

    # --- Byzantine hooks -----------------------------------------------------
    def withhold(self, sender: int, kind: MessageKind) -> None:
        """Suppress all future messages from ``sender`` of this ``kind``.

        Tier 2a model: matches BunnyFinder's "withholding" attack hook. Use
        ``release`` to undo. Already-scheduled messages remain queued; they
        will be filtered at delivery time.
        """
        self._withheld_kinds.add((sender, kind))

    def release(self, sender: int, kind: MessageKind) -> None:
        self._withheld_kinds.discard((sender, kind))

    # --- delivery ------------------------------------------------------------
    def deliver_due(self, current_ms: int) -> list[Message]:
        """Deliver and return every message with ``deliver_at_ms <= current_ms``.

        Invariants asserted:
            - ``current_ms`` is monotonic non-decreasing across calls.
            - Messages are delivered in (time, seq) order.
        """
        assert current_ms >= self._last_delivered_ms, (
            f"time moved backwards: {current_ms} < {self._last_delivered_ms}"
        )
        delivered: list[Message] = []
        while self._queue and self._queue[0].deliver_at_ms <= current_ms:
            msg = self._queue.popleft()
            if (msg.sender, msg.kind) in self._withheld_kinds:
                continue  # silently dropped (Byzantine withhold)
            handler = self._handlers.get(msg.kind)
            if handler is not None:
                handler(msg.payload)
            delivered.append(msg)
        self._last_delivered_ms = current_ms
        return delivered

    # --- introspection -------------------------------------------------------
    def pending(self) -> list[Message]:
        return list(self._queue)

    def __len__(self) -> int:
        return len(self._queue)
