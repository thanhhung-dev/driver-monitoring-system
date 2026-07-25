from collections import defaultdict
from typing import Any, Callable


class EventBus:
    """Lightweight publish/subscribe event bus for pipeline monitoring.

    Phase 1: used for logging/monitoring stage completion events.
    Phase 2: will be used for threaded stage coordination.
    """

    def __init__(self) -> None:
        self._subscribers: dict[str, list[Callable]] = defaultdict(list)

    def subscribe(self, event: str, callback: Callable) -> None:
        self._subscribers[event].append(callback)

    def publish(self, event: str, data: Any = None) -> None:
        for cb in self._subscribers[event]:
            cb(data)
