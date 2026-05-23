"""Connection counter — isolated to break the main ↔ websocket_handler circular import."""

_total_connections: int = 0


def increment_connections() -> None:
    global _total_connections
    _total_connections += 1


def get_total_connections() -> int:
    return _total_connections
