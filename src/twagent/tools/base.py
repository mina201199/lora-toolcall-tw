"""The contract between the agent loop and whatever executes tools."""

from __future__ import annotations

from typing import Any, Protocol


class ToolError(Exception):
    """A tool ran and failed (timeout, bad upstream data, impossible request).

    The agent reports the message back to the model as the tool result. It
    never substitutes data from another executor: a live failure must not be
    papered over with mock output.
    """


class ToolExecutor(Protocol):
    def call(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        """Run one already-validated call and return a JSON-serialisable result."""
        ...
