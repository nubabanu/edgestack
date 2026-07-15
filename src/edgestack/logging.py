"""Structured logging for EdgeStack.

Emits single-line ``key=value`` records (human readable, machine greppable)
and redacts anything that looks like a secret before it reaches a handler.
"""

from __future__ import annotations

import logging
import re
import sys
from typing import Any

_SECRET_KEY_RE = re.compile(r"(api[_-]?key|secret|token|password|credential)", re.IGNORECASE)
_SECRET_VALUE_RE = re.compile(
    r"""((?:api[_-]?key|secret|token|password)\s*[=:]\s*)(\S+)""", re.IGNORECASE
)

_REDACTED = "***REDACTED***"


def redact(text: str) -> str:
    """Redact secret-looking ``key=value`` pairs inside a message string."""
    return _SECRET_VALUE_RE.sub(rf"\g<1>{_REDACTED}", text)


class _StructuredFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        base = f"ts={self.formatTime(record, '%Y-%m-%dT%H:%M:%S')}"
        parts = [base, f"level={record.levelname}", f"logger={record.name}"]
        extra = getattr(record, "fields", None)
        message = record.getMessage()
        parts.append(f'msg="{redact(message)}"')
        if isinstance(extra, dict):
            for key, value in extra.items():
                if _SECRET_KEY_RE.search(str(key)):
                    value = _REDACTED
                parts.append(f"{key}={value}")
        if record.exc_info:
            parts.append(f'exc="{self.formatException(record.exc_info)}"')
        return " ".join(parts)


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(f"edgestack.{name}")


def log_event(logger: logging.Logger, level: int, message: str, **fields: Any) -> None:
    """Log a structured event with arbitrary key/value fields."""
    logger.log(level, message, extra={"fields": fields})


def configure(level: str = "INFO") -> None:
    """Configure root EdgeStack logging once. Safe to call repeatedly."""
    root = logging.getLogger("edgestack")
    root.setLevel(level.upper())
    if not root.handlers:
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(_StructuredFormatter())
        root.addHandler(handler)
        root.propagate = False
