"""EdgeStack exception hierarchy."""

from __future__ import annotations


class EdgeStackError(Exception):
    """Base class for all EdgeStack errors."""


class ConfigError(EdgeStackError):
    """Invalid or missing configuration."""


class DataError(EdgeStackError):
    """Problems loading, validating or storing data."""


class SchemaError(DataError):
    """A frame does not match its declared schema."""


class DataQualityError(DataError):
    """Data failed quality gates and was quarantined."""


class ProviderError(DataError):
    """A data provider failed (network, parsing, rate limits)."""


class TestPeriodLockedError(EdgeStackError):
    """Attempted access to the final untouched test period without an unlock key."""

    __test__ = False  # keep pytest from collecting this as a test class


class LeakageError(EdgeStackError):
    """A computation would use information not available at the prediction time."""


class ValidationError(EdgeStackError):
    """Statistical validation could not be performed as requested."""


class ExecutionModelError(EdgeStackError):
    """An order or fill violates the configured execution assumptions."""


class LiveTradingDisabledError(EdgeStackError):
    """Live trading is disabled and no live adapter exists in this repository."""
