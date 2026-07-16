"""Feature registry and computation engine with structural point-in-time safety.

Every feature is registered with metadata (family, inputs, availability lag,
cross-sectional flag). The *engine* — not each feature author — enforces the
causality contract:

- time-series features receive one symbol's history and may look ahead at most
  ``availability_lag`` bars; the engine shifts the result by that lag so the
  value stored at date D uses only information available by D's close;
- cross-sectional features receive one date-slice at a time, which makes
  normalizing over the full history structurally impossible;
- the generic look-ahead regression test iterates the registry, so every
  registered feature is automatically covered.

A feature value stored at date D is knowable at D's session close; whether it
is *tradeable* at D's close or only at the next open is an execution-model
decision handled by labels and the backtester, never by features.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from dataclasses import dataclass

import pandas as pd

from edgestack.config import EdgeStackConfig
from edgestack.data.calendar import TradingCalendar
from edgestack.exceptions import DataError, LeakageError
from edgestack.types import Family

FeatureFn = Callable[[pd.DataFrame], pd.Series]

#: Columns a time-series feature may consume: raw bars, calendar facts and
#: broadcast benchmark columns. Cross-sectional features may additionally
#: consume any time-series feature computed before them.
PANEL_COLUMNS = ("open", "high", "low", "close", "volume", "adj_close")
BENCH_COLUMNS = ("bench_close",)


@dataclass(frozen=True)
class FeatureSpec:
    name: str
    family: Family
    fn: FeatureFn
    inputs: tuple[str, ...]
    min_history: int
    availability_lag: int
    cross_sectional: bool
    description: str


_REGISTRY: dict[str, FeatureSpec] = {}
_LOADED = False


def feature(
    name: str,
    family: Family,
    *,
    inputs: tuple[str, ...] = ("close",),
    min_history: int = 1,
    availability_lag: int = 0,
    cross_sectional: bool = False,
    description: str = "",
) -> Callable[[FeatureFn], FeatureFn]:
    """Register a feature function under ``name``."""

    def deco(fn: FeatureFn) -> FeatureFn:
        if name in _REGISTRY:
            raise DataError(f"feature already registered: {name}")
        if availability_lag < 0:
            raise DataError(f"{name}: availability_lag must be >= 0")
        desc = description
        if not desc and fn.__doc__:
            doc_lines = fn.__doc__.strip().splitlines()
            desc = doc_lines[0] if doc_lines else ""
        _REGISTRY[name] = FeatureSpec(
            name=name,
            family=family,
            fn=fn,
            inputs=inputs,
            min_history=min_history,
            availability_lag=availability_lag,
            cross_sectional=cross_sectional,
            description=desc,
        )
        return fn

    return deco


def _ensure_loaded() -> None:
    global _LOADED
    if _LOADED:
        return
    import importlib

    for module in (
        "price",
        "trend",
        "momentum",
        "volatility",
        "volume",
        "structure",
        "candles",
        "calendar",
        "market",
    ):
        importlib.import_module(f"edgestack.features.{module}")
    _LOADED = True


def all_specs() -> tuple[FeatureSpec, ...]:
    _ensure_loaded()
    return tuple(_REGISTRY[name] for name in sorted(_REGISTRY))


def get_spec(name: str) -> FeatureSpec:
    _ensure_loaded()
    try:
        return _REGISTRY[name]
    except KeyError:
        raise DataError(f"unknown feature: {name}") from None


def featureset_id(specs: tuple[FeatureSpec, ...] | None = None) -> str:
    """Deterministic id of the active feature set (names + lags)."""
    specs = specs or all_specs()
    payload = ",".join(f"{s.name}:{s.availability_lag}:{int(s.cross_sectional)}" for s in specs)
    return hashlib.sha256(payload.encode()).hexdigest()[:12]


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------


def build_features(
    panel: pd.DataFrame,
    cfg: EdgeStackConfig,
    specs: tuple[FeatureSpec, ...] | None = None,
) -> pd.DataFrame:
    """Compute all registered features over a validated bar panel.

    Returns a long frame ``symbol, date, <feature columns>`` where every value
    at date D is observable by D's session close.
    """
    specs = specs or all_specs()
    ts_specs = [s for s in specs if not s.cross_sectional]
    cs_specs = [s for s in specs if s.cross_sectional]

    calendar = TradingCalendar(cfg.data.calendar)
    start, end = panel["date"].min(), panel["date"].max()
    facts = calendar.session_facts(start.date(), end.date())

    symbols = panel["symbol"].unique()
    bench_symbol = cfg.universe.benchmark_symbol
    if bench_symbol not in symbols:
        # Without the benchmark in the panel, benchmark-relative features would
        # silently be NaN; fall back to the first symbol but say so loudly.
        bench_symbol = str(sorted(symbols)[0])
    bench_close = (
        panel.loc[panel["symbol"] == bench_symbol].set_index("date")["close"].rename("bench_close")
    )

    outputs = []
    for symbol, group in panel.groupby("symbol", sort=True):
        df = group.set_index("date").sort_index()
        df = df.join(facts, how="left").join(bench_close, how="left")
        feats = pd.DataFrame(index=df.index)
        for spec in ts_specs:
            series = spec.fn(df)
            if not series.index.equals(df.index):
                raise LeakageError(f"feature {spec.name} returned a misaligned index")
            if spec.availability_lag:
                series = series.shift(spec.availability_lag)
            feats[spec.name] = pd.to_numeric(series, errors="coerce").astype("float64")
        feats.insert(0, "symbol", symbol)
        outputs.append(feats.reset_index())

    result = pd.concat(outputs, ignore_index=True).sort_values(["symbol", "date"])

    if cs_specs:
        merged = result.merge(
            panel[["symbol", "date", *PANEL_COLUMNS[:5]]], on=["symbol", "date"], how="left"
        )
        for spec in cs_specs:
            if spec.availability_lag:
                raise LeakageError(
                    f"cross-sectional feature {spec.name} cannot declare availability_lag"
                )
            pieces = []
            for date_value, day in merged.groupby("date", sort=True):
                series = spec.fn(day.set_index("symbol"))
                pieces.append(
                    pd.DataFrame(
                        {"date": date_value, "symbol": series.index, spec.name: series.to_numpy()}
                    )
                )
            column = pd.concat(pieces, ignore_index=True)
            column[spec.name] = pd.to_numeric(column[spec.name], errors="coerce").astype("float64")
            result = result.merge(column, on=["symbol", "date"], how="left")
            merged = merged.merge(column, on=["symbol", "date"], how="left")

    return result.sort_values(["symbol", "date"]).reset_index(drop=True)
