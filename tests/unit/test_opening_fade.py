"""Deterministic contracts for the standalone opening-fade campaign."""

from __future__ import annotations

from datetime import date, time

import pandas as pd
import pytest

from edgestack.config import EdgeStackConfig
from edgestack.data.catalog import DataCatalog
from edgestack.exceptions import ValidationError
from edgestack.execution.costs import CostModel
from edgestack.research.opening_fade import (
    CsvEventCalendar,
    DefinitionConfig,
    aggregate_session_returns,
    align_cross_index,
    build_session_features,
    calculate_gap,
    cumulative_vwap,
    detect_false_breakout,
    detect_overnight_rejection,
    gap_fill_flags,
    next_bar_entry,
    normalize_intraday,
    opening_range,
    retracement_flags,
    simulate_intraday_trade,
)
from edgestack.types import CostScenario


def _bars(
    values: list[tuple[float, float, float, float]],
    *,
    session: str = "2026-03-09",
    start: str = "09:30",
    minutes: int = 5,
    symbol: str = "SPY",
) -> pd.DataFrame:
    start_et = pd.Timestamp(f"{session} {start}", tz="America/New_York")
    rows = []
    for index, (open_, high, low, close) in enumerate(values):
        rows.append(
            {
                "symbol": symbol,
                "timestamp": (start_et + pd.Timedelta(minutes=index * minutes)).tz_convert("UTC"),
                "interval_minutes": minutes,
                "open": open_,
                "high": high,
                "low": low,
                "close": close,
                "volume": 100_000.0 + index,
            }
        )
    return pd.DataFrame(rows)


def _regular_day(
    *,
    session: str = "2026-03-09",
    symbol: str = "SPY",
    minutes: int = 15,
    end: time = time(15, 45),
) -> pd.DataFrame:
    count = int(((end.hour * 60 + end.minute) - (9 * 60 + 30)) / minutes) + 1
    values = []
    current = 100.2
    for index in range(count):
        close = current + (0.15 if index < 2 else -0.04)
        values.append((current, max(current, close) + 0.03, min(current, close) - 0.03, close))
        current = close
    return _bars(values, session=session, symbol=symbol, minutes=minutes)


def _context(session: str = "2026-03-09", symbol: str = "SPY") -> pd.DataFrame:
    return pd.DataFrame(
        {
            "symbol": [symbol],
            "session": [pd.Timestamp(session)],
            "previous_close": [100.0],
            "daily_atr20": [2.0],
            "ma20": [99.0],
            "ma50": [98.0],
            "realized_vol20": [0.15],
            "previous_day_return": [0.002],
            "previous_day_range": [0.01],
        }
    )


def _cost(scenario: CostScenario) -> CostModel:
    return CostModel(
        scenario=scenario,
        commission_bps=0,
        half_spread_bps=1,
        base_slippage_bps=1,
        impact_coeff_bps=5,
        short_borrow_annualized=0.03,
        sec_fee_bps=0.03,
    )


def test_positive_gap_calculation() -> None:
    result = calculate_gap(previous_close=100, official_open=100.5, premarket_last=None)
    assert result["official_open_gap"] == pytest.approx(0.005)
    assert result["primary_gap"] == pytest.approx(0.005)


def test_official_open_gap_is_distinct_from_true_premarket_gap() -> None:
    result = calculate_gap(previous_close=100, official_open=100.5, premarket_last=100.2)
    assert result["premarket_gap"] == pytest.approx(0.002)
    assert result["official_open_gap"] == pytest.approx(0.005)
    assert result["gap_source"] == "PREMARKET_LAST"


def test_opening_range_construction() -> None:
    frame = _bars(
        [(100, 101, 99.8, 100.7), (100.7, 101.2, 100.5, 101.0), (101, 101.1, 100.4, 100.5)]
    )
    result = opening_range(frame, minutes=15, interval_minutes=5)
    assert result == {
        "complete": True,
        "bars": 3,
        "open": 100.0,
        "high": 101.2,
        "low": 99.8,
        "close": 100.5,
        "volume": pytest.approx(300_003.0),
    }


def test_vwap_uses_only_current_and_prior_bar_volume() -> None:
    frame = _bars([(100, 101, 99, 100), (100, 102, 100, 101)])
    first = (101 + 99 + 100) / 3
    expected_second = (first * 100_000 + ((102 + 100 + 101) / 3) * 100_001) / 200_001
    values = cumulative_vwap(frame)
    assert values.iloc[0] == pytest.approx(first)
    assert values.iloc[1] == pytest.approx(expected_second)


def test_retracement_25_50_75_and_full_classification() -> None:
    assert retracement_flags(regular_open=100, opening_high=102, evaluation_price=101)[
        "retraced_50"
    ]
    assert retracement_flags(regular_open=100, opening_high=102, evaluation_price=100.5)[
        "retraced_75"
    ]
    assert retracement_flags(regular_open=100, opening_high=102, evaluation_price=100)[
        "retraced_100"
    ]
    zero = retracement_flags(regular_open=100, opening_high=100, evaluation_price=99)
    assert zero["retracement_fraction"] is None


def test_gap_fill_classification() -> None:
    halfway = gap_fill_flags(previous_close=100, regular_open=102, evaluation_price=101)
    full = gap_fill_flags(previous_close=100, regular_open=102, evaluation_price=99.9)
    assert halfway == {"half_gap_fill": True, "full_gap_fill": False}
    assert full == {"half_gap_fill": True, "full_gap_fill": True}


def test_false_breakout_requires_completed_close_back_inside() -> None:
    frame = _bars(
        [
            (100, 101, 99.8, 100.8),
            (100.8, 101.1, 100.5, 101.0),
            (101, 101.2, 100.8, 101.1),
            (101.1, 101.5, 100.7, 100.9),
        ]
    )
    assert detect_false_breakout(frame, opening_range_bars=3) == 3


def test_overnight_high_rejection_requires_real_level() -> None:
    frame = _bars([(100, 101.2, 99.9, 100.8), (100.8, 101.1, 100.2, 100.4)])
    assert detect_overnight_rejection(frame, overnight_high=None) is None
    assert detect_overnight_rejection(frame, overnight_high=101.0) == 0


def test_entry_is_delayed_to_next_bar_open() -> None:
    frame = _bars([(100, 101, 99, 100.5), (100.8, 101, 100, 100.2)])
    position, price = next_bar_entry(frame, 0)
    assert position == 1
    assert price == 100.8


def test_signal_on_final_bar_cannot_same_bar_fill() -> None:
    frame = _bars([(100, 101, 99, 100.5)])
    with pytest.raises(ValidationError, match="same-bar entry is forbidden"):
        next_bar_entry(frame, 0)


def test_stop_and_target_same_bar_uses_stop_first() -> None:
    frame, _ = normalize_intraday(
        _bars([(100, 100.2, 99.8, 100), (100, 102, 98, 99), (99, 100, 98, 99)])
    )
    result = simulate_intraday_trade(
        frame,
        signal_position=0,
        direction="SHORT",
        stop=101,
        target=99,
        exit_time=time(11),
        cost_model=_cost(CostScenario.BASE),
        participation=0,
    )
    assert result["exit_reason"] == "stop"
    assert result["exit"] == 101


def test_gap_through_stop_fills_at_worse_open() -> None:
    frame, _ = normalize_intraday(
        _bars([(100, 100.2, 99.8, 100), (102, 103, 101.5, 102.5), (102.5, 103, 102, 102)])
    )
    result = simulate_intraday_trade(
        frame,
        signal_position=0,
        direction="SHORT",
        stop=101,
        target=99,
        exit_time=time(11),
        cost_model=_cost(CostScenario.BASE),
        participation=0,
    )
    assert result["exit_reason"] == "gap_through_stop"
    assert result["exit"] == 102


def test_missing_opening_bar_is_excluded() -> None:
    frame = _regular_day().iloc[1:].reset_index(drop=True)
    features, _ = build_session_features(
        frame, daily_context=_context(), definition=DefinitionConfig()
    )
    assert not features.iloc[0]["eligible"]
    assert "MISSING_OPENING_BAR" in features.iloc[0]["exclusion_reason"]


def test_early_close_is_separately_classified() -> None:
    frame = _regular_day(end=time(13))
    features, _ = build_session_features(
        frame, daily_context=_context(), definition=DefinitionConfig()
    )
    assert not features.iloc[0]["eligible"]
    assert features.iloc[0]["early_close"]
    assert "EARLY_CLOSE" in features.iloc[0]["exclusion_reason"]


def test_us_eu_dst_mismatch_still_maps_open_to_0930_new_york() -> None:
    before_eu_switch = pd.Timestamp("2026-03-09T13:30:00Z").tz_convert("America/New_York")
    after_eu_switch = pd.Timestamp("2026-03-30T13:30:00Z").tz_convert("America/New_York")
    assert before_eu_switch.time() == time(9, 30)
    assert after_eu_switch.time() == time(9, 30)


def test_duplicate_bars_drop_and_catalog_write_is_idempotent(
    cfg: EdgeStackConfig,
) -> None:
    catalog = DataCatalog(cfg)
    frame = _bars([(100, 101, 99, 100.5), (100.5, 101, 100, 100.2)])
    duplicated = pd.concat([frame, frame.iloc[[0]]], ignore_index=True)
    normalized, quality = normalize_intraday(duplicated)
    assert quality["duplicate_bars_dropped"] == 1
    catalog.write_intraday_bars(normalized, provider="fixture")
    catalog.write_intraday_bars(normalized, provider="fixture")
    assert len(catalog.load_intraday_bars("SPY", interval_minutes=5)) == 2


def test_corporate_action_boundary_is_excluded() -> None:
    actions = pd.DataFrame(
        {
            "symbol": ["SPY"],
            "date": [pd.Timestamp("2026-03-09")],
            "action_type": ["split"],
            "value": [2.0],
        }
    )
    features, _ = build_session_features(
        _regular_day(),
        daily_context=_context(),
        definition=DefinitionConfig(),
        corporate_actions=actions,
    )
    assert not features.iloc[0]["eligible"]
    assert "CORPORATE_ACTION_BOUNDARY" in features.iloc[0]["exclusion_reason"]


def test_individual_stock_liquidity_floor_is_enforced() -> None:
    features, _ = build_session_features(
        _regular_day(symbol="AAPL"),
        daily_context=_context(symbol="AAPL"),
        definition=DefinitionConfig(),
        individual_stock_symbols=("AAPL",),
        minimum_stock_opening_dollar_volume=1_000_000_000.0,
    )

    assert not features.iloc[0]["eligible"]
    assert "INSUFFICIENT_STOCK_OPENING_DOLLAR_VOLUME" in features.iloc[0]["exclusion_reason"]


def test_event_calendar_exposes_only_flags_known_at_decision_time(tmp_path) -> None:
    path = tmp_path / "events.csv"
    pd.DataFrame(
        {
            "event_time_utc": ["2026-03-09T14:00:00Z", "2026-03-09T15:00:00Z"],
            "known_at_utc": ["2026-03-08T12:00:00Z", "2026-03-09T14:00:00Z"],
            "event_type": ["SCHEDULED_MACRO", "LATE_PUBLICATION"],
            "symbol": ["", ""],
        }
    ).to_csv(path, index=False)

    calendar = CsvEventCalendar(path)
    flags = calendar.flags_known_at("SPY", pd.Timestamp("2026-03-09T13:30:00Z"))

    assert flags == ("SCHEDULED_MACRO",)


def test_cross_index_alignment_uses_inner_timestamp_join() -> None:
    spy, _ = normalize_intraday(_bars([(100, 101, 99, 100), (100, 101, 99, 100)]))
    qqq, _ = normalize_intraday(
        _bars([(110, 111, 109, 110), (110, 111, 109, 110)], symbol="QQQ").iloc[1:]
    )
    spy["vwap"] = cumulative_vwap(spy)
    qqq["vwap"] = cumulative_vwap(qqq)
    assert len(align_cross_index({"SPY": spy, "QQQ": qqq})) == 1


def test_cost_monotonicity_higher_cost_never_improves_result() -> None:
    frame, _ = normalize_intraday(
        _bars([(100, 100.2, 99.8, 100), (100, 100.1, 98.5, 99), (99, 99.2, 98, 98.5)])
    )
    results = []
    for scenario in CostScenario:
        result = simulate_intraday_trade(
            frame,
            signal_position=0,
            direction="SHORT",
            stop=102,
            target=98.5,
            exit_time=time(11),
            cost_model=_cost(scenario),
            participation=0.001,
        )
        results.append(result["net_return"])
    assert results == sorted(results, reverse=True)


def test_session_aggregation_prevents_correlated_double_counting() -> None:
    trades = pd.DataFrame(
        {
            "session": [date(2026, 1, 2), date(2026, 1, 2), date(2026, 1, 5)],
            "net_return": [0.01, 0.03, -0.01],
        }
    )
    result = aggregate_session_returns(trades)
    assert len(result) == 2
    assert result.iloc[0] == pytest.approx(0.02)
