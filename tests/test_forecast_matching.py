from dataclasses import dataclass

import pytest

from chap_core.xai.forecast_matching import find_forecast_row_index


@dataclass
class _FakeForecast:
    org_unit: str
    period: str


def test_exact_match():
    forecasts = [_FakeForecast("A", "202407"), _FakeForecast("A", "202408")]
    assert find_forecast_row_index(forecasts, "A", "202407") == 0
    assert find_forecast_row_index(forecasts, "A", "202408") == 1


def test_horizon_step_maps_to_calendar_month():
    """202406_1 → July 2024 (step 1 = origin), 202406_2 → August 2024, 202406_3 → September 2024."""
    forecasts = [
        _FakeForecast("A", "202407"),
        _FakeForecast("A", "202408"),
        _FakeForecast("A", "202409"),
    ]
    assert find_forecast_row_index(forecasts, "A", "202406_1") == 0
    assert find_forecast_row_index(forecasts, "A", "202406_2") == 1
    assert find_forecast_row_index(forecasts, "A", "202406_3") == 2


def test_horizon_step_different_org_units():
    forecasts = [
        _FakeForecast("A", "202407"),
        _FakeForecast("B", "202407"),
        _FakeForecast("A", "202408"),
        _FakeForecast("B", "202408"),
    ]
    assert find_forecast_row_index(forecasts, "A", "202406_1") == 0
    assert find_forecast_row_index(forecasts, "B", "202406_1") == 1
    assert find_forecast_row_index(forecasts, "A", "202406_2") == 2
    assert find_forecast_row_index(forecasts, "B", "202406_2") == 3


def test_unknown_org_unit_returns_none():
    forecasts = [_FakeForecast("A", "202407")]
    assert find_forecast_row_index(forecasts, "Z", "202407") is None


def test_no_match_falls_back_to_first():
    forecasts = [_FakeForecast("A", "202407"), _FakeForecast("A", "202408")]
    # A period that cannot be resolved to any known calendar month
    result = find_forecast_row_index(forecasts, "A", "unknown_period")
    assert result == 0
