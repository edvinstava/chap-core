from chap_core.xai.forecast_matching import find_forecast_row_index


def test_exact_match(make_fake_forecasts):
    forecasts = make_fake_forecasts([("A", "202407"), ("A", "202408")])
    assert find_forecast_row_index(forecasts, "A", "202407") == 0
    assert find_forecast_row_index(forecasts, "A", "202408") == 1


def test_horizon_step_maps_to_calendar_month(make_fake_forecasts):
    forecasts = make_fake_forecasts([("A", "202407"), ("A", "202408"), ("A", "202409")])
    assert find_forecast_row_index(forecasts, "A", "202406_1") == 0
    assert find_forecast_row_index(forecasts, "A", "202406_2") == 1
    assert find_forecast_row_index(forecasts, "A", "202406_3") == 2


def test_horizon_step_different_org_units(make_fake_forecasts):
    forecasts = make_fake_forecasts([("A", "202407"), ("B", "202407"), ("A", "202408"), ("B", "202408")])
    assert find_forecast_row_index(forecasts, "A", "202406_1") == 0
    assert find_forecast_row_index(forecasts, "B", "202406_1") == 1
    assert find_forecast_row_index(forecasts, "A", "202406_2") == 2
    assert find_forecast_row_index(forecasts, "B", "202406_2") == 3


def test_unknown_org_unit_returns_none(make_fake_forecasts):
    forecasts = make_fake_forecasts([("A", "202407")])
    assert find_forecast_row_index(forecasts, "Z", "202407") is None


def test_no_match_falls_back_to_first(make_fake_forecasts):
    forecasts = make_fake_forecasts([("A", "202407"), ("A", "202408")])
    result = find_forecast_row_index(forecasts, "A", "unknown_period")
    assert result == 0
