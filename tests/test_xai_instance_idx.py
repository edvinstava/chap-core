from types import SimpleNamespace

from chap_core.xai.forecast_matching import find_forecast_row_index, norm_period_id


def test_norm_period_id_hyphen():
    assert norm_period_id("202405-2") == "202405_2"


def test_find_forecast_row_index_exact_order_independent():
    fcs = [
        SimpleNamespace(org_unit="A", period="202405_2"),
        SimpleNamespace(org_unit="A", period="202405_1"),
    ]
    assert find_forecast_row_index(fcs, "A", "202405_1") == 1


def test_find_forecast_row_index_hyphen_normalized():
    fcs = [
        SimpleNamespace(org_unit="A", period="202405_1"),
        SimpleNamespace(org_unit="A", period="202405_2"),
    ]
    assert find_forecast_row_index(fcs, "A", "202405-2") == 1


def test_find_forecast_row_index_does_not_pick_first_horizon_on_shared_calendar_prefix():
    fcs = [
        SimpleNamespace(org_unit="W6", period="202405_1"),
        SimpleNamespace(org_unit="W6", period="202405_2"),
        SimpleNamespace(org_unit="W6", period="202405_3"),
    ]
    assert find_forecast_row_index(fcs, "W6", "202405_1") == 0
    assert find_forecast_row_index(fcs, "W6", "202405_2") == 1
    assert find_forecast_row_index(fcs, "W6", "202405_3") == 2


def test_find_forecast_row_index_unknown_org():
    fcs = [SimpleNamespace(org_unit="A", period="202405_1")]
    assert find_forecast_row_index(fcs, "B", "202405_1") is None
