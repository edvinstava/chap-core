from chap_core.xai.covariate_fallback import _target_signature, resolve_covariate_row


def test_target_signature_advances_calendar_month_per_horizon_step():
    # Step k means k months ahead of the base: 202406_1 → July, _2 → August, _3 → September
    assert _target_signature("202406_1") == ("month", 2024, 7)
    assert _target_signature("202406_2") == ("month", 2024, 8)
    assert _target_signature("202406_3") == ("month", 2024, 9)


def test_resolve_dataset_match_exact_period(make_covariate_df):
    df = make_covariate_df(["202405", "202406"], [5.0, 42.0])
    loc = df[df["location"] == "A"]
    row, prov = resolve_covariate_row(loc, "time_period", ["rainfall"], "202406", "A", df)
    assert prov["source"] == "dataset_match"
    assert row["rainfall"] == 42.0


def test_resolve_historical_same_month_mean(make_covariate_df):
    df = make_covariate_df(
        ["202401", "202402", "202406", "202306", "202206"],
        [1.0, 2.0, 100.0, 10.0, 20.0],
    )
    loc = df[df["location"] == "A"]
    row, prov = resolve_covariate_row(loc, "time_period", ["rainfall"], "202706", "A", df)
    assert prov["source"] == "historical_same_month_mean"
    assert prov["yearsUsed"] == [2022, 2023, 2024]
    expected = (100.0 + 10.0 + 20.0) / 3.0
    assert abs(row["rainfall"] - expected) < 1e-9


def test_resolve_historical_with_timestamp_period_column(make_covariate_df):
    df = make_covariate_df(
        ["2023-07-01", "2022-07-15", "2024-01-01", "2024-06-01"],
        [30.0, 20.0, 1.0, 2.0],
        as_datetime=True,
    )
    loc = df[df["location"] == "A"]
    row, prov = resolve_covariate_row(loc, "time_period", ["rainfall"], "202507", "A", df)
    assert prov["source"] == "historical_same_month_mean"
    assert abs(row["rainfall"] - 25.0) < 1e-9


def test_resolve_dataset_match_yyyy_mm_string(make_covariate_df):
    df = make_covariate_df(["2024-06-01", "2024-07-01"], [1.0, 99.0])
    loc = df[df["location"] == "A"]
    row, prov = resolve_covariate_row(loc, "time_period", ["rainfall"], "202407", "A", df)
    assert prov["source"] == "dataset_match"
    assert row["rainfall"] == 99.0


def test_resolve_last_row_when_no_historical_month(make_covariate_df):
    df = make_covariate_df(["202404", "202405"], [7.0, 8.0])
    loc = df[df["location"] == "A"]
    row, prov = resolve_covariate_row(loc, "time_period", ["rainfall"], "202406", "A", df)
    assert prov["source"] == "last_available_row"
    assert row["rainfall"] == 8.0
