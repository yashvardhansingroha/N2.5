from datetime import date, datetime, timedelta, timezone
from pathlib import Path
import sqlite3

import numpy as np
import pandas as pd
import pytest

import app

try:
    import atmospheric_hero
except ImportError:
    atmospheric_hero = None


def complete_daily(start: date, count: int) -> pd.DataFrame:
    rows = []
    for offset in range(count):
        day = start + timedelta(days=offset)
        rows.append(
            {
                "Date": day,
                "CAMS_PM2.5": float(20 + offset),
                "Temperature": 25.0,
                "Humidity": 55.0,
                "Wind_Speed": 8.0,
                "Precipitation": 0.0,
                "Pressure_MSL": 1005.0,
                "Cloud_Cover": 30.0,
                "Wind_Direction": 300.0,
                "Wind_Gusts": 15.0,
                "Boundary_Layer_Height": 700.0,
                "Timezone": "Asia/Kolkata",
            }
        )
    return pd.DataFrame(rows)


@pytest.mark.skipif(atmospheric_hero is None, reason="Optional video hero is not installed")
def test_atmospheric_hero_assets_and_disable_switch(monkeypatch, tmp_path: Path):
    assert atmospheric_hero.VIDEO_PATH.is_file()
    assert atmospheric_hero.POSTER_PATH.is_file()
    assert atmospheric_hero.VIDEO_PATH.stat().st_size < 15 * 1024 * 1024

    monkeypatch.setenv("AQ_VIDEO_HERO", "0")
    assert not atmospheric_hero.feature_enabled()

    monkeypatch.setenv("AQ_VIDEO_HERO", "1")
    missing_poster = tmp_path / "missing-poster.jpg"
    monkeypatch.setattr(atmospheric_hero, "POSTER_PATH", missing_poster)
    assert not atmospheric_hero.assets_available()
    assert not atmospheric_hero.render_atmospheric_hero("Delhi", True)


def test_open_meteo_daily_aggregation_and_timezone():
    payload = {
        "timezone": "Asia/Kolkata",
        "hourly": {
            "time": pd.date_range("2026-01-01", periods=24, freq="h").strftime("%Y-%m-%dT%H:%M").tolist(),
            "precipitation": [1.0] * 24,
            "wind_gusts_10m": list(range(24)),
            "wind_direction_10m": [350.0] * 12 + [10.0] * 12,
        },
    }
    daily, timezone_name = app.parse_hourly_daily(
        payload,
        {
            "precipitation": "Precipitation",
            "wind_gusts_10m": "Wind_Gusts",
            "wind_direction_10m": "Wind_Direction",
        },
        {
            "Precipitation": "sum",
            "Wind_Gusts": "max",
            "Wind_Direction": "circular_mean",
        },
    )
    assert timezone_name == "Asia/Kolkata"
    assert daily.iloc[0]["Precipitation"] == 24.0
    assert daily.iloc[0]["Wind_Gusts"] == 23.0
    assert min(abs(daily.iloc[0]["Wind_Direction"]), abs(daily.iloc[0]["Wind_Direction"] - 360)) < 1e-6


def test_local_today_uses_response_timezone_at_midnight_boundary():
    instant = datetime(2026, 10, 8, 18, 31, tzinfo=timezone.utc)
    assert app.local_today_for_timezone("Asia/Kolkata", instant) == date(2026, 10, 9)


def test_database_migration_and_hourly_upsert_are_idempotent(tmp_path: Path):
    path = tmp_path / "cache.sqlite"
    with sqlite3.connect(path) as connection:
        connection.execute(
            """CREATE TABLE hourly_history (
                location_key TEXT, local_timestamp TEXT, local_date TEXT,
                cams_pm25 REAL, temperature REAL, humidity REAL, wind_speed REAL,
                timezone TEXT, source TEXT, fetched_at_utc TEXT,
                PRIMARY KEY(location_key, local_timestamp))"""
        )
    app.initialize_database(path)
    columns = {
        row[1] for row in sqlite3.connect(path).execute("PRAGMA table_info(hourly_history)")
    }
    assert {"precipitation", "pressure_msl", "boundary_layer_height"}.issubset(columns)

    times = pd.date_range("2026-01-01", periods=24, freq="h")
    frame = pd.DataFrame(
        {
            "local_timestamp": times.strftime("%Y-%m-%dT%H:%M:%S"),
            "local_date": ["2026-01-01"] * 24,
            "cams_pm25": np.arange(24, dtype=float),
            "cams_pm10": np.arange(24, dtype=float) + 20,
            **{column: np.ones(24) for column in app.WEATHER_DAILY_NAMES},
        }
    )
    frame.loc[0, "boundary_layer_height"] = np.nan
    app.store_hourly_rows("x", frame, "Asia/Kolkata", path)
    app.store_hourly_rows("x", frame, "Asia/Kolkata", path)
    with app.database_connection(path) as connection:
        assert connection.execute("SELECT COUNT(*) FROM hourly_history").fetchone()[0] == 24
    assert app.complete_cached_dates(
        "x", date(2026, 1, 1), date(2026, 1, 1), path
    ) == {date(2026, 1, 1)}
    daily = app.daily_history_from_cache(
        "x", date(2026, 1, 1), date(2026, 1, 1), path
    )
    assert len(daily) == 1
    assert daily.iloc[0]["CAMS_PM10"] == 31.5
    assert daily.iloc[0]["Boundary_Layer_Height"] == 1.0


def test_cams_coverage_start_uses_first_complete_local_day(monkeypatch):
    times = pd.date_range("2022-08-04", periods=48, freq="h")
    values = [np.nan] * 5 + [10.0] * 43
    payload = {
        "timezone": "Asia/Kolkata",
        "hourly": {
            "time": times.strftime("%Y-%m-%dT%H:%M").tolist(),
            "pm2_5": values,
        },
    }
    monkeypatch.setattr(app, "get_json", lambda *_args, **_kwargs: payload)
    start, timezone_name = app.discover_cams_coverage_start.__wrapped__(
        28.6139, 77.2090, 0
    )
    assert start == date(2022, 8, 5)
    assert timezone_name == "Asia/Kolkata"


def test_multifactor_lags_are_date_aware_and_target_safe():
    data = complete_daily(date(2026, 1, 1), 10)
    engineered = app.add_multifactor_features(data)
    target = engineered.loc[engineered["Date"] == date(2026, 1, 10)].iloc[0]
    assert target["PM2.5_Lag1"] == 28.0
    assert target["PM2.5_Lag7"] == 22.0
    assert target["PM2.5_Mean3"] == np.mean([28.0, 27.0, 26.0])

    changed = data.copy()
    changed.loc[changed["Date"] == date(2026, 1, 10), "CAMS_PM2.5"] = 9999.0
    changed_target = app.add_multifactor_features(changed).iloc[-1]
    assert changed_target["PM2.5_Mean7"] == target["PM2.5_Mean7"]

    missing = data.loc[data["Date"] != date(2026, 1, 9)]
    missing_target = app.add_multifactor_features(missing).iloc[-1]
    assert pd.isna(missing_target["PM2.5_Lag1"])
    assert pd.isna(missing_target["PM2.5_Mean3"])


def test_fire_windows_exclude_target_and_future_detections():
    data = complete_daily(date(2026, 1, 1), 5)
    fires = pd.DataFrame(
        {
            "Local_Date": [date(2026, 1, 1), date(2026, 1, 4), date(2026, 1, 5)],
            "latitude": [29.0, 29.0, 29.0],
            "longitude": [77.0, 77.0, 77.0],
            "frp": [10.0, 100.0, 1000.0],
            "distance_km": [50.0, 50.0, 50.0],
            "bearing": [300.0, 300.0, 300.0],
        }
    )
    engineered = app.add_multifactor_features(data, fires, True)
    jan4 = engineered.loc[engineered["Date"] == date(2026, 1, 4)].iloc[0]
    assert jan4["Fire_Count_3d"] == 1.0
    assert jan4["Fire_FRP_3d"] == 10.0


def test_firms_csv_shape_and_numeric_cleaning(monkeypatch):
    class Response:
        status_code = 200
        ok = True
        text = (
            "latitude,longitude,bright_ti4,scan,track,acq_date,acq_time,satellite,"
            "confidence,version,bright_ti5,frp,daynight\n"
            "28.8,77.1,330,0.4,0.4,2026-01-01,45,N20,nominal,2.0NRT,290,12.5,D\n"
            "bad,77.2,330,0.4,0.4,2026-01-01,130,N20,nominal,2.0NRT,bad,D\n"
        )

    class Session:
        def get(self, *_args, **_kwargs):
            return Response()

    monkeypatch.setattr(app, "request_session", lambda *_args, **_kwargs: Session())
    result = app.fetch_firms_range.__wrapped__(
        "key", 28.6139, 77.2090, date(2026, 1, 1), date(2026, 1, 1), 0
    )
    assert len(result) == 1
    assert result.iloc[0]["frp"] == 12.5
    assert result.iloc[0]["acquisition_timestamp_utc"].hour == 0
    assert result.iloc[0]["acquisition_timestamp_utc"].minute == 45


def test_isolation_forest_small_sample_fallback():
    engineered = app.add_multifactor_features(complete_daily(date(2026, 1, 1), 4))
    fitted = app.fit_period_model(engineered, app.BASELINE_FEATURES)
    assert fitted["used_fallback"]
    assert "deferred" in fitted["anomaly_note"]


def test_optional_weather_missing_values_are_training_imputed():
    engineered = app.add_multifactor_features(
        complete_daily(date(2025, 1, 1), 400)
    )
    engineered.loc[20:80, "Boundary_Layer_Height"] = np.nan
    fitted = app.fit_period_model(engineered, app.EXPANDED_WEATHER_FEATURES)
    assert len(fitted["model_rows"]) == 393
    assert fitted["imputer"].statistics_[
        app.EXPANDED_WEATHER_FEATURES.index("Boundary_Layer_Height")
    ] == 700.0


def test_production_training_uses_maximum_contiguous_history_and_five_year_cap():
    full = app.add_multifactor_features(complete_daily(date(2025, 1, 1), 380))
    selected, days = app.select_production_training(full, app.BASELINE_FEATURES)
    assert days == 379
    assert len(selected) == 379

    long_history = app.add_multifactor_features(
        complete_daily(date(2020, 1, 1), 2200)
    )
    selected, days = app.select_production_training(
        long_history, app.BASELINE_FEATURES
    )
    cutoff = selected["Date"].max()
    expected_start = app.five_year_training_start(cutoff)
    assert selected["Date"].min() == expected_start
    assert days == len(pd.date_range(expected_start, cutoff, freq="D"))

    short = app.add_multifactor_features(complete_daily(date(2026, 1, 1), 100))
    with pytest.raises(ValueError, match="365 are required"):
        app.select_production_training(short, app.BASELINE_FEATURES)


def test_rolling_windows_are_dynamic_complete_and_year_season_balanced():
    history = complete_daily(date(2022, 8, 5), 1477)
    windows = app.generate_rolling_season_windows(history)
    assert len(windows) == 12
    assert {window["year_label"] for window in windows} == {
        "2023-24", "2024-25", "2025-26"
    }
    assert {window["season"] for window in windows} == {
        "Monsoon transition", "Post-monsoon", "Winter", "Spring"
    }
    assert all((window["holdout_end"] - window["holdout_start"]).days == 59 for window in windows)

    missing = history.loc[history["Date"] != date(2024, 8, 20)]
    reduced = app.generate_rolling_season_windows(missing)
    assert len(reduced) == 11
    assert "2024_monsoon_transition" not in {window["id"] for window in reduced}


def test_expanding_backtest_uses_only_prior_dates_and_caps_at_five_years():
    history = app.add_multifactor_features(
        complete_daily(date(2018, 1, 1), 2600)
    )
    holdout_start = date(2024, 8, 12)
    specification = {
        "id": "2024_monsoon_transition",
        "cycle_year": 2024,
        "year_label": "2024-25",
        "season": "Monsoon transition",
        "holdout_start": holdout_start,
        "holdout_end": holdout_start + timedelta(days=59),
        "training_start": app.five_year_training_start(holdout_start - timedelta(days=1)),
        "training_end": holdout_start - timedelta(days=1),
    }
    result = app.run_seasonal_backtest(
        history, specification, app.BASELINE_FEATURES, "Baseline"
    )
    assert result["training_start"] == app.five_year_training_start(
        specification["training_end"]
    )
    assert result["model_rows"]["Date"].max() < result["holdout_start"]
    assert set(result["model_rows"]["Date"]).isdisjoint(result["results"]["Date"])


def test_candidate_gate_selects_only_consistent_improvement(monkeypatch):
    def fake_backtests(_data, _location, _features, group):
        group_mae = {
            "Baseline": [10.0, 10.0, 10.0, 10.0],
            "Pollution history": [8.0, 8.0, 11.0, 11.0],
            "Expanded weather": [9.0, 9.0, 9.0, 9.0],
            "Calendar and festivals": [8.5, 8.5, 8.5, 8.5],
        }[group]
        seasons = ["Monsoon transition", "Post-monsoon", "Winter", "Spring"]
        results = [
            {
                "window_id": f"2025_{index}",
                "season": seasons[index],
                "year_label": "2025-26",
                "model_mae": value,
                "classification": {"Recall": 0.8},
            }
            for index, value in enumerate(group_mae)
        ]
        return results, 0

    monkeypatch.setattr(app, "build_seasonal_backtests", fake_backtests)
    result = app.evaluate_feature_candidates(pd.DataFrame(), "x", False)
    assert result["selected_group"] == "Calendar and festivals"
    history_decision = result["decisions"].loc[
        result["decisions"]["Feature group"] == "Pollution history"
    ].iloc[0]
    assert not bool(history_decision["Passes gate"])


def test_permutation_importance_artifact_is_reused(monkeypatch, tmp_path: Path):
    calls = {"count": 0}

    def fake_importance(_results):
        calls["count"] += 1
        return pd.DataFrame(
            {"Factor group": ["Current weather"], "MAE increase when shuffled": [1.2]}
        )

    monkeypatch.setattr(app, "grouped_permutation_importance", fake_importance)
    results = [{"window_id": "window", "artifact_fingerprint": "fingerprint"}]
    first, first_hit = app.cached_grouped_permutation_importance(results, tmp_path)
    second, second_hit = app.cached_grouped_permutation_importance(results, tmp_path)
    assert not first_hit
    assert second_hit
    assert calls["count"] == 1
    pd.testing.assert_frame_equal(first, second)


def test_recursive_production_returns_seven_non_narrowing_intervals():
    history = complete_daily(date(2025, 1, 1), 380)
    engineered = app.add_multifactor_features(history)
    cutoff = history["Date"].max()
    local_today = cutoff + timedelta(days=1)
    forecast_dates = [local_today + timedelta(days=offset) for offset in range(8)]
    weather = complete_daily(forecast_dates[0], len(forecast_dates)).drop(
        columns=["CAMS_PM2.5", "Timezone"]
    )
    recursive = pd.DataFrame(
        {"Day_Ahead": range(1, 8), "MAE": np.linspace(5.0, 11.0, 7)}
    )
    forecast, fitted = app.create_multifactor_production_forecast(
        engineered, weather, local_today, recursive, app.BASELINE_FEATURES
    )
    widths = forecast["Upper_80"] - forecast["Lower_80"]
    assert len(forecast) == 7
    assert fitted["training_days"] == 379
    assert np.all(np.diff(widths) >= -1e-9)
