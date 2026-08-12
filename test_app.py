from datetime import date, timedelta
from pathlib import Path
import sqlite3

import numpy as np
import pandas as pd

import app


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
            **{column: np.ones(24) for column in app.WEATHER_DAILY_NAMES},
        }
    )
    app.store_hourly_rows("x", frame, "Asia/Kolkata", path)
    app.store_hourly_rows("x", frame, "Asia/Kolkata", path)
    with app.database_connection(path) as connection:
        assert connection.execute("SELECT COUNT(*) FROM hourly_history").fetchone()[0] == 24


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


def test_production_training_prefers_365_then_90():
    full = app.add_multifactor_features(complete_daily(date(2025, 1, 1), 380))
    selected, days = app.select_production_training(full, app.BASELINE_FEATURES)
    assert days == 365
    assert len(selected) == 365
    short = app.add_multifactor_features(complete_daily(date(2026, 1, 1), 100))
    selected, days = app.select_production_training(short, app.BASELINE_FEATURES)
    assert days == 90
    assert len(selected) == 90


def test_candidate_gate_selects_only_consistent_improvement(monkeypatch):
    def fake_backtests(_data, _location, _features, group):
        group_mae = {
            "Baseline": [10.0, 10.0, 10.0, 10.0],
            "Pollution history": [8.0, 8.0, 11.0, 11.0],
            "Expanded weather": [9.0, 9.0, 9.0, 9.0],
            "Calendar and festivals": [8.5, 8.5, 8.5, 8.5],
        }[group]
        results = [
            {
                "model_mae": value,
                "classification": {"Recall": 0.8},
            }
            for value in group_mae
        ]
        return results, 0

    monkeypatch.setattr(app, "build_seasonal_backtests", fake_backtests)
    result = app.evaluate_feature_candidates(pd.DataFrame(), "x", False)
    assert result["selected_group"] == "Calendar and festivals"
    history_decision = result["decisions"].loc[
        result["decisions"]["Feature group"] == "Pollution history"
    ].iloc[0]
    assert not bool(history_decision["Passes gate"])


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
    assert fitted["training_days"] == 365
    assert np.all(np.diff(widths) >= -1e-9)
