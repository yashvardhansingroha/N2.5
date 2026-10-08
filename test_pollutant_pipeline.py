from datetime import date, timedelta
from io import BytesIO

import numpy as np
import pandas as pd
import pytest

import pollutant_pipeline as pipeline
import station_archive as station
import archived_forecast_weather as archived


def sample_daily(start: date, count: int, target: str = "CAMS_PM2.5") -> pd.DataFrame:
    rows = []
    for offset in range(count):
        day = start + timedelta(days=offset)
        rows.append({
            "Date": day, target: 30.0 + offset % 30,
            "Temperature": 25.0, "Humidity": 60.0, "Wind_Speed": 8.0,
            "Precipitation": 0.0, "Pressure_MSL": 1005.0,
            "Cloud_Cover": 30.0, "Wind_Direction": 270.0,
            "Wind_Gusts": 15.0, "Boundary_Layer_Height": 700.0,
        })
    return pd.DataFrame(rows)


def test_calendar_lags_do_not_jump_missing_dates():
    data = sample_daily(date(2025, 1, 1), 10)
    data = data.loc[data["Date"] != date(2025, 1, 3)]
    engineered = pipeline.engineer(data, pipeline.CAMS_POLLUTANTS["PM2.5"])
    fourth = engineered.loc[engineered["Date"] == date(2025, 1, 4)].iloc[0]
    assert np.isnan(fourth["Lag1"])
    assert fourth["Lag2"] == 31.0
    assert np.isnan(fourth["Mean7"])
    eighth = engineered.loc[engineered["Date"] == date(2025, 1, 8)].iloc[0]
    assert eighth["Lag7"] == 30.0


def test_seven_day_mean_requires_all_seven_previous_dates():
    data = sample_daily(date(2025, 1, 1), 8)
    engineered = pipeline.engineer(data, pipeline.CAMS_POLLUTANTS["PM2.5"])
    assert engineered.iloc[-1]["Mean7"] == 33.0


def test_known_holiday_and_diwali_features():
    row = pipeline.feature_row(date(2025, 10, 20), {}, {})
    assert row["Indian_Holiday"] == 1.0
    assert row["Diwali"] == 1.0
    assert row["Days_From_Diwali"] == 0.0


def test_threshold_metrics_undefined_denominators():
    no_events = pipeline.score(np.array([20.0, 30.0]), np.array([10.0, 20.0]), 60.0)
    assert no_events["recall"] is None
    assert no_events["precision"] is None
    all_events = pipeline.score(np.array([80.0, 90.0]), np.array([70.0, 80.0]), 60.0)
    assert all_events["false_positive_rate"] is None


def test_final_holdout_excluded_from_frozen_fit(monkeypatch, tmp_path):
    original = pipeline.fit_models
    fitted_ranges = []

    def record_fit(training, pollutant):
        fitted_ranges.append((min(training["Date"]), max(training["Date"])))
        return original(training, pollutant)

    monkeypatch.setattr(pipeline, "fit_models", record_fit)
    analysis = pipeline.analyze(
        sample_daily(date(2024, 1, 1), 650),
        pipeline.CAMS_POLLUTANTS["PM2.5"], tmp_path,
    )
    assert len(fitted_ranges) == 2
    assert fitted_ranges[0][1] < analysis["final_start"]
    assert fitted_ranges[1][1] == analysis["cutoff"]
    assert analysis["final_scored_days"] == 180
    assert analysis["frozen"]["forest"] is not analysis["production"]["forest"]
    assert analysis["final_results"].iloc[0]["Persistence"] == 30.0 + (650 - 181) % 30


def test_anomaly_fallback_uses_all_training_rows(monkeypatch):
    class AlmostAllAnomalous:
        def __init__(self, **_kwargs):
            pass

        def fit_predict(self, x):
            result = np.full(len(x), -1)
            result[0] = 1
            return result

    monkeypatch.setattr(pipeline, "IsolationForest", AlmostAllAnomalous)
    rows = pipeline.engineer(
        sample_daily(date(2025, 1, 1), 60), pipeline.CAMS_POLLUTANTS["PM2.5"]
    )
    fitted = pipeline.fit_models(rows, pipeline.CAMS_POLLUTANTS["PM2.5"])
    assert fitted["anomaly_fallback"]
    assert fitted["fit_rows"] == 59


def test_recursive_bridge_uses_only_prior_prediction(monkeypatch):
    pollutant = pipeline.CAMS_POLLUTANTS["PM2.5"]
    cutoff = date(2026, 10, 4)
    prior = sample_daily(cutoff - timedelta(days=19), 20)
    engineered = pipeline.engineer(prior, pollutant)
    analysis = {
        "cutoff": cutoff, "pollutant": pollutant, "engineered": engineered,
        "production": object(), "champion": "Persistence",
    }
    weather = sample_daily(cutoff + timedelta(days=1), 11).drop(columns=[pollutant.target])
    lags = []

    def predict(_fitted, _method, row, _pollutant):
        lags.append(row["Lag1"])
        return row["Lag1"] + 1

    monkeypatch.setattr(pipeline, "predict_one", predict)
    monkeypatch.setattr(pipeline, "recursive_residuals", lambda _analysis: {})
    forecast = pipeline.forecast(analysis, weather, date(2026, 10, 8))
    assert len(forecast) == 7
    assert forecast.iloc[0]["Lead"] == 5
    assert forecast.iloc[-1]["Lead"] == 11
    assert np.allclose(np.diff(lags), 1.0)


def test_openaq_parser_deduplicates_subhourly_and_drops_invalid():
    frame = pd.DataFrame({
        "location_id": [17, 17, 17, 17],
        "datetime": [
            "2025-04-01T00:00:00+05:30", "2025-04-01T00:15:00+05:30",
            "2025-04-01T00:30:00+05:30", "2025-04-01T01:00:00+05:30",
        ],
        "parameter": ["pm25"] * 4, "value": [10, 10, "bad", -9],
    })
    buffer = BytesIO()
    frame.to_csv(buffer, index=False, compression="gzip")
    parsed = station.parse_openaq(buffer.getvalue())
    assert len(parsed) == 1
    assert parsed.iloc[0]["value"] == 10.0
    assert parsed.iloc[0]["local_hour"] == "2025-04-01T00:00:00"


def test_station_audit_and_18_distinct_hour_rule(tmp_path):
    path = tmp_path / "station.sqlite"
    with station.connect(path) as connection:
        for pollutant in station.POLLUTANTS:
            for hour in range(24):
                timestamp = f"2025-04-01T{hour:02d}:00:00"
                connection.execute(
                    "INSERT INTO hours VALUES (?, ?, ?, ?)",
                    ("XKDR", pollutant, timestamp, 50.0),
                )
                connection.execute(
                    "INSERT INTO hours VALUES (?, ?, ?, ?)",
                    ("OpenAQ", pollutant, timestamp, 50.0),
                )
            for hour in range(17 if pollutant == "pm25" else 18):
                connection.execute(
                    "INSERT INTO hours VALUES (?, ?, ?, ?)",
                    ("XKDR", pollutant, f"2025-04-02T{hour:02d}:00:00", 40.0),
                )
    audit = station.audit_overlap(path)
    assert audit["passed"]
    daily = station.daily_station(path).set_index("Date")
    assert np.isnan(daily.loc[date(2025, 4, 2), "Observed_PM2.5"])
    assert daily.loc[date(2025, 4, 2), "Observed_PM10"] == 40.0
    with station.connect(path) as connection:
        for day in ("2025-04-01", "2025-04-02"):
            connection.execute(
                "INSERT INTO weather_daily VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (day, 25.0, 60.0, 8.0, 0.0, 1005.0, 30.0, 270.0, 15.0, 700.0, "Asia/Kolkata"),
            )
    common, _ = station.cached_station_dataset(path)
    assert common["Date"].tolist() == [date(2025, 4, 1)]
    with station.connect(path) as connection:
        for hour in (0, 1):
            connection.execute(
                "UPDATE hours SET value=80 WHERE source='OpenAQ' AND pollutant='pm10' AND local_hour=?",
                (f"2025-04-01T{hour:02d}:00:00",),
            )
    assert not station.audit_overlap(path)["passed"]


def test_xkdr_fetch_skips_cached_dates(monkeypatch, tmp_path):
    calls = []

    def fake_request(start, end):
        calls.append((start, end))
        return pd.DataFrame(columns=["pollutant", "local_hour", "value"])

    monkeypatch.setattr(station, "_xkdr_request", fake_request)
    path = tmp_path / "station.sqlite"
    station.fetch_xkdr(date(2025, 1, 1), date(2025, 1, 3), path)
    station.fetch_xkdr(date(2025, 1, 1), date(2025, 1, 3), path)
    assert calls == [(date(2025, 1, 1), date(2025, 1, 3))]


def test_openaq_fetch_is_idempotent_and_failure_keeps_prior_rows(monkeypatch, tmp_path):
    day = date(2025, 4, 1)
    path = tmp_path / "station.sqlite"
    calls = []
    monkeypatch.setattr(station, "list_openaq_days", lambda *_args: [day])

    def fake_request(request_day):
        calls.append(request_day)
        return pd.DataFrame({
            "pollutant": ["pm25"],
            "local_hour": ["2025-04-01T00:00:00"],
            "value": [42.0],
        })

    monkeypatch.setattr(station, "_openaq_request", fake_request)
    station.fetch_openaq(day, day, path)
    station.fetch_openaq(day, day, path)
    assert calls == [day]
    with station.connect(path) as connection:
        assert connection.execute("SELECT COUNT(*) FROM hours").fetchone()[0] == 1

    later = date(2025, 4, 2)
    monkeypatch.setattr(station, "list_openaq_days", lambda *_args: [day, later])
    monkeypatch.setattr(station, "_openaq_request", lambda _day: (_ for _ in ()).throw(OSError("offline")))
    with pytest.raises(OSError, match="offline"):
        station.fetch_openaq(day, later, path)
    with station.connect(path) as connection:
        assert connection.execute("SELECT COUNT(*) FROM hours").fetchone()[0] == 1


def test_archived_weather_requires_24_numeric_values():
    times = pd.date_range("2026-04-01", periods=48, freq="h")
    payload = {
        "timezone": "Asia/Kolkata",
        "hourly": {"time": times.strftime("%Y-%m-%dT%H:%M").tolist()},
    }
    for api_field in station.WEATHER_FIELDS:
        payload["hourly"][api_field] = [10.0] * 48
    payload["hourly"]["wind_speed_10m"][4] = None
    result = archived.parse_payload(payload)
    assert result["Date"].tolist() == [date(2026, 4, 2)]


def test_archived_weather_cache_fetches_only_missing_dates(monkeypatch, tmp_path):
    requests_made = []

    class Response:
        def __init__(self, payload):
            self.payload = payload

        def raise_for_status(self):
            pass

        def json(self):
            return self.payload

    def fake_get(_url, params, timeout):
        requests_made.append((params["start_date"], params["end_date"]))
        times = pd.date_range(params["start_date"], periods=24 * (
            (date.fromisoformat(params["end_date"]) - date.fromisoformat(params["start_date"])).days + 1
        ), freq="h")
        hourly = {"time": times.strftime("%Y-%m-%dT%H:%M").tolist()}
        for name in station.WEATHER_FIELDS:
            hourly[name] = [10.0] * len(times)
        return Response({"timezone": "Asia/Kolkata", "hourly": hourly})

    monkeypatch.setattr(archived.requests, "get", fake_get)
    path = tmp_path / "weather.sqlite"
    archived.fetch_archived_daily(28.56, 77.18, date(2026, 4, 1), date(2026, 4, 2), path)
    archived.fetch_archived_daily(28.56, 77.18, date(2026, 4, 1), date(2026, 4, 2), path)
    archived.fetch_archived_daily(28.56, 77.18, date(2026, 4, 1), date(2026, 4, 3), path)
    assert requests_made == [("2026-04-01", "2026-04-02"), ("2026-04-03", "2026-04-03")]


def test_final_test_uses_archived_weather_but_production_does_not(tmp_path):
    source = sample_daily(date(2024, 1, 1), 650)
    cutoff = max(source["Date"])
    final_start = cutoff - timedelta(days=179)
    archived_weather = source.loc[source["Date"].between(final_start, cutoff)].drop(
        columns=["CAMS_PM2.5"]
    ).copy()
    archived_weather["Temperature"] = 99.0
    analysis = pipeline.analyze(
        source, pipeline.CAMS_POLLUTANTS["PM2.5"], tmp_path,
        archived_weather,
    )
    holdout = analysis["validation_engineered"].loc[
        analysis["validation_engineered"]["Date"] >= final_start
    ]
    assert set(holdout["Temperature"]) == {99.0}
    assert set(analysis["engineered"]["Temperature"]) == {25.0}
    assert "stitched historical-forecast" in analysis["weather_test_note"]
    with pytest.raises(ValueError, match="incomplete"):
        pipeline.analyze(
            source, pipeline.CAMS_POLLUTANTS["PM2.5"], tmp_path,
            archived_weather.iloc[:-1],
        )


def test_five_year_cap_excludes_lag_context_from_fitting(monkeypatch, tmp_path):
    source = sample_daily(date(2020, 1, 1), 2000)
    fits = []
    original = pipeline.fit_models

    def record(training, pollutant):
        fits.append(min(training["Date"]))
        return original(training, pollutant)

    monkeypatch.setattr(pipeline, "fit_models", record)
    monkeypatch.setattr(
        pipeline, "evaluate_development",
        lambda *_args: (pd.DataFrame(), "Persistence"),
    )
    analysis = pipeline.analyze(source, pipeline.CAMS_POLLUTANTS["PM2.5"], tmp_path)
    assert all(day >= analysis["start"] for day in fits)


def test_development_artifact_reuse_and_invalidation(monkeypatch, tmp_path):
    pollutant = pipeline.CAMS_POLLUTANTS["PM2.5"]
    data = pipeline.engineer(sample_daily(date(2024, 1, 1), 450), pollutant)
    start = date(2025, 1, 5)
    window = {
        "id": "window", "year": "2025-26", "season": "Winter",
        "start": start, "end": start + timedelta(days=59),
        "training_start": date(2024, 1, 1),
    }
    monkeypatch.setattr(pipeline, "rolling_windows", lambda *_args: [window])
    calls = []

    def fake_fit(training, _pollutant):
        assert max(training["Date"]) < start
        calls.append(len(training))
        return object()

    def fake_evaluate(_fitted, _holdout, _pollutant):
        return pd.DataFrame(), {
            method: {"mae": 10.0, "recall": 0.8, "exceedance_rate": 0.3, "n": 60}
            for method in pipeline.METHODS
        }

    monkeypatch.setattr(pipeline, "fit_models", fake_fit)
    monkeypatch.setattr(pipeline, "evaluate", fake_evaluate)
    pipeline.evaluate_development(data, pollutant, date(2025, 6, 1), tmp_path)
    pipeline.evaluate_development(data, pollutant, date(2025, 6, 1), tmp_path)
    assert len(calls) == 1
    changed = data.copy()
    changed.loc[changed["Date"] == start, pollutant.target] += 1
    pipeline.evaluate_development(changed, pollutant, date(2025, 6, 1), tmp_path)
    assert len(calls) == 2
    monkeypatch.setattr(pipeline, "RIDGE_ALPHA", 11.0)
    pipeline.evaluate_development(changed, pollutant, date(2025, 6, 1), tmp_path)
    assert len(calls) == 3
