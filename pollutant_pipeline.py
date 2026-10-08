"""Shared date-aware modeling for PM2.5 and PM10 daily forecasts."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from functools import lru_cache
import hashlib
import json
from pathlib import Path
from typing import Any

import joblib
import holidays
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler


MODEL_VERSION = 2
RF_ESTIMATORS = 300
RF_MIN_SAMPLES_LEAF = 2
RIDGE_ALPHA = 10.0
ANOMALY_CONTAMINATION = "auto"
FINAL_TEST_DAYS = 180
FORECAST_DAYS = 7
METHODS = ("Persistence", "Seven-day mean", "Climatology", "Ridge", "Random Forest")
FEATURES = (
    "Temperature", "Humidity", "Wind_Speed", "Precipitation", "Pressure_MSL",
    "Cloud_Cover", "Wind_Gusts", "Boundary_Layer_Height", "Wind_Direction_Sin",
    "Wind_Direction_Cos", "Lag1", "Lag2", "Lag3", "Lag7", "Mean7",
    "Day_Sin", "Day_Cos", "Weekend", "Indian_Holiday",
    "Diwali", "Days_From_Diwali",
)
SEASONS = (
    ("Monsoon transition", 8, 12, 0),
    ("Post-monsoon", 10, 21, 0),
    ("Winter", 12, 20, 0),
    ("Spring", 3, 15, 1),
)


@dataclass(frozen=True)
class Pollutant:
    key: str
    api_field: str
    target: str
    threshold: float
    source: str


CAMS_POLLUTANTS = {
    "PM2.5": Pollutant("PM2.5", "pm2_5", "CAMS_PM2.5", 60.0, "CAMS modeled"),
    "PM10": Pollutant("PM10", "pm10", "CAMS_PM10", 100.0, "CAMS modeled"),
}
STATION_POLLUTANTS = {
    "PM2.5": Pollutant("PM2.5", "pm25", "Observed_PM2.5", 60.0, "R.K. Puram station"),
    "PM10": Pollutant("PM10", "pm10", "Observed_PM10", 100.0, "R.K. Puram station"),
}


def five_year_start(cutoff: date) -> date:
    return (pd.Timestamp(cutoff) - pd.DateOffset(years=5)).date() + timedelta(days=1)


def _weather_value(row: dict[str, Any], key: str) -> float:
    return float(pd.to_numeric(row.get(key), errors="coerce"))


@lru_cache(maxsize=20)
def _calendar(year: int) -> tuple[frozenset[date], tuple[date, ...]]:
    india = holidays.country_holidays(
        "IN", years=[year - 1, year, year + 1], language="en_US"
    )
    diwali_days = tuple(sorted(
        day for day, name in india.items()
        if "diwali" in name.casefold() or "deepavali" in name.casefold()
    ))
    return frozenset(india), diwali_days


def feature_row(day: date, weather: dict[str, Any], known: dict[date, float]) -> dict[str, Any]:
    row = {"Date": day}
    for name in (
        "Temperature", "Humidity", "Wind_Speed", "Precipitation", "Pressure_MSL",
        "Cloud_Cover", "Wind_Gusts", "Boundary_Layer_Height",
    ):
        row[name] = _weather_value(weather, name)
    direction = _weather_value(weather, "Wind_Direction")
    row["Wind_Direction_Sin"] = np.sin(np.deg2rad(direction))
    row["Wind_Direction_Cos"] = np.cos(np.deg2rad(direction))
    for offset in (1, 2, 3, 7):
        row[f"Lag{offset}"] = known.get(day - timedelta(days=offset), np.nan)
    prior = [known.get(day - timedelta(days=offset), np.nan) for offset in range(1, 8)]
    numeric = [float(value) for value in prior if pd.notna(value)]
    row["Mean7"] = float(np.mean(numeric)) if len(numeric) == 7 else np.nan
    annual = 2.0 * np.pi * day.timetuple().tm_yday / 365.25
    row["Day_Sin"] = float(np.sin(annual))
    row["Day_Cos"] = float(np.cos(annual))
    row["Weekend"] = float(day.weekday() >= 5)
    holidays_in_year, diwali_days = _calendar(day.year)
    row["Indian_Holiday"] = float(day in holidays_in_year)
    distance = min(((day - festival).days for festival in diwali_days), key=abs) if diwali_days else 30
    row["Diwali"] = float(abs(distance) <= 1)
    row["Days_From_Diwali"] = float(np.clip(distance, -30, 30))
    return row


def engineer(frame: pd.DataFrame, pollutant: Pollutant) -> pd.DataFrame:
    data = frame.copy().sort_values("Date").drop_duplicates("Date", keep="last")
    data[pollutant.target] = pd.to_numeric(data[pollutant.target], errors="coerce")
    known = {
        row["Date"]: float(row[pollutant.target])
        for row in data.to_dict("records")
        if pd.notna(row[pollutant.target])
    }
    rows = []
    for original in data.to_dict("records"):
        row = feature_row(original["Date"], original, known)
        row[pollutant.target] = original[pollutant.target]
        rows.append(row)
    return pd.DataFrame(rows)


def eligible(frame: pd.DataFrame, pollutant: Pollutant) -> pd.DataFrame:
    return frame.dropna(subset=[pollutant.target, "Lag1"]).copy()


def fit_models(training: pd.DataFrame, pollutant: Pollutant) -> dict[str, Any]:
    data = eligible(training, pollutant)
    if len(data) < 30:
        raise ValueError(f"Only {len(data)} usable training rows for {pollutant.key}.")
    imputer = SimpleImputer(strategy="mean", keep_empty_features=True)
    x = imputer.fit_transform(data[list(FEATURES)])
    y = data[pollutant.target].to_numpy(dtype=float)
    if len(data) < 8:
        normal = np.ones(len(data), dtype=bool)
    else:
        normal = IsolationForest(random_state=42, contamination=ANOMALY_CONTAMINATION).fit_predict(
            np.column_stack([y, x])
        ) == 1
    fallback = int(normal.sum()) < 10
    fit_mask = np.ones(len(data), dtype=bool) if fallback else normal
    forest = RandomForestRegressor(
        n_estimators=RF_ESTIMATORS,
        min_samples_leaf=RF_MIN_SAMPLES_LEAF, random_state=42, n_jobs=-1
    ).fit(x[fit_mask], y[fit_mask])
    forest.n_jobs = 1
    scaler = StandardScaler().fit(x[fit_mask])
    ridge = Ridge(alpha=RIDGE_ALPHA).fit(scaler.transform(x[fit_mask]), y[fit_mask])
    return {
        "imputer": imputer, "scaler": scaler, "forest": forest, "ridge": ridge,
        "training": data[["Date", pollutant.target]].copy(),
        "normal_rows": int(normal.sum()), "fit_rows": int(fit_mask.sum()),
        "anomaly_fallback": fallback,
    }


def _climatology(training: pd.DataFrame, day: date, pollutant: Pollutant) -> float:
    dates = pd.to_datetime(training["Date"])
    days = dates.dt.dayofyear.to_numpy()
    target_day = day.timetuple().tm_yday
    distance = np.abs(days - target_day)
    local = np.minimum(distance, 366 - distance) <= 15
    values = training[pollutant.target].to_numpy(dtype=float)
    return float(np.mean(values[local])) if local.any() else float(np.mean(values))


def predict_methods(
    fitted: dict[str, Any], rows: pd.DataFrame, pollutant: Pollutant
) -> dict[str, np.ndarray]:
    x = fitted["imputer"].transform(rows[list(FEATURES)])
    lag = rows["Lag1"].to_numpy(dtype=float)
    rolling = rows["Mean7"].to_numpy(dtype=float)
    return {
        "Persistence": lag,
        "Seven-day mean": np.where(np.isfinite(rolling), rolling, lag),
        "Climatology": np.array([
            _climatology(fitted["training"], day, pollutant) for day in rows["Date"]
        ]),
        "Ridge": np.maximum(0, fitted["ridge"].predict(fitted["scaler"].transform(x))),
        "Random Forest": np.maximum(0, fitted["forest"].predict(x)),
    }


def score(actual: np.ndarray, predicted: np.ndarray, threshold: float) -> dict[str, Any]:
    positive = actual > threshold
    predicted_positive = predicted > threshold
    tp = int(np.sum(positive & predicted_positive))
    fp = int(np.sum(~positive & predicted_positive))
    fn = int(np.sum(positive & ~predicted_positive))
    tn = int(np.sum(~positive & ~predicted_positive))
    return {
        "n": int(len(actual)),
        "mae": float(mean_absolute_error(actual, predicted)),
        "rmse": float(np.sqrt(mean_squared_error(actual, predicted))),
        "r2": float(r2_score(actual, predicted)) if len(actual) >= 2 else None,
        "exceedance_rate": float(positive.mean()) if len(actual) else None,
        "recall": tp / (tp + fn) if tp + fn else None,
        "precision": tp / (tp + fp) if tp + fp else None,
        "false_positive_rate": fp / (fp + tn) if fp + tn else None,
    }


def evaluate(
    fitted: dict[str, Any], holdout: pd.DataFrame, pollutant: Pollutant
) -> tuple[pd.DataFrame, dict[str, dict[str, Any]]]:
    usable = eligible(holdout, pollutant).sort_values("Date").copy()
    if usable.empty:
        raise ValueError("The holdout has no dates with a target and preceding-day lag.")
    forecasts = predict_methods(fitted, usable, pollutant)
    results = pd.DataFrame({"Date": usable["Date"], "Actual": usable[pollutant.target]})
    results = results.reset_index(drop=True)
    for method, values in forecasts.items():
        results[method] = values
    actual = results["Actual"].to_numpy(dtype=float)
    metrics = {
        method: score(actual, results[method].to_numpy(dtype=float), pollutant.threshold)
        for method in METHODS
    }
    return results, metrics


def rolling_windows(
    engineered: pd.DataFrame, pollutant: Pollutant, final_start: date
) -> list[dict[str, Any]]:
    first = min(engineered["Date"])
    windows = []
    for cycle_year in range(first.year, final_start.year + 1):
        for season, month, day, offset in SEASONS:
            start = date(cycle_year + offset, month, day)
            end = start + timedelta(days=59)
            if end >= final_start:
                continue
            training_start = max(first, five_year_start(start - timedelta(days=1)))
            training = engineered.loc[engineered["Date"].between(training_start, start - timedelta(days=1))]
            holdout = engineered.loc[engineered["Date"].between(start, end)]
            if len(eligible(training, pollutant)) >= 365 and len(eligible(holdout, pollutant)) >= 45:
                windows.append({
                    "id": f"{cycle_year}_{season.lower().replace(' ', '_')}",
                    "year": f"{cycle_year}-{str(cycle_year + 1)[-2:]}",
                    "season": season, "start": start, "end": end,
                    "training_start": training_start,
                })
    return windows


def _artifact_path(
    relevant: pd.DataFrame, pollutant: Pollutant, identity: str, cache_dir: Path
) -> Path:
    selected = relevant[["Date", pollutant.target, *FEATURES]].copy()
    selected["Date"] = selected["Date"].astype(str)
    payload = pd.util.hash_pandas_object(selected, index=False).values.tobytes()
    configuration = {
        "version": MODEL_VERSION, "source": pollutant.source,
        "pollutant": pollutant.key, "window": identity,
        "threshold": pollutant.threshold, "features": FEATURES,
        "rf_estimators": RF_ESTIMATORS,
        "rf_min_samples_leaf": RF_MIN_SAMPLES_LEAF,
        "ridge_alpha": RIDGE_ALPHA,
        "anomaly_contamination": ANOMALY_CONTAMINATION,
    }
    fingerprint = hashlib.sha256(
        payload + json.dumps(configuration, sort_keys=True).encode()
    ).hexdigest()
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / f"shared_{pollutant.source.replace(' ', '_')}_{pollutant.key}_{identity}_{fingerprint}.joblib"


def evaluate_development(
    engineered: pd.DataFrame,
    pollutant: Pollutant,
    final_start: date,
    cache_dir: Path,
) -> tuple[pd.DataFrame, str]:
    summaries = []
    for window in rolling_windows(engineered, pollutant, final_start):
        relevant = engineered.loc[engineered["Date"].between(window["training_start"], window["end"])]
        path = _artifact_path(relevant, pollutant, window["id"], cache_dir)
        if path.exists():
            metrics = joblib.load(path)
        else:
            training = relevant.loc[relevant["Date"] < window["start"]]
            holdout = relevant.loc[relevant["Date"].between(window["start"], window["end"])]
            _, metrics = evaluate(fit_models(training, pollutant), holdout, pollutant)
            joblib.dump(metrics, path)
        for method, values in metrics.items():
            summaries.append({
                "Year": window["year"], "Season": window["season"],
                "Method": method, "MAE": values["mae"], "Recall": values["recall"],
                "Exceedance rate": values["exceedance_rate"], "Scored days": values["n"],
            })
    summary = pd.DataFrame(summaries)
    if summary.empty:
        return summary, "Persistence"
    medians = summary.groupby("Method")["MAE"].median()
    baseline = float(medians["Persistence"])
    eligible_methods = []
    baseline_windows = summary.loc[summary["Method"] == "Persistence"]
    for method in METHODS:
        if method == "Persistence" or float(medians[method]) >= baseline * 0.98:
            continue
        candidate = summary.loc[summary["Method"] == method]
        merged = candidate.merge(
            baseline_windows, on=["Year", "Season"], suffixes=("", "_baseline")
        )
        recall_pairs = merged.dropna(subset=["Recall", "Recall_baseline"])
        if not recall_pairs.empty and (recall_pairs["Recall_baseline"] - recall_pairs["Recall"]).max() > 0.05:
            continue
        if (merged["MAE"] <= merged["MAE_baseline"]).sum() < int(np.ceil(0.75 * len(merged))):
            continue
        eligible_methods.append(method)
    champion = min(eligible_methods, key=lambda method: medians[method]) if eligible_methods else "Persistence"
    return summary, champion


def analyze(
    daily: pd.DataFrame,
    pollutant: Pollutant,
    cache_dir: Path = Path("model_cache"),
    historical_forecast_weather: pd.DataFrame | None = None,
) -> dict[str, Any]:
    clean = daily.dropna(subset=["Date"]).copy()
    clean[pollutant.target] = pd.to_numeric(clean[pollutant.target], errors="coerce")
    observed = clean.dropna(subset=[pollutant.target])
    if observed.empty:
        raise ValueError(f"No {pollutant.source} {pollutant.key} target is available.")
    cutoff = max(observed["Date"])
    start = five_year_start(cutoff)
    clean = clean.loc[clean["Date"].between(start - timedelta(days=7), cutoff)]
    engineered = engineer(clean, pollutant)
    final_start = cutoff - timedelta(days=FINAL_TEST_DAYS - 1)
    training = engineered.loc[engineered["Date"].between(start, final_start - timedelta(days=1))]
    validation_engineered = engineered
    weather_note = "Observed historical weather; optimistic forecast proxy"
    if historical_forecast_weather is not None:
        weather = historical_forecast_weather.set_index("Date")
        expected = set(pd.date_range(final_start, cutoff, freq="D").date)
        weather_columns = (
            "Temperature", "Humidity", "Wind_Speed", "Precipitation",
            "Pressure_MSL", "Cloud_Cover", "Wind_Direction", "Wind_Gusts",
            "Boundary_Layer_Height",
        )
        if set(weather.index) != expected or weather[list(weather_columns)].isna().any().any():
            raise ValueError("Archived forecast weather is incomplete for the 180-day test.")
        validation_input = clean.copy()
        holdout_mask = validation_input["Date"].between(final_start, cutoff)
        for column in weather_columns:
            validation_input.loc[holdout_mask, column] = validation_input.loc[
                holdout_mask, "Date"
            ].map(weather[column]).to_numpy()
        validation_engineered = engineer(validation_input, pollutant)
        weather_note = (
            "Open-Meteo stitched historical-forecast weather; near-term archived "
            "runs, not a fixed one-day-ahead issue-time forecast"
        )
    holdout = validation_engineered.loc[
        validation_engineered["Date"].between(final_start, cutoff)
    ]
    if len(eligible(training, pollutant)) < 365:
        raise ValueError("Fewer than 365 usable training dates precede the final test.")
    if len(eligible(holdout, pollutant)) < 90:
        raise ValueError("Fewer than 90 scored dates remain in the final 180-day test.")
    development, champion = evaluate_development(engineered, pollutant, final_start, cache_dir)
    frozen = fit_models(training, pollutant)
    test_results, test_metrics = evaluate(frozen, holdout, pollutant)
    production = fit_models(engineered.loc[engineered["Date"] >= start], pollutant)
    return {
        "pollutant": pollutant, "start": start, "cutoff": cutoff,
        "calendar_days": (cutoff - start).days + 1,
        "valid_days": int(observed["Date"].between(start, cutoff).sum()),
        "training_rows": int(len(production["training"])),
        "final_train_rows": int(len(frozen["training"])),
        "final_start": final_start, "final_scored_days": len(test_results),
        "development": development, "champion": champion,
        "final_results": test_results, "final_metrics": test_metrics,
        "frozen": frozen, "production": production, "engineered": engineered,
        "validation_engineered": validation_engineered, "weather_test_note": weather_note,
    }


def predict_one(
    fitted: dict[str, Any], method: str, row: dict[str, Any], pollutant: Pollutant
) -> float:
    if method == "Persistence":
        return max(0.0, float(row["Lag1"]))
    if method == "Seven-day mean":
        rolling = row["Mean7"]
        return max(0.0, float(rolling if np.isfinite(rolling) else row["Lag1"]))
    if method == "Climatology":
        return max(0.0, _climatology(fitted["training"], row["Date"], pollutant))
    frame = pd.DataFrame([row])
    x = fitted["imputer"].transform(frame[list(FEATURES)])
    if method == "Ridge":
        value = fitted["ridge"].predict(fitted["scaler"].transform(x))[0]
    elif method == "Random Forest":
        value = fitted["forest"].predict(x)[0]
    else:
        raise ValueError(f"Unknown forecast method: {method}")
    return max(0.0, float(value))


def recursive_residuals(
    analysis: dict[str, Any], max_lead: int = 12
) -> dict[int, tuple[float, float, int]]:
    data = analysis.get("validation_engineered", analysis["engineered"])
    pollutant = analysis["pollutant"]
    method = analysis["champion"]
    known_actual = {
        row["Date"]: float(row[pollutant.target])
        for row in data.to_dict("records") if pd.notna(row[pollutant.target])
    }
    weather = data.set_index("Date")
    errors: dict[int, list[float]] = {lead: [] for lead in range(1, max_lead + 1)}
    for origin in pd.date_range(analysis["final_start"], analysis["cutoff"] - timedelta(days=max_lead), freq="7D").date:
        if origin - timedelta(days=1) not in known_actual:
            continue
        history = {day: value for day, value in known_actual.items() if day < origin}
        for lead in range(1, max_lead + 1):
            day = origin + timedelta(days=lead - 1)
            if day not in weather.index:
                break
            row = feature_row(day, weather.loc[day].to_dict(), history)
            if pd.isna(row["Lag1"]):
                break
            prediction = predict_one(analysis["frozen"], method, row, pollutant)
            history[day] = prediction
            if day in known_actual:
                errors[lead].append(known_actual[day] - prediction)
    result = {}
    for lead, values in errors.items():
        if values:
            result[lead] = (
                float(np.quantile(values, 0.10)), float(np.quantile(values, 0.90)), len(values)
            )
    return result


def forecast(
    analysis: dict[str, Any], forecast_weather: pd.DataFrame, today: date,
    max_bridge_days: int = 5,
) -> pd.DataFrame:
    cutoff = analysis["cutoff"]
    if (today - cutoff).days > max_bridge_days:
        raise ValueError(f"Latest {analysis['pollutant'].source} observation is {(today-cutoff).days} days old.")
    weather = forecast_weather.set_index("Date")
    actual = analysis["engineered"].dropna(subset=[analysis["pollutant"].target])
    known = {row["Date"]: float(row[analysis["pollutant"].target]) for row in actual.to_dict("records")}
    residuals = recursive_residuals(analysis)
    rows = []
    lead = 0
    lower_offset = 0.0
    upper_offset = 0.0
    for day in pd.date_range(cutoff + timedelta(days=1), today + timedelta(days=FORECAST_DAYS), freq="D").date:
        lead += 1
        if day not in weather.index:
            raise ValueError(f"Weather forecast is missing {day}.")
        row = feature_row(day, weather.loc[day].to_dict(), known)
        if pd.isna(row["Lag1"]):
            raise ValueError(f"Previous-day {analysis['pollutant'].key} is missing for {day}.")
        prediction = predict_one(analysis["production"], analysis["champion"], row, analysis["pollutant"])
        known[day] = prediction
        if day > today:
            lower, upper, count = residuals.get(lead, (0.0, 0.0, 0))
            if count >= 5:
                lower_offset = min(lower_offset, lower)
                upper_offset = max(upper_offset, upper)
            rows.append({
                "Date": day, "Lead": lead, "Prediction": prediction,
                "Lower_80": max(0.0, prediction + lower_offset) if count >= 5 else np.nan,
                "Upper_80": prediction + upper_offset if count >= 5 else np.nan,
                "Interval windows": count,
                "Exceeds": prediction > analysis["pollutant"].threshold,
            })
    return pd.DataFrame(rows)


def summary_json(analysis: dict[str, Any]) -> dict[str, Any]:
    p = analysis["pollutant"]
    development = analysis["development"]
    seasonal = json.loads(development.to_json(orient="records")) if not development.empty else []
    seasonal_medians = (
        {method: float(value) for method, value in development.groupby("Method")["MAE"].median().items()}
        if not development.empty else {}
    )
    return {
        "source": p.source, "pollutant": p.key, "threshold_ug_m3": p.threshold,
        "date_start": analysis["start"].isoformat(),
        "date_end": analysis["cutoff"].isoformat(),
        "calendar_days": analysis["calendar_days"], "valid_days": analysis["valid_days"],
        "production_training_rows": analysis["training_rows"],
        "final_train_rows": analysis["final_train_rows"],
        "final_test_start": analysis["final_start"].isoformat(),
        "final_scored_days": analysis["final_scored_days"],
        "development_windows": int(len(development.drop_duplicates(["Year", "Season"]))) if not development.empty else 0,
        "development_median_mae": seasonal_medians,
        "development_results": seasonal,
        "selected_method": analysis["champion"],
        "final_test_metrics": analysis["final_metrics"],
        "weather_test_note": analysis["weather_test_note"],
    }
