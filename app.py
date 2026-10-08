"""Urban Air Quality Early Warning System.

Run with:
    streamlit run app.py

The app caches CAMS, weather, and optional NASA FIRMS observations; compares
leakage-safe feature groups across rolling yearly seasonal holdouts; and
produces a separate seven-day forward forecast. CPCB observations are stored
separately and are never substituted for the modeled CAMS target.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
import hashlib
from io import StringIO
import json
import math
from pathlib import Path
import sqlite3
from typing import Any
from zoneinfo import ZoneInfo

import holidays
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import streamlit as st
from requests.adapters import HTTPAdapter
from sklearn.ensemble import IsolationForest, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from urllib3.util.retry import Retry

try:
    from atmospheric_hero import feature_enabled as video_hero_feature_enabled
    from atmospheric_hero import render_atmospheric_hero
except ImportError:
    video_hero_feature_enabled = None
    render_atmospheric_hero = None


DATA_FILE = Path("cpcb_data.csv")
CACHE_DB_FILE = Path("air_quality_cache.sqlite")
MODEL_CACHE_DIR = Path("model_cache")
CPCB_API_URL = (
    "https://api.data.gov.in/resource/"
    "3b01bcb8-0b14-4abf-b6f2-c1bfd384ba69"
)
AIR_QUALITY_URL = "https://air-quality-api.open-meteo.com/v1/air-quality"
HISTORICAL_WEATHER_URL = "https://archive-api.open-meteo.com/v1/archive"
FORECAST_WEATHER_URL = "https://api.open-meteo.com/v1/forecast"
FIRMS_AREA_URL = "https://firms.modaps.eosdis.nasa.gov/api/area/csv"
FIRMS_SOURCE = "VIIRS_NOAA20_NRT"
FIRMS_RADIUS_KM = 500.0

WEATHER_API_FIELDS = {
    "temperature_2m": "temperature",
    "relative_humidity_2m": "humidity",
    "wind_speed_10m": "wind_speed",
    "precipitation": "precipitation",
    "pressure_msl": "pressure_msl",
    "cloud_cover": "cloud_cover",
    "wind_direction_10m": "wind_direction",
    "wind_gusts_10m": "wind_gusts",
    "boundary_layer_height": "boundary_layer_height",
}
WEATHER_DAILY_NAMES = {
    "temperature": "Temperature",
    "humidity": "Humidity",
    "wind_speed": "Wind_Speed",
    "precipitation": "Precipitation",
    "pressure_msl": "Pressure_MSL",
    "cloud_cover": "Cloud_Cover",
    "wind_direction": "Wind_Direction",
    "wind_gusts": "Wind_Gusts",
    "boundary_layer_height": "Boundary_Layer_Height",
}

HISTORY_COLUMNS = [
    "Date",
    "CAMS_PM2.5",
    "CPCB_PM2.5",
    "Manual_PM2.5",
    "Temperature",
    "Humidity",
    "Wind_Speed",
    "Precipitation",
    "Pressure_MSL",
    "Cloud_Cover",
    "Wind_Direction",
    "Wind_Gusts",
    "Boundary_Layer_Height",
    "Timezone",
    "CPCB_Station_Count",
]
MODEL_FEATURES = [
    "Temperature",
    "Humidity",
    "Wind_Speed",
    "PM2.5_Yesterday",
]
WEATHER_FEATURES = list(WEATHER_DAILY_NAMES.values())
CORE_WEATHER_FEATURES = ["Temperature", "Humidity", "Wind_Speed"]
BASELINE_FEATURES = MODEL_FEATURES.copy()
HISTORY_FEATURES = [
    "Temperature", "Humidity", "Wind_Speed",
    "PM2.5_Lag1", "PM2.5_Lag2", "PM2.5_Lag3", "PM2.5_Lag7",
    "PM2.5_Mean3", "PM2.5_Mean7", "PM2.5_Std7",
]
EXPANDED_WEATHER_FEATURES = HISTORY_FEATURES + [
    "Precipitation", "Pressure_MSL", "Cloud_Cover", "Wind_Gusts",
    "Wind_Direction_Sin", "Wind_Direction_Cos", "Boundary_Layer_Height",
]
CALENDAR_FEATURES = EXPANDED_WEATHER_FEATURES + [
    "Is_Weekend", "Is_Indian_Holiday", "Is_Diwali",
    "Days_From_Diwali", "Day_Of_Year_Sin", "Day_Of_Year_Cos",
]
FIRE_FEATURES = CALENDAR_FEATURES + [
    "Fire_Count_1d", "Fire_Count_3d", "Fire_Count_7d",
    "Fire_FRP_3d", "Fire_FRP_7d", "Fire_Distance_Weighted_FRP_3d",
    "Fire_Upwind_FRP_3d",
]
FEATURE_GROUPS = {
    "Baseline": BASELINE_FEATURES,
    "Pollution history": HISTORY_FEATURES,
    "Expanded weather": EXPANDED_WEATHER_FEATURES,
    "Calendar and festivals": CALENDAR_FEATURES,
    "Satellite fires": FIRE_FEATURES,
}
HISTORY_DAYS = 90
TRAIN_DAYS = 60
TEST_DAYS = 30
BUFFER_DAYS = 120
FORECAST_DAYS = 7
CPCB_SAFE_LIMIT = 60.0
MIN_SEASON_TRAINING_ROWS = 365
CACHE_SCHEMA_VERSION = 2
MODEL_CACHE_VERSION = 3
MAX_TRAINING_YEARS = 5
MIN_PRODUCTION_TRAINING_ROWS = 365
LAG_CONTEXT_DAYS = 7
CAMS_COVERAGE_PROBE_START = date(2022, 8, 1)
CAMS_COVERAGE_PROBE_END = date(2022, 9, 15)
SEASON_TEMPLATES = [
    ("monsoon_transition", "Monsoon transition", 0, 8, 12),
    ("post_monsoon", "Post-monsoon", 0, 10, 21),
    ("winter", "Winter", 0, 12, 20),
    ("spring", "Spring", 1, 3, 15),
]


class MissingAPIKeyError(ValueError):
    """Raised when no data.gov.in key was supplied."""


class InvalidAPIKeyError(ValueError):
    """Raised when data.gov.in rejects an API key."""


class APIUnavailableError(ConnectionError):
    """Raised when an external service cannot be reached or parsed."""


class NoCityDataError(LookupError):
    """Raised when CPCB responds but has no usable city PM2.5."""


class IncompleteHistoryError(ValueError):
    """Raised when a complete contiguous 90-day history cannot be built."""

    def __init__(self, message: str, missing_dates: list[date] | None = None):
        super().__init__(message)
        self.missing_dates = missing_dates or []


def request_session(retry_reads: bool = True) -> requests.Session:
    """Create a requests session with one retry for transient failures."""
    retry = Retry(
        total=1,
        connect=1,
        read=1 if retry_reads else 0,
        status=1,
        backoff_factor=0.4,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset({"GET"}),
    )
    session = requests.Session()
    session.mount("https://", HTTPAdapter(max_retries=retry))
    return session


def get_json(
    url: str,
    params: dict[str, Any],
    service_name: str,
    timeout: tuple[float, float] = (10.0, 30.0),
    retry_reads: bool = True,
) -> dict[str, Any]:
    """Fetch JSON and translate request/format failures into app errors."""
    try:
        response = request_session(retry_reads=retry_reads).get(
            url, params=params, timeout=timeout
        )
    except requests.RequestException as exc:
        raise APIUnavailableError(
            f"{service_name} request failed ({exc.__class__.__name__})."
        ) from exc

    try:
        payload = response.json()
    except ValueError as exc:
        raise APIUnavailableError(
            f"{service_name} returned an unreadable response."
        ) from exc

    if service_name == "data.gov.in":
        error_text = str(payload.get("error", ""))
        if response.status_code in (401, 403) or "key" in error_text.casefold():
            raise InvalidAPIKeyError("The data.gov.in API key was rejected.")

    if not response.ok or payload.get("error") is True:
        reason = payload.get("reason") or payload.get("message") or response.reason
        raise APIUnavailableError(f"{service_name} returned an error: {reason}")
    return payload


def parse_hourly_daily(
    payload: dict[str, Any],
    field_map: dict[str, str],
    aggregations: dict[str, str] | None = None,
) -> tuple[pd.DataFrame, str]:
    """Aggregate only local dates containing 24 numeric hourly values."""
    hourly = payload.get("hourly")
    timezone_name = payload.get("timezone")
    if not isinstance(hourly, dict) or not timezone_name:
        raise APIUnavailableError("API response omitted hourly data or timezone.")

    data: dict[str, Any] = {"time": hourly.get("time", [])}
    for api_field, output_field in field_map.items():
        data[output_field] = hourly.get(api_field, [])
    try:
        frame = pd.DataFrame(data)
    except ValueError as exc:
        raise APIUnavailableError("Hourly API arrays had inconsistent lengths.") from exc

    frame["time"] = pd.to_datetime(frame["time"], errors="coerce")
    value_columns = list(field_map.values())
    for column in value_columns:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.dropna(subset=["time"])
    frame["Date"] = frame["time"].dt.date

    rows: list[dict[str, Any]] = []
    for local_date, group in frame.groupby("Date", sort=True):
        complete = group["time"].nunique() == 24
        complete = complete and all(group[column].notna().sum() == 24 for column in value_columns)
        if complete:
            row: dict[str, Any] = {"Date": local_date}
            for column in value_columns:
                method = (aggregations or {}).get(column, "mean")
                if method == "sum":
                    value = group[column].sum()
                elif method == "max":
                    value = group[column].max()
                elif method == "circular_mean":
                    radians = np.deg2rad(group[column].to_numpy(dtype=float))
                    value = np.degrees(
                        np.arctan2(np.sin(radians).mean(), np.cos(radians).mean())
                    ) % 360.0
                else:
                    value = group[column].mean()
                row[column] = float(value)
            rows.append(row)
    return pd.DataFrame(rows), str(timezone_name)


def location_key(latitude: float, longitude: float) -> str:
    """Return a stable cache key for a rounded coordinate pair."""
    return f"{latitude:.4f},{longitude:.4f}"


def years_ago(day: date, years: int) -> date:
    """Subtract whole calendar years while handling leap-day cutoffs."""
    return (pd.Timestamp(day) - pd.DateOffset(years=years)).date()


def five_year_training_start(cutoff: date) -> date:
    """Inclusive first target date in a five-calendar-year training window."""
    return years_ago(cutoff, MAX_TRAINING_YEARS) + timedelta(days=1)


@st.cache_data(ttl=86400, show_spinner=False)
def discover_cams_coverage_start(
    latitude: float, longitude: float, refresh_token: int
) -> tuple[date, str]:
    """Find the first complete local CAMS day near the documented global boundary."""
    del refresh_token
    payload = get_json(
        AIR_QUALITY_URL,
        {
            "latitude": latitude,
            "longitude": longitude,
            "start_date": CAMS_COVERAGE_PROBE_START.isoformat(),
            "end_date": CAMS_COVERAGE_PROBE_END.isoformat(),
            "hourly": "pm2_5",
            "timezone": "auto",
            "domains": "cams_global",
        },
        "Open-Meteo CAMS coverage probe",
    )
    daily, timezone_name = parse_hourly_daily(payload, {"pm2_5": "CAMS_PM2.5"})
    if daily.empty:
        raise APIUnavailableError(
            "Open-Meteo returned no complete CAMS days near its global archive boundary."
        )
    return min(daily["Date"]), timezone_name


def database_connection(path: Path = CACHE_DB_FILE) -> sqlite3.Connection:
    connection = sqlite3.connect(path, timeout=30)
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("PRAGMA busy_timeout=30000")
    return connection


def initialize_database(path: Path = CACHE_DB_FILE) -> None:
    """Create caches and migrate older databases without dropping observations."""
    with database_connection(path) as connection:
        connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS hourly_history (
                location_key TEXT NOT NULL,
                local_timestamp TEXT NOT NULL,
                local_date TEXT NOT NULL,
                cams_pm25 REAL,
                temperature REAL,
                humidity REAL,
                wind_speed REAL,
                precipitation REAL,
                pressure_msl REAL,
                cloud_cover REAL,
                wind_direction REAL,
                wind_gusts REAL,
                boundary_layer_height REAL,
                timezone TEXT NOT NULL,
                source TEXT NOT NULL,
                fetched_at_utc TEXT NOT NULL,
                PRIMARY KEY (location_key, local_timestamp)
            );
            CREATE INDEX IF NOT EXISTS idx_hourly_location_date
                ON hourly_history(location_key, local_date);

            CREATE TABLE IF NOT EXISTS fire_observations (
                location_key TEXT NOT NULL,
                acquisition_timestamp_utc TEXT NOT NULL,
                latitude REAL NOT NULL,
                longitude REAL NOT NULL,
                frp REAL,
                confidence TEXT,
                satellite TEXT,
                source TEXT NOT NULL,
                fetched_at_utc TEXT NOT NULL,
                PRIMARY KEY (
                    location_key, acquisition_timestamp_utc, latitude, longitude, source
                )
            );
            CREATE INDEX IF NOT EXISTS idx_fire_location_time
                ON fire_observations(location_key, acquisition_timestamp_utc);

            CREATE TABLE IF NOT EXISTS fire_fetch_days (
                location_key TEXT NOT NULL,
                local_date TEXT NOT NULL,
                source TEXT NOT NULL,
                fetched_at_utc TEXT NOT NULL,
                PRIMARY KEY (location_key, local_date, source)
            );

            CREATE TABLE IF NOT EXISTS prospective_forecasts (
                location_key TEXT NOT NULL,
                target_date TEXT NOT NULL,
                prediction REAL NOT NULL,
                lower_80 REAL NOT NULL,
                upper_80 REAL NOT NULL,
                issued_at_utc TEXT NOT NULL,
                timezone TEXT NOT NULL,
                model_hash TEXT NOT NULL,
                PRIMARY KEY (location_key, target_date)
            );

            CREATE TABLE IF NOT EXISTS prospective_actuals (
                location_key TEXT NOT NULL,
                target_date TEXT NOT NULL,
                actual_pm25 REAL NOT NULL,
                recorded_at_utc TEXT NOT NULL,
                source TEXT NOT NULL,
                PRIMARY KEY (location_key, target_date)
            );

            CREATE TABLE IF NOT EXISTS cache_metadata (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );
            """
        )
        existing_columns = {
            row[1] for row in connection.execute("PRAGMA table_info(hourly_history)")
        }
        for column in (
            "precipitation", "pressure_msl", "cloud_cover", "wind_direction",
            "wind_gusts", "boundary_layer_height",
        ):
            if column not in existing_columns:
                connection.execute(f"ALTER TABLE hourly_history ADD COLUMN {column} REAL")
        connection.execute(
            "INSERT OR REPLACE INTO cache_metadata(key, value) VALUES (?, ?)",
            ("schema_version", str(CACHE_SCHEMA_VERSION)),
        )


def payload_to_hourly(
    payload: dict[str, Any], field_map: dict[str, str]
) -> tuple[pd.DataFrame, str]:
    """Parse hourly API arrays without discarding partial dates."""
    hourly = payload.get("hourly")
    timezone_name = payload.get("timezone")
    if not isinstance(hourly, dict) or not timezone_name:
        raise APIUnavailableError("API response omitted hourly data or timezone.")
    data: dict[str, Any] = {"local_timestamp": hourly.get("time", [])}
    for api_field, output_field in field_map.items():
        data[output_field] = hourly.get(api_field, [])
    try:
        frame = pd.DataFrame(data)
    except ValueError as exc:
        raise APIUnavailableError("Hourly API arrays had inconsistent lengths.") from exc
    timestamps = pd.to_datetime(frame["local_timestamp"], errors="coerce")
    frame = frame.loc[timestamps.notna()].copy()
    timestamps = timestamps.loc[timestamps.notna()]
    frame["local_timestamp"] = timestamps.dt.strftime("%Y-%m-%dT%H:%M:%S")
    frame["local_date"] = timestamps.dt.strftime("%Y-%m-%d")
    for output_field in field_map.values():
        frame[output_field] = pd.to_numeric(frame[output_field], errors="coerce")
    return frame, str(timezone_name)


@st.cache_data(ttl=1800, show_spinner=False)
def fetch_hourly_archive_range(
    latitude: float,
    longitude: float,
    start_date: date,
    end_date: date,
    refresh_token: int,
) -> tuple[pd.DataFrame, str]:
    """Fetch one missing date range from both historical APIs."""
    del refresh_token
    common = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": start_date.isoformat(),
        "end_date": end_date.isoformat(),
        "timezone": "auto",
    }
    air_payload = get_json(
        AIR_QUALITY_URL,
        {**common, "hourly": "pm2_5", "domains": "cams_global"},
        "Open-Meteo Air Quality",
    )
    weather_payload = get_json(
        HISTORICAL_WEATHER_URL,
        {
            **common,
            "hourly": ",".join(WEATHER_API_FIELDS),
        },
        "Open-Meteo Historical Weather",
    )
    air, air_timezone = payload_to_hourly(
        air_payload, {"pm2_5": "cams_pm25"}
    )
    weather, weather_timezone = payload_to_hourly(weather_payload, WEATHER_API_FIELDS)
    if air_timezone != weather_timezone:
        raise APIUnavailableError(
            f"Historical API timezone mismatch: {air_timezone} and {weather_timezone}."
        )
    merged = air.merge(weather, on=["local_timestamp", "local_date"], how="outer")
    return merged.sort_values("local_timestamp").reset_index(drop=True), air_timezone


def complete_cached_dates(
    cache_location: str,
    start_date: date,
    end_date: date,
    path: Path = CACHE_DB_FILE,
) -> set[date]:
    """Return dates with 24 unique numeric rows for every required variable."""
    if not path.exists():
        return set()
    query = """
        SELECT local_date
        FROM hourly_history
        WHERE location_key = ? AND local_date BETWEEN ? AND ?
        GROUP BY local_date
        HAVING COUNT(DISTINCT local_timestamp) = 24
           AND COUNT(cams_pm25) = 24
           AND COUNT(temperature) = 24
           AND COUNT(humidity) = 24
           AND COUNT(wind_speed) = 24
    """
    with database_connection(path) as connection:
        rows = connection.execute(
            query,
            (cache_location, start_date.isoformat(), end_date.isoformat()),
        ).fetchall()
    return {date.fromisoformat(row[0]) for row in rows}


def contiguous_ranges(days: list[date]) -> list[tuple[date, date]]:
    if not days:
        return []
    ordered = sorted(set(days))
    ranges: list[tuple[date, date]] = []
    range_start = ordered[0]
    previous = ordered[0]
    for current in ordered[1:]:
        if current != previous + timedelta(days=1):
            ranges.append((range_start, previous))
            range_start = current
        previous = current
    ranges.append((range_start, previous))
    return ranges


def store_hourly_rows(
    cache_location: str,
    frame: pd.DataFrame,
    timezone_name: str,
    path: Path = CACHE_DB_FILE,
) -> int:
    """Idempotently upsert fetched hourly rows into SQLite."""
    if frame.empty:
        return 0
    fetched_at = datetime.now(timezone.utc).isoformat()
    columns = ["cams_pm25", *WEATHER_DAILY_NAMES]
    records = []
    for row in frame.to_dict("records"):
        records.append(
            (
                cache_location,
                row["local_timestamp"],
                row["local_date"],
                *[
                    None if pd.isna(row.get(column)) else float(row[column])
                    for column in columns
                ],
                timezone_name,
                "Open-Meteo CAMS + Historical Weather",
                fetched_at,
            )
        )
    with database_connection(path) as connection:
        connection.executemany(
            """
            INSERT INTO hourly_history(
                location_key, local_timestamp, local_date, cams_pm25,
                temperature, humidity, wind_speed, precipitation, pressure_msl,
                cloud_cover, wind_direction, wind_gusts, boundary_layer_height,
                timezone, source, fetched_at_utc
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(location_key, local_timestamp) DO UPDATE SET
                local_date = excluded.local_date,
                cams_pm25 = COALESCE(excluded.cams_pm25, hourly_history.cams_pm25),
                temperature = COALESCE(excluded.temperature, hourly_history.temperature),
                humidity = COALESCE(excluded.humidity, hourly_history.humidity),
                wind_speed = COALESCE(excluded.wind_speed, hourly_history.wind_speed),
                precipitation = COALESCE(excluded.precipitation, hourly_history.precipitation),
                pressure_msl = COALESCE(excluded.pressure_msl, hourly_history.pressure_msl),
                cloud_cover = COALESCE(excluded.cloud_cover, hourly_history.cloud_cover),
                wind_direction = COALESCE(excluded.wind_direction, hourly_history.wind_direction),
                wind_gusts = COALESCE(excluded.wind_gusts, hourly_history.wind_gusts),
                boundary_layer_height = COALESCE(
                    excluded.boundary_layer_height, hourly_history.boundary_layer_height
                ),
                timezone = excluded.timezone,
                source = excluded.source,
                fetched_at_utc = excluded.fetched_at_utc
            """,
            records,
        )
    return len(records)


def ensure_hourly_cache(
    latitude: float,
    longitude: float,
    start_date: date,
    end_date: date,
    refresh_token: int,
    path: Path = CACHE_DB_FILE,
) -> dict[str, Any]:
    """Fetch only incomplete date ranges and persist every returned hourly row."""
    initialize_database(path)
    cache_location = location_key(latitude, longitude)
    expected = {
        start_date + timedelta(days=offset)
        for offset in range((end_date - start_date).days + 1)
    }
    before = complete_cached_dates(cache_location, start_date, end_date, path)
    missing_ranges = contiguous_ranges(list(expected - before))
    rows_written = 0
    timezone_name: str | None = None
    for missing_start, missing_end in missing_ranges:
        frame, timezone_name = fetch_hourly_archive_range(
            latitude,
            longitude,
            missing_start,
            missing_end,
            refresh_token,
        )
        rows_written += store_hourly_rows(
            cache_location, frame, timezone_name, path
        )
    after = complete_cached_dates(cache_location, start_date, end_date, path)
    return {
        "location_key": cache_location,
        "requested_days": len(expected),
        "already_complete_days": len(before),
        "complete_days": len(after),
        "missing_dates": sorted(expected - after),
        "fetched_ranges": missing_ranges,
        "rows_written": rows_written,
        "timezone": timezone_name,
    }


def daily_history_from_cache(
    cache_location: str,
    start_date: date,
    end_date: date,
    path: Path = CACHE_DB_FILE,
) -> pd.DataFrame:
    """Aggregate complete hourly cache rows into local daily model features."""
    query = """
        SELECT local_date AS Date,
               AVG(cams_pm25) AS "CAMS_PM2.5",
               AVG(temperature) AS Temperature,
               AVG(humidity) AS Humidity,
               AVG(wind_speed) AS Wind_Speed,
               SUM(precipitation) AS Precipitation,
               AVG(pressure_msl) AS Pressure_MSL,
               AVG(cloud_cover) AS Cloud_Cover,
               AVG(wind_direction) AS Wind_Direction,
               MAX(wind_gusts) AS Wind_Gusts,
               AVG(boundary_layer_height) AS Boundary_Layer_Height,
               MIN(timezone) AS Timezone,
               COUNT(DISTINCT local_timestamp) AS hours,
               COUNT(cams_pm25) AS pm_count,
               COUNT(temperature) AS temperature_count,
               COUNT(humidity) AS humidity_count,
               COUNT(wind_speed) AS wind_count
               ,COUNT(precipitation) AS precipitation_count
               ,COUNT(pressure_msl) AS pressure_count
               ,COUNT(cloud_cover) AS cloud_count
               ,COUNT(wind_direction) AS direction_count
               ,COUNT(wind_gusts) AS gust_count
               ,COUNT(boundary_layer_height) AS boundary_count
        FROM hourly_history
        WHERE location_key = ? AND local_date BETWEEN ? AND ?
        GROUP BY local_date
        ORDER BY local_date
    """
    with database_connection(path) as connection:
        frame = pd.read_sql_query(
            query,
            connection,
            params=(cache_location, start_date.isoformat(), end_date.isoformat()),
        )
    if frame.empty:
        return pd.DataFrame(columns=["Date", "CAMS_PM2.5", *WEATHER_FEATURES, "Timezone"])
    valid = (
        (frame["hours"] == 24)
        & (frame["pm_count"] == 24)
        & (frame["temperature_count"] == 24)
        & (frame["humidity_count"] == 24)
        & (frame["wind_count"] == 24)
    )
    frame = frame.loc[valid, ["Date", "CAMS_PM2.5", *WEATHER_FEATURES, "Timezone"]].copy()
    frame["Date"] = pd.to_datetime(frame["Date"]).dt.date
    return frame.reset_index(drop=True)


def haversine_km(
    latitude: float, longitude: float, other_latitude: pd.Series, other_longitude: pd.Series
) -> np.ndarray:
    """Vectorized great-circle distance from the forecast location."""
    lat1 = math.radians(latitude)
    lat2 = np.radians(other_latitude.to_numpy(dtype=float))
    delta_lat = lat2 - lat1
    delta_lon = np.radians(other_longitude.to_numpy(dtype=float) - longitude)
    value = np.sin(delta_lat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(delta_lon / 2.0) ** 2
    return 6371.0088 * 2.0 * np.arcsin(np.sqrt(np.clip(value, 0.0, 1.0)))


def bearing_degrees(
    latitude: float, longitude: float, other_latitude: pd.Series, other_longitude: pd.Series
) -> np.ndarray:
    """Initial bearing from the forecast location to each fire detection."""
    lat1 = math.radians(latitude)
    lat2 = np.radians(other_latitude.to_numpy(dtype=float))
    delta_lon = np.radians(other_longitude.to_numpy(dtype=float) - longitude)
    y = np.sin(delta_lon) * np.cos(lat2)
    x = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(delta_lon)
    return (np.degrees(np.arctan2(y, x)) + 360.0) % 360.0


@st.cache_data(ttl=1800, show_spinner=False)
def fetch_firms_range(
    map_key: str,
    latitude: float,
    longitude: float,
    start_date: date,
    end_date: date,
    refresh_token: int,
) -> pd.DataFrame:
    """Fetch and validate one five-day-or-shorter NASA FIRMS area response."""
    del refresh_token
    day_count = (end_date - start_date).days + 1
    if not 1 <= day_count <= 5:
        raise ValueError("NASA FIRMS area requests must cover one through five days.")
    latitude_delta = FIRMS_RADIUS_KM / 111.0
    longitude_delta = FIRMS_RADIUS_KM / max(20.0, 111.0 * math.cos(math.radians(latitude)))
    area = ",".join(
        f"{value:.4f}" for value in (
            longitude - longitude_delta,
            latitude - latitude_delta,
            longitude + longitude_delta,
            latitude + latitude_delta,
        )
    )
    url = f"{FIRMS_AREA_URL}/{map_key}/{FIRMS_SOURCE}/{area}/{day_count}/{start_date.isoformat()}"
    try:
        response = request_session().get(url, timeout=(10.0, 45.0))
    except requests.RequestException as exc:
        raise APIUnavailableError(
            f"NASA FIRMS request failed ({exc.__class__.__name__})."
        ) from exc
    if response.status_code in (401, 403):
        raise InvalidAPIKeyError("The NASA FIRMS MAP_KEY was rejected.")
    if not response.ok:
        raise APIUnavailableError(f"NASA FIRMS returned HTTP {response.status_code}.")
    text = response.text.strip()
    if not text or text.casefold().startswith("no data"):
        return pd.DataFrame(
            columns=["latitude", "longitude", "acq_date", "acq_time", "frp", "confidence", "satellite"]
        )
    try:
        frame = pd.read_csv(StringIO(text), dtype={"acq_time": "string"})
    except Exception as exc:
        raise APIUnavailableError("NASA FIRMS returned unreadable CSV data.") from exc
    required = {"latitude", "longitude", "acq_date", "acq_time", "frp", "confidence", "satellite"}
    missing = required - set(frame.columns)
    if missing:
        raise APIUnavailableError(
            "NASA FIRMS response omitted fields: " + ", ".join(sorted(missing))
        )
    for column in ("latitude", "longitude", "frp"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame["acq_time"] = frame["acq_time"].astype("string").str.zfill(4)
    frame["acquisition_timestamp_utc"] = pd.to_datetime(
        frame["acq_date"].astype(str) + " " + frame["acq_time"].str[:2] + ":" + frame["acq_time"].str[2:],
        errors="coerce",
        utc=True,
    )
    frame = frame.dropna(
        subset=["latitude", "longitude", "frp", "acquisition_timestamp_utc"]
    ).copy()
    if not frame.empty:
        frame["distance_km"] = haversine_km(
            latitude, longitude, frame["latitude"], frame["longitude"]
        )
        frame = frame.loc[frame["distance_km"] <= FIRMS_RADIUS_KM].copy()
    return frame.reset_index(drop=True)


def fire_cached_dates(
    cache_location: str,
    start_date: date,
    end_date: date,
    path: Path = CACHE_DB_FILE,
) -> set[date]:
    query = """
        SELECT local_date FROM fire_fetch_days
        WHERE location_key = ? AND source = ? AND local_date BETWEEN ? AND ?
    """
    with database_connection(path) as connection:
        rows = connection.execute(
            query, (cache_location, FIRMS_SOURCE, start_date.isoformat(), end_date.isoformat())
        ).fetchall()
    return {date.fromisoformat(row[0]) for row in rows}


def store_fire_range(
    cache_location: str,
    frame: pd.DataFrame,
    start_date: date,
    end_date: date,
    path: Path = CACHE_DB_FILE,
) -> int:
    """Insert detections and mark every successfully requested date, including zero-fire days."""
    fetched_at = datetime.now(timezone.utc).isoformat()
    records = []
    for row in frame.to_dict("records"):
        records.append(
            (
                cache_location,
                pd.Timestamp(row["acquisition_timestamp_utc"]).isoformat(),
                float(row["latitude"]),
                float(row["longitude"]),
                float(row["frp"]),
                str(row.get("confidence", "")),
                str(row.get("satellite", "")),
                FIRMS_SOURCE,
                fetched_at,
            )
        )
    days = [
        (cache_location, (start_date + timedelta(days=offset)).isoformat(), FIRMS_SOURCE, fetched_at)
        for offset in range((end_date - start_date).days + 1)
    ]
    with database_connection(path) as connection:
        connection.executemany(
            """
            INSERT OR IGNORE INTO fire_observations(
                location_key, acquisition_timestamp_utc, latitude, longitude, frp,
                confidence, satellite, source, fetched_at_utc
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            records,
        )
        connection.executemany(
            """
            INSERT OR REPLACE INTO fire_fetch_days(
                location_key, local_date, source, fetched_at_utc
            ) VALUES (?, ?, ?, ?)
            """,
            days,
        )
    return len(records)


def ensure_fire_cache(
    map_key: str,
    latitude: float,
    longitude: float,
    start_date: date,
    end_date: date,
    refresh_token: int,
    path: Path = CACHE_DB_FILE,
) -> dict[str, Any]:
    initialize_database(path)
    cache_location = location_key(latitude, longitude)
    expected = {
        start_date + timedelta(days=offset)
        for offset in range((end_date - start_date).days + 1)
    }
    before = fire_cached_dates(cache_location, start_date, end_date, path)
    requests_made = 0
    rows_written = 0
    for range_start, range_end in contiguous_ranges(list(expected - before)):
        chunk_start = range_start
        while chunk_start <= range_end:
            chunk_end = min(range_end, chunk_start + timedelta(days=4))
            frame = fetch_firms_range(
                map_key, latitude, longitude, chunk_start, chunk_end, refresh_token
            )
            rows_written += store_fire_range(
                cache_location, frame, chunk_start, chunk_end, path
            )
            requests_made += 1
            chunk_start = chunk_end + timedelta(days=1)
    after = fire_cached_dates(cache_location, start_date, end_date, path)
    return {
        "complete_days": len(after),
        "requested_days": len(expected),
        "requests_made": requests_made,
        "rows_written": rows_written,
        "complete": after == expected,
    }


def load_fire_observations(
    cache_location: str,
    start_date: date,
    end_date: date,
    timezone_name: str,
    latitude: float,
    longitude: float,
    path: Path = CACHE_DB_FILE,
) -> pd.DataFrame:
    query = """
        SELECT acquisition_timestamp_utc, latitude, longitude, frp
        FROM fire_observations
        WHERE location_key = ? AND acquisition_timestamp_utc >= ?
          AND acquisition_timestamp_utc < ? AND source = ?
    """
    utc_start = datetime.combine(start_date, datetime.min.time(), ZoneInfo(timezone_name)).astimezone(timezone.utc)
    utc_end = datetime.combine(end_date + timedelta(days=1), datetime.min.time(), ZoneInfo(timezone_name)).astimezone(timezone.utc)
    with database_connection(path) as connection:
        frame = pd.read_sql_query(
            query,
            connection,
            params=(cache_location, utc_start.isoformat(), utc_end.isoformat(), FIRMS_SOURCE),
        )
    if frame.empty:
        return pd.DataFrame(columns=["Local_Date", "latitude", "longitude", "frp", "distance_km", "bearing"])
    timestamps = pd.to_datetime(frame["acquisition_timestamp_utc"], utc=True, errors="coerce")
    frame = frame.loc[timestamps.notna()].copy()
    frame["Local_Date"] = timestamps.loc[timestamps.notna()].dt.tz_convert(timezone_name).dt.date
    frame["distance_km"] = haversine_km(latitude, longitude, frame["latitude"], frame["longitude"])
    frame["bearing"] = bearing_degrees(latitude, longitude, frame["latitude"], frame["longitude"])
    return frame.reset_index(drop=True)


@st.cache_data(ttl=1800, show_spinner=False)
def fetch_forecast_weather(
    latitude: float, longitude: float, refresh_token: int
) -> dict[str, Any]:
    """Fetch local time and weather from seven past through seven future days."""
    del refresh_token
    payload = get_json(
        FORECAST_WEATHER_URL,
        {
            "latitude": latitude,
            "longitude": longitude,
            "current": "temperature_2m",
            "hourly": ",".join(WEATHER_API_FIELDS),
            "past_days": 7,
            # Today plus tomorrow through day +7 requires nine local dates.
            "forecast_days": 9,
            "timezone": "auto",
        },
        "Open-Meteo Forecast",
    )
    current_time = pd.to_datetime(payload.get("current", {}).get("time"), errors="coerce")
    if pd.isna(current_time):
        raise APIUnavailableError("Open-Meteo Forecast omitted its local current time.")
    daily, timezone_name = parse_hourly_daily(
        payload,
        {api: WEATHER_DAILY_NAMES[stored] for api, stored in WEATHER_API_FIELDS.items()},
        {
            "Precipitation": "sum",
            "Wind_Gusts": "max",
            "Wind_Direction": "circular_mean",
        },
    )
    return {
        "today": current_time.date(),
        "timezone": timezone_name,
        "daily": daily,
    }


@st.cache_data(ttl=1800, show_spinner=False)
def fetch_historical_bundle(
    latitude: float,
    longitude: float,
    local_today: date,
    refresh_token: int,
) -> dict[str, Any]:
    """Fetch a buffer and return the latest complete contiguous 90 days."""
    del refresh_token
    end_date = local_today - timedelta(days=1)
    start_date = end_date - timedelta(days=BUFFER_DAYS - 1)
    common_params = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": start_date.isoformat(),
        "end_date": end_date.isoformat(),
        "timezone": "auto",
    }
    air_payload = get_json(
        AIR_QUALITY_URL,
        {**common_params, "hourly": "pm2_5", "domains": "cams_global"},
        "Open-Meteo Air Quality",
    )
    weather_payload = get_json(
        HISTORICAL_WEATHER_URL,
        {
            **common_params,
            "hourly": ",".join(WEATHER_API_FIELDS),
        },
        "Open-Meteo Historical Weather",
    )
    air_daily, air_timezone = parse_hourly_daily(
        air_payload, {"pm2_5": "CAMS_PM2.5"}
    )
    weather_daily, weather_timezone = parse_hourly_daily(
        weather_payload,
        {api: WEATHER_DAILY_NAMES[stored] for api, stored in WEATHER_API_FIELDS.items()},
        {
            "Precipitation": "sum",
            "Wind_Gusts": "max",
            "Wind_Direction": "circular_mean",
        },
    )
    if air_timezone != weather_timezone:
        raise IncompleteHistoryError(
            f"API timezone mismatch: {air_timezone} versus {weather_timezone}."
        )

    merged = air_daily.merge(weather_daily, on="Date", how="inner")
    if merged.empty:
        raise IncompleteHistoryError("No common complete historical dates were returned.")
    cutoff = max(merged["Date"])
    expected = [cutoff - timedelta(days=offset) for offset in range(HISTORY_DAYS - 1, -1, -1)]
    available = set(merged["Date"])
    missing = [day for day in expected if day not in available]
    if missing:
        raise IncompleteHistoryError(
            "The latest 90-calendar-day window contains incomplete or missing dates.",
            missing,
        )

    selected = merged.loc[merged["Date"].isin(expected)].copy().sort_values("Date")
    selected["Timezone"] = air_timezone
    if len(selected) != HISTORY_DAYS:
        raise IncompleteHistoryError(
            f"Expected 90 complete dates but assembled {len(selected)}."
        )
    return {
        "history": selected.reset_index(drop=True),
        "cutoff": cutoff,
        "timezone": air_timezone,
        "buffer_start": start_date,
        "buffer_end": end_date,
        "complete_days_in_buffer": int(len(merged)),
    }


@st.cache_data(ttl=1800, show_spinner=False)
def fetch_cpcb_pm25(api_key: str, city: str, refresh_token: int) -> dict[str, Any]:
    """Fetch all CPCB pages and average deduplicated numeric PM2.5 stations."""
    del refresh_token
    if not api_key.strip():
        raise MissingAPIKeyError("A data.gov.in API key is required.")

    page_size = 1000
    max_records = 10_000
    offset = 0
    records_all: list[dict[str, Any]] = []
    while offset < max_records:
        payload = get_json(
            CPCB_API_URL,
            {
                "api-key": api_key.strip(),
                "format": "json",
                "limit": page_size,
                "offset": offset,
            },
            "data.gov.in",
            timeout=(5.0, 8.0),
            retry_reads=False,
        )
        records = payload.get("records")
        if not isinstance(records, list):
            raise APIUnavailableError("data.gov.in response omitted its records list.")
        records_all.extend(records)
        count = len(records)
        try:
            total = int(payload.get("total", offset + count))
        except (TypeError, ValueError):
            total = offset + count
        offset += count
        if count == 0 or count < page_size or offset >= total:
            break

    rows = pd.DataFrame(records_all)
    if rows.empty:
        raise NoCityDataError(f"No CPCB records were returned for {city.strip()}.")

    def first_column(candidates: tuple[str, ...]) -> str | None:
        return next((name for name in candidates if name in rows.columns), None)

    city_column = first_column(("city", "City"))
    station_column = first_column(("station", "station_id", "station_name"))
    pollutant_column = first_column(("pollutant_id", "pollutant"))
    average_column = first_column(("avg_value", "pollutant_avg", "average"))
    if not all((city_column, station_column, pollutant_column, average_column)):
        raise APIUnavailableError("data.gov.in returned an unexpected CPCB schema.")

    target_city = city.strip().casefold()
    city_mask = rows[city_column].astype(str).str.strip().str.casefold() == target_city
    pollutant_mask = (
        rows[pollutant_column].astype(str).str.strip().str.casefold() == "pm2.5"
    )
    usable = rows.loc[city_mask & pollutant_mask].copy()
    usable["numeric_pm25"] = pd.to_numeric(usable[average_column], errors="coerce")
    usable = usable.dropna(subset=["numeric_pm25"])
    if usable.empty:
        raise NoCityDataError(
            f"{city.strip()} had no numeric PM2.5 station readings."
        )

    usable["station_key"] = usable[station_column].astype(str).str.strip().str.casefold()
    timestamp_column = next(
        (name for name in ("last_update", "last_updated", "updated_at") if name in usable),
        None,
    )
    usable = usable.sort_values(timestamp_column or station_column)
    usable = usable.drop_duplicates("station_key", keep="last")
    observed_date: date | None = None
    if timestamp_column:
        timestamps = pd.to_datetime(usable[timestamp_column], errors="coerce", dayfirst=True)
        if timestamps.notna().any():
            observed_date = timestamps.max().date()
    return {
        "pm25": float(usable["numeric_pm25"].mean()),
        "station_count": int(len(usable)),
        "observed_date": observed_date,
        "records_scanned": int(len(records_all)),
    }


def empty_history() -> pd.DataFrame:
    return pd.DataFrame(columns=HISTORY_COLUMNS)


def load_history(path: Path = DATA_FILE) -> pd.DataFrame:
    """Load the new schema and migrate the prior ambiguous PM2.5 column."""
    if not path.exists():
        return empty_history()
    frame = pd.read_csv(path)
    if "PM2.5" in frame.columns and "Manual_PM2.5" not in frame.columns:
        frame["Manual_PM2.5"] = frame["PM2.5"]
    for column in HISTORY_COLUMNS:
        if column not in frame:
            frame[column] = np.nan
    frame = frame[HISTORY_COLUMNS].copy()
    frame["Date"] = pd.to_datetime(frame["Date"], errors="coerce").dt.date
    frame = frame.dropna(subset=["Date"])
    numeric_columns = [
        "CAMS_PM2.5",
        "CPCB_PM2.5",
        "Manual_PM2.5",
        *WEATHER_FEATURES,
        "CPCB_Station_Count",
    ]
    for column in numeric_columns:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame.sort_values("Date").drop_duplicates("Date", keep="last").reset_index(drop=True)


def save_history(frame: pd.DataFrame, path: Path = DATA_FILE) -> None:
    output = frame[HISTORY_COLUMNS].copy().sort_values("Date")
    output["Date"] = output["Date"].map(lambda value: value.isoformat())
    output.to_csv(path, index=False)


def merge_historical_rows(
    existing: pd.DataFrame, historical: pd.DataFrame
) -> pd.DataFrame:
    """Upsert CAMS/weather fields while preserving CPCB and manual columns."""
    history = existing.copy()
    if history.empty:
        history = empty_history()
    history["Timezone"] = history["Timezone"].astype("object")
    for row in historical.to_dict("records"):
        local_date = row["Date"]
        matches = history.index[history["Date"] == local_date].tolist()
        if matches:
            index = matches[0]
        else:
            history = pd.concat(
                [history, pd.DataFrame([{column: np.nan for column in HISTORY_COLUMNS}])],
                ignore_index=True,
            )
            index = history.index[-1]
            history.at[index, "Date"] = local_date
        for column in ["CAMS_PM2.5", *WEATHER_FEATURES, "Timezone"]:
            history.at[index, column] = row[column]
    return history.sort_values("Date").drop_duplicates("Date", keep="last").reset_index(drop=True)


def upsert_observation(
    history: pd.DataFrame,
    observation_date: date,
    column: str,
    value: float,
    timezone_name: str,
    station_count: int | None = None,
) -> pd.DataFrame:
    """Upsert one source-specific observation without replacing other sources."""
    result = history.copy()
    result["Timezone"] = result["Timezone"].astype("object")
    matches = result.index[result["Date"] == observation_date].tolist()
    if matches:
        index = matches[0]
    else:
        new_row = {name: np.nan for name in HISTORY_COLUMNS}
        new_row["Date"] = observation_date
        result = pd.concat([result, pd.DataFrame([new_row])], ignore_index=True)
        index = result.index[-1]
    result.at[index, column] = float(value)
    result.at[index, "Timezone"] = timezone_name
    if station_count is not None:
        result.at[index, "CPCB_Station_Count"] = int(station_count)
    return result.sort_values("Date").drop_duplicates("Date", keep="last").reset_index(drop=True)


def select_stored_window(history: pd.DataFrame) -> tuple[pd.DataFrame, date]:
    """Select the latest strict 90-calendar-day CAMS/weather window."""
    required = ["CAMS_PM2.5", *CORE_WEATHER_FEATURES]
    usable = history.dropna(subset=required).copy().sort_values("Date")
    if usable.empty:
        raise IncompleteHistoryError("No complete CAMS history is stored locally.")
    cutoff = usable["Date"].max()
    expected = [cutoff - timedelta(days=offset) for offset in range(89, -1, -1)]
    available = set(usable["Date"])
    missing = [day for day in expected if day not in available]
    if missing:
        raise IncompleteHistoryError("Stored history has missing calendar dates.", missing)
    window = usable.loc[usable["Date"].isin(expected)].sort_values("Date").reset_index(drop=True)
    if len(window) != HISTORY_DAYS:
        raise IncompleteHistoryError("Stored history does not contain 90 distinct dates.")
    return window, cutoff


def add_calendar_lag(
    frame: pd.DataFrame, context: pd.DataFrame | None = None
) -> pd.DataFrame:
    """Join PM2.5 from exactly Date - 1; never shift across a calendar gap."""
    result = frame.copy()
    lag_source = (context if context is not None else frame)[
        ["Date", "CAMS_PM2.5"]
    ].copy()
    lag_source["Date"] = lag_source["Date"].map(
        lambda value: value + timedelta(days=1)
    )
    lag_source = lag_source.rename(columns={"CAMS_PM2.5": "PM2.5_Yesterday"})
    result = result.drop(columns=["PM2.5_Yesterday"], errors="ignore")
    return result.merge(lag_source, on="Date", how="left", validate="one_to_one")


def add_multifactor_features(
    frame: pd.DataFrame,
    fire_observations: pd.DataFrame | None = None,
    fire_data_complete: bool = False,
) -> pd.DataFrame:
    """Build target-date features using pollution and fires from earlier dates only."""
    data = frame.copy().sort_values("Date").drop_duplicates("Date", keep="last").reset_index(drop=True)
    target_by_date = data.set_index("Date")["CAMS_PM2.5"].to_dict()
    years = sorted({day.year for day in data["Date"]})
    india_holidays = holidays.country_holidays("IN", years=years, language="en_US")
    diwali_dates = sorted(
        holiday_date
        for holiday_date, name in india_holidays.items()
        if "diwali" in str(name).casefold() or "deepavali" in str(name).casefold()
    )
    rows: list[dict[str, Any]] = []
    for row in data.to_dict("records"):
        target_date = row["Date"]
        for lag in (1, 2, 3, 7):
            row[f"PM2.5_Lag{lag}"] = target_by_date.get(target_date - timedelta(days=lag), np.nan)
        row["PM2.5_Yesterday"] = row["PM2.5_Lag1"]
        for window in (3, 7):
            prior = [target_by_date.get(target_date - timedelta(days=offset), np.nan) for offset in range(1, window + 1)]
            row[f"PM2.5_Mean{window}"] = float(np.mean(prior)) if pd.notna(prior).all() else np.nan
        prior_seven = [target_by_date.get(target_date - timedelta(days=offset), np.nan) for offset in range(1, 8)]
        row["PM2.5_Std7"] = float(np.std(prior_seven, ddof=1)) if pd.notna(prior_seven).all() else np.nan
        direction = pd.to_numeric(row.get("Wind_Direction"), errors="coerce")
        row["Wind_Direction_Sin"] = np.sin(np.deg2rad(direction)) if pd.notna(direction) else np.nan
        row["Wind_Direction_Cos"] = np.cos(np.deg2rad(direction)) if pd.notna(direction) else np.nan
        day_of_year = target_date.timetuple().tm_yday
        row["Day_Of_Year_Sin"] = math.sin(2.0 * math.pi * day_of_year / 365.25)
        row["Day_Of_Year_Cos"] = math.cos(2.0 * math.pi * day_of_year / 365.25)
        row["Is_Weekend"] = float(target_date.weekday() >= 5)
        row["Is_Indian_Holiday"] = float(target_date in india_holidays)
        distances = [(target_date - event_date).days for event_date in diwali_dates]
        nearest_diwali = min(distances, key=abs) if distances else 31
        row["Days_From_Diwali"] = float(np.clip(nearest_diwali, -30, 30))
        row["Is_Diwali"] = float(abs(nearest_diwali) <= 1)

        fire_columns = FIRE_FEATURES[len(CALENDAR_FEATURES):]
        if not fire_data_complete:
            row.update({column: np.nan for column in fire_columns})
        else:
            fires = fire_observations if fire_observations is not None else pd.DataFrame()
            windows: dict[int, pd.DataFrame] = {}
            for window in (1, 3, 7):
                start = target_date - timedelta(days=window)
                windows[window] = fires.loc[
                    fires["Local_Date"].between(start, target_date - timedelta(days=1))
                ] if not fires.empty else fires
                row[f"Fire_Count_{window}d"] = float(len(windows[window]))
            for window in (3, 7):
                row[f"Fire_FRP_{window}d"] = float(windows[window]["frp"].sum()) if not windows[window].empty else 0.0
            recent = windows[3]
            if recent.empty:
                row["Fire_Distance_Weighted_FRP_3d"] = 0.0
                row["Fire_Upwind_FRP_3d"] = 0.0
            else:
                distance_weight = np.exp(-recent["distance_km"].to_numpy(dtype=float) / 150.0)
                weighted_frp = recent["frp"].to_numpy(dtype=float) * distance_weight
                row["Fire_Distance_Weighted_FRP_3d"] = float(weighted_frp.sum())
                if pd.notna(direction):
                    delta = np.deg2rad(recent["bearing"].to_numpy(dtype=float) - float(direction))
                    upwind = np.maximum(0.0, np.cos(delta))
                    row["Fire_Upwind_FRP_3d"] = float((weighted_frp * upwind).sum())
                else:
                    row["Fire_Upwind_FRP_3d"] = np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def required_model_values(feature_names: list[str]) -> list[str]:
    """Return target-history inputs that cannot be safely mean-imputed."""
    return [
        feature
        for feature in feature_names
        if feature == "PM2.5_Yesterday" or feature.startswith("PM2.5_")
    ]


def fit_period_model(
    period: pd.DataFrame, feature_names: list[str] | None = None
) -> dict[str, Any]:
    """Fit imputer, training-only anomaly detector, and Random Forest."""
    selected_features = feature_names or MODEL_FEATURES
    mandatory_features = required_model_values(selected_features)
    data = period.copy().sort_values("Date").reset_index(drop=True)
    if "PM2.5_Yesterday" not in data:
        data = add_calendar_lag(data)
    model_rows = data.dropna(
        subset=["CAMS_PM2.5", *mandatory_features]
    ).copy()
    if len(model_rows) < 2:
        raise ValueError("Not enough lagged rows to fit the model.")

    imputer = SimpleImputer(strategy="mean")
    transformed = imputer.fit_transform(model_rows[selected_features])
    anomaly_matrix = np.column_stack([model_rows["CAMS_PM2.5"].to_numpy(), transformed])
    if len(model_rows) < 8:
        anomaly_labels = np.ones(len(model_rows), dtype=int)
        anomaly_note = "Isolation Forest deferred for the small sample."
    else:
        detector = IsolationForest(contamination="auto", random_state=42)
        anomaly_labels = detector.fit_predict(anomaly_matrix)
        anomaly_note = "Isolation Forest fitted only on this training period."
    normal_mask = anomaly_labels == 1
    use_fallback = int(normal_mask.sum()) < 10
    fit_mask = np.ones(len(model_rows), dtype=bool) if use_fallback else normal_mask

    model = RandomForestRegressor(
        n_estimators=400,
        min_samples_leaf=1,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(transformed[fit_mask], model_rows.loc[fit_mask, "CAMS_PM2.5"])
    return {
        "model": model,
        "imputer": imputer,
        "model_rows": model_rows,
        "normal_rows": int(normal_mask.sum()),
        "fit_rows": int(fit_mask.sum()),
        "used_fallback": use_fallback,
        "anomaly_note": anomaly_note,
        "features": selected_features,
    }


def threshold_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float | None]:
    """Calculate exceedance classification metrics with safe denominators."""
    actual_positive = actual > CPCB_SAFE_LIMIT
    predicted_positive = predicted > CPCB_SAFE_LIMIT
    tp = int(np.sum(actual_positive & predicted_positive))
    tn = int(np.sum(~actual_positive & ~predicted_positive))
    fp = int(np.sum(~actual_positive & predicted_positive))
    fn = int(np.sum(actual_positive & ~predicted_positive))
    total = len(actual)
    return {
        "Accuracy": (tp + tn) / total if total else None,
        "Recall": tp / (tp + fn) if tp + fn else None,
        "Precision": tp / (tp + fp) if tp + fp else None,
        "False-positive rate": fp / (fp + tn) if fp + tn else None,
        "TP": float(tp),
        "TN": float(tn),
        "FP": float(fp),
        "FN": float(fn),
    }


def data_fingerprint(frame: pd.DataFrame, configuration: dict[str, Any]) -> str:
    """Hash exact model data and configuration for persistent artifact reuse."""
    ordered = frame.copy().sort_values("Date").reset_index(drop=True)
    ordered["Date"] = ordered["Date"].astype(str)
    payload = pd.util.hash_pandas_object(ordered, index=False).values.tobytes()
    config_bytes = json.dumps(configuration, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(payload + config_bytes).hexdigest()


def invalidate_model_cache(path: Path = MODEL_CACHE_DIR) -> None:
    if not path.exists():
        return
    for artifact in path.glob("*.joblib"):
        artifact.unlink(missing_ok=True)


def generate_rolling_season_windows(daily_context: pd.DataFrame) -> list[dict[str, Any]]:
    """Generate every complete 60-day seasonal holdout with at least one prior year."""
    required = ["CAMS_PM2.5", *CORE_WEATHER_FEATURES]
    complete = (
        daily_context.dropna(subset=required)
        .sort_values("Date")
        .drop_duplicates("Date", keep="last")
    )
    if complete.empty:
        return []
    available = set(complete["Date"])
    first_date = min(available)
    last_date = max(available)
    specifications: list[dict[str, Any]] = []
    for cycle_year in range(first_date.year, last_date.year + 1):
        for season_id, season, year_offset, month, day_of_month in SEASON_TEMPLATES:
            holdout_start = date(cycle_year + year_offset, month, day_of_month)
            holdout_end = holdout_start + timedelta(days=59)
            if holdout_end > last_date:
                continue
            expected_holdout = {
                holdout_start + timedelta(days=offset) for offset in range(60)
            }
            if not expected_holdout.issubset(available):
                continue
            training_end = holdout_start - timedelta(days=1)
            training_start = max(first_date, five_year_training_start(training_end))
            training_dates = {
                current
                for current in available
                if training_start <= current <= training_end
            }
            if len(training_dates) < MIN_SEASON_TRAINING_ROWS:
                continue
            specifications.append(
                {
                    "id": f"{cycle_year}_{season_id}",
                    "cycle_year": cycle_year,
                    "year_label": f"{cycle_year}-{str(cycle_year + 1)[-2:]}",
                    "season": season,
                    "holdout_start": holdout_start,
                    "holdout_end": holdout_end,
                    "training_start": training_start,
                    "training_end": training_end,
                }
            )
    return specifications


def run_seasonal_backtest(
    daily_context: pd.DataFrame,
    specification: dict[str, Any],
    feature_names: list[str] | None = None,
    feature_group: str = "Baseline",
) -> dict[str, Any]:
    """Fit one expanding-history model and freeze it for a 60-day holdout.

    Training contains every earlier complete date available at that origin,
    capped at five calendar years. The first holdout lag is the true value from
    the immediately preceding calendar date.
    """
    holdout_start = specification["holdout_start"]
    holdout_end = specification["holdout_end"]
    training_start = specification["training_start"]
    training_end = specification["training_end"]
    expected_holdout = [
        holdout_start + timedelta(days=offset) for offset in range(60)
    ]
    if expected_holdout[-1] != holdout_end:
        raise AssertionError(f"{specification['season']} holdout is not 60 days.")

    selected_features = feature_names or MODEL_FEATURES
    mandatory_features = required_model_values(selected_features)
    data = daily_context.copy().sort_values("Date").reset_index(drop=True)
    lagged = data if "PM2.5_Yesterday" in data else add_calendar_lag(data, data)
    training = lagged.loc[
        lagged["Date"].between(training_start, training_end)
    ].copy()
    holdout = lagged.loc[
        lagged["Date"].between(holdout_start, holdout_end)
    ].copy()
    expected_set = set(expected_holdout)
    missing_holdout = sorted(expected_set - set(holdout["Date"]))
    if missing_holdout or len(holdout) != 60:
        raise IncompleteHistoryError(
            f"{specification['season']} holdout is not 60 complete dates.",
            missing_holdout,
        )
    if set(training["Date"]) & set(holdout["Date"]):
        raise AssertionError("Seasonal training and holdout date sets overlap.")

    valid_training = training.dropna(
        subset=["CAMS_PM2.5", *mandatory_features]
    ).copy()
    if len(valid_training) < MIN_SEASON_TRAINING_ROWS:
        raise IncompleteHistoryError(
            f"{specification['season']} has only {len(valid_training)} valid "
            f"training rows; {MIN_SEASON_TRAINING_ROWS} are required."
        )
    if holdout[["CAMS_PM2.5", *mandatory_features]].isna().any().any():
        raise IncompleteHistoryError(
            f"{specification['season']} holdout or lag context is incomplete."
        )
    boundary_lag = float(holdout.iloc[0]["PM2.5_Yesterday"])
    preceding = data.loc[data["Date"] == training_end, "CAMS_PM2.5"]
    if preceding.empty or not np.isclose(boundary_lag, float(preceding.iloc[0])):
        raise AssertionError(
            f"{specification['season']} first lag is not the preceding true date."
        )

    fitted = fit_period_model(valid_training, selected_features)
    transformed = fitted["imputer"].transform(holdout[selected_features])
    predictions = fitted["model"].predict(transformed)
    actual = holdout["CAMS_PM2.5"].to_numpy(dtype=float)
    persistence = holdout["PM2.5_Yesterday"].to_numpy(dtype=float)
    model_mae = float(mean_absolute_error(actual, predictions))
    baseline_mae = float(mean_absolute_error(actual, persistence))
    improvement = (
        100.0 * (baseline_mae - model_mae) / baseline_mae
        if baseline_mae > 0
        else None
    )
    results = pd.DataFrame(
        {
            "Date": holdout["Date"].to_list(),
            "Actual": actual,
            "Model": predictions,
            "Persistence": persistence,
        }
    )
    results["Residual"] = results["Actual"] - results["Model"]
    results["Absolute_Error"] = results["Residual"].abs()
    return {
        **fitted,
        "window_id": specification["id"],
        "cycle_year": specification["cycle_year"],
        "year_label": specification["year_label"],
        "feature_group": feature_group,
        "season": specification["season"],
        "training_start": training_start,
        "training_end": training_end,
        "holdout_start": holdout_start,
        "holdout_end": holdout_end,
        "calendar_training_dates": int(len(training)),
        "valid_training_rows": int(len(valid_training)),
        "holdout_rows": int(len(holdout)),
        "boundary_lag": boundary_lag,
        "boundary_actual": float(preceding.iloc[0]),
        "model_mae": model_mae,
        "model_rmse": float(np.sqrt(mean_squared_error(actual, predictions))),
        "model_r2": float(r2_score(actual, predictions)),
        "baseline_mae": baseline_mae,
        "improvement_pct": improvement,
        "classification": threshold_metrics(actual, predictions),
        "results": results,
        "holdout_features": holdout[selected_features].reset_index(drop=True),
    }


def cached_seasonal_backtest(
    daily_context: pd.DataFrame,
    specification: dict[str, Any],
    cache_location: str,
    feature_names: list[str] | None = None,
    feature_group: str = "Baseline",
    path: Path = MODEL_CACHE_DIR,
) -> tuple[dict[str, Any], bool]:
    """Load or build one versioned per-window model and metrics artifact."""
    start = specification["training_start"]
    end = specification["holdout_end"]
    relevant = daily_context.loc[daily_context["Date"].between(start, end)].copy()
    selected_features = feature_names or MODEL_FEATURES
    configuration = {
        "cache_version": MODEL_CACHE_VERSION,
        "location": cache_location,
        "window": specification,
        "features": selected_features,
        "feature_group": feature_group,
        "estimators": 400,
        "random_state": 42,
        "minimum_training_rows": MIN_SEASON_TRAINING_ROWS,
    }
    fingerprint_columns = list(dict.fromkeys(["Date", "CAMS_PM2.5", *selected_features]))
    fingerprint = data_fingerprint(relevant[fingerprint_columns], configuration)
    path.mkdir(parents=True, exist_ok=True)
    group_slug = feature_group.casefold().replace(" ", "_")
    artifact_path = path / f"{specification['id']}_{group_slug}_{fingerprint}.joblib"
    if artifact_path.exists():
        artifact = joblib.load(artifact_path)
        return artifact, True
    artifact = run_seasonal_backtest(
        daily_context, specification, selected_features, feature_group
    )
    artifact["artifact_fingerprint"] = fingerprint
    artifact["artifact_path"] = str(artifact_path)
    joblib.dump(artifact, artifact_path)
    for stale in path.glob(f"{specification['id']}_{group_slug}_*.joblib"):
        if stale != artifact_path:
            stale.unlink(missing_ok=True)
    return artifact, False


def build_seasonal_backtests(
    daily_context: pd.DataFrame,
    cache_location: str,
    feature_names: list[str] | None = None,
    feature_group: str = "Baseline",
) -> tuple[list[dict[str, Any]], int]:
    results: list[dict[str, Any]] = []
    cache_hits = 0
    model_ids: set[int] = set()
    specifications = generate_rolling_season_windows(daily_context)
    if not specifications:
        raise IncompleteHistoryError(
            "No complete rolling seasonal holdout has at least 365 earlier training rows."
        )
    for specification in specifications:
        result, cache_hit = cached_seasonal_backtest(
            daily_context, specification, cache_location, feature_names, feature_group
        )
        if id(result["model"]) in model_ids:
            raise AssertionError("Seasonal windows unexpectedly share a fitted model.")
        model_ids.add(id(result["model"]))
        cache_hits += int(cache_hit)
        results.append(result)
    return results, cache_hits


def evaluate_feature_candidates(
    engineered: pd.DataFrame,
    cache_location: str,
    fire_data_complete: bool,
) -> dict[str, Any]:
    """Compare feature groups and apply the documented production promotion gate."""
    groups = {
        name: features
        for name, features in FEATURE_GROUPS.items()
        if name != "Satellite fires" or fire_data_complete
    }
    by_group: dict[str, list[dict[str, Any]]] = {}
    cache_hits = 0
    for name, features in groups.items():
        results, hits = build_seasonal_backtests(
            engineered, cache_location, features, name
        )
        by_group[name] = results
        cache_hits += hits

    baseline = by_group["Baseline"]
    baseline_by_id = {item["window_id"]: item for item in baseline}
    baseline_median = float(np.median([item["model_mae"] for item in baseline]))
    window_count = len(baseline)
    season_names = sorted({item["season"] for item in baseline})
    year_labels = sorted({item["year_label"] for item in baseline})
    required_windows = math.ceil(0.75 * window_count)
    required_seasons = math.ceil(0.75 * len(season_names))
    required_years = math.ceil(0.75 * len(year_labels))
    decisions: list[dict[str, Any]] = []
    eligible: list[dict[str, Any]] = []
    for name, results in by_group.items():
        maes = [item["model_mae"] for item in results]
        median_mae = float(np.median(maes))
        improvement = 100.0 * (baseline_median - median_mae) / baseline_median if baseline_median else 0.0
        windows_not_worse = sum(
            candidate["model_mae"]
            <= baseline_by_id[candidate["window_id"]]["model_mae"] + 1e-9
            for candidate in results
        )
        seasons_not_worse = 0
        for season in season_names:
            candidate_values = [
                item["model_mae"] for item in results if item["season"] == season
            ]
            baseline_values = [
                item["model_mae"] for item in baseline if item["season"] == season
            ]
            seasons_not_worse += int(
                np.median(candidate_values) <= np.median(baseline_values) + 1e-9
            )
        years_not_worse = 0
        for year_label in year_labels:
            candidate_values = [
                item["model_mae"] for item in results if item["year_label"] == year_label
            ]
            baseline_values = [
                item["model_mae"] for item in baseline if item["year_label"] == year_label
            ]
            years_not_worse += int(
                np.median(candidate_values) <= np.median(baseline_values) + 1e-9
            )
        recall_drop = 0.0
        recall_valid = True
        for candidate in results:
            reference_recall = baseline_by_id[candidate["window_id"]]["classification"]["Recall"]
            candidate_recall = candidate["classification"]["Recall"]
            if reference_recall is not None and candidate_recall is None:
                recall_valid = False
            elif reference_recall is not None and candidate_recall is not None:
                recall_drop = max(recall_drop, float(reference_recall - candidate_recall))
        promoted = (
            name != "Baseline"
            and improvement >= 2.0
            and windows_not_worse >= required_windows
            and seasons_not_worse >= required_seasons
            and years_not_worse >= required_years
            and recall_valid
            and recall_drop <= 0.05 + 1e-12
        )
        decision = {
            "Feature group": name,
            "Median MAE": median_mae,
            "Median improvement (%)": improvement,
            "Windows improved or tied": f"{windows_not_worse}/{window_count}",
            "Seasons improved or tied": f"{seasons_not_worse}/{len(season_names)}",
            "Years improved or tied": f"{years_not_worse}/{len(year_labels)}",
            "Worst recall drop (points)": 100.0 * recall_drop,
            "Passes gate": promoted,
        }
        decisions.append(decision)
        if promoted:
            eligible.append(decision)
    selected = min(eligible, key=lambda item: item["Median MAE"])["Feature group"] if eligible else "Baseline"
    return {
        "by_group": by_group,
        "decisions": pd.DataFrame(decisions),
        "selected_group": selected,
        "selected_features": groups[selected],
        "cache_hits": cache_hits,
        "artifact_count": window_count * len(groups),
        "window_count": window_count,
        "year_count": len(year_labels),
        "season_count": len(season_names),
    }


def grouped_permutation_importance(
    seasonal_results: list[dict[str, Any]],
) -> pd.DataFrame:
    """Aggregate holdout permutation effects into understandable factor families."""
    feature_family = {
        **{feature: "Current weather" for feature in ("Temperature", "Humidity", "Wind_Speed")},
        **{feature: "Pollution history" for feature in HISTORY_FEATURES if feature not in ("Temperature", "Humidity", "Wind_Speed")},
        **{feature: "Expanded weather" for feature in EXPANDED_WEATHER_FEATURES if feature not in HISTORY_FEATURES},
        **{feature: "Calendar and festivals" for feature in CALENDAR_FEATURES if feature not in EXPANDED_WEATHER_FEATURES},
        **{feature: "Satellite fires" for feature in FIRE_FEATURES if feature not in CALENDAR_FEATURES},
    }
    rows: list[dict[str, Any]] = []
    for result in seasonal_results:
        features = result["features"]
        transformed = result["imputer"].transform(result["holdout_features"][features])
        actual = result["results"]["Actual"].to_numpy(dtype=float)
        importance = permutation_importance(
            result["model"], transformed, actual,
            scoring="neg_mean_absolute_error", n_repeats=3, random_state=42, n_jobs=1,
        )
        for feature, value in zip(features, importance.importances_mean):
            rows.append(
                {
                    "Season": result["season"],
                    "Factor group": feature_family.get(feature, "Other"),
                    "MAE increase when shuffled": max(0.0, float(value)),
                }
            )
    if not rows:
        return pd.DataFrame(columns=["Factor group", "MAE increase when shuffled"])
    return (
        pd.DataFrame(rows)
        .groupby("Factor group", as_index=False)["MAE increase when shuffled"]
        .mean()
        .sort_values("MAE increase when shuffled", ascending=False)
        .reset_index(drop=True)
    )


def cached_grouped_permutation_importance(
    seasonal_results: list[dict[str, Any]],
    path: Path = MODEL_CACHE_DIR,
) -> tuple[pd.DataFrame, bool]:
    """Persist grouped permutation importance for an unchanged artifact set."""
    fingerprints = [
        result.get("artifact_fingerprint", result["window_id"])
        for result in seasonal_results
    ]
    signature = hashlib.sha256(
        json.dumps(
            {
                "cache_version": MODEL_CACHE_VERSION,
                "repeats": 3,
                "jobs": 1,
                "artifacts": fingerprints,
            },
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()
    path.mkdir(parents=True, exist_ok=True)
    artifact_path = path / f"permutation_importance_{signature}.joblib"
    if artifact_path.exists():
        return joblib.load(artifact_path), True
    importance = grouped_permutation_importance(seasonal_results)
    joblib.dump(importance, artifact_path)
    for stale in path.glob("permutation_importance_*.joblib"):
        if stale != artifact_path:
            stale.unlink(missing_ok=True)
    return importance, False


def run_backtest(window: pd.DataFrame) -> dict[str, Any]:
    """Run a fixed 60/30 calendar split with no test-period fitting."""
    data = window.copy().sort_values("Date").reset_index(drop=True)
    if len(data) != HISTORY_DAYS:
        raise ValueError("Backtest requires exactly 90 complete calendar dates.")
    expected = pd.date_range(data.loc[0, "Date"], periods=HISTORY_DAYS, freq="D").date
    if list(data["Date"]) != list(expected):
        raise ValueError("Backtest dates are not contiguous calendar dates.")

    data["PM2.5_Yesterday"] = data["CAMS_PM2.5"].shift(1)
    train_end = data.loc[0, "Date"] + timedelta(days=TRAIN_DAYS - 1)
    test_start = train_end + timedelta(days=1)
    train = data.loc[data["Date"] <= train_end].copy()
    test = data.loc[data["Date"] >= test_start].copy()
    if len(train) != TRAIN_DAYS or len(test) != TEST_DAYS:
        raise AssertionError("Calendar threshold did not produce a 60/30 split.")
    assert np.isclose(
        test.iloc[0]["PM2.5_Yesterday"], train.iloc[-1]["CAMS_PM2.5"]
    ), "Day 61 lag does not equal day 60 true PM2.5."

    fitted = fit_period_model(train)
    test_x = fitted["imputer"].transform(test[MODEL_FEATURES])
    predictions = fitted["model"].predict(test_x)
    actual = test["CAMS_PM2.5"].to_numpy(dtype=float)
    persistence = test["PM2.5_Yesterday"].to_numpy(dtype=float)
    results = pd.DataFrame(
        {
            "Date": test["Date"].to_list(),
            "Actual": actual,
            "Model": predictions,
            "Persistence": persistence,
        }
    )
    results["Residual"] = results["Actual"] - results["Model"]
    results["Absolute_Error"] = results["Residual"].abs()
    return {
        **fitted,
        "train_start": train["Date"].min(),
        "train_end": train["Date"].max(),
        "test_start": test["Date"].min(),
        "test_end": test["Date"].max(),
        "day61_lag": float(test.iloc[0]["PM2.5_Yesterday"]),
        "day60_pm25": float(train.iloc[-1]["CAMS_PM2.5"]),
        "results": results,
        "model_mae": float(mean_absolute_error(actual, predictions)),
        "model_rmse": float(np.sqrt(mean_squared_error(actual, predictions))),
        "model_r2": float(r2_score(actual, predictions)),
        "baseline_mae": float(mean_absolute_error(actual, persistence)),
        "baseline_rmse": float(np.sqrt(mean_squared_error(actual, persistence))),
        "classification": threshold_metrics(actual, predictions),
        "test_frame": test,
    }


def run_recursive_validation(backtest: dict[str, Any]) -> pd.DataFrame:
    """Evaluate all 24 seven-day rolling recursive windows in the test period."""
    test = backtest["test_frame"].reset_index(drop=True)
    errors: dict[int, list[float]] = {horizon: [] for horizon in range(1, 8)}
    model = backtest["model"]
    imputer = backtest["imputer"]
    for origin in range(len(test) - FORECAST_DAYS + 1):
        lag = float(test.loc[origin, "PM2.5_Yesterday"])
        for step in range(FORECAST_DAYS):
            row = test.loc[origin + step]
            features = pd.DataFrame(
                [{
                    "Temperature": row["Temperature"],
                    "Humidity": row["Humidity"],
                    "Wind_Speed": row["Wind_Speed"],
                    "PM2.5_Yesterday": lag,
                }]
            )
            transformed = imputer.transform(features[MODEL_FEATURES])
            prediction = float(model.predict(transformed)[0])
            errors[step + 1].append(abs(float(row["CAMS_PM2.5"]) - prediction))
            lag = prediction
    return pd.DataFrame(
        {
            "Horizon": [f"+{horizon}" for horizon in range(1, 8)],
            "Day_Ahead": list(range(1, 8)),
            "MAE": [float(np.mean(errors[horizon])) for horizon in range(1, 8)],
            "Windows": [len(errors[horizon]) for horizon in range(1, 8)],
        }
    )


def create_production_forecast(
    window: pd.DataFrame,
    forecast_weather: pd.DataFrame,
    local_today: date,
    recursive_metrics: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Fit all 90 days and recursively forecast tomorrow through day +7."""
    fitted = fit_period_model(window)
    cutoff = window["Date"].max()
    target_end = local_today + timedelta(days=FORECAST_DAYS)
    bridge_dates = pd.date_range(cutoff + timedelta(days=1), target_end, freq="D").date
    weather = forecast_weather.set_index("Date")
    missing_weather = [day for day in bridge_dates if day not in weather.index]
    if missing_weather:
        raise ValueError(
            "Forecast weather is missing: " + ", ".join(day.isoformat() for day in missing_weather)
        )

    lag = float(window.sort_values("Date").iloc[-1]["CAMS_PM2.5"])
    rows: list[dict[str, Any]] = []
    for forecast_date in bridge_dates:
        weather_row = weather.loc[forecast_date]
        features = pd.DataFrame(
            [{
                "Temperature": weather_row["Temperature"],
                "Humidity": weather_row["Humidity"],
                "Wind_Speed": weather_row["Wind_Speed"],
                "PM2.5_Yesterday": lag,
            }]
        )
        transformed = fitted["imputer"].transform(features[MODEL_FEATURES])
        tree_predictions = np.array(
            [tree.predict(transformed)[0] for tree in fitted["model"].estimators_],
            dtype=float,
        )
        prediction = float(tree_predictions.mean())
        rows.append(
            {
                "Date": forecast_date,
                "Prediction": prediction,
                "Tree_P10": float(np.quantile(tree_predictions, 0.10)),
                "Tree_P90": float(np.quantile(tree_predictions, 0.90)),
                "Is_Bridge": forecast_date <= local_today,
            }
        )
        lag = prediction

    output = pd.DataFrame(rows)
    displayed = output.loc[output["Date"] > local_today].head(FORECAST_DAYS).copy()
    if len(displayed) != FORECAST_DAYS:
        raise ValueError("Could not construct all seven forward forecast dates.")
    horizon_mae = recursive_metrics.sort_values("Day_Ahead")["MAE"].to_numpy()
    expansion = np.maximum.accumulate(np.maximum(0.0, horizon_mae - horizon_mae[0]))
    tree_half_width = np.maximum(
        displayed["Prediction"].to_numpy() - displayed["Tree_P10"].to_numpy(),
        displayed["Tree_P90"].to_numpy() - displayed["Prediction"].to_numpy(),
    )
    half_width = np.maximum.accumulate(tree_half_width + expansion)
    displayed["Lower_80"] = np.maximum(0.0, displayed["Prediction"].to_numpy() - half_width)
    displayed["Upper_80"] = displayed["Prediction"].to_numpy() + half_width
    displayed["Day_Ahead"] = np.arange(1, FORECAST_DAYS + 1)
    displayed["Exceeds_60"] = displayed["Prediction"] > CPCB_SAFE_LIMIT
    return displayed.reset_index(drop=True), fitted


def select_production_training(
    engineered: pd.DataFrame, feature_names: list[str]
) -> tuple[pd.DataFrame, int]:
    """Select the latest contiguous usable history, capped at five calendar years."""
    mandatory_features = required_model_values(feature_names)
    complete = (
        engineered.dropna(subset=["CAMS_PM2.5", *mandatory_features])
        .sort_values("Date")
        .drop_duplicates("Date", keep="last")
        .copy()
    )
    if complete.empty:
        raise ValueError("No complete rows are available for the selected production model.")
    cutoff = complete["Date"].max()
    available = set(complete["Date"])
    cap_start = five_year_training_start(cutoff)
    contiguous_start = cutoff
    while (
        contiguous_start - timedelta(days=1) in available
        and contiguous_start > cap_start
    ):
        contiguous_start -= timedelta(days=1)
    selected = complete.loc[
        complete["Date"].between(contiguous_start, cutoff)
    ].sort_values("Date")
    if len(selected) < MIN_PRODUCTION_TRAINING_ROWS:
        raise ValueError(
            f"Only {len(selected)} contiguous usable production rows are available; "
            f"{MIN_PRODUCTION_TRAINING_ROWS} are required."
        )
    expected = pd.date_range(contiguous_start, cutoff, freq="D").date
    if len(expected) != len(selected) or set(expected) != set(selected["Date"]):
        raise AssertionError("Production training dates are not contiguous.")
    return selected.reset_index(drop=True), int(len(selected))


def create_multifactor_production_forecast(
    engineered_history: pd.DataFrame,
    forecast_weather: pd.DataFrame,
    local_today: date,
    recursive_metrics: pd.DataFrame,
    feature_names: list[str],
    fire_observations: pd.DataFrame | None = None,
    fire_data_complete: bool = False,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Fit the promoted feature schema and recursively forecast tomorrow through +7."""
    training, training_days = select_production_training(engineered_history, feature_names)
    fitted = fit_period_model(training, feature_names)
    fitted["training_days"] = training_days
    cutoff = engineered_history["Date"].max()
    target_end = local_today + timedelta(days=FORECAST_DAYS)
    bridge_dates = list(pd.date_range(cutoff + timedelta(days=1), target_end, freq="D").date)
    weather = forecast_weather.set_index("Date")
    missing_weather = [day for day in bridge_dates if day not in weather.index]
    if missing_weather:
        raise ValueError(
            "Forecast weather is missing: " + ", ".join(day.isoformat() for day in missing_weather)
        )

    working = engineered_history[["Date", "CAMS_PM2.5", *WEATHER_FEATURES, "Timezone"]].copy()
    timezone_name = str(engineered_history.iloc[-1].get("Timezone", "Asia/Kolkata"))
    rows: list[dict[str, Any]] = []
    for forecast_date in bridge_dates:
        weather_row = weather.loc[forecast_date]
        new_row = {
            "Date": forecast_date,
            "CAMS_PM2.5": np.nan,
            **{column: weather_row[column] for column in WEATHER_FEATURES},
            "Timezone": timezone_name,
        }
        candidate = pd.concat([working, pd.DataFrame([new_row])], ignore_index=True)
        feature_row = add_multifactor_features(
            candidate, fire_observations, fire_data_complete
        ).loc[lambda value: value["Date"] == forecast_date]
        mandatory_features = required_model_values(feature_names)
        if feature_row.empty or feature_row[mandatory_features].isna().any().any():
            missing = feature_row[mandatory_features].columns[
                feature_row[mandatory_features].isna().any()
            ].tolist() if not feature_row.empty else mandatory_features
            raise ValueError("Forecast factors are missing: " + ", ".join(missing))
        transformed = fitted["imputer"].transform(feature_row[feature_names])
        tree_predictions = np.array(
            [tree.predict(transformed)[0] for tree in fitted["model"].estimators_],
            dtype=float,
        )
        prediction = max(0.0, float(tree_predictions.mean()))
        rows.append(
            {
                "Date": forecast_date,
                "Prediction": prediction,
                "Tree_P10": max(0.0, float(np.quantile(tree_predictions, 0.10))),
                "Tree_P90": max(0.0, float(np.quantile(tree_predictions, 0.90))),
                "Is_Bridge": forecast_date <= local_today,
            }
        )
        candidate.loc[candidate["Date"] == forecast_date, "CAMS_PM2.5"] = prediction
        working = candidate

    output = pd.DataFrame(rows)
    displayed = output.loc[output["Date"] > local_today].head(FORECAST_DAYS).copy()
    if len(displayed) != FORECAST_DAYS:
        raise ValueError("Could not construct all seven forward forecast dates.")
    horizon_mae = recursive_metrics.sort_values("Day_Ahead")["MAE"].to_numpy()
    expansion = np.maximum.accumulate(np.maximum(0.0, horizon_mae - horizon_mae[0]))
    tree_half_width = np.maximum(
        displayed["Prediction"].to_numpy() - displayed["Tree_P10"].to_numpy(),
        displayed["Tree_P90"].to_numpy() - displayed["Prediction"].to_numpy(),
    )
    half_width = np.maximum.accumulate(tree_half_width + expansion)
    displayed["Lower_80"] = np.maximum(0.0, displayed["Prediction"].to_numpy() - half_width)
    displayed["Upper_80"] = displayed["Prediction"].to_numpy() + half_width
    displayed["Day_Ahead"] = np.arange(1, FORECAST_DAYS + 1)
    displayed["Exceeds_60"] = displayed["Prediction"] > CPCB_SAFE_LIMIT
    return displayed.reset_index(drop=True), fitted


def log_prospective_forecast(
    cache_location: str,
    forecast_row: pd.Series,
    timezone_name: str,
    model_hash: str,
    latest_actual_date: date,
    path: Path = CACHE_DB_FILE,
) -> bool:
    """Insert the official next-day forecast once; never update it."""
    target_date = forecast_row["Date"]
    if target_date <= latest_actual_date:
        return False
    initialize_database(path)
    with database_connection(path) as connection:
        cursor = connection.execute(
            """
            INSERT OR IGNORE INTO prospective_forecasts(
                location_key, target_date, prediction, lower_80, upper_80,
                issued_at_utc, timezone, model_hash
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                cache_location,
                target_date.isoformat(),
                float(forecast_row["Prediction"]),
                float(forecast_row["Lower_80"]),
                float(forecast_row["Upper_80"]),
                datetime.now(timezone.utc).isoformat(),
                timezone_name,
                model_hash,
            ),
        )
    return cursor.rowcount == 1


def record_prospective_actuals(
    cache_location: str,
    daily_history: pd.DataFrame,
    path: Path = CACHE_DB_FILE,
) -> int:
    """Insert newly completed CAMS actuals once without mutating prior values."""
    initialize_database(path)
    if daily_history.empty:
        return 0
    actual_by_date = {
        row["Date"].isoformat(): float(row["CAMS_PM2.5"])
        for row in daily_history.to_dict("records")
        if not pd.isna(row["CAMS_PM2.5"])
    }
    inserted = 0
    with database_connection(path) as connection:
        targets = connection.execute(
            """
            SELECT target_date FROM prospective_forecasts
            WHERE location_key = ?
            """,
            (cache_location,),
        ).fetchall()
        for (target_date,) in targets:
            if target_date not in actual_by_date:
                continue
            cursor = connection.execute(
                """
                INSERT OR IGNORE INTO prospective_actuals(
                    location_key, target_date, actual_pm25,
                    recorded_at_utc, source
                ) VALUES (?, ?, ?, ?, ?)
                """,
                (
                    cache_location,
                    target_date,
                    actual_by_date[target_date],
                    datetime.now(timezone.utc).isoformat(),
                    "Open-Meteo CAMS completed daily mean",
                ),
            )
            inserted += int(cursor.rowcount == 1)
    return inserted


def prospective_evaluation(
    cache_location: str, path: Path = CACHE_DB_FILE
) -> pd.DataFrame:
    initialize_database(path)
    query = """
        SELECT f.target_date AS Date,
               f.prediction AS Prediction,
               f.lower_80 AS Lower_80,
               f.upper_80 AS Upper_80,
               f.issued_at_utc AS Issued_At_UTC,
               f.model_hash AS Model_Hash,
               a.actual_pm25 AS Actual,
               a.recorded_at_utc AS Actual_Recorded_At_UTC
        FROM prospective_forecasts AS f
        LEFT JOIN prospective_actuals AS a
          ON a.location_key = f.location_key
         AND a.target_date = f.target_date
        WHERE f.location_key = ?
        ORDER BY f.target_date
    """
    with database_connection(path) as connection:
        frame = pd.read_sql_query(query, connection, params=(cache_location,))
    if not frame.empty:
        frame["Date"] = pd.to_datetime(frame["Date"]).dt.date
    return frame


def style_chart(figure: plt.Figure, axis: plt.Axes) -> None:
    """Match Matplotlib output to the dashboard without hiding chart structure."""
    figure.patch.set_facecolor("#101719")
    axis.set_facecolor("#101719")
    axis.tick_params(colors="#9badb0", labelsize=9)
    axis.xaxis.label.set_color("#9badb0")
    axis.yaxis.label.set_color("#9badb0")
    axis.grid(axis="y", color="#293439", linewidth=0.8, alpha=0.9)
    axis.spines[["top", "right"]].set_visible(False)
    axis.spines[["left", "bottom"]].set_color("#405158")
    legend = axis.legend(
        frameon=False,
        labelcolor="#dce8e6",
        loc="upper left",
        ncol=min(4, len(axis.get_legend_handles_labels()[0])),
    )
    if legend:
        legend.get_frame().set_facecolor("#101719")
    figure.tight_layout()


def backtest_chart(results: pd.DataFrame) -> plt.Figure:
    figure, axis = plt.subplots(figsize=(11, 4.5))
    axis.plot(results["Date"], results["Actual"], label="Actual CAMS", color="#62d6d6", linewidth=2.3)
    axis.plot(results["Date"], results["Model"], label="Random Forest", color="#ff6b5f", linewidth=1.9)
    axis.plot(results["Date"], results["Persistence"], label="Persistence", color="#f2c14e", linestyle="--", linewidth=1.4)
    axis.axhline(CPCB_SAFE_LIMIT, color="#ff4057", linestyle=":", label="60 ug/m3 threshold")
    axis.set_ylabel("Daily mean PM2.5 (ug/m3)")
    figure.autofmt_xdate(rotation=25)
    style_chart(figure, axis)
    return figure


def recursive_chart(metrics: pd.DataFrame) -> plt.Figure:
    figure, axis = plt.subplots(figsize=(8, 3.8))
    axis.plot(metrics["Day_Ahead"], metrics["MAE"], color="#62d6d6", marker="s", markersize=5, linewidth=2.2, label="Recursive MAE")
    axis.set_xticks(range(1, 8))
    axis.set_xlabel("Recursive forecast horizon (days ahead)")
    axis.set_ylabel("MAE (ug/m3)")
    style_chart(figure, axis)
    return figure


def forecast_chart(forecast: pd.DataFrame) -> plt.Figure:
    figure, axis = plt.subplots(figsize=(10, 4.3))
    axis.fill_between(
        forecast["Date"],
        forecast["Lower_80"],
        forecast["Upper_80"],
        color="#315c60",
        alpha=0.72,
        label="Widened 80% interval",
    )
    axis.plot(forecast["Date"], forecast["Prediction"], color="#70d6a5", marker="s", markersize=5, linewidth=2.3, label="Forecast")
    axis.axhline(CPCB_SAFE_LIMIT, color="#ff4057", linestyle="--", label="60 ug/m3 threshold")
    axis.set_ylabel("Forecast PM2.5 (ug/m3)")
    figure.autofmt_xdate(rotation=20)
    style_chart(figure, axis)
    return figure


def read_secret_api_key() -> str:
    """Read Streamlit secrets safely; absent secrets files raise exceptions."""
    try:
        return str(st.secrets.get("DATA_GOV_IN_API_KEY", ""))
    except Exception:
        return ""


def read_firms_map_key() -> str:
    """Read the optional FIRMS key without assuming a secrets file exists."""
    try:
        return str(st.secrets.get("FIRMS_MAP_KEY", ""))
    except Exception:
        return ""


def format_rate(value: float | None) -> str:
    return "N/A" if value is None else f"{100 * value:.1f}%"


def render_visual_system() -> None:
    """Apply the dashboard's atmospheric operations-console visual system."""
    st.markdown(
        """
        <style>
        :root {
            --aq-bg: #0b1012;
            --aq-panel: #151b1e;
            --aq-panel-2: #101618;
            --aq-line: #293439;
            --aq-text: #f3f7f5;
            --aq-muted: #9badb0;
            --aq-cyan: #62d6d6;
            --aq-coral: #ff6b5f;
            --aq-amber: #f2c14e;
            --aq-green: #70d6a5;
        }
        html, body, [class*="css"] {
            letter-spacing: 0 !important;
        }
        [data-testid="stAppViewContainer"] {
            background: var(--aq-bg);
            color: var(--aq-text);
        }
        [data-testid="stHeader"] {
            background: rgba(11, 16, 18, 0.94);
            border-bottom: 1px solid var(--aq-line);
        }
        [data-testid="stMainBlockContainer"] {
            max-width: 1260px;
            padding-top: 1.25rem;
            padding-bottom: 4rem;
        }
        [data-testid="stSidebar"] {
            background: #101517;
            border-right: 1px solid var(--aq-line);
        }
        [data-testid="stSidebar"] [data-testid="stVerticalBlock"] {
            gap: 0.65rem;
        }
        [data-testid="stSidebar"] h2 {
            font-size: 1.05rem;
            color: var(--aq-text);
            border: 0;
            padding: 0;
        }
        [data-testid="stSidebar"] h3 {
            font-size: 0.85rem;
            color: var(--aq-cyan);
            text-transform: uppercase;
        }
        .aq-masthead {
            position: relative;
            min-height: 230px;
            overflow: hidden;
            border-top: 1px solid #395158;
            border-bottom: 1px solid var(--aq-line);
            padding: 2.4rem 2rem 2rem;
            margin: 0 0 1.8rem;
            background: #101719;
        }
        .aq-masthead-content {
            position: relative;
            z-index: 2;
            max-width: 760px;
        }
        .aq-kicker {
            display: inline-flex;
            align-items: center;
            gap: 0.55rem;
            color: var(--aq-cyan);
            font-size: 0.76rem;
            font-weight: 700;
            text-transform: uppercase;
        }
        .aq-kicker-mark {
            display: inline-block;
            width: 18px;
            height: 5px;
            background: var(--aq-coral);
        }
        .aq-masthead h1 {
            max-width: 760px;
            margin: 0.8rem 0 0.7rem;
            color: var(--aq-text);
            font-size: 2.7rem;
            line-height: 1.04;
            font-weight: 760;
            letter-spacing: 0;
        }
        .aq-masthead h1 span {
            color: var(--aq-cyan);
        }
        .aq-masthead p {
            max-width: 680px;
            margin: 0;
            color: var(--aq-muted);
            font-size: 1rem;
            line-height: 1.65;
        }
        .aq-airflow {
            position: absolute;
            inset: 0;
            overflow: hidden;
            opacity: 0.54;
            pointer-events: none;
        }
        .aq-stream {
            position: absolute;
            left: -42%;
            width: 34%;
            height: 2px;
            background: var(--aq-cyan);
            animation: aq-drift 10s linear infinite;
        }
        .aq-stream::after {
            content: "";
            position: absolute;
            right: -34px;
            top: -2px;
            width: 24px;
            height: 6px;
            background: var(--aq-coral);
        }
        .aq-stream.s1 { top: 20%; animation-duration: 12s; }
        .aq-stream.s2 { top: 42%; animation-delay: -7s; animation-duration: 15s; opacity: 0.65; }
        .aq-stream.s3 { top: 67%; animation-delay: -3s; animation-duration: 11s; opacity: 0.42; }
        .aq-stream.s4 { top: 84%; animation-delay: -10s; animation-duration: 18s; opacity: 0.3; }
        @keyframes aq-drift {
            from { transform: translateX(0); }
            to { transform: translateX(430%); }
        }
        section.main h2 {
            margin-top: 2.1rem;
            padding-left: 0.8rem;
            border-left: 4px solid var(--aq-coral);
            color: var(--aq-text);
            font-size: 1.65rem;
            line-height: 1.2;
        }
        section.main h3 {
            margin-top: 1.5rem;
            color: var(--aq-cyan);
            font-size: 1.08rem;
            text-transform: uppercase;
        }
        section.main p, [data-testid="stCaptionContainer"] {
            color: var(--aq-muted);
        }
        [data-testid="stMetric"] {
            min-height: 112px;
            padding: 1rem 1rem 0.85rem;
            background: var(--aq-panel);
            border: 1px solid var(--aq-line);
            border-top: 3px solid var(--aq-cyan);
            border-radius: 6px;
        }
        [data-testid="stMetricLabel"] {
            color: var(--aq-muted);
            font-size: 0.76rem;
            text-transform: uppercase;
        }
        [data-testid="stMetricValue"] {
            color: var(--aq-text);
            font-variant-numeric: tabular-nums;
        }
        [data-testid="stAlert"] {
            border-radius: 6px;
            border-width: 1px;
            border-left-width: 4px;
            background: var(--aq-panel);
        }
        [data-testid="stDataFrame"] {
            border: 1px solid var(--aq-line);
            border-radius: 6px;
            overflow: hidden;
        }
        [data-testid="stExpander"] {
            border: 1px solid var(--aq-line);
            border-radius: 6px;
            background: var(--aq-panel-2);
        }
        [data-baseweb="input"], [data-baseweb="select"] > div {
            border-radius: 5px !important;
            border-color: var(--aq-line) !important;
            background: #0d1214 !important;
        }
        div.stButton > button, div.stDownloadButton > button {
            min-height: 2.65rem;
            border-radius: 5px;
            border: 1px solid var(--aq-coral);
            font-weight: 700;
            letter-spacing: 0;
        }
        div.stButton > button:hover, div.stDownloadButton > button:hover {
            border-color: var(--aq-cyan);
            color: var(--aq-cyan);
        }
        hr {
            border-color: var(--aq-line) !important;
        }
        ::-webkit-scrollbar { width: 10px; height: 10px; }
        ::-webkit-scrollbar-track { background: var(--aq-bg); }
        ::-webkit-scrollbar-thumb { background: #405158; border-radius: 4px; }
        @media (max-width: 700px) {
            [data-testid="stMainBlockContainer"] {
                padding-left: 1rem;
                padding-right: 1rem;
            }
            .aq-masthead {
                min-height: 245px;
                padding: 2rem 1.25rem 1.6rem;
            }
            .aq-masthead h1 {
                font-size: 2rem;
                line-height: 1.08;
            }
            .aq-masthead p { font-size: 0.92rem; }
            .aq-airflow { opacity: 0.38; }
            .aq-stream.s1 { top: 15%; }
            .aq-stream.s4 { top: 92%; }
            .aq-stream.s2, .aq-stream.s3 { display: none; }
            section.main h2 { font-size: 1.35rem; }
            [data-testid="stMetric"] { min-height: 96px; }
        }
        @media (prefers-reduced-motion: reduce) {
            .aq-stream { animation: none; }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_masthead() -> None:
    st.markdown(
        """
        <div class="aq-masthead">
            <div class="aq-airflow" aria-hidden="true">
                <span class="aq-stream s1"></span>
                <span class="aq-stream s2"></span>
                <span class="aq-stream s3"></span>
                <span class="aq-stream s4"></span>
            </div>
            <div class="aq-masthead-content">
                <div class="aq-kicker"><span class="aq-kicker-mark"></span>PM2.5 / Urban Atmosphere</div>
                <h1>Urban Air Quality<br><span>Early Warning System</span></h1>
                <p>Maximum-available CAMS training under a five-year cap, rolling yearly seasonal backtests, prospective evaluation, and a seven-day PM2.5 forecast.</p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def main() -> None:
    st.set_page_config(
        page_title="Urban Air Quality Early Warning",
        page_icon="AQ",
        layout="wide",
    )
    render_visual_system()
    initialize_database()

    if "refresh_token" not in st.session_state:
        st.session_state.refresh_token = 0
    if "cpcb_attempt_date" not in st.session_state:
        st.session_state.cpcb_attempt_date = None
    if "cpcb_status" not in st.session_state:
        st.session_state.cpcb_status = None

    video_hero_active = bool(
        render_atmospheric_hero
        and video_hero_feature_enabled
        and video_hero_feature_enabled()
    )
    motion_enabled = True
    with st.sidebar:
        st.header("Location and data")
        city = st.text_input("City", value="Delhi").strip()
        latitude = st.number_input("Latitude", value=28.6139, format="%.4f")
        longitude = st.number_input("Longitude", value=77.2090, format="%.4f")
        secret_key = read_secret_api_key()
        api_key = secret_key or st.text_input("data.gov.in API key", type="password")
        firms_secret = read_firms_map_key()
        firms_key = firms_secret or st.text_input(
            "NASA FIRMS MAP_KEY (optional)", type="password",
            help="Enables satellite-detected thermal activity factors. The key is never stored by the app.",
        )
        if video_hero_active:
            st.divider()
            st.subheader("Display")
            motion_enabled = st.checkbox(
                "Atmospheric motion",
                value=True,
                help="Pause the decorative city video while retaining its poster image.",
            )
        st.divider()
        st.subheader("Manual observation")
        save_manual = st.checkbox("Save a separate manual PM2.5 value")
        manual_pm25 = st.number_input(
            "Manual PM2.5 (ug/m3)", min_value=0.0, value=50.0, step=1.0,
            disabled=not save_manual,
        )
        fetch_clicked = st.button("Refresh Data", type="primary", use_container_width=True)
        st.caption(
            "The app checks for stale data when opened. Updates while it is closed "
            "require an external scheduler."
        )

    hero_rendered = bool(
        video_hero_active
        and render_atmospheric_hero
        and render_atmospheric_hero(city, motion_enabled)
    )
    if not hero_rendered:
        render_masthead()

    if fetch_clicked:
        st.session_state.refresh_token += 1
        st.session_state.cpcb_attempt_date = None
        st.session_state.cpcb_status = None
        invalidate_model_cache()
    refresh_token = st.session_state.refresh_token

    history = load_history(DATA_FILE)
    forecast_context: dict[str, Any] | None = None
    cached_daily = pd.DataFrame()
    cache_summary: dict[str, Any] | None = None
    fire_summary: dict[str, Any] | None = None
    fire_observations = pd.DataFrame()
    fire_data_complete = False
    cams_source_start: date | None = None
    archive_start: date | None = None
    archive_end: date | None = None
    requested_training_start: date | None = None
    cache_location = location_key(latitude, longitude)
    data_errors: list[str] = []

    with st.spinner("Checking historical air quality and weather data..."):
        try:
            forecast_context = fetch_forecast_weather(latitude, longitude, refresh_token)
        except APIUnavailableError as exc:
            data_errors.append(str(exc))

        if forecast_context:
            try:
                archive_end = forecast_context["today"] - timedelta(days=1)
                requested_training_start = five_year_training_start(archive_end)
                cams_source_start, coverage_timezone = discover_cams_coverage_start(
                    latitude, longitude, refresh_token
                )
                if coverage_timezone != forecast_context["timezone"]:
                    raise APIUnavailableError(
                        "CAMS coverage and forecast responses returned different timezones."
                    )
                archive_start = max(
                    cams_source_start,
                    requested_training_start - timedelta(days=LAG_CONTEXT_DAYS),
                )
                cache_summary = ensure_hourly_cache(
                    latitude,
                    longitude,
                    archive_start,
                    archive_end,
                    refresh_token,
                )
                cached_daily = daily_history_from_cache(
                    cache_location, archive_start, archive_end
                )
                history = merge_historical_rows(history, cached_daily)
                save_history(history, DATA_FILE)
            except APIUnavailableError as exc:
                data_errors.append(str(exc))

            fire_start = archive_start or CAMS_COVERAGE_PROBE_START
            fire_end = forecast_context["today"]
            if firms_key:
                try:
                    fire_summary = ensure_fire_cache(
                        firms_key, latitude, longitude, fire_start, fire_end, refresh_token
                    )
                except (APIUnavailableError, InvalidAPIKeyError) as exc:
                    data_errors.append(f"Satellite fire factors unavailable: {exc}")
            fire_data_complete = len(
                fire_cached_dates(cache_location, fire_start, fire_end)
            ) == (fire_end - fire_start).days + 1
            if fire_data_complete:
                try:
                    fire_observations = load_fire_observations(
                        cache_location, fire_start, fire_end,
                        forecast_context["timezone"], latitude, longitude,
                    )
                except Exception as exc:
                    data_errors.append(f"Stored fire observations could not be prepared: {exc}")
                    fire_data_complete = False

    if cached_daily.empty:
        cached_daily = daily_history_from_cache(
            cache_location, CAMS_COVERAGE_PROBE_START, date.today()
        )
        if not cached_daily.empty:
            history = merge_historical_rows(history, cached_daily)
            save_history(history, DATA_FILE)

    if cached_daily.empty and not history.empty:
        cached_daily = history.dropna(
            subset=["CAMS_PM2.5", *CORE_WEATHER_FEATURES]
        )[["Date", "CAMS_PM2.5", *WEATHER_FEATURES, "Timezone"]].copy()

    local_today = forecast_context["today"] if forecast_context else None
    timezone_name = forecast_context["timezone"] if forecast_context else "Unknown"

    cpcb_today_missing = bool(
        local_today
        and history.loc[
            history["Date"] == local_today, "CPCB_PM2.5"
        ].dropna().empty
    )
    should_attempt_cpcb = bool(
        api_key
        and local_today
        and (
            fetch_clicked
            or (
                cpcb_today_missing
                and st.session_state.cpcb_attempt_date != local_today
            )
        )
    )
    if should_attempt_cpcb:
        st.session_state.cpcb_attempt_date = local_today
        try:
            cpcb = fetch_cpcb_pm25(api_key, city, refresh_token)
            cpcb_date = cpcb["observed_date"] or local_today
            history = upsert_observation(
                history,
                cpcb_date,
                "CPCB_PM2.5",
                cpcb["pm25"],
                timezone_name,
                cpcb["station_count"],
            )
            save_history(history, DATA_FILE)
            st.session_state.cpcb_status = {
                "state": "available",
                "message": (
                    f"CPCB observation saved for {cpcb_date}: {cpcb['pm25']:.1f} "
                    f"ug/m3 from {cpcb['station_count']} stations."
                ),
            }
        except (MissingAPIKeyError, InvalidAPIKeyError) as exc:
            st.session_state.cpcb_status = {
                "state": "key_error",
                "message": f"CPCB API key problem: {exc}",
            }
        except NoCityDataError as exc:
            st.session_state.cpcb_status = {
                "state": "no_data",
                "message": f"No CPCB city data: {exc}",
            }
        except APIUnavailableError:
            st.session_state.cpcb_status = {
                "state": "offline",
                "message": (
                    "The CPCB data.gov.in service is temporarily not responding. "
                    "CAMS history, backtesting, and forecasting remain operational. "
                    "Use Refresh Data later to retry the CPCB-only observation."
                ),
            }

    if fetch_clicked and save_manual and local_today:
        history = upsert_observation(
            history,
            local_today,
            "Manual_PM2.5",
            float(manual_pm25),
            timezone_name,
        )
        save_history(history, DATA_FILE)
        st.success(f"Manual observation saved separately for {local_today}.")

    for error in data_errors:
        st.warning(error)

    cpcb_status = st.session_state.cpcb_status
    if cpcb_status:
        if cpcb_status["state"] == "available":
            st.success(cpcb_status["message"])
        elif cpcb_status["state"] == "key_error":
            st.error(cpcb_status["message"])
        elif cpcb_status["state"] == "no_data":
            st.warning(cpcb_status["message"])
        else:
            st.info(cpcb_status["message"])

    st.subheader("Historical Coverage")
    window: pd.DataFrame | None = None
    cutoff: date | None = None
    try:
        window, cutoff = select_stored_window(history)
        complete_dates = sorted(set(cached_daily["Date"])) if not cached_daily.empty else []
        earliest_complete = min(complete_dates) if complete_dates else window.iloc[0]["Date"]
        latest_complete = max(complete_dates) if complete_dates else cutoff
        cap_start = five_year_training_start(latest_complete)
        expected_cap_dates = {
            cap_start + timedelta(days=offset)
            for offset in range((latest_complete - cap_start).days + 1)
        }
        missing_cap_dates = expected_cap_dates - set(complete_dates)
        status_a, status_b, status_c, status_d = st.columns(4)
        status_a.metric("Complete CAMS days", len(complete_dates))
        status_b.metric("Earliest complete date", earliest_complete.isoformat())
        status_c.metric("Latest complete date", latest_complete.isoformat())
        status_d.metric("Unavailable days", len(missing_cap_dates))
        st.caption(
            f"Requested training cap: {cap_start} through {latest_complete}. Source boundary "
            f"detected at {cams_source_start or earliest_complete}. API timezone: "
            f"{window.iloc[-1]['Timezone']}. The separate development validation uses the "
            f"latest 90 days. CPCB and manual observations remain separate."
        )
    except IncompleteHistoryError as exc:
        st.error(str(exc))
        if exc.missing_dates:
            st.write("Missing dates: " + ", ".join(day.isoformat() for day in exc.missing_dates))

    if cache_summary:
        fetched_text = (
            f"Fetched {len(cache_summary['fetched_ranges'])} missing range(s) and "
            f"wrote {cache_summary['rows_written']} hourly rows."
            if cache_summary["fetched_ranges"]
            else "Hourly SQLite cache was already complete; no archive download was needed."
        )
        st.caption(fetched_text)
    if fire_summary:
        st.caption(
            f"NASA FIRMS cache: {fire_summary['complete_days']}/"
            f"{fire_summary['requested_days']} days; {fire_summary['rows_written']} new "
            "satellite thermal detections stored."
        )
    elif not firms_key:
        st.caption(
            "Satellite fire factors are optional and currently disabled. Add FIRMS_MAP_KEY "
            "to Streamlit secrets or the sidebar to enable them."
        )

    seasonal_results: list[dict[str, Any]] = []
    candidate_evaluation: dict[str, Any] | None = None
    engineered_daily = (
        add_multifactor_features(cached_daily, fire_observations, fire_data_complete)
        if not cached_daily.empty else pd.DataFrame()
    )
    selected_features = BASELINE_FEATURES
    selected_group = "Baseline"
    if not cached_daily.empty:
        st.divider()
        st.header("Rolling Yearly Seasonal Backtests")
        st.write(
            "Each eligible 60-day holdout uses an independently fitted model trained on "
            "all earlier complete history available at that origin, capped at five years."
        )
        try:
            with st.spinner("Comparing leakage-safe feature groups across seasonal holdouts..."):
                candidate_evaluation = evaluate_feature_candidates(
                    engineered_daily, cache_location, fire_data_complete
                )
                selected_group = candidate_evaluation["selected_group"]
                selected_features = candidate_evaluation["selected_features"]
                seasonal_results = candidate_evaluation["by_group"][selected_group]
                seasonal_cache_hits = candidate_evaluation["cache_hits"]
            st.subheader("Feature Promotion Gate")
            gate_display = candidate_evaluation["decisions"].copy()
            for column in ("Median MAE", "Median improvement (%)", "Worst recall drop (points)"):
                gate_display[column] = gate_display[column].round(2)
            st.dataframe(gate_display, hide_index=True, use_container_width=True)
            season_comparison = []
            baseline_windows = candidate_evaluation["by_group"]["Baseline"]
            for baseline_result in baseline_windows:
                row = {
                    "Year": baseline_result["year_label"],
                    "Season": baseline_result["season"],
                }
                for group_name, group_results in candidate_evaluation["by_group"].items():
                    matched = next(
                        item for item in group_results
                        if item["window_id"] == baseline_result["window_id"]
                    )
                    row[f"{group_name} MAE"] = round(matched["model_mae"], 2)
                season_comparison.append(row)
            st.dataframe(
                pd.DataFrame(season_comparison), hide_index=True, use_container_width=True
            )
            if selected_group == "Baseline":
                st.info(
                    "No enhanced feature group passed every promotion rule, so the baseline "
                    "feature schema remains in production."
                )
            else:
                st.success(f"Promoted production feature group: {selected_group}.")
            st.caption(
                "Promotion requires at least 2% lower overall median MAE, improvement or a "
                "tie in at least 75% of individual windows, years, and seasons, and no "
                "exceedance-recall loss greater than five percentage points."
            )
            year_options = sorted({result["year_label"] for result in seasonal_results})
            season_options = sorted({result["season"] for result in seasonal_results})
            filter_a, filter_b = st.columns(2)
            selected_years = filter_a.multiselect(
                "Backtest years", year_options, default=year_options
            )
            selected_seasons = filter_b.multiselect(
                "Backtest seasons", season_options, default=season_options
            )
            visible_results = [
                result for result in seasonal_results
                if result["year_label"] in selected_years
                and result["season"] in selected_seasons
            ]
            summary_rows = []
            for result in visible_results:
                classification = result["classification"]
                summary_rows.append(
                    {
                        "Year": result["year_label"],
                        "Season": result["season"],
                        "Holdout": (
                            f"{result['holdout_start']} to {result['holdout_end']}"
                        ),
                        "Training dates": result["calendar_training_dates"],
                        "Usable rows": result["valid_training_rows"],
                        "Model MAE": round(result["model_mae"], 2),
                        "Persistence MAE": round(result["baseline_mae"], 2),
                        "RMSE": round(result["model_rmse"], 2),
                        "R2": round(result["model_r2"], 3),
                        "MAE improvement": (
                            "N/A"
                            if result["improvement_pct"] is None
                            else f"{result['improvement_pct']:.1f}%"
                        ),
                        "Recall": format_rate(classification["Recall"]),
                        "Precision": format_rate(classification["Precision"]),
                        "FPR": format_rate(classification["False-positive rate"]),
                    }
                )
            st.dataframe(pd.DataFrame(summary_rows), hide_index=True, use_container_width=True)
            st.caption(
                f"Loaded {seasonal_cache_hits}/{candidate_evaluation['artifact_count']} candidate models "
                "from disk cache. Historical "
                "actual weather is used as a forecast proxy, so these results are optimistic."
            )

            aggregate_rows = []
            aggregate_fields = {
                "Model MAE": "model_mae",
                "Persistence MAE": "baseline_mae",
                "RMSE": "model_rmse",
                "R2": "model_r2",
                "MAE improvement (%)": "improvement_pct",
            }
            for label, key in aggregate_fields.items():
                values = pd.Series(
                    [result[key] for result in visible_results], dtype="float64"
                ).dropna()
                aggregate_rows.append(
                    {
                        "Metric": label,
                        "Mean": round(float(values.mean()), 3) if not values.empty else "N/A",
                        "Median": round(float(values.median()), 3) if not values.empty else "N/A",
                        "Sample std": (
                            round(float(values.std(ddof=1)), 3)
                            if len(values) > 1
                            else "N/A"
                        ),
                    }
                )
            st.subheader("Selected Backtest Summary")
            st.dataframe(pd.DataFrame(aggregate_rows), hide_index=True, use_container_width=True)
            st.caption(
                f"The current selection contains {len(visible_results)} holdouts from "
                f"{candidate_evaluation['year_count']} eligible yearly cycles. Expanding "
                "training histories overlap, so dispersion is descriptive rather than an "
                "independent uncertainty estimate. Historical actual weather makes these "
                "development results optimistic."
            )

            st.subheader("Forecast Factors")
            factor_status = pd.DataFrame(
                [
                    {"Factor group": "Pollution history", "Available": True, "Used in production": "PM2.5_Lag2" in selected_features},
                    {"Factor group": "Expanded weather", "Available": True, "Used in production": "Pressure_MSL" in selected_features},
                    {"Factor group": "Calendar and festivals", "Available": True, "Used in production": "Is_Indian_Holiday" in selected_features},
                    {"Factor group": "Satellite fires", "Available": fire_data_complete, "Used in production": "Fire_Count_1d" in selected_features},
                ]
            )
            st.dataframe(factor_status, hide_index=True, use_container_width=True)
            latest = engineered_daily.sort_values("Date").iloc[-1]
            latest_factors = {
                "Latest PM2.5": latest.get("CAMS_PM2.5"),
                "7-day PM2.5 mean": latest.get("PM2.5_Mean7"),
                "Precipitation": latest.get("Precipitation"),
                "Boundary layer height": latest.get("Boundary_Layer_Height"),
                "3-day fire detections": latest.get("Fire_Count_3d") if fire_data_complete else "Unavailable",
            }
            st.dataframe(
                pd.DataFrame([latest_factors]), hide_index=True, use_container_width=True
            )
            importance, importance_cache_hit = cached_grouped_permutation_importance(
                seasonal_results
            )
            if not importance.empty:
                st.dataframe(importance.round(3), hide_index=True, use_container_width=True)
            st.caption(
                "Permutation importance reports the average holdout MAE increase after a factor "
                "family is disrupted. It describes predictive usefulness, not causation. "
                f"Importance artifact reused: {'yes' if importance_cache_hit else 'no'}. "
                "Holiday and weekend signals are activity proxies, not measured traffic. "
                "FIRMS detections are satellite-observed thermal activity, not confirmed crop burning."
            )

            for result in visible_results:
                with st.expander(
                    f"{result['year_label']} {result['season']}: actual vs model vs persistence"
                ):
                    chart = backtest_chart(result["results"])
                    st.pyplot(chart, use_container_width=True)
                    plt.close(chart)
                    st.caption(
                        f"Boundary verified: first holdout lag {result['boundary_lag']:.2f} "
                        f"equals the preceding true PM2.5 {result['boundary_actual']:.2f}. "
                        f"Fitted rows: {result['fit_rows']}; normal rows: "
                        f"{result['normal_rows']}."
                    )
        except (IncompleteHistoryError, ValueError, AssertionError) as exc:
            seasonal_results = []
            st.error(f"Season-balanced backtests withheld: {exc}")

    if window is not None:
        try:
            backtest = run_backtest(window)
            recursive = run_recursive_validation(backtest)
        except (ValueError, AssertionError) as exc:
            st.error(f"Backtest withheld: {exc}")
            backtest = None
            recursive = None

        if backtest is not None and recursive is not None:
            st.divider()
            st.header("Development Validation")
            st.write(
                f"Frozen model trained on {backtest['train_start']} to {backtest['train_end']} "
                f"and tested on untouched dates {backtest['test_start']} to {backtest['test_end']}."
            )
            metric_columns = st.columns(5)
            metric_columns[0].metric("Model MAE", f"{backtest['model_mae']:.1f} ug/m3")
            metric_columns[1].metric("Persistence MAE", f"{backtest['baseline_mae']:.1f} ug/m3")
            metric_columns[2].metric("Model RMSE", f"{backtest['model_rmse']:.1f} ug/m3")
            metric_columns[3].metric("Persistence RMSE", f"{backtest['baseline_rmse']:.1f} ug/m3")
            metric_columns[4].metric("Model R2", f"{backtest['model_r2']:.2f}")
            st.caption(
                f"Verified boundary: day 61 lag {backtest['day61_lag']:.2f} equals "
                f"day 60 true PM2.5 {backtest['day60_pm25']:.2f}. "
                f"Normal training rows: {backtest['normal_rows']}; fitted rows: {backtest['fit_rows']}."
            )
            if backtest["used_fallback"]:
                st.info("Isolation Forest left fewer than 10 normal rows, so all training rows were used.")

            classification = backtest["classification"]
            class_table = pd.DataFrame(
                [{
                    "Accuracy": format_rate(classification["Accuracy"]),
                    "Exceedance recall": format_rate(classification["Recall"]),
                    "Precision": format_rate(classification["Precision"]),
                    "False-positive rate": format_rate(classification["False-positive rate"]),
                }]
            )
            st.subheader("60 ug/m3 Exceedance Classification")
            st.dataframe(class_table, hide_index=True, use_container_width=True)

            chart = backtest_chart(backtest["results"])
            st.pyplot(chart, use_container_width=True)
            plt.close(chart)
            st.caption(
                "These 30 dates are sequential and autocorrelated, not 30 independent trials. "
                "The backtest uses historical actual weather instead of archived weather "
                "forecasts, so real forward accuracy is likely lower."
            )

            st.subheader("Largest Backtest Errors")
            largest_errors = (
                backtest["results"]
                .nlargest(5, "Absolute_Error")[
                    ["Date", "Actual", "Model", "Residual", "Absolute_Error"]
                ]
                .copy()
            )
            st.dataframe(largest_errors, hide_index=True, use_container_width=True)

            st.divider()
            st.header("Recursive Horizon Validation")
            st.write(
                "Twenty-four rolling seven-day windows use the frozen 60-day model. "
                "Only the value before each window is known; later lags are predictions."
            )
            recursive_plot = recursive_chart(recursive)
            st.pyplot(recursive_plot, use_container_width=True)
            plt.close(recursive_plot)
            st.dataframe(
                recursive[["Horizon", "MAE", "Windows"]],
                hide_index=True,
                use_container_width=True,
            )
            st.caption("The rolling windows overlap, so their errors are correlated.")

            st.divider()
            st.header("Forward-Looking 7-Day Forecast")
            forward_forecast: pd.DataFrame | None = None
            if forecast_context and local_today:
                try:
                    forecast, production = create_multifactor_production_forecast(
                        engineered_daily,
                        forecast_context["daily"],
                        local_today,
                        recursive,
                        selected_features,
                        fire_observations,
                        fire_data_complete,
                    )
                    forward_forecast = forecast
                    forecast_plot = forecast_chart(forecast)
                    st.pyplot(forecast_plot, use_container_width=True)
                    plt.close(forecast_plot)
                    forecast_display = forecast[
                        ["Date", "Day_Ahead", "Prediction", "Lower_80", "Upper_80", "Exceeds_60"]
                    ].copy()
                    forecast_display.columns = [
                        "Date",
                        "Day ahead",
                        "Predicted PM2.5",
                        "Lower 80%",
                        "Upper 80%",
                        "Exceeds 60",
                    ]
                    st.dataframe(forecast_display, hide_index=True, use_container_width=True)
                    exceedance_days = forecast.loc[forecast["Exceeds_60"], "Date"].tolist()
                    if exceedance_days:
                        st.error(
                            "Hazardous: predicted to exceed 60 ug/m3 on "
                            + ", ".join(day.isoformat() for day in exceedance_days)
                        )
                    else:
                        st.success("All seven modeled daily forecasts are at or below 60 ug/m3.")
                    if production["used_fallback"]:
                        st.info("The production fit used the Isolation Forest fallback.")
                    production_rows = production["model_rows"].sort_values("Date")
                    st.caption(
                        f"Production group: {selected_group}; trained on "
                        f"{production['training_days']} contiguous complete days from "
                        f"{production_rows.iloc[0]['Date']} through "
                        f"{production_rows.iloc[-1]['Date']}, under a five-calendar-year cap. "
                        "This is a CAMS-trained "
                        "modeled baseline, not a CPCB-observation forecast. "
                        "Tree-based 80% bands are widened using recursive error growth, but they "
                        "remain heuristic and may understate real uncertainty."
                    )
                except ValueError as exc:
                    st.warning(f"Forward forecast unavailable: {exc}")
            else:
                st.warning("Forward weather could not be fetched, so no production forecast is shown.")

            st.divider()
            st.header("Prospective Live Evaluation")
            if not cached_daily.empty:
                record_prospective_actuals(cache_location, cached_daily)
            if forward_forecast is not None and cutoff is not None:
                production_training, _ = select_production_training(
                    engineered_daily, selected_features
                )
                production_hash = data_fingerprint(
                    production_training[["Date", "CAMS_PM2.5", *selected_features]],
                    {
                        "purpose": "official_next_day_forecast",
                        "model_cache_version": MODEL_CACHE_VERSION,
                        "features": selected_features,
                        "feature_group": selected_group,
                        "estimators": 400,
                    },
                )
                log_prospective_forecast(
                    cache_location,
                    forward_forecast.iloc[0],
                    timezone_name,
                    production_hash,
                    cutoff,
                )
            prospective = prospective_evaluation(cache_location)
            if prospective.empty:
                st.info(
                    "No official next-day forecasts have been logged yet. The first "
                    "eligible forecast will be stored before its actual CAMS value exists."
                )
            else:
                verified = prospective.dropna(subset=["Actual"]).copy()
                st.metric("Verified prospective days", f"{len(verified)}/30")
                if len(verified) >= 30:
                    actual = verified["Actual"].to_numpy(dtype=float)
                    predicted = verified["Prediction"].to_numpy(dtype=float)
                    live_metrics = threshold_metrics(actual, predicted)
                    live_columns = st.columns(4)
                    live_columns[0].metric(
                        "Prospective MAE",
                        f"{mean_absolute_error(actual, predicted):.1f} ug/m3",
                    )
                    live_columns[1].metric(
                        "Prospective RMSE",
                        f"{np.sqrt(mean_squared_error(actual, predicted)):.1f} ug/m3",
                    )
                    live_columns[2].metric(
                        "Exceedance recall", format_rate(live_metrics["Recall"])
                    )
                    live_columns[3].metric(
                        "False-positive rate",
                        format_rate(live_metrics["False-positive rate"]),
                    )
                else:
                    st.info(
                        "Final unbiased accuracy is withheld until 30 immutable "
                        "forecast/actual pairs have accumulated."
                    )
                st.dataframe(
                    prospective[
                        [
                            "Date",
                            "Prediction",
                            "Lower_80",
                            "Upper_80",
                            "Actual",
                            "Issued_At_UTC",
                        ]
                    ],
                    hide_index=True,
                    use_container_width=True,
                )
                st.caption(
                    "Forecasts and completed CAMS actuals are insert-only SQLite "
                    "records and cannot be silently regenerated or overwritten."
                )

    if not history.empty:
        with st.expander("CPCB and manual observations"):
            observations = history.loc[
                history[["CPCB_PM2.5", "Manual_PM2.5"]].notna().any(axis=1),
                ["Date", "CPCB_PM2.5", "Manual_PM2.5", "CPCB_Station_Count", "Timezone"],
            ]
            if observations.empty:
                st.write("No separate CPCB or manual observations have been stored yet.")
            else:
                st.dataframe(observations, hide_index=True, use_container_width=True)

        with st.expander("View and download normalized history"):
            display_history = history.copy()
            display_history["Date"] = display_history["Date"].astype(str)
            st.dataframe(display_history, hide_index=True, use_container_width=True)
            st.download_button(
                "Download CSV",
                data=display_history.to_csv(index=False).encode("utf-8"),
                file_name="cpcb_data.csv",
                mime="text/csv",
            )

    st.caption(
        "CAMS PM2.5 is modeled atmospheric data and historical weather is reanalysis. "
        "Neither is equivalent to CPCB station observations. This educational forecast "
        "is not an official health alert."
    )


if __name__ == "__main__":
    main()
