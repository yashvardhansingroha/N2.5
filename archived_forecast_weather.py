"""Cache Open-Meteo's stitched historical-forecast weather for holdout tests."""

from __future__ import annotations

from datetime import date
from pathlib import Path
import sqlite3

import numpy as np
import pandas as pd
import requests

from station_archive import WEATHER_FIELDS


DB_FILE = Path("historical_forecast_cache.sqlite")
URL = "https://historical-forecast-api.open-meteo.com/v1/forecast"


def _key(latitude: float, longitude: float) -> str:
    return f"{latitude:.5f},{longitude:.5f}"


def _connect(path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(path)
    connection.execute("""CREATE TABLE IF NOT EXISTS daily_weather (
        location TEXT NOT NULL, local_date TEXT NOT NULL,
        Temperature REAL, Humidity REAL, Wind_Speed REAL, Precipitation REAL,
        Pressure_MSL REAL, Cloud_Cover REAL, Wind_Direction REAL,
        Wind_Gusts REAL, Boundary_Layer_Height REAL, Timezone TEXT NOT NULL,
        PRIMARY KEY(location, local_date))""")
    return connection


def parse_payload(payload: dict) -> pd.DataFrame:
    hourly = payload.get("hourly", {})
    if payload.get("timezone") != "Asia/Kolkata" or not hourly.get("time"):
        raise ValueError("Archived forecast weather lacks Asia/Kolkata hourly timestamps.")
    frame = pd.DataFrame({"Date": pd.to_datetime(hourly["time"]).date})
    for api_field, column in WEATHER_FIELDS.items():
        values = hourly.get(api_field)
        if values is None or len(values) != len(frame):
            raise ValueError(f"Archived forecast weather lacks {api_field}.")
        frame[column] = pd.to_numeric(values, errors="coerce")
    counts = frame.groupby("Date")[list(WEATHER_FIELDS.values())].count()
    valid = counts.index[(counts == 24).all(axis=1)]
    if not len(valid):
        return pd.DataFrame(columns=["Date", *WEATHER_FIELDS.values(), "Timezone"])
    aggregates = {column: "mean" for column in WEATHER_FIELDS.values()}
    aggregates["Precipitation"] = "sum"
    aggregates["Wind_Gusts"] = "max"
    daily = frame.groupby("Date", as_index=False).agg(aggregates)
    daily = daily.loc[daily["Date"].isin(valid)].copy()
    direction = frame.groupby("Date")["Wind_Direction"].apply(
        lambda values: float(np.rad2deg(np.arctan2(
            np.mean(np.sin(np.deg2rad(values))),
            np.mean(np.cos(np.deg2rad(values))),
        )) % 360)
    )
    daily["Wind_Direction"] = daily["Date"].map(direction)
    daily["Timezone"] = payload["timezone"]
    return daily


def fetch_archived_daily(
    latitude: float, longitude: float, start: date, end: date,
    path: Path = DB_FILE,
) -> pd.DataFrame:
    location = _key(latitude, longitude)
    with _connect(path) as connection:
        existing = {
            date.fromisoformat(row[0]) for row in connection.execute(
                "SELECT local_date FROM daily_weather WHERE location=? AND local_date BETWEEN ? AND ?",
                (location, start.isoformat(), end.isoformat()),
            )
        }
    missing = [day for day in pd.date_range(start, end, freq="D").date if day not in existing]
    for index in range(0, len(missing), 90):
        chunk = missing[index:index + 90]
        if not chunk:
            continue
        response = requests.get(
            URL,
            params={
                "latitude": latitude, "longitude": longitude,
                "start_date": chunk[0].isoformat(), "end_date": chunk[-1].isoformat(),
                "timezone": "auto", "hourly": ",".join(WEATHER_FIELDS),
            },
            timeout=90,
        )
        response.raise_for_status()
        daily = parse_payload(response.json())
        with _connect(path) as connection:
            connection.executemany(
                """INSERT OR IGNORE INTO daily_weather VALUES
                (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                [(
                    location, row.Date.isoformat(),
                    *[float(getattr(row, column)) for column in WEATHER_FIELDS.values()],
                    row.Timezone,
                ) for row in daily.itertuples(index=False)],
            )
    with _connect(path) as connection:
        result = pd.read_sql_query(
            "SELECT * FROM daily_weather WHERE location=? AND local_date BETWEEN ? AND ? ORDER BY local_date",
            connection, params=(location, start.isoformat(), end.isoformat()),
        )
    result["Date"] = pd.to_datetime(result.pop("local_date")).dt.date
    result = result.drop(columns="location")
    required = set(pd.date_range(start, end, freq="D").date)
    if set(result["Date"]) != required:
        raise ValueError(f"Archived forecast weather covers {len(result)}/{len(required)} required days.")
    return result
