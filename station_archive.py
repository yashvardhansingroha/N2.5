"""Audited, source-labelled R.K. Puram hourly station observations."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime, timedelta
from io import BytesIO
from pathlib import Path
import sqlite3
from xml.etree import ElementTree

import numpy as np
import pandas as pd
import requests


DB_FILE = Path("observed_data_cache.sqlite")
XKDR_URL = "https://airquality.xkdr.org/v1/measurements"
XKDR_DEMO_KEY = "aqi_demo_wbf92Qx21zX-Wa_Tg8Dx1nXe"
OPENAQ_URL = "https://openaq-data-archive.s3.amazonaws.com"
OPENAQ_PREFIX = "records/csv.gz/locationid=17"
STATION_NAME = "R K Puram, Delhi - DPCC"
XKDR_STATION = "site_124"
XKDR_END = date(2025, 9, 1)
OPENAQ_START = date(2025, 2, 19)
POLLUTANTS = {"pm25": "Observed_PM2.5", "pm10": "Observed_PM10"}
WEATHER_FIELDS = {
    "temperature_2m": "Temperature", "relative_humidity_2m": "Humidity",
    "wind_speed_10m": "Wind_Speed", "precipitation": "Precipitation",
    "pressure_msl": "Pressure_MSL", "cloud_cover": "Cloud_Cover",
    "wind_direction_10m": "Wind_Direction", "wind_gusts_10m": "Wind_Gusts",
    "boundary_layer_height": "Boundary_Layer_Height",
}
STATION_LATITUDE = 28.563262
STATION_LONGITUDE = 77.186937


def connect(path: Path = DB_FILE) -> sqlite3.Connection:
    connection = sqlite3.connect(path)
    connection.execute("""CREATE TABLE IF NOT EXISTS hours (
        source TEXT NOT NULL, pollutant TEXT NOT NULL, local_hour TEXT NOT NULL,
        value REAL NOT NULL, PRIMARY KEY(source, pollutant, local_hour))""")
    connection.execute("""CREATE TABLE IF NOT EXISTS fetched_days (
        source TEXT NOT NULL, local_date TEXT NOT NULL,
        PRIMARY KEY(source, local_date))""")
    connection.execute("""CREATE TABLE IF NOT EXISTS weather_daily (
        local_date TEXT PRIMARY KEY, Temperature REAL, Humidity REAL,
        Wind_Speed REAL, Precipitation REAL, Pressure_MSL REAL,
        Cloud_Cover REAL, Wind_Direction REAL, Wind_Gusts REAL,
        Boundary_Layer_Height REAL, Timezone TEXT NOT NULL)""")
    return connection


def _days(start: date, end: date) -> list[date]:
    return list(pd.date_range(start, end, freq="D").date)


def _mark_days(connection: sqlite3.Connection, source: str, days: list[date]) -> None:
    connection.executemany(
        "INSERT OR IGNORE INTO fetched_days VALUES (?, ?)",
        [(source, day.isoformat()) for day in days],
    )


def _store(connection: sqlite3.Connection, source: str, frame: pd.DataFrame) -> None:
    if frame.empty:
        return
    records = frame[["pollutant", "local_hour", "value"]].drop_duplicates(
        ["pollutant", "local_hour"], keep="last"
    )
    connection.executemany(
        "INSERT OR IGNORE INTO hours VALUES (?, ?, ?, ?)",
        [(source, row.pollutant, row.local_hour, float(row.value))
         for row in records.itertuples(index=False)],
    )


def parse_xkdr(content: bytes) -> pd.DataFrame:
    frame = pd.read_csv(BytesIO(content), encoding="latin-1")
    required = {"station_id", "parameter_name", "period_start", "mean"}
    if not required.issubset(frame):
        raise ValueError("XKDR response schema changed.")
    frame = frame.loc[frame["station_id"] == XKDR_STATION].copy()
    frame["pollutant"] = frame["parameter_name"].str.lower().str.replace(".", "", regex=False)
    frame["local_hour"] = pd.to_datetime(frame["period_start"], errors="coerce").dt.strftime("%Y-%m-%dT%H:00:00")
    frame["value"] = pd.to_numeric(frame["mean"], errors="coerce")
    return frame.loc[
        frame["pollutant"].isin(POLLUTANTS)
        & frame["local_hour"].notna()
        & np.isfinite(frame["value"])
        & (frame["value"] >= 0),
        ["pollutant", "local_hour", "value"],
    ]


def parse_openaq(content: bytes) -> pd.DataFrame:
    frame = pd.read_csv(BytesIO(content), compression="gzip", encoding="latin-1")
    required = {"location_id", "datetime", "parameter", "value"}
    if not required.issubset(frame):
        raise ValueError("OpenAQ response schema changed.")
    frame = frame.loc[frame["location_id"] == 17].copy()
    frame["pollutant"] = frame["parameter"].astype(str).str.lower().str.replace(".", "", regex=False)
    timestamps = pd.to_datetime(frame["datetime"], errors="coerce", utc=True).dt.tz_convert("Asia/Kolkata")
    frame["local_hour"] = timestamps.dt.strftime("%Y-%m-%dT%H:00:00")
    frame["value"] = pd.to_numeric(frame["value"], errors="coerce")
    frame = frame.loc[
        frame["pollutant"].isin(POLLUTANTS)
        & frame["local_hour"].notna()
        & np.isfinite(frame["value"])
        & (frame["value"] >= 0),
        ["pollutant", "local_hour", "value"],
    ]
    return frame.groupby(["pollutant", "local_hour"], as_index=False)["value"].mean()


def _xkdr_request(start: date, end: date) -> pd.DataFrame:
    response = requests.get(
        XKDR_URL,
        headers={"Authorization": f"Bearer {XKDR_DEMO_KEY}"},
        params={
            "station": XKDR_STATION, "parameter": ["PM2.5", "PM10"],
            "start": start.isoformat(), "end": end.isoformat(),
            "agg": "hourly", "format": "csv",
        },
        timeout=60,
    )
    response.raise_for_status()
    return parse_xkdr(response.content)


def fetch_xkdr(start: date, end: date, path: Path = DB_FILE) -> int:
    end = min(end, XKDR_END)
    if start > end:
        return 0
    written = 0
    with connect(path) as connection:
        fetched = {date.fromisoformat(row[0]) for row in connection.execute(
            "SELECT local_date FROM fetched_days WHERE source='XKDR'"
        )}
        missing = [day for day in _days(start, end) if day not in fetched]
        for index in range(0, len(missing), 75):
            chunk = missing[index:index + 75]
            if not chunk:
                continue
            # Query one continuous span, but mark only requested missing dates.
            result = _xkdr_request(chunk[0], chunk[-1])
            _store(connection, "XKDR", result)
            _mark_days(connection, "XKDR", chunk)
            written += len(result)
    return written


def list_openaq_days(start: date, end: date) -> list[date]:
    found: set[date] = set()
    for year in range(start.year, end.year + 1):
        token = None
        while True:
            params = {"list-type": "2", "prefix": f"{OPENAQ_PREFIX}/year={year}/", "max-keys": 1000}
            if token:
                params["continuation-token"] = token
            response = requests.get(OPENAQ_URL, params=params, timeout=60)
            response.raise_for_status()
            root = ElementTree.fromstring(response.content)
            namespace = {"s": "http://s3.amazonaws.com/doc/2006-03-01/"}
            for item in root.findall("s:Contents/s:Key", namespace):
                stem = item.text.rsplit("/", 1)[-1] if item.text else ""
                try:
                    day = datetime.strptime(stem, "location-17-%Y%m%d.csv.gz").date()
                except ValueError:
                    continue
                if start <= day <= end:
                    found.add(day)
            token = root.findtext("s:NextContinuationToken", namespaces=namespace)
            if not token:
                break
    return sorted(found)


def _openaq_request(day: date) -> pd.DataFrame:
    url = (
        f"{OPENAQ_URL}/{OPENAQ_PREFIX}/year={day.year}/month={day.month:02d}/"
        f"location-17-{day:%Y%m%d}.csv.gz"
    )
    response = requests.get(url, timeout=60)
    response.raise_for_status()
    return parse_openaq(response.content)


def fetch_openaq(start: date, end: date, path: Path = DB_FILE) -> int:
    available = list_openaq_days(max(start, OPENAQ_START), end)
    with connect(path) as connection:
        fetched = {date.fromisoformat(row[0]) for row in connection.execute(
            "SELECT local_date FROM fetched_days WHERE source='OpenAQ'"
        )}
    missing = [day for day in available if day not in fetched]
    written = 0
    with ThreadPoolExecutor(max_workers=12) as executor:
        futures = {executor.submit(_openaq_request, day): day for day in missing}
        for future in as_completed(futures):
            day = futures[future]
            frame = future.result()
            with connect(path) as connection:
                _store(connection, "OpenAQ", frame)
                _mark_days(connection, "OpenAQ", [day])
            written += len(frame)
    return written


def audit_overlap(path: Path = DB_FILE) -> dict:
    with connect(path) as connection:
        overlap = pd.read_sql_query("""SELECT x.pollutant, x.local_hour, x.value AS xkdr,
                   o.value AS openaq FROM hours x JOIN hours o
                   ON x.pollutant=o.pollutant AND x.local_hour=o.local_hour
                   WHERE x.source='XKDR' AND o.source='OpenAQ'""", connection)
        source_hours = pd.read_sql_query(
            """SELECT source, pollutant, local_hour FROM hours
            WHERE local_hour >= ? AND local_hour < ?""",
            connection,
            params=(OPENAQ_START.isoformat(), (XKDR_END + timedelta(days=1)).isoformat()),
        )
    outcome = {
        "station": STATION_NAME, "matched_hours": len(overlap),
        "overlap_start": OPENAQ_START.isoformat(), "overlap_end": XKDR_END.isoformat(),
        "expected_units": "micrograms per cubic metre (ug/m3)",
        "unit_note": (
            "XKDR and OpenAQ both identify particulate concentration in ug/m3. "
            "The OpenAQ archive's raw unit text contains replacement characters, "
            "so strict byte-for-byte unit comparison is unavailable."
        ),
        "pollutants": {},
    }
    for pollutant in POLLUTANTS:
        subset = overlap.loc[overlap["pollutant"] == pollutant]
        x_hours = set(source_hours.loc[
            (source_hours["source"] == "XKDR") & (source_hours["pollutant"] == pollutant),
            "local_hour",
        ])
        o_hours = set(source_hours.loc[
            (source_hours["source"] == "OpenAQ") & (source_hours["pollutant"] == pollutant),
            "local_hour",
        ])
        errors = (subset["xkdr"] - subset["openaq"]).abs()
        passed = len(subset) >= 24 and float((errors <= 1.0).mean()) >= 0.95 and abs(float((subset["xkdr"] - subset["openaq"]).median())) <= 1.0
        outcome["pollutants"][pollutant] = {
            "matched_hours": len(subset),
            "xkdr_only_hours": len(x_hours - o_hours),
            "openaq_only_hours": len(o_hours - x_hours),
            "within_1_ug_m3": float((errors <= 1.0).mean()) if len(subset) else None,
            "median_signed_difference": float((subset["xkdr"] - subset["openaq"]).median()) if len(subset) else None,
            "passed": passed,
        }
    outcome["passed"] = all(row["passed"] for row in outcome["pollutants"].values())
    return outcome


def daily_station(path: Path = DB_FILE) -> pd.DataFrame:
    audit = audit_overlap(path)
    if not audit["passed"]:
        raise ValueError("XKDR/OpenAQ station overlap failed the source audit.")
    with connect(path) as connection:
        raw = pd.read_sql_query("SELECT * FROM hours", connection)
    raw["Date"] = pd.to_datetime(raw["local_hour"]).dt.date
    raw["Hour"] = pd.to_datetime(raw["local_hour"]).dt.hour
    raw = raw.loc[
        ((raw["source"] == "XKDR") & (raw["Date"] <= XKDR_END))
        | ((raw["source"] == "OpenAQ") & (raw["Date"] > XKDR_END))
    ]
    grouped = raw.groupby(["Date", "pollutant"]).agg(
        value=("value", "mean"), hours=("Hour", "nunique")
    ).reset_index()
    grouped.loc[grouped["hours"] < 18, "value"] = np.nan
    daily = grouped.pivot(index="Date", columns="pollutant", values="value").reset_index()
    for source, target in POLLUTANTS.items():
        daily[target] = daily[source] if source in daily else np.nan
    return daily[["Date", *POLLUTANTS.values()]].sort_values("Date")


def _weather_request(start: date, end: date) -> pd.DataFrame:
    response = requests.get(
        "https://archive-api.open-meteo.com/v1/archive",
        params={
            "latitude": STATION_LATITUDE, "longitude": STATION_LONGITUDE,
            "start_date": start.isoformat(), "end_date": end.isoformat(),
            "timezone": "auto", "hourly": ",".join(WEATHER_FIELDS),
        },
        timeout=90,
    )
    response.raise_for_status()
    payload = response.json()
    hourly = payload.get("hourly", {})
    if not hourly.get("time") or payload.get("timezone") != "Asia/Kolkata":
        raise ValueError("Station weather response lacks local hourly data or timezone.")
    frame = pd.DataFrame({"Date": pd.to_datetime(hourly["time"]).date})
    for api_field, column in WEATHER_FIELDS.items():
        values = hourly.get(api_field)
        if values is None or len(values) != len(frame):
            raise ValueError(f"Station weather response lacks {api_field}.")
        frame[column] = pd.to_numeric(values, errors="coerce")
    aggregates = {column: "mean" for column in WEATHER_FIELDS.values()}
    aggregates["Precipitation"] = "sum"
    aggregates["Wind_Gusts"] = "max"
    daily = frame.groupby("Date", as_index=False).agg(aggregates)
    # Core weather must be present for a day to enter the station model.
    counts = frame.groupby("Date")[["Temperature", "Humidity", "Wind_Speed"]].count()
    valid = counts.index[(counts == 24).all(axis=1)]
    daily = daily.loc[daily["Date"].isin(valid)].copy()
    direction = frame.groupby("Date")["Wind_Direction"].apply(
        lambda values: np.rad2deg(np.arctan2(
            np.nanmean(np.sin(np.deg2rad(values))),
            np.nanmean(np.cos(np.deg2rad(values))),
        )) % 360
    )
    daily["Wind_Direction"] = daily["Date"].map(direction)
    daily["Timezone"] = payload["timezone"]
    return daily


def fetch_station_weather(start: date, end: date, path: Path = DB_FILE) -> pd.DataFrame:
    with connect(path) as connection:
        stored = pd.read_sql_query("SELECT * FROM weather_daily", connection)
    existing = set(pd.to_datetime(stored["local_date"]).dt.date) if not stored.empty else set()
    missing = [day for day in _days(start, end) if day not in existing]
    for index in range(0, len(missing), 90):
        chunk = missing[index:index + 90]
        if not chunk:
            continue
        frame = _weather_request(chunk[0], chunk[-1])
        with connect(path) as connection:
            connection.executemany(
                """INSERT OR IGNORE INTO weather_daily VALUES
                (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                [(
                    row.Date.isoformat(), *[
                        None if pd.isna(getattr(row, column)) else float(getattr(row, column))
                        for column in WEATHER_FIELDS.values()
                    ], row.Timezone,
                ) for row in frame.itertuples(index=False)],
            )
    with connect(path) as connection:
        daily = pd.read_sql_query(
            "SELECT * FROM weather_daily WHERE local_date BETWEEN ? AND ? ORDER BY local_date",
            connection, params=(start.isoformat(), end.isoformat()),
        )
    daily["Date"] = pd.to_datetime(daily.pop("local_date")).dt.date
    return daily


def prepare_station(start: date, end: date, path: Path = DB_FILE) -> tuple[pd.DataFrame, dict]:
    fetch_xkdr(start, min(end, XKDR_END), path)
    fetch_openaq(OPENAQ_START, end, path)
    audit = audit_overlap(path)
    if not audit["passed"]:
        return pd.DataFrame(), audit
    daily = daily_station(path)
    return daily.loc[daily["Date"].between(start, end)], audit


def cached_station_dataset(path: Path = DB_FILE) -> tuple[pd.DataFrame, dict]:
    audit = audit_overlap(path)
    if not audit["passed"]:
        return pd.DataFrame(), audit
    station = daily_station(path)
    common = station.dropna(subset=list(POLLUTANTS.values()))
    if common.empty:
        return pd.DataFrame(), audit
    station = station.loc[station["Date"] <= max(common["Date"])]
    with connect(path) as connection:
        weather = pd.read_sql_query("SELECT * FROM weather_daily", connection)
    if weather.empty:
        return pd.DataFrame(), audit
    weather["Date"] = pd.to_datetime(weather.pop("local_date")).dt.date
    return station.merge(weather, on="Date", how="left"), audit
