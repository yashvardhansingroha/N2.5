"""Presentation dashboard for the shared PM2.5 and PM10 analysis."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from pollutant_pipeline import Pollutant, analyze, forecast, summary_json


RESULTS_FILE = Path("results.json")


def save_result(analysis: dict[str, Any], extra: dict[str, Any] | None = None) -> None:
    existing: dict[str, Any] = {}
    if RESULTS_FILE.exists():
        try:
            existing = json.loads(RESULTS_FILE.read_text(encoding="utf-8"))
        except (ValueError, OSError):
            existing = {}
    identity = f"{analysis['pollutant'].source}:{analysis['pollutant'].key}"
    updated = {**summary_json(analysis), **(extra or {})}
    if existing.get("analyses", {}).get(identity) == updated:
        return
    existing.setdefault("analyses", {})[identity] = updated
    existing["generated_at_utc"] = datetime.now(timezone.utc).isoformat()
    RESULTS_FILE.write_text(json.dumps(existing, indent=2, default=str) + "\n", encoding="utf-8")


def _format_percent(value: float | None) -> str:
    return "N/A" if value is None else f"{value * 100:.1f}%"


def render(
    daily: pd.DataFrame,
    pollutant: Pollutant,
    forecast_weather: pd.DataFrame | None,
    today,
    source_audit: dict[str, Any] | None = None,
    historical_forecast_weather: pd.DataFrame | None = None,
) -> dict[str, Any] | None:
    st.header(f"{pollutant.key} | {pollutant.source}")
    st.caption(
        f"Daily concentration in ug/m3. Exceedance threshold: {pollutant.threshold:.0f} ug/m3. "
        "A concentration is not an AQI value."
    )
    if pollutant.target not in daily:
        st.warning(f"{pollutant.key} data has not been collected for this source.")
        return None
    try:
        with st.spinner(f"Loading {pollutant.key} backtests and production model..."):
            analysis = analyze(
                daily, pollutant,
                historical_forecast_weather=historical_forecast_weather,
            )
    except (ValueError, KeyError) as exc:
        st.warning(f"{pollutant.key} analysis unavailable: {exc}")
        return None

    coverage = st.columns(4)
    coverage[0].metric("First date", str(analysis["start"]))
    coverage[1].metric("Latest complete date", str(analysis["cutoff"]))
    coverage[2].metric("Valid target days", f"{analysis['valid_days']}/{analysis['calendar_days']}")
    coverage[3].metric("Production training rows", analysis["training_rows"])
    st.caption(
        f"Target: {pollutant.source} {pollutant.key}. The five-calendar-year range includes "
        "only dates with a measured or modeled target; missing targets are never filled."
    )

    st.subheader("Untouched 180-day test")
    st.write(
        f"Training: {analysis['final_train_rows']} usable dates ending before "
        f"{analysis['final_start']}. "
        f"Scored {analysis['final_scored_days']} of 180 calendar dates. "
        f"Method chosen from earlier seasonal windows: {analysis['champion']}."
    )
    table = []
    persistence_mae = analysis["final_metrics"]["Persistence"]["mae"]
    for method, metric in analysis["final_metrics"].items():
        table.append({
            "Method": method,
            "MAE": round(metric["mae"], 2),
            "MAE improvement": f"{(persistence_mae - metric['mae']) / persistence_mae * 100:+.1f}%" if persistence_mae else "N/A",
            "RMSE": round(metric["rmse"], 2),
            "R2": None if metric["r2"] is None else round(metric["r2"], 3),
            "Exceedance rate": _format_percent(metric["exceedance_rate"]),
            "Recall": _format_percent(metric["recall"]),
            "Precision": _format_percent(metric["precision"]),
            "False-positive rate": _format_percent(metric["false_positive_rate"]),
        })
    st.dataframe(pd.DataFrame(table), hide_index=True, use_container_width=True)
    chart_columns = list(dict.fromkeys(["Actual", analysis["champion"], "Persistence"]))
    chart = analysis["final_results"].set_index("Date")[chart_columns].copy()
    chart[f"{pollutant.threshold:.0f} threshold"] = pollutant.threshold
    st.line_chart(chart)
    st.caption(analysis["weather_test_note"] + ".")

    with st.expander("Earlier rolling seasonal backtests"):
        dev = analysis["development"]
        if dev.empty:
            st.write("No earlier seasonal windows passed the data coverage rule.")
        else:
            medians = dev.groupby("Method", as_index=False)["MAE"].median()
            medians = medians.rename(columns={"MAE": "Median seasonal MAE"})
            medians["Median seasonal MAE"] = medians["Median seasonal MAE"].round(2)
            st.dataframe(medians, hide_index=True, use_container_width=True)
            st.caption(
                "Seasonal windows select the production method; their training histories "
                "overlap. They are development evidence, not independent live trials."
            )
            st.dataframe(
                dev.assign(
                    MAE=dev["MAE"].round(2),
                    Recall=dev["Recall"].map(_format_percent),
                    **{"Exceedance rate": dev["Exceedance rate"].map(_format_percent)},
                ),
                hide_index=True,
                use_container_width=True,
            )

    st.subheader("Forward-looking seven-day forecast")
    if analysis["champion"] == "Persistence":
        st.caption(
            "Production method: persistence. This recursive forecast carries forward "
            "the latest PM value; weather inputs do not change its point prediction."
        )
    else:
        st.caption(f"Production method: {analysis['champion']}.")
    forward = None
    if forecast_weather is not None and today is not None:
        try:
            forward = forecast(analysis, forecast_weather, today)
        except ValueError as exc:
            st.warning(str(exc))
    else:
        st.warning("Forecast weather is unavailable.")
    if forward is not None and not forward.empty:
        display = forward.copy()
        display["Prediction"] = display["Prediction"].round(1)
        display["Lower_80"] = display["Lower_80"].round(1)
        display["Upper_80"] = display["Upper_80"].round(1)
        st.dataframe(display, hide_index=True, use_container_width=True)
        st.line_chart(forward.set_index("Date")[["Prediction", "Lower_80", "Upper_80"]])
        exceedance_dates = [str(day) for day in forward.loc[forward["Exceeds"], "Date"]]
        if exceedance_dates:
            st.error(
                f"{pollutant.key} may exceed {pollutant.threshold:.0f} ug/m3 on "
                + ", ".join(exceedance_dates)
            )
        else:
            st.success(f"No {pollutant.key} exceedance is predicted in the next seven days.")
        st.caption(
            "The effective lead includes any bridge from the latest completed target. "
            "Intervals are empirical only when at least five recursive validation windows "
            "exist for that lead."
        )
    save_result(
        analysis,
        {
            "forecast_dates": [str(day) for day in forward["Date"]] if forward is not None else [],
            "forecast": [
                {
                    "date": str(row.Date), "effective_lead": int(row.Lead),
                    "prediction_ug_m3": float(row.Prediction),
                    "lower_80_ug_m3": None if pd.isna(row.Lower_80) else float(row.Lower_80),
                    "upper_80_ug_m3": None if pd.isna(row.Upper_80) else float(row.Upper_80),
                    "exceeds_threshold": bool(row.Exceeds),
                }
                for row in forward.itertuples(index=False)
            ] if forward is not None else [],
            "source_audit": source_audit,
        },
    )
    analysis["forward_ready"] = forward is not None and not forward.empty
    return analysis
