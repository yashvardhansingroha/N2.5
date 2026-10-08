# Urban Air Quality Early Warning System

Local Streamlit analysis for PM2.5 and PM10. It keeps CAMS modeled concentrations and R.K. Puram station observations as separate targets.

## Run from Git

```powershell
git clone https://github.com/yashvardhansingroha/N2.5.git
cd N2.5
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

Open <http://localhost:8501>. No API key is required for the CAMS and public-archive station modes. First launch downloads historical data and can take several minutes. Later runs use local SQLite caches. Select **R.K. Puram station** to fetch and audit the observed archive; if that source fails, the app retains the CAMS outlook.

An optional `DATA_GOV_IN_API_KEY` can be put in an untracked `.streamlit/secrets.toml` for a separate live CPCB city reading. The optional `FIRMS_MAP_KEY` caches satellite thermal detections for later research; it is **not** a feature in the current five-year comparison. Do not commit keys or local caches.

## Data provenance

- [Open-Meteo Air Quality API](https://open-meteo.com/en/docs/air-quality-api): CAMS modeled hourly PM2.5 and PM10. This is not a CPCB measurement. CAMS coverage currently starts in August 2022, so its five-year calendar span has fewer valid days.
- [Open-Meteo Historical Weather API](https://open-meteo.com/en/docs/historical-weather-api): training weather at the relevant coordinates.
- [Open-Meteo Historical Forecast API](https://open-meteo.com/en/docs/historical-forecast-api): stitched near-term forecast weather for the final test. It is not a fixed next-day forecast issued at a specific hour.
- [XKDR India Air Quality API](https://airquality.xkdr.org/): CPCB CAAQM R.K. Puram hourly observations through its last listed station date. Requests stay below the public demo per-query row cap.
- [OpenAQ archive](https://docs.openaq.org/aws/about): later R.K. Puram DPCC observations, public S3 location 17.

The station sources are merged only after an hourly overlap audit passes: at least 95% of matched values within 1 ug/m3 for each pollutant and median signed difference no larger than 1 ug/m3. Each pollutant needs at least 18 distinct valid local hours for a daily target. Missing target days are not imputed. The audit records unmatched hours and a raw-unit encoding caveat in `results.json`.

## Evaluation

The five-year calendar window ends on the latest complete target date for each source. Earlier rolling seasonal windows select among persistence, seven-day mean, training-only day-of-year climatology, Ridge, and Random Forest. The latest 180 calendar days are then held out; every method is scored on the same available dates. The app reports MAE, RMSE, R2, exceedance base rate, recall, precision, and false-positive rate. Undefined ratios show `N/A`.

Historical PM lags use exact preceding calendar dates. Weather imputation, anomaly filtering, scaling, and model fitting learn from training dates only. The seven-day forecast is recursive, with a maximum five-day bridge from the latest station observation. The displayed lead includes that bridge. Empirical intervals come from recursive validation residuals, not a calibrated clinical or regulatory guarantee.

Open-Meteo's historical-forecast sequence improves the weather proxy for the final holdout, but it does not reproduce a fixed next-day issue time. Seasonal windows have overlapping training histories. This is a research demo, not an official health alert. `results.json` is the offline metrics and coverage summary; the raw caches and keys are intentionally excluded from Git.

## Verify

```powershell
python -m py_compile app.py pollutant_pipeline.py station_archive.py archived_forecast_weather.py
python -m pytest -q -p no:cacheprovider
```
