# Presentation guide: Urban Air Quality Early Warning System

Use this for the 10 October 2026, 11:00 AM IST presentation. Figures below are the saved 9 October snapshot in `results.json`. The live app can refresh; if it does, use the numbers shown on screen and state the new cutoff date. Allow 6-8 minutes for slides and 2 minutes for the demo.

## The central claim

We built a reproducible two-pollutant forecasting prototype. It distinguishes modeled CAMS concentrations from R.K. Puram station observations, audits the two station archives before merging them, tests forecasts chronologically against simple baselines, and issues a seven-day outlook with source and lead-time labels. The strongest result is station PM10. The selected Random Forest reduced final-test MAE by 26.2% versus persistence. Complexity did not consistently win for the other three source-pollutant combinations.

Do not say that we have five complete years of measurements, that CAMS is CPCB ground truth, that the app has proven prospective accuracy, or that its warnings are official health alerts.

## Slide 1: Problem and question

**On slide:** Urban Air Quality Early Warning System. PM2.5 and PM10 daily concentration forecasts for Delhi. Research question: Can a model improve on simply carrying forward yesterday's value?

**Say:** "We forecast next-day daily particulate concentration and flag possible exceedances. The project tests whether weather, pollution history, and calendar information beat a simple persistence forecast. We evaluate modeled city-level data and a fixed observed station separately."

## Slide 2: Data sources and units

**On slide:** Two targets, never mixed. CAMS modeled PM2.5/PM10 from [Open-Meteo Air Quality](https://open-meteo.com/en/docs/air-quality-api). Observed R.K. Puram DPCC readings compiled by [XKDR](https://airquality.xkdr.org/) and the [OpenAQ archive](https://docs.openaq.org/aws/about). Weather from Open-Meteo. Target unit: ug/m3, not AQI.

**Say:** "CAMS is an atmospheric model, not a station measurement. The R.K. Puram series is our field-observation analysis. Open-Meteo weather supplies predictors. The alert thresholds are India's 24-hour concentration standards: PM2.5 above 60 and PM10 above 100 ug/m3, from the [CPCB NAAQS table](https://cpcb.gov.in/upload/NAAQS_2019.pdf). These are not AQI values."

## Slide 3: Five-year coverage and station audit

| Source | Calendar span | PM2.5 valid days | PM10 valid days |
| --- | --- | ---: | ---: |
| CAMS modeled | 9 Oct 2021 to 8 Oct 2026 | 1,526 / 1,826 | 1,526 / 1,826 |
| R.K. Puram station | 6 Oct 2021 to 5 Oct 2026 | 1,610 / 1,826 | 1,594 / 1,826 |

**Say:** "Five years describes the calendar span. CAMS global history begins in August 2022, so we do not have five complete years of CAMS targets. For the station, we combine XKDR's earlier records with OpenAQ's later records only after checking every matched hour in their overlap. There were 6,167 matched hours. Within 1 ug/m3 agreement was 98.9% for PM2.5 and 97.7% for PM10, above our 95% gate. Both median signed differences were zero. We also report unmatched hours. OpenAQ's raw unit text has encoding damage, so exact unit-string equivalence could not be checked."

**Quality rules:** CAMS needs 24 numeric hourly readings per day for both targets and required weather. Each station pollutant needs 18 distinct valid local hours. Missing pollutant targets are never filled. Dates use Asia/Kolkata local time. SQLite caches keep repeated launches from redownloading the archive.

## Slide 4: Features and models

**On slide:** Input groups: weather, prior PM, calendar. Candidates: persistence, seven-day trailing mean, day-of-year climatology, Ridge, Random Forest.

**Say:** "The model has 21 features. Weather includes temperature, humidity, wind speed and direction, rain, pressure, cloud cover, gusts, and boundary-layer height. Pollution history uses exact calendar-day lags of 1, 2, 3, and 7 days plus a complete prior seven-day mean. Calendar variables cover season, weekend, Indian public holidays, and Diwali proximity. A missing calendar day never becomes a fake 'yesterday.' We impute missing predictor values from training data only. Isolation Forest flags unusual training rows; if too few normal rows remain, fitting falls back to all usable training rows. Ridge and Random Forest are trained on the resulting training rows."

**Do not claim:** Measured traffic is a feature. Holiday and weekend values are activity proxies. NASA FIRMS fire detections can be cached for research but do not enter this five-year model. Isolation Forest cannot prove an anomaly was caused by crop burning or fireworks.

## Slide 5: Leakage controls and scoring

**On slide:** Earlier rolling seasonal windows select a method. The latest 180 calendar days form a separate chronological test. Compare every method on identical scored dates. Seven-day forecasts are recursive.

**Say:** "We never random-shuffle dates. Each seasonal model fits its imputer, anomaly detector, scaler, and estimator on dates before its holdout. Earlier windows choose the method; the latest 180-day period does not choose it. One-day-ahead test predictions use the true previous calendar day's PM, which is information a live next-day forecast would have. The seven-day forecast instead feeds each prediction into the next day's lag. For the final test, weather comes from Open-Meteo's [stitched historical-forecast archive](https://open-meteo.com/en/docs/historical-forecast-api). That is closer to operational weather than reanalysis, but it is not a fixed day-ahead issue-time forecast."

**Metric definitions:** MAE is average absolute error in ug/m3, lower is better. RMSE penalizes large misses. R2 describes explained variation. Exceedance rate is the fraction of actual test dates above the threshold. Recall is the fraction of real exceedances flagged. False-positive rate is the fraction of non-exceedance dates falsely flagged. A high recall needs the exceedance base rate beside it.

**Integrity note:** Code excludes the final 180 days from fitting and method selection. We did inspect those results while developing the project, so do not call this a pristine one-time blind test. Only later append-only live evaluation could establish prospective accuracy.

## Slide 6: Final test results

| Target | Scored dates | Selected method | Selected MAE | Persistence MAE | Random Forest MAE |
| --- | ---: | --- | ---: | ---: | ---: |
| CAMS PM2.5 | 180 / 180 | Persistence | 16.62 | 16.62 | 16.91 |
| CAMS PM10 | 180 / 180 | Persistence | 86.99 | 86.99 | 95.62 |
| Station PM2.5 | 131 / 180 | Persistence | 12.12 | 12.12 | 11.52 |
| Station PM10 | 133 / 180 | Random Forest | 27.61 | 37.41 | 27.61 |

**Say:** "Station PM10 is the clear positive result. Its selected Random Forest reduced MAE from 37.41 to 27.61 ug/m3, a 26.2% improvement, and achieved R2 of 0.732. Exceedance recall was 92.3%, precision 88.4%, and false-positive rate 26.2% at a 68.4% exceedance base rate. For the CAMS targets, persistence won the selection gate. This is an honest result: more features do not guarantee a better forecast."

**If asked about station PM2.5:** Random Forest happened to have a lower MAE on the final test, 11.52 versus 12.12, but it improved only 7 of 10 earlier seasonal windows. The preset promotion rule needed at least 8 of 10, so the app kept persistence. We did not change the choice after seeing the test result.

**Do not compare PM2.5 and PM10 MAE directly:** Their concentration scales differ. Compare methods within a target.

## Slide 7: Live forecast and demo

**On slide:** Screenshot or live app at <http://localhost:8501/>. Show source, latest complete target date, selected method, effective lead, seven-day table, and threshold warning.

**Say:** "The app refreshes available history, reuses cached model evidence, and forecasts tomorrow through day seven. If the latest station observation is several days old, it predicts the intervening bridge internally. The 'effective lead' counts that bridge. If the station is more than five days old or unavailable, the app displays a CAMS outlook instead. The model is not an official CPCB alert."

**Demo clicks:** Open the app. Show the four-row Source coverage table first. Select PM10 and R.K. Puram station. Scroll to the 133-day final test and point to the Random Forest and persistence rows. Scroll to the forecast and point to the effective lead. Expand the XKDR/OpenAQ audit. Switch to PM2.5 or CAMS modeled to show honest persistence selection. Avoid opening the separate live CPCB observation panel unless asked; data.gov.in may be unavailable without affecting these analyses.

**Snapshot only:** The saved 9 October results predict station PM10 at 168.0 ug/m3 for 10 October and station PM2.5 at 60.3 ug/m3. These values are not fixed presentation claims. Verify the current live date and value before speaking. If the station remains at 5 October on 10 October, the first displayed station forecast has effective lead 6, not 1.

## Slide 8: Limits and next steps

**Say:** "This is a tested prototype, not a regulatory forecast. CAMS is modeled data. The station has missing days, leaving only 131 and 133 scored dates in its 180-day test. Seasonal training histories overlap. Stitched historical weather does not reproduce one frozen next-day forecast. We need prospective, timestamped forecasts paired later with observed outcomes before claiming live accuracy. Next steps are a longer prospective evaluation, a proper archive of issue-time weather forecasts, and better station and fire coverage."

## Questions to rehearse

1. **Why five years but only 1,526 CAMS days?** The five-year range is a calendar window. The CAMS global source begins in August 2022. We show valid-day counts and never invent earlier targets.
2. **Why does the station test score only 131/133 of 180 days?** A target and the true preceding calendar day's PM are both required. Missing or incomplete station days cannot be imputed into a trustworthy test.
3. **How do you avoid leakage?** Chronological splits, date-aware lag joins, and training-only imputation, anomaly detection, scaling, and fitting. The code never fits on final-test targets.
4. **Why did PM2.5 keep persistence when Random Forest looked better on the final test?** Selection came from earlier windows. Its 7/10 seasonal wins missed the 8/10 consistency gate. Test results did not change the selected method.
5. **Does Isolation Forest identify crop burning?** No. It finds statistical outliers. Their causes need independent evidence.
6. **Is this AQI?** No. It predicts daily PM concentration in ug/m3 and compares that concentration to 24-hour standards.
7. **Does the seven-day PM2.5 forecast use weather?** If persistence is selected, its point forecast carries forward the last PM value and does not respond to weather. The candidate weather models are still evaluated; the app does not force them into production when they fail the gate.
8. **Is the seven-day accuracy proven?** No. Recursive historical residuals support indicative intervals, but we lack a long prospective record of forecasts issued before outcomes occurred.
9. **Why trust the combined station sources?** The 6,167-hour overlap passed a preset numerical agreement gate for both pollutants. Unmatched hours and unit-text encoding problems remain disclosed.
10. **What if APIs fail during the demo?** The local SQLite caches support the app. The committed `results.json` holds four coverage and metric summaries plus seven forecast rows per mode. A fresh clone without caches still needs internet for the first collection.

## Morning checklist

1. On the presentation laptop, confirm the repository, virtual environment, local SQLite caches, and video asset are present. Run `python -m streamlit run app.py` and open <http://localhost:8501/>.
2. Before 8:30 AM IST, load all four mode combinations once. Note each latest complete date. Run `python -m pytest -q -p no:cacheprovider`.
3. If refreshed values differ from this guide, read the live `results.json` and update the one results slide. Do not combine an old slide number with a new app cutoff.
4. Keep `results.json` and this guide open locally. Do not rely on the live CPCB data.gov.in feed for the demo. Do not show API keys.
