# Initial Project Plan: Urban Air Quality Early Warning System

## Objective

The project will develop a Streamlit-based system to predict daily PM2.5 concentrations for the next seven days and warn when the predicted value exceeds 60 ug/m3.

## Data Sources

| Data | Source | Planned Use |
|---|---|---|
| Historical and forecast PM2.5 | Open-Meteo Air Quality API using CAMS data | Main target for training, backtesting, and forecasting |
| Historical and forecast weather | Open-Meteo Historical Weather and Forecast APIs | Temperature, humidity, wind, rainfall, pressure, cloud cover, and boundary-layer information |
| Ground-station PM2.5 observations | CPCB dataset through data.gov.in | Separate comparison with real monitoring-station observations |
| Satellite thermal detections | NASA FIRMS VIIRS API | Optional indicator of nearby fire or burning activity |
| Holidays and festivals | Indian calendar data generated with the `holidays` Python package | Proxy for changes in traffic and human activity |

## Proposed Method

1. Collect hourly PM2.5 and weather data for the selected city coordinates.
2. Clean nonnumeric values and accept only dates with complete hourly observations.
3. Convert hourly data into daily values and store it locally in SQLite.
4. Create weather, seasonal, festival, fire, PM2.5 lag, and rolling-average features.
5. Use Isolation Forest to identify unusual pollution events.
6. Train a Random Forest model using chronological training data.
7. Test the frozen model on later dates that were not used during training.
8. Compare its predictions with a persistence baseline where tomorrow's PM2.5 equals today's value.
9. Generate a recursive seven-day forecast using forecast weather.

## How Findings Will Be Obtained

The findings will come from leakage-free seasonal backtests and future recorded forecasts. Performance will be measured using MAE, RMSE, R2, exceedance recall, precision, and false-positive rate. Different feature groups will be compared to determine whether pollution history, expanded weather, calendar events, or fire activity improve predictions.

## Expected Output

The final output will be an interactive Streamlit dashboard containing historical evaluation results, model comparisons, seven-day PM2.5 forecasts, uncertainty ranges, and threshold warnings. It will be presented as an educational forecasting prototype rather than an official CPCB health-alert system.
