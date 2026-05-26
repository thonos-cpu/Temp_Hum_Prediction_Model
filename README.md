# IoT Sensor Data Preprocessing, Feature Engineering, and Forecasting

This repository contains a Jupyter notebook for cleaning, enriching, analyzing, and forecasting time-series data collected from an IoT environmental sensing setup. The workflow is centered on measurements stored in MongoDB and produces a cleaned sensor dataset, engineered features, correlation diagnostics, visual reports, and forecasting benchmarks for temperature, humidity, and light.

The notebook file is:

```text
preproccess.ipynb
```

## What The Notebook Does

The notebook implements an end-to-end data science pipeline for IoT telemetry. It starts by loading raw sensor records from MongoDB, removes known invalid readings, repairs irregular sampling intervals, filters statistical outliers, generates predictive features, evaluates feature usefulness, and trains time-series forecasting models.

At a high level, the workflow performs the following stages:

1. Connects to a MongoDB collection using environment variables.
2. Loads sensor documents into a Pandas DataFrame.
3. Removes MongoDB `_id` values from the analysis dataset.
4. Converts timestamps into timezone-aware datetime values.
5. Removes readings after the known sensor failure cutoff.
6. Removes extreme raw light and solar readings above the configured threshold.
7. Detects suspicious timestamp gaps and duplicate-like measurements.
8. Cleans timestamp spacing and interpolates missing numeric readings.
9. Removes outliers using z-score filtering.
10. Removes additional outliers using an IQR-based rule.
11. Re-interpolates after outlier removal to restore a regular time series.
12. Plots cleaned sensor signals.
13. Computes initial sensor correlations.
14. Engineers rolling, lag, delta, diurnal, statistical, cross-sensor, and IQR features.
15. Evaluates feature importance with correlation, ANOVA F-score, mutual information, and random forest importance.
16. Saves correlation and diagnostic plots.
17. Builds future targets for temperature, humidity, and light.
18. Trains a multi-output linear regression model with time-series cross-validation.
19. Saves fold-level prediction plots and model metrics.
20. Runs SARIMAX forecasting with exogenous variables.
21. Exports SARIMAX evaluation results to Excel.

## Data Source

The notebook expects the source data to be stored in a MongoDB collection. Each document should represent one sensor measurement.

The expected fields are:

| Column | Type | Description |
| --- | --- | --- |
| `timestamp` | datetime or datetime-like string | Measurement time |
| `light_raw` | numeric | Raw light sensor reading |
| `temperature_c` | numeric | Temperature in Celsius |
| `humidity_percent` | numeric | Relative humidity percentage |
| `solar_raw` | numeric | Raw solar sensor reading |
| `battery_v` | numeric | Sensor battery voltage |

The notebook reads all documents from the configured collection and excludes `_id`:

```python
cursor = input.find({}, {"_id": 0})
df = pd.DataFrame(list(cursor))
```

## Important Safety Note

One notebook cell modifies the MongoDB source collection:

```python
input.delete_many(time_cutoff)
```

This deletes records from MongoDB after the configured cutoff timestamp. If you are running the notebook on your own dataset, either:

1. Run it against a copied collection, or
2. Remove/comment out the `input.delete_many(time_cutoff)` line before execution.

For reproducible research, the safest approach is to preserve the raw collection untouched and perform all deletions only inside the DataFrame.

## Local Setup

### 1. Install Python

Use Python 3.10 or newer.

Check your version:

```bash
python --version
```

If Python is not installed, download it from:

```text
https://www.python.org/downloads/
```

### 2. Clone Or Download This Repository

If the project is hosted on GitHub:

```bash
git clone <https://github.com/thonos-cpu/Temp_Hum_Prediction_Model>
cd <Temp_Hum_Prediction_Model>
```

If you downloaded a ZIP file, extract it and open a terminal inside the project folder.

### 3. Create A Virtual Environment

On Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

On macOS or Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 4. Install Required Packages

Install the libraries used by the notebook:

```bash
pip install jupyter pymongo python-dotenv matplotlib pandas numpy scipy seaborn scikit-learn statsmodels openpyxl
```

Optional but recommended:

```bash
pip install notebook ipykernel
```

Register the virtual environment as a Jupyter kernel:

```bash
python -m ipykernel install --user --name iot-preprocessing --display-name "Python (IoT Preprocessing)"
```

## MongoDB Setup

### Option A: Use A Local MongoDB Server

Install MongoDB Community Server from:

```text
https://www.mongodb.com/try/download/community
```

Start MongoDB locally. The default host and port are usually:

```text
host: localhost
port: 27017
```

### Option B: Use MongoDB Atlas

If you use MongoDB Atlas, configure your connection details in the same environment file described below. You may need to adapt the URI construction cell in the notebook if your Atlas connection string uses the `mongodb+srv://` format.

## Environment Configuration

Create a file named `credentials.env` in the same folder as `preproccess.ipynb`.

For a local MongoDB server without authentication:

```env
MONGO_HOST=localhost
MONGO_PORT=27017
MONGO_DB=your_database_name
MONGO_COLLECTION=your_collection_name
MONGO_USERNAME=
MONGO_PASSWORD=
```

For a MongoDB server with username and password:

```env
MONGO_HOST=localhost
MONGO_PORT=27017
MONGO_DB=your_database_name
MONGO_COLLECTION=your_collection_name
MONGO_USERNAME=your_username
MONGO_PASSWORD=your_password
```

The notebook loads this file with:

```python
load_dotenv("credentials.env")
```

## Preparing Your Own Dataset

To run the notebook on your own sensor data, your MongoDB collection should contain one document per measurement. A minimal example document looks like this:

```json
{
  "timestamp": "2026-05-17T16:08:45.889Z",
  "light_raw": 42.0,
  "temperature_c": 23.4,
  "humidity_percent": 51.2,
  "solar_raw": 37.0,
  "battery_v": 2.91
}
```

Before running the notebook, make sure that:

1. `timestamp` values are valid datetimes or strings Pandas can parse.
2. Numeric sensor fields do not contain text labels.
3. All required columns exist.
4. The collection contains enough chronological data for rolling windows and time-series splits.
5. The records are from one consistent sensing setup or from sensors that are comparable.

The notebook assumes a sampling interval around 20 seconds. If your device samples at a different rate, update the timestamp-cleaning thresholds:

```python
if diff < 17:
    continue
```

```python
if diff > 24:
    missing_times = pd.date_range(
        start=cur + pd.Timedelta(seconds=20),
        end=nxt - pd.Timedelta(seconds=17),
        freq="20s"
    )
```

For example, if your device samples every 60 seconds, you should adjust the duplicate threshold, missing-data threshold, and interpolation frequency accordingly.

## Dataset-Specific Parameters To Review

Before using the notebook with a new dataset, review these hard-coded assumptions.

### Sensor Failure Cutoff

The current notebook removes data after:

```python
2026-05-20 09:49:55.647 UTC
```

This was chosen for the original TelosB experiment because the sensor stopped working after battery drain. For your own data, change or remove this cutoff:

```python
cutoff = pd.Timestamp(
    year=2026,
    month=5,
    day=20,
    hour=9,
    minute=49,
    second=55,
    microsecond=647000,
    tz="UTC"
)
```

### Raw Light And Solar Threshold

The notebook removes readings where:

```python
light_raw >= 150
solar_raw >= 150
```

This was chosen because the original dataset had an abnormal spike above 1200. For another device, inspect your sensor range before keeping this rule.

### Outlier Rules

The notebook applies:

1. z-score filtering with threshold `3`.
2. IQR filtering with multiplier `2.2`.

These are reasonable general-purpose filters, but they can remove valid rare events. If rare spikes matter in your domain, tune these thresholds or disable one of the filters.

### Forecast Horizon

The linear regression forecasting section uses:

```python
horizon = 100
```

Because the cleaned data is approximately 20-second sampled, this predicts roughly:

```text
100 * 20 seconds = 2000 seconds = about 33 minutes
```

If your sampling interval changes, the real-world prediction horizon changes too.

## Running The Notebook

Start Jupyter:

```bash
jupyter notebook
```

Then open:

```text
preproccess.ipynb
```

Select the kernel:

```text
Python (IoT Preprocessing)
```

Run the notebook from top to bottom.

In Jupyter Notebook, use:

```text
Kernel > Restart & Run All
```

In JupyterLab, use:

```text
Run > Restart Kernel and Run All Cells
```

## Generated Files

The notebook may generate the following artifacts.

### Cached Pipeline Output

```text
pipeline_results.pkl
```

This stores feature matrices, correlation matrices, and feature scoring results so the expensive feature-analysis pipeline does not need to be recomputed every time.

### Correlation And Diagnostic Plots

```text
01_raw_timeseries.png
02_corr_heatmap_pearson.png
02_corr_heatmap_spearman.png
03_sensor_corr_heatmap.png
05_diurnal_boxplots.png
```

These visualize raw time-series behavior, feature correlations, raw sensor correlations, and hourly sensor distributions.

### Linear Regression Forecasting Outputs

For each target and fold, the notebook saves prediction plots such as:

```text
target_temp_fold_1_pred.png
target_humidity_fold_1_pred.png
target_light_fold_1_pred.png
```

It also saves full prediction summaries:

```text
target_temp_predicted_summary.png
target_humidity_predicted_summary.png
target_light_predicted_summary.png
```

The regression metrics are exported to:

```text
time_series_metrics.xlsx
```

### SARIMAX Outputs

SARIMAX fold-level scores are exported to:

```text
sarimax.xlsx
```

## Feature Engineering Details

The notebook creates several feature families.

### Rolling Features

For each sensor column, rolling features are computed over time windows:

```text
15 minutes
60 minutes
360 minutes
```

For each window, the notebook computes:

```text
rolling mean
rolling standard deviation
rolling range
```

### Lag Features

Lag features preserve previous measurements:

```text
lag 1
lag 5
lag 15
```

These help the model learn temporal dependence.

### Delta Features

The notebook computes first-order and second-order changes:

```text
delta1
delta2
```

These capture short-term movement and acceleration in the signal.

### Diurnal Features

The notebook encodes time-of-day behavior with:

```text
hour_sin
hour_cos
day_of_week
is_daytime
hour_bin
```

The sine and cosine features preserve the circular nature of daily time.

### Statistical Features

For one-hour rolling windows, the notebook computes:

```text
skewness
kurtosis
coefficient of variation
```

These describe distribution shape and local volatility.

### Cross-Sensor Features

The notebook derives physically motivated features:

```text
heat_index_proxy
solar_efficiency
battery_drain_rate
humidity_temp_interaction
```

These combine sensors into higher-level indicators.

### IQR Features

Rolling interquartile range features are computed for multiple windows to quantify local spread and variability.

## Modeling Approach

### Linear Regression With Time-Series Cross-Validation

The notebook predicts future values of:

```text
temperature_c
humidity_percent
light_raw
```

using a multi-output linear regression model.

The targets are shifted forward by `horizon = 100`, and the data is evaluated with:

```python
TimeSeriesSplit(n_splits=6)
```

This preserves chronological ordering and avoids training on future data.

Metrics include:

```text
MAE
MSE
RMSE
Train R2
Test R2
Median Absolute Error
Explained Variance
Max Error
Residual Mean
Overfitting gap
```

### SARIMAX With Exogenous Variables

The notebook also evaluates SARIMAX models for:

```text
temperature_c
humidity_percent
light_raw
```

The exogenous variables are:

```text
solar_raw
battery_v
time_sin
time_cos
temp_lag_1
humidity_lag_1
light_lag_1
```

The configured SARIMAX order is:

```python
order=(2, 0, 1)
```

The SARIMAX evaluation uses:

```python
TimeSeriesSplit(n_splits=5, test_size=300)
```

and reports:

```text
MAE
MSE
RMSE
R2
```

## Recommended Workflow For New Experiments

For a clean experiment on a new dataset:

1. Import your raw measurements into a MongoDB collection.
2. Back up the raw collection.
3. Create `credentials.env`.
4. Open the notebook.
5. Remove or modify the original experiment cutoff.
6. Adjust sampling-interval thresholds if your device does not sample every 20 seconds.
7. Inspect raw plots before keeping the outlier thresholds.
8. Run preprocessing cells.
9. Validate the cleaned signal plots.
10. Run feature engineering and correlation analysis.
11. Review the selected features and correlation heatmaps.
12. Run forecasting models.
13. Compare regression and SARIMAX metrics.
14. Use exported plots and Excel files for reporting.

## Reproducibility Notes

The notebook uses deterministic settings where applicable, including:

```python
random_state=42
```

for mutual information and random forest feature scoring.

Some results may still vary across library versions, especially SARIMAX optimization results and floating-point numerical routines. For publishable experiments, record your package versions:

```bash
pip freeze > requirements-lock.txt
```

## Troubleshooting

### `ModuleNotFoundError`

Install the missing package:

```bash
pip install <package-name>
```

For the notebook as written, the most important packages are:

```text
pymongo
python-dotenv
pandas
numpy
matplotlib
scipy
seaborn
scikit-learn
statsmodels
openpyxl
```

### MongoDB Connection Fails

Check that:

1. MongoDB is running.
2. `MONGO_HOST` and `MONGO_PORT` are correct.
3. The database name exists.
4. The collection name exists.
5. Username and password are correct if authentication is enabled.
6. Your IP address is allowed if using MongoDB Atlas.

### Empty DataFrame

If `df` has zero rows:

1. Confirm the collection contains documents.
2. Confirm the environment variables point to the right database and collection.
3. Confirm the notebook connects to the expected MongoDB instance.

### Timestamp Errors

If timestamp parsing fails, normalize your timestamps before import or adjust:

```python
pd.to_datetime(df["timestamp"], utc=True)
```

### Too Many Rows Dropped

If the cleaned dataset becomes too small, review:

1. The sensor failure cutoff.
2. The `light_raw < 150` and `solar_raw < 150` filters.
3. The z-score threshold.
4. The IQR multiplier.
5. The rolling-window sizes.
6. The forecast horizon.

## Suggested Repository Structure

A clean GitHub version of this project could use:

```text
.
├── README.md
├── preproccess.ipynb
├── requirements.txt
├── .gitignore
└── examples/
    └── sample_document.json
```

Do not commit real credentials. Add this to `.gitignore`:

```gitignore
credentials.env
*.pkl
*.xlsx
*.png
.venv/
__pycache__/
```

## Project Summary

This notebook is a complete IoT time-series preparation and forecasting pipeline. It is especially suited for environmental sensor data where measurements arrive at near-regular intervals, contain occasional missing packets, and benefit from temporal feature engineering. The result is a reproducible analysis path from raw MongoDB records to cleaned signals, interpretable features, visual diagnostics, and model evaluation artifacts.
